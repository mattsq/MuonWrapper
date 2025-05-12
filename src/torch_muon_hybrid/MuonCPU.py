# muon_cpu.py
import torch
from torch import Tensor
from torch.optim import Optimizer

# ---------- Newton–Schulz helper (unchanged except for dtype) ----------
@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Return Z ≈ G⁰  (orthogonalised copy) via a quintic Newton-Schulz."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()                         # CPU bf16 is slow ⇒ keep fp32
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    I = None                              # not needed but keeps jit happy
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

# ------------------------------ Optimiser ------------------------------
class MuonCPU(Optimizer):
    r"""CPU-only rewrite of KellerJordan/Muon (2024-12-08).

    Args:
        params (iterable):   ≥2-D tensors to update.
        lr (float):          SGD learning rate (μP-scaled, default 0.02).
        weight_decay (float)
        momentum (float)
        nesterov (bool)
        ns_steps (int):      Newton–Schulz iterations (default 5).
    """
    def __init__(
        self, params, *, lr=0.02, weight_decay=0.01,
        momentum=0.95, nesterov=True, ns_steps=5
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # ------- momentum buffer -------
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - group["momentum"])      # buf = m·buf + (1-m)·g
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf

                # ------- orthogonalise update -------
                if g.ndim == 4:                            # conv kernel
                    flat_g = g.reshape(len(g), -1)
                    flat_g = zeropower_via_newtonschulz5(flat_g, group["ns_steps"])
                    g_precond = flat_g.view_as(g)
                elif g.ndim >= 2:
                    g_precond = zeropower_via_newtonschulz5(g, group["ns_steps"])
                else:
                    # 0-/1-D tensors shouldn't be here by design
                    g_precond = g

                # ------- weight decay + update -------
                p.mul_(1.0 - group["lr"] * group["weight_decay"])
                scaling = max(1, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(g_precond, alpha=-group["lr"] * scaling)

        return loss
