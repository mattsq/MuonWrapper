# muon_hybrid.py
import torch
from typing import Iterable, Mapping, Any, Type
import os

# ----------------------------------------------------------------------
#  pick the concrete Muon implementation once, expose it as `Muon`
# ----------------------------------------------------------------------
_FORCE_CPU = os.getenv("MUON_HYBRID_FORCE_CPU", "0") == "1"

if not _FORCE_CPU and torch.cuda.is_available():
    try:
        # local, package-relative import
        from .MuonGPU import MuonGPU as Muon           # noqa: F401
    except (ImportError, RuntimeError):
        # GPU build not present / wrong compute capability / etc.
        from .MuonCPU import MuonCPU as Muon           # noqa: F401
else:
    from .MuonCPU import MuonCPU as Muon               # noqa: F401


# ----------------------------------------------------------------------
# 2.  Generic wrapper
# ----------------------------------------------------------------------
class MuonHybrid(torch.optim.Optimizer):
    r"""
    A composite optimiser that routes
      * tensors with ``ndim >= 2``  →  **Muon**
      * tensors with ``ndim  <  2`` →  *any* optimiser you pass in

    Parameters
    ----------
    params : iterable of Tensor
        Model parameters (mixed).
    small_opt_cls : Type[torch.optim.Optimizer], default ``torch.optim.AdamW``
        The optimiser to apply on 0-/1-D tensors.
    muon_kwargs : dict, optional
        Passed verbatim to ``Muon`` (e.g. ``lr``, ``momentum``).
        If omitted, Muon uses its own defaults (lr=0.02, …).
    small_opt_kwargs : dict, optional
        Passed verbatim to ``small_opt_cls`` (e.g. ``lr``, ``betas``, …).
    select_fn : callable, optional
        Custom splitter:  ``select_fn(param: Tensor) -> bool``.
        True → Muon, False → small optimiser.
        By default it uses ``param.ndim >= 2``.

    Example
    -------
    >>> opt = MuonHybrid(
    ...     model.parameters(),
    ...     small_opt_cls=torch.optim.SGD,
    ...     muon_kwargs={'lr': 0.02, 'momentum': 0.95},
    ...     small_opt_kwargs={'lr': 1e-2, 'momentum': 0.9}
    ... )
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        small_opt_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        muon_kwargs: Mapping[str, Any] | None = None,
        small_opt_kwargs: Mapping[str, Any] | None = None,
        select_fn=None,
    ):
        if muon_kwargs is None:
            muon_kwargs = {}
        if small_opt_kwargs is None:
            small_opt_kwargs = {}
        if select_fn is None:
            select_fn = lambda p: p.ndim >= 2      # default splitter

        # ---------------- split parameter iterables -----------------
        muon_params, small_params = [], []
        for p in params:
            (muon_params if select_fn(p) else small_params).append(p)

        # ---------------- instantiate the two child opts ------------
        self.muon = Muon(muon_params, **muon_kwargs) if muon_params else None
        self.small_opt = (
            small_opt_cls(small_params, **small_opt_kwargs) if small_params else None
        )

        # --------------- register with Optimizer base ---------------
        merged_defaults = muon_kwargs.copy()
        merged_defaults.update({f"small_{k}": v for k, v in small_opt_kwargs.items()})
        all_params = muon_params + small_params        # may be just one side
        super().__init__(all_params, merged_defaults)  # real init

        # Share the *actual* param_group dicts so schedulers can mutate lrs
        self.param_groups = []
        if self.muon:
            self.param_groups += self.muon.param_groups
        if self.small_opt:
            self.param_groups += self.small_opt.param_groups

        # Expose a merged view of .state (read-only)
        self.state = {**(self.muon.state if self.muon else {}),
                      **(self.small_opt.state if self.small_opt else {})}

        # keep profiler hook alive when present (PyTorch ≤2.2 Linux build)
        try:
            from torch.optim.optimizer import _hook_for_profile  # most wheels
        except (ImportError, AttributeError):
            # Windows wheels or future versions: profiler hook not exposed
            _hook_for_profile = lambda *args, **kwargs: None

        self._hook_for_profile = _hook_for_profile
    # ------------------------------------------------------------------
    #  standard API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        if self.muon:
            self.muon.step()
        if self.small_opt:
            self.small_opt.step()
        return loss

    def zero_grad(self, *args, **kwargs):
        if self.muon:
            self.muon.zero_grad(*args, **kwargs)
        if self.small_opt:
            self.small_opt.zero_grad(*args, **kwargs)

    # --- replace the whole add_param_group in hybrid.py -------------------
    def add_param_group(self, group):
        """
        Route *new* parameters to the correct child optimiser.
        Duplicate tensors are silently ignored (PyTorch would error).
        """
        params = list(group["params"])          # normalise

        # --------- utility to test membership fast -----------------------
        def _in(opt, p):
            return any(p is q for g in opt.param_groups for q in g["params"])

        # --------- split & deduplicate -----------------------------------
        muon_bucket, small_bucket = [], []
        for p in params:
            if p.ndim >= 2:
                if self.muon and not _in(self.muon, p):
                    muon_bucket.append(p)
            else:
                if self.small_opt and not _in(self.small_opt, p):
                    small_bucket.append(p)

        # --------- forward only truly new params -------------------------
        if muon_bucket:
            g = group.copy()
            g["params"] = muon_bucket
            self.muon.add_param_group(g)
            self.param_groups.append(self.muon.param_groups[-1])

        if small_bucket:
            g = group.copy()
            g["params"] = small_bucket
            self.small_opt.add_param_group(g)
            self.param_groups.append(self.small_opt.param_groups[-1])


    # ------------------------------------------------------------------
    #  (de-)serialization helpers
    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            "muon": self.muon.state_dict() if self.muon else None,
            "small_opt": self.small_opt.state_dict() if self.small_opt else None,
        }

    def load_state_dict(self, state_dict):
        if self.muon and state_dict["muon"]:
            self.muon.load_state_dict(state_dict["muon"])
        if self.small_opt and state_dict["small_opt"]:
            self.small_opt.load_state_dict(state_dict["small_opt"])

    # optional cosmetic repr
    def __repr__(self):
        parts = []
        if self.muon:
            parts.append("Muon:\n  " + repr(self.muon).replace("\n", "\n  "))
        if self.small_opt:
            parts.append(
                f"{self.small_opt.__class__.__name__}:\n  "
                + repr(self.small_opt).replace("\n", "\n  ")
            )
        return "MuonHybrid(\n  " + "\n  ".join(parts) + "\n)"
