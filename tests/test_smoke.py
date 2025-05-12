import torch, torch_muon_hybrid as tmh

def test_forward_step():
    m = torch.nn.Linear(4, 4)
    opt = tmh.MuonHybrid(m.parameters())
    x = torch.randn(2, 4)
    loss = m(x).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
