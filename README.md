# MuonWrapper

## Introduction

This is a simple wrapper for Muon (https://github.com/KellerJordan/Muon) that does a couple of things:

1. Allows Muon to be used in CPU-only environments by re-writing KellerJordan's code to remove all the CUDA/distributed boilerplate
2. Provides a wrapper to allow Muon to be used once and a separate optimizer to be passed for parameters Muon won't be used for, rather than writing boilerplate inside the optimization loop.

There are good reasons not to do 1., but I've found Muon works nicely in other, nonGPT-environments so coded this up for me so I didn't have to keep re-pasting Muon.py everywhere.

Extremely vibe-coded, so use at your own etc. All actual work is credit to Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse and Jeremy Bernstein.

## Usage

```{python}
import torch, torch_muon_hybrid as tmh

def test_forward_step():
    m = torch.nn.Linear(4, 4)
    opt = tmh.MuonHybrid(m.parameters(), small_opt_cls=torch.optim.AdamW)
    x = torch.randn(2, 4)
    loss = m(x).mean()
    loss.backward()
    opt.step()
    opt.zero_grad()
```