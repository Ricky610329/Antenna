"""Round 116 — 3-bit phase at inc=0 (does it fix R110 catastrophic?)."""

import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config
from methodology_demo import worst_case_loss, evaluate_metrics

config.device = "cuda:0"

n = 51
freq = 38e9
main_lo, main_hi = 162, 192
gd_steps = 1500
n_restarts = 5

def quantize(params, n_levels):
    phase = (params * torch.pi) % (2 * torch.pi)
    if n_levels == 0:
        return phase / torch.pi  # continuous
    levels = torch.linspace(0, 2*torch.pi * (n_levels-1)/n_levels, n_levels, device=params.device)
    dist = torch.stack([torch.minimum(torch.abs(phase - l), 2*torch.pi - torch.abs(phase - l)) for l in levels])
    idx = dist.argmin(0)
    return levels[idx] / torch.pi

def deploy(inc, n_levels, rw):
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    best = None
    seed_results = []
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=rw)
            loss.backward()
            opt.step()
        with torch.no_grad():
            quantized = quantize(params, n_levels)
            resp_q = sim(quantized)["response"].cpu().numpy()
        m = evaluate_metrics(resp_q, main_lo, main_hi)
        seed_results.append(m)
        if best is None or m["worst_supp"] > best["worst_supp"]:
            best = m
    flats = sum(1 for r in seed_results if r['flat_top_compliant'])
    return best, flats

print("=" * 75)
print("R116 — Does 3-bit phase fix inc=0 catastrophic?")
print("=" * 75)

trials = [
    {"name": "1bit_inc0_rw2 (R110)",   "n_levels": 2, "inc": 0,  "rw": 2.0},
    {"name": "1bit_inc0_rw5 (R111)",   "n_levels": 2, "inc": 0,  "rw": 5.0},
    {"name": "2bit_inc0_rw2",           "n_levels": 4, "inc": 0,  "rw": 2.0},
    {"name": "3bit_inc0_rw2 (NEW)",     "n_levels": 8, "inc": 0,  "rw": 2.0},
    {"name": "3bit_inc51_rw2 (sweet ref)","n_levels":8,"inc":51, "rw": 2.0},
]

print(f"\n{'config':<35} | {'best worst':>10} | {'best ripple':>11} | {'flat-top':>9}")
print("-" * 75)
for t in trials:
    best, flats = deploy(t["inc"], t["n_levels"], t["rw"])
    print(f"{t['name']:<35} | {best['worst_supp']:>+10.2f} | {best['main_ripple']:>11.2f} | {flats}/5")
