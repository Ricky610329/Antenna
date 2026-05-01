"""Round 115 — 3-bit phase (8 levels)."""

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
inc = 51.0
main_lo, main_hi = 162, 192
gd_steps = 1500
n_restarts = 5

def quantize_to_levels(params, n_levels):
    """Quantize phase to n_levels evenly spaced in [0, 2π)."""
    phase = (params * torch.pi) % (2 * torch.pi)
    levels = torch.linspace(0, 2*torch.pi * (n_levels-1)/n_levels, n_levels, device=params.device)
    # Find closest level (with wrap-around)
    dist = torch.stack([
        torch.minimum(torch.abs(phase - l), 2*torch.pi - torch.abs(phase - l))
        for l in levels
    ])
    idx = dist.argmin(0)
    quantized_phase = levels[idx]
    return quantized_phase / torch.pi  # pattern = phase/π for sim

def deploy(quantize_fn, name, rw=2.0):
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    best = None
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
            quantized = quantize_fn(params)
            resp_q = sim(quantized)["response"].cpu().numpy()
        m = evaluate_metrics(resp_q, main_lo, main_hi)
        if best is None or m["worst_supp"] > best["worst_supp"]:
            best = m
    return best

print("=" * 70)
print("R115 — Complete phase resolution curve (1/2/3/4-bit + continuous)")
print("=" * 70)

results = []
for bits, n_levels in [(1, 2), (2, 4), (3, 8), (4, 16)]:
    b = deploy(lambda p, nl=n_levels: quantize_to_levels(p, nl), f"{bits}-bit")
    results.append({"bits": bits, "n_levels": n_levels, **b})
    print(f"\n{bits}-bit ({n_levels} levels): worst={b['worst_supp']:+.2f}, ripple={b['main_ripple']:.2f}")

# Continuous reference
def quant_cont(p):
    return ((p * torch.pi) % (2*torch.pi)) / torch.pi
b_cont = deploy(quant_cont, "continuous")
results.append({"bits": "cont", "n_levels": "inf", **b_cont})
print(f"\nContinuous: worst={b_cont['worst_supp']:+.2f}, ripple={b_cont['main_ripple']:.2f}")

print("\n" + "=" * 70)
print(f"{'bits':>5} | {'levels':>6} | {'best worst':>10} | {'ripple':>7} | {'Δ vs continuous':>16}")
print("-" * 70)
for r in results:
    delta = r['worst_supp'] - b_cont['worst_supp']
    print(f"{str(r['bits']):>5} | {str(r['n_levels']):>6} | {r['worst_supp']:>+10.2f} | "
          f"{r['main_ripple']:>7.2f} | {delta:>+16.2f}")
