"""Round 114 — 2-bit phase RIS (4 levels: 0, π/2, π, 3π/2)."""

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

def deploy_with_quantize(quantize_fn, name, rw=2.0):
    """Generic deploy with custom quantize."""
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
            quantized = quantize_fn(params)
            resp_q = sim(quantized)["response"].cpu().numpy()
        m = evaluate_metrics(resp_q, main_lo, main_hi)
        seed_results.append(m)
        if best is None or m["worst_supp"] > best["worst_supp"]:
            best = m
    return best, seed_results

def quantize_1bit(params):
    """1-bit: phase ∈ {0, π}, pattern ∈ {0, 1}."""
    phase = (params * torch.pi) % (2 * torch.pi)
    return ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()

def quantize_2bit(params):
    """2-bit: phase ∈ {0, π/2, π, 3π/2}, pattern ∈ {0, 0.5, 1, 1.5}."""
    phase = (params * torch.pi) % (2 * torch.pi)  # [0, 2π)
    # Snap to nearest of 4 levels
    levels = torch.tensor([0.0, torch.pi/2, torch.pi, 3*torch.pi/2], device=params.device)
    # Distance to each level (with wrap-around)
    dist = torch.stack([
        torch.minimum(torch.abs(phase - l), 2*torch.pi - torch.abs(phase - l))
        for l in levels
    ])
    idx = dist.argmin(0)
    quantized_phase = levels[idx]
    return quantized_phase / torch.pi  # pattern = phase/π for sim

def quantize_continuous(params):
    """No quantization (sigmoid output, half-circle)."""
    # Just use params as-is (free phase, no quantize)
    return ((params * torch.pi) % (2 * torch.pi)) / torch.pi  # pattern in [0, 2)

print("=" * 70)
print("R114 — Phase resolution comparison (1-bit vs 2-bit vs continuous)")
print("=" * 70)

print("\n--- 1-bit (binary, baseline) ---")
b1, sr1 = deploy_with_quantize(quantize_1bit, "1bit", rw=2.0)
flats1 = sum(1 for r in sr1 if r['flat_top_compliant'])
print(f"  Best: worst={b1['worst_supp']:+.2f}, ripple={b1['main_ripple']:.2f}, flat-top={flats1}/{n_restarts}")

print("\n--- 2-bit (4-level phase) ---")
b2, sr2 = deploy_with_quantize(quantize_2bit, "2bit", rw=2.0)
flats2 = sum(1 for r in sr2 if r['flat_top_compliant'])
print(f"  Best: worst={b2['worst_supp']:+.2f}, ripple={b2['main_ripple']:.2f}, flat-top={flats2}/{n_restarts}")

print("\n--- Continuous (no quantize, upper bound reference) ---")
bc, src = deploy_with_quantize(quantize_continuous, "continuous", rw=2.0)
flatsc = sum(1 for r in src if r['flat_top_compliant'])
print(f"  Best: worst={bc['worst_supp']:+.2f}, ripple={bc['main_ripple']:.2f}, flat-top={flatsc}/{n_restarts}")

print("\n" + "=" * 70)
print(f"{'config':<25} | {'best worst':>10} | {'best ripple':>11} | {'flat-top':>9}")
print("-" * 70)
print(f"{'1-bit (binary)':<25} | {b1['worst_supp']:>+10.2f} | {b1['main_ripple']:>11.2f} | {flats1}/5")
print(f"{'2-bit (4-level)':<25} | {b2['worst_supp']:>+10.2f} | {b2['main_ripple']:>11.2f} | {flats2}/5")
print(f"{'Continuous':<25} | {bc['worst_supp']:>+10.2f} | {bc['main_ripple']:>11.2f} | {flatsc}/5")
