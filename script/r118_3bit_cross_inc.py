"""Round 118 — Verify 3-bit eliminates inc-dependence at all inc values."""

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

def quantize_3bit(params):
    phase = (params * torch.pi) % (2 * torch.pi)
    levels = torch.linspace(0, 2*torch.pi*7/8, 8, device=params.device)
    dist = torch.stack([torch.minimum(torch.abs(phase-l), 2*torch.pi-torch.abs(phase-l)) for l in levels])
    return levels[dist.argmin(0)] / torch.pi

def deploy(inc, n_levels):
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
            loss = worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=2.0)
            loss.backward()
            opt.step()
        with torch.no_grad():
            quantized = quantize_3bit(params)
            resp_q = sim(quantized)["response"].cpu().numpy()
        m = evaluate_metrics(resp_q, main_lo, main_hi)
        seed_results.append(m)
        if best is None or m["worst_supp"] > best["worst_supp"]:
            best = m
    return best, sum(1 for r in seed_results if r['flat_top_compliant'])

print("=" * 70)
print("R118 — 3-bit phase cross-inc verification (uniformity check)")
print("=" * 70)

print(f"\n{'inc':>5} | {'1-bit (R106)':>15} | {'3-bit (R118)':>15} | {'Δ flat':>8}")
print("-" * 70)
# R106 1-bit data
r106_data = {0.0: "0/5 ✗", 30.0: "2/5", 51.0: "5/5 ★", 60.0: "3/5", 70.0: "1/5"}

for inc in [0, 30, 51, 60, 70]:
    b, flats = deploy(inc, 8)
    r106 = r106_data.get(float(inc), "n/a")
    flat_str = f"yes {flats}/5" if b['flat_top_compliant'] else f"no {flats}/5"
    print(f"{inc:>4}° | {r106:>15} | worst={b['worst_supp']:>+5.2f}, {flat_str:>10}")
