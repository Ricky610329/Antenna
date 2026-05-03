"""
Round 118 — Sidelobe area minimization

新方向: 壓 sidelobe 整體 distribution 而非只 worst (max(side)).
試 4 種 loss formulations:
  A. baseline (R94): -(min_main - max_side) + λ_r * ripple
  B. + mean_side: A + λ_m * mean(side)
  C. + L2 energy: A + λ_l2 * mean(side²)
  D. + threshold violations: A + λ_t * relu(side - thresh).pow(2).mean()

對每個跑 5 seeds, 評估:
  - worst supp (vs baseline trade-off)
  - mean(side), L2(side), max(side)
  - main 是否仍 flat-top compliant
"""

from __future__ import annotations

import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"

n = 51
freq = 38e9
inc = 51.0
main_lo, main_hi = 162, 192
gd_steps = 1500
n_restarts = 5


def soft_max(x, beta=20.0):
    return (1/beta) * torch.logsumexp(beta * x, dim=-1)

def soft_min(x, beta=20.0):
    return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def loss_A_baseline(resp, main_lo, main_hi, ripple_w=2.0):
    """R94 baseline: only max(side) suppression + ripple."""
    main = resp[main_lo:main_hi]
    side = torch.cat([resp[:main_lo], resp[main_hi:]])
    main_min = soft_min(main)
    side_max = soft_max(side)
    main_max = soft_max(main)
    return -(main_min - side_max) + ripple_w * (main_max - main_min)


def loss_B_mean(resp, main_lo, main_hi, ripple_w=2.0, mean_w=0.3):
    """A + mean(side) penalty."""
    main = resp[main_lo:main_hi]
    side = torch.cat([resp[:main_lo], resp[main_hi:]])
    main_min = soft_min(main)
    side_max = soft_max(side)
    main_max = soft_max(main)
    return -(main_min - side_max) + ripple_w * (main_max - main_min) + mean_w * side.mean()


def loss_C_L2(resp, main_lo, main_hi, ripple_w=2.0, l2_w=0.05):
    """A + L2 energy penalty (Parseval-style)."""
    main = resp[main_lo:main_hi]
    side = torch.cat([resp[:main_lo], resp[main_hi:]])
    main_min = soft_min(main)
    side_max = soft_max(side)
    main_max = soft_max(main)
    # side response in dB; squaring penalizes high values heavily
    return -(main_min - side_max) + ripple_w * (main_max - main_min) + l2_w * (side ** 2).mean()


def loss_D_violation(resp, main_lo, main_hi, ripple_w=2.0, viol_w=0.3, threshold=-25.0):
    """A + above-threshold ReLU penalty."""
    main = resp[main_lo:main_hi]
    side = torch.cat([resp[:main_lo], resp[main_hi:]])
    main_min = soft_min(main)
    side_max = soft_max(side)
    main_max = soft_max(main)
    above = F.relu(side - threshold).pow(2).mean()
    return -(main_min - side_max) + ripple_w * (main_max - main_min) + viol_w * above


def deploy(loss_fn, name):
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    best = None
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            loss = loss_fn(resp, main_lo, main_hi)
            loss.backward()
            opt.step()
        with torch.no_grad():
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            resp_bin = sim(binary)["response"].cpu().numpy()
        # Compute all sidelobe metrics
        main = resp_bin[main_lo:main_hi]
        side = np.delete(resp_bin, np.arange(main_lo, main_hi))
        score = {
            "worst_supp": float(main.min() - side.max()),
            "main_min": float(main.min()),
            "main_max": float(main.max()),
            "main_ripple": float(main.max() - main.min()),
            "side_max": float(side.max()),
            "side_mean": float(side.mean()),
            "side_l2": float(np.mean(side ** 2)),
            "side_above_25": int(np.sum(side > -25)),
            "main_below_3": int(np.sum(main < -3)),
            "flat_top": int(np.sum(main < -3)) == 0,
        }
        if best is None or score["worst_supp"] > best["worst_supp"]:
            best = score
    return best


print("=" * 90)
print("R118 — Sidelobe area minimization (4 loss formulations)")
print("  n=51, 38GHz, inc=51, broadside 15°, 5 restarts × 1500 steps")
print("=" * 90)

print(f"\n{'loss':<25} | {'worst':>7} | {'side_max':>9} | {'side_mean':>10} | "
      f"{'side_l2':>9} | {'>{-25}':>6} | {'flat-top':>9}")
print("-" * 100)

for loss_fn, name in [
    (loss_A_baseline, "A: baseline (R94)"),
    (loss_B_mean, "B: + mean(side) λ=0.3"),
    (loss_C_L2, "C: + L2 energy λ=0.05"),
    (loss_D_violation, "D: + violation λ=0.3"),
]:
    s = deploy(loss_fn, name)
    flat = "yes" if s['flat_top'] else "no"
    print(f"{name:<25} | {s['worst_supp']:>+7.2f} | {s['side_max']:>+9.2f} | "
          f"{s['side_mean']:>+10.2f} | {s['side_l2']:>9.2f} | "
          f"{s['side_above_25']:>6d} | {flat + ' (' + str(0 if s['flat_top'] else s.get('main_below_3', 0)) + '/30)':>9}")
