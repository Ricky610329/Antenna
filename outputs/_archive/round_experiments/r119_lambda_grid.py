"""Round 119 — Sweep (λ_mean, ripple_weight) grid for sidelobe area + flat-top sweet spot."""

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

def loss_fn(resp, main_lo, main_hi, ripple_w, mean_w):
    main = resp[main_lo:main_hi]
    side = torch.cat([resp[:main_lo], resp[main_hi:]])
    main_min = soft_min(main)
    side_max = soft_max(side)
    main_max = soft_max(main)
    return -(main_min - side_max) + ripple_w * (main_max - main_min) + mean_w * side.mean()

def deploy(ripple_w, mean_w):
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
            loss = loss_fn(resp, main_lo, main_hi, ripple_w, mean_w)
            loss.backward()
            opt.step()
        with torch.no_grad():
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            resp_bin = sim(binary)["response"].cpu().numpy()
        main = resp_bin[main_lo:main_hi]
        side = np.delete(resp_bin, np.arange(main_lo, main_hi))
        s = {
            "worst": float(main.min() - side.max()),
            "side_max": float(side.max()),
            "side_mean": float(side.mean()),
            "main_below_3": int(np.sum(main < -3)),
            "flat_top": int(np.sum(main < -3)) == 0,
        }
        seed_results.append(s)
        if best is None or s["worst"] > best["worst"]:
            best = s
    flat_count = sum(1 for r in seed_results if r['flat_top'])
    return best, flat_count

print("=" * 90)
print(f"R119 — (λ_mean, rw) grid search for sidelobe area + flat-top sweet spot")
print(f"  n=51, broadside, 5 restarts × 1500 steps")
print("=" * 90)

print(f"\n{'rw':>4} | {'λ_mean':>7} | {'best worst':>10} | {'side_max':>9} | {'side_mean':>10} | {'flat-top':>9}")
print("-" * 80)

for rw in [2.0, 3.0, 5.0]:
    for lam in [0.0, 0.1, 0.3, 0.5, 1.0]:
        best, flats = deploy(rw, lam)
        flat_marker = "✓" if flats == 5 else f"{flats}/5"
        print(f"{rw:>4.1f} | {lam:>7.2f} | {best['worst']:>+10.2f} | {best['side_max']:>+9.2f} | "
              f"{best['side_mean']:>+10.2f} | {flat_marker:>9}")
