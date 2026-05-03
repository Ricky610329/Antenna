"""Round 121 — Extend λ_mean + combine with multi-bit phase."""

import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"

n = 51
freq = 38e9
inc = 51.0
main_lo, main_hi = 162, 192
gd_steps = 1500
n_restarts = 5

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def quantize(params, n_levels):
    phase = (params * torch.pi) % (2 * torch.pi)
    if n_levels == 0: return phase / torch.pi
    levels = torch.linspace(0, 2*torch.pi*(n_levels-1)/n_levels, n_levels, device=params.device)
    dist = torch.stack([torch.minimum(torch.abs(phase-l), 2*torch.pi-torch.abs(phase-l)) for l in levels])
    return levels[dist.argmin(0)] / torch.pi

def deploy(rw, mean_w, n_levels=2):
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
            main = resp[main_lo:main_hi]; side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + rw * (mx - mm) + mean_w * side.mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            quantized = quantize(params, n_levels)
            resp_b = sim(quantized)["response"].cpu().numpy()
        main = resp_b[main_lo:main_hi]; side = np.delete(resp_b, np.arange(main_lo, main_hi))
        s = {"worst": float(main.min()-side.max()), "side_max": float(side.max()),
             "side_mean": float(side.mean()), "ripple": float(main.max()-main.min()),
             "main_below_3": int(np.sum(main<-3)), "flat_top": int(np.sum(main<-3))==0}
        seed_results.append(s)
        if best is None or s["worst"] > best["worst"]: best = s
    flats = sum(1 for r in seed_results if r['flat_top'])
    return best, flats

print("=" * 90)
print("R121 — Extend λ_mean + multi-bit phase stacking (n=51, rw=2)")
print("=" * 90)

print(f"\n{'bits':>4} | {'λ_mean':>7} | {'best worst':>10} | {'side_max':>9} | {'side_mean':>10} | {'ripple':>7} | {'flat-top':>9}")
print("-" * 95)
for bits, n_lev in [(1, 2), (2, 4), (3, 8)]:
    for lam in [0.0, 1.0, 2.0, 3.0]:
        best, flats = deploy(rw=2.0, mean_w=lam, n_levels=n_lev)
        flat = "✓ 5/5" if flats == 5 else f"{flats}/5"
        print(f"{bits:>4} | {lam:>7.2f} | {best['worst']:>+10.2f} | {best['side_max']:>+9.2f} | "
              f"{best['side_mean']:>+10.2f} | {best['ripple']:>7.2f} | {flat:>9}")
    print()
