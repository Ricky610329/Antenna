"""Round 126 -- Probe the +45 deg steering boundary discovered in R125.

R125 finding: at +45 deg steering, R121 CHAMPION (2-bit + lambda=1) ties
baseline on worst-case (+1.17 vs +1.22). side_mean still improved (-28.64 vs -20.55)
but worst-case stuck.

R126 tests whether more hardware / stronger loss can break the boundary:
  A. R121 baseline:        2-bit + lambda=1.0 + rw=2.0  (R125 result)
  B. 3-bit upgrade:        3-bit + lambda=1.0 + rw=2.0
  C. Stronger ripple:      2-bit + lambda=1.0 + rw=3.0
  D. Stronger mean:        2-bit + lambda=1.5 + rw=2.0  (R121 said lambda=2 lost flat-top)
  E. Continuous phase:     continuous + lambda=1.0 + rw=2.0

Reveal: what's the physical worst-case ceiling at +45 deg steering?
"""
import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn
from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
n = 51; freq = 38e9; inc = 51.0
main_lo, main_hi = 262, 278  # +45 deg steering, width 15 deg
gd_steps = 1500; n_restarts = 5

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def quantize(params, n_levels):
    """n_levels=0: continuous; >0: discrete."""
    phase = (params * torch.pi) % (2 * torch.pi)
    if n_levels == 0: return phase / torch.pi
    levels = torch.linspace(0, 2*torch.pi*(n_levels-1)/n_levels, n_levels, device=params.device)
    dist = torch.stack([torch.minimum(torch.abs(phase-l), 2*torch.pi-torch.abs(phase-l)) for l in levels])
    return levels[dist.argmin(0)] / torch.pi

def deploy(name, n_levels, mean_w, ripple_w):
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    best = None; seed_results = []
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            main = resp[main_lo:main_hi]; side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + ripple_w * (mx - mm) + mean_w * side.mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            quantized = quantize(params, n_levels)
            resp_b = sim(quantized)["response"].cpu().numpy()
        main = resp_b[main_lo:main_hi]; side = np.delete(resp_b, np.arange(main_lo, main_hi))
        s = {"worst": float(main.min()-side.max()), "side_max": float(side.max()),
             "side_mean": float(side.mean()), "ripple": float(main.max()-main.min()),
             "main_min": float(main.min()), "flat_top": int(np.sum(main<-3))==0}
        seed_results.append(s)
        if best is None or s["worst"] > best["worst"]: best = s
    flats = sum(1 for r in seed_results if r['flat_top'])
    return best, flats

print("=" * 105)
print("R126 -- +45 deg steering boundary probe (R121 CHAMPION ties baseline at this point)")
print(f"  n=51, freq=38GHz, inc=51deg, main_center=+45deg, width=15deg")
print("=" * 105)

experiments = [
    ("A: R121 baseline (2-bit, lam=1, rw=2)", 4, 1.0, 2.0),
    ("B: 3-bit upgrade",                       8, 1.0, 2.0),
    ("C: stronger ripple (rw=3)",              4, 1.0, 3.0),
    ("D: stronger mean (lam=1.5)",             4, 1.5, 2.0),
    ("E: continuous phase",                    0, 1.0, 2.0),
]

print(f"\n{'recipe':<45} | {'worst':>7} | {'side_max':>9} | {'side_mean':>10} | {'ripple':>7} | {'flat':>5}")
print("-" * 105)
for name, n_levels, mw, rw in experiments:
    b, f = deploy(name, n_levels, mw, rw)
    print(f"{name:<45} | {b['worst']:>+7.2f} | {b['side_max']:>+9.2f} | {b['side_mean']:>+10.2f} | "
          f"{b['ripple']:>+7.2f} | {'OK' if f==5 else f'{f}/5':>5}")
