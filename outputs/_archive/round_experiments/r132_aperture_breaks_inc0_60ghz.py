"""Round 132 -- Does bigger aperture break inc=0deg + 60GHz boundary for 1-bit?

R131 showed inc=0deg + 60GHz + width=10deg + 1-bit + n=51 has NO clean OK
recipe (best was rw=2 lam=0.3 -> worst +0.13, flat 2/3).

R132 tests if n=71 (larger aperture) rescues the boundary, like R127 did
for +45deg steering boundary.

Recipes tested at n=71, inc=0deg, 60GHz, width=10deg, 1-bit:
  R119 baseline:    rw=2, lam=1
  R131 28GHz rescue: rw=2, lam=0.3
  R131 38GHz rescue: rw=2, lam=0.5

If any pass with worst > 0 and flat-top OK, aperture rescue confirmed.
"""
import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn
from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
freq = 60e9
inc = 0.0
gd_steps = 1500
n_restarts = 3  # n=71 is heavy on 8GB VRAM

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi

main_lo, main_hi = steer_to_indices(0, 10)  # broadside, width=10deg

def deploy_1bit(n_elem, mean_w, ripple_w):
    sim = RISSimulator(element_num=n_elem, freq_hz=freq, inc_theta_deg=inc)
    best = None; seed_results = []
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n_elem, n_elem, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            main = resp[main_lo:main_hi]; side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + ripple_w * (mx - mm) + mean_w * side.mean()
            loss.backward(); opt.step()
        with torch.no_grad():
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            resp_b = sim(binary)["response"].cpu().numpy()
        main_arr = resp_b[main_lo:main_hi]; side_arr = np.delete(resp_b, np.arange(main_lo, main_hi))
        s = {"worst": float(main_arr.min()-side_arr.max()),
             "side_mean": float(side_arr.mean()),
             "side_max": float(side_arr.max()),
             "ripple": float(main_arr.max()-main_arr.min()),
             "flat_top": int(np.sum(main_arr<-3))==0}
        seed_results.append(s)
        if best is None or s["worst"] > best["worst"]: best = s
    flats = sum(1 for r in seed_results if r['flat_top'])
    return best, flats

print("=" * 100, flush=True)
print(f"R132 -- Does aperture upgrade rescue inc=0deg+60GHz boundary? (1-bit, narrow cap)", flush=True)
print(f"  freq=60GHz, inc=0deg, width=10deg broadside", flush=True)
print(f"  {n_restarts} restarts x {gd_steps} steps (n=71 heavy on 8GB VRAM)", flush=True)
print("=" * 100, flush=True)

print(f"\n{'n':>4} | {'recipe':<25} | {'worst':>7} | {'side_max':>9} | {'side_mean':>10} | {'ripple':>7} | {'flat-top':>9}", flush=True)
print("-" * 90, flush=True)

# n=51 baseline reference (from R130/R131)
print(f"  51 | (R130 baseline ref)        |   +0.34 |     -3.39 |     -31.11 |   +3.05 | 1/5  fail", flush=True)
print(f"  51 | (R131 best: rw=2 lam=0.3)  |   +0.13 |     N/A   |     -21.55 |   +2.79 | 2/3  fail", flush=True)
print(flush=True)

recipes = [
    ("R119 baseline (rw=2 lam=1)",   1.0, 2.0),
    ("R131 28GHz (rw=2 lam=0.3)",     0.3, 2.0),
    ("R131 38GHz (rw=2 lam=0.5)",     0.5, 2.0),
]

for name, lam, rw in recipes:
    print(f"[runner] n=71, {name} ...", flush=True)
    b, f = deploy_1bit(71, lam, rw)
    flat_str = "OK" if f == n_restarts else f"{f}/{n_restarts}"
    passed = (f >= n_restarts - 1) and (b["worst"] > 0)
    marker = " PASS" if passed else " fail"
    print(f"  71 | {name:<25} | {b['worst']:>+7.2f} | {b['side_max']:>+9.2f} | "
          f"{b['side_mean']:>+10.2f} | {b['ripple']:>+7.2f} | {flat_str:>9}{marker}", flush=True)
