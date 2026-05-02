"""Round 129 — at wide cap (broadside), re-grid (rw, lambda) for 1-bit recipe.

R128 found R119 recipe (rw=2, lambda=1) loses flat-top at wide caps:
  width=20deg broadside: flat-top 3/5
  width=30deg broadside: flat-top 2/5

Hypothesis: ripple penalty too weak relative to wider main region.
R129 grid-searches (rw, lambda) at width=20deg and 30deg to find new sweet spots.

Constraint: 1-bit ONLY (0 or pi phases).
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
gd_steps = 1500; n_restarts = 3  # reduced for grid

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def deploy_1bit(main_lo, main_hi, mean_w, ripple_w):
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
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            resp_b = sim(binary)["response"].cpu().numpy()
        main_arr = resp_b[main_lo:main_hi]; side_arr = np.delete(resp_b, np.arange(main_lo, main_hi))
        s = {"worst": float(main_arr.min()-side_arr.max()),
             "side_mean": float(side_arr.mean()),
             "ripple": float(main_arr.max()-main_arr.min()),
             "flat_top": int(np.sum(main_arr<-3))==0}
        seed_results.append(s)
        if best is None or s["worst"] > best["worst"]: best = s
    flats = sum(1 for r in seed_results if r['flat_top'])
    return best, flats

def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi

# Grid: rw in {2, 3, 5, 8}, lambda in {0.5, 1, 1.5}
rws = [2.0, 3.0, 5.0, 8.0]
lams = [0.5, 1.0, 1.5]

for width in [20, 30]:
    lo, hi = steer_to_indices(0, width)
    print("=" * 105, flush=True)
    print(f"R129 -- width={width}deg, broadside, indices=[{lo},{hi}]  ({n_restarts} restarts x {gd_steps} steps)", flush=True)
    print("=" * 105, flush=True)
    print(f"\n{'rw':>4} | {'lam':>4} | {'worst':>7} | {'side_mean':>10} | {'ripple':>7} | {'flat-top':>9}", flush=True)
    print("-" * 60, flush=True)
    for rw in rws:
        for lam in lams:
            b, f = deploy_1bit(lo, hi, lam, rw)
            flat_str = "OK" if f == n_restarts else f"{f}/{n_restarts}"
            print(f"{rw:>4.1f} | {lam:>4.1f} | {b['worst']:>+7.2f} | {b['side_mean']:>+10.2f} | "
                  f"{b['ripple']:>+7.2f} | {flat_str:>9}", flush=True)
        print(flush=True)
    print(flush=True)
