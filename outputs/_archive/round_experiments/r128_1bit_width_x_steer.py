"""Round 128 — 1-bit ONLY: width x steering joint sweep with R119 recipe.

COURSE CORRECTION: actual deployable RIS hardware is 1-bit (0 or pi phases only).
R121 CHAMPION used 2-bit which doesn't deploy. Pivot back to 1-bit baseline.

Best 1-bit recipe so far: R119 (1-bit + lambda_mean=1 + rw=2),
side_mean -23.70 dB at n=51 broadside width 15deg.

R128 sweeps two axes never jointly validated for 1-bit:
  - cap width: 10deg, 20deg, 30deg
  - steering: 0deg (broadside), +30deg, +45deg

Total 9 configs x 5 seeds x 1500 steps. n=51, 38GHz, inc=51deg.

Goal: identify where 1-bit recipe holds vs breaks across width+angle.
"""
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
gd_steps = 1500
n_restarts = 5

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def deploy_1bit(main_lo, main_hi, mean_w=1.0, ripple_w=2.0):
    """1-bit (0 or pi) RIS with R119 recipe."""
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
            # Hard 1-bit: phase mod 2pi, then 0 or pi based on which half-circle
            phase = (params * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            resp_b = sim(binary)["response"].cpu().numpy()
        main_arr = resp_b[main_lo:main_hi]; side_arr = np.delete(resp_b, np.arange(main_lo, main_hi))
        s = {"worst": float(main_arr.min()-side_arr.max()),
             "side_max": float(side_arr.max()),
             "side_mean": float(side_arr.mean()),
             "ripple": float(main_arr.max()-main_arr.min()),
             "main_min": float(main_arr.min()),
             "flat_top": int(np.sum(main_arr<-3))==0,
             "main_below_3": int(np.sum(main_arr<-3))}
        seed_results.append(s)
        if best is None or s["worst"] > best["worst"]: best = s
    flats = sum(1 for r in seed_results if r['flat_top'])
    return best, flats, seed_results

# steering->index map: theta_deg = -90 + index * 0.5
# For width W and center C, indices = round(((C-W/2) - (-90))/0.5), round(((C+W/2) - (-90))/0.5)
def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi

print("=" * 110, flush=True)
print("R128 -- 1-bit RIS (0 or pi only): width x steering joint sweep", flush=True)
print(f"  n={n}, freq=38GHz, inc=51deg, recipe=R119 (1-bit + lambda=1 + rw=2)", flush=True)
print(f"  {n_restarts} restarts x {gd_steps} steps", flush=True)
print("=" * 110, flush=True)

widths = [10, 20, 30]
steerings = [0, 30, 45]

print(f"\n{'width':>6} | {'steer':>5} | {'worst':>7} | {'side_max':>9} | {'side_mean':>10} | {'ripple':>7} | {'flat-top':>9}", flush=True)
print("-" * 80, flush=True)

results_grid = {}
for w in widths:
    for steer in steerings:
        lo, hi = steer_to_indices(steer, w)
        print(f"[runner] width={w}deg, steer={steer}deg, indices=[{lo},{hi}]", flush=True)
        b, f, _ = deploy_1bit(lo, hi)
        flat_str = "OK" if f == n_restarts else f"{f}/{n_restarts}"
        print(f"{w:>5}d | {steer:>+4}d | {b['worst']:>+7.2f} | {b['side_max']:>+9.2f} | "
              f"{b['side_mean']:>+10.2f} | {b['ripple']:>+7.2f} | {flat_str:>9}", flush=True)
        results_grid[(w, steer)] = (b, f)
    print(flush=True)

print("\n=== Summary table ===", flush=True)
print(f"{'metric':<12} | " + " | ".join(f"{s:>+4}d" for s in steerings), flush=True)
for w in widths:
    print(f"\nwidth {w:>2}d:", flush=True)
    for metric in ["worst", "side_mean", "flat_top"]:
        vals = []
        for s in steerings:
            b, f = results_grid[(w, s)]
            if metric == "flat_top":
                vals.append("OK" if f == n_restarts else f"{f}/{n_restarts}")
            else:
                vals.append(f"{b[metric]:+.2f}")
        print(f"  {metric:<10} | " + " | ".join(f"{v:>6}" for v in vals), flush=True)
