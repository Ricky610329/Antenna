"""Round 130 -- 1-bit narrow-cap recipe universal validation across inc x freq.

R128 found that R119 (1-bit, rw=2, lam=1) is robust at narrow cap (10deg)
across steering. R130 verifies it also generalizes across:
  - incidence: 0, 30, 51, 70 deg
  - frequency: 5.8, 28, 38, 60 GHz

Total: 4 inc x 4 freq = 16 configs at width=10deg broadside, n=51, 1-bit.

Goal: declare a "narrow-cap 1-bit universal recipe" if all 16 pass with
flat-top >= 4/5 and worst > 0.
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
gd_steps = 1500
n_restarts = 5
width = 10  # narrow cap (10 deg)

def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)

def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi

main_lo, main_hi = steer_to_indices(0, width)  # broadside, width=10deg

def deploy_1bit(freq_hz, inc_deg, mean_w=1.0, ripple_w=2.0):
    """R119 recipe: 1-bit + lam=1 + rw=2."""
    sim = RISSimulator(element_num=n, freq_hz=freq_hz, inc_theta_deg=inc_deg)
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
             "side_max": float(side_arr.max()),
             "ripple": float(main_arr.max()-main_arr.min()),
             "flat_top": int(np.sum(main_arr<-3))==0}
        seed_results.append(s)
        if best is None or s["worst"] > best["worst"]: best = s
    flats = sum(1 for r in seed_results if r['flat_top'])
    return best, flats

freqs = [(5.8e9, "5.8GHz"), (28e9, "28GHz"), (38e9, "38GHz"), (60e9, "60GHz")]
incs = [0, 30, 51, 70]

print("=" * 110, flush=True)
print(f"R130 -- 1-bit narrow-cap (width={width}deg broadside) universal validation", flush=True)
print(f"  Recipe: R119 (1-bit, rw=2, lam=1), n={n}, {n_restarts} restarts x {gd_steps} steps", flush=True)
print(f"  Sweeping inc x freq: 4 x 4 = 16 configs", flush=True)
print("=" * 110, flush=True)

print(f"\n{'inc':>4} | {'freq':>8} | {'worst':>7} | {'side_max':>9} | {'side_mean':>10} | {'ripple':>7} | {'flat-top':>9}", flush=True)
print("-" * 80, flush=True)

results = {}
all_pass_count = 0
for inc in incs:
    for freq_hz, freq_label in freqs:
        print(f"[runner] inc={inc}deg, freq={freq_label} ...", flush=True)
        b, f = deploy_1bit(freq_hz, inc)
        flat_str = "OK" if f == n_restarts else f"{f}/{n_restarts}"
        # Pass criterion: flat-top >= 4/5 AND worst > 0
        passed = (f >= 4) and (b["worst"] > 0)
        if passed: all_pass_count += 1
        marker = " PASS" if passed else " fail"
        print(f"{inc:>3}d | {freq_label:>8} | {b['worst']:>+7.2f} | {b['side_max']:>+9.2f} | "
              f"{b['side_mean']:>+10.2f} | {b['ripple']:>+7.2f} | {flat_str:>9}{marker}", flush=True)
        results[(inc, freq_label)] = (b, f, passed)
    print(flush=True)

print("=" * 80, flush=True)
print(f"PASS rate: {all_pass_count}/16 configs (flat-top >= 4/5 AND worst > 0)", flush=True)
print("=" * 80, flush=True)

print(f"\n=== Worst-case heatmap (inc rows x freq cols) ===", flush=True)
print(f"{'':>5} | " + " | ".join(f"{f[1]:>8}" for f in freqs), flush=True)
for inc in incs:
    row = [f"{results[(inc, f[1])][0]['worst']:+.2f}" for f in freqs]
    print(f"{inc:>3}d | " + " | ".join(f"{v:>8}" for v in row), flush=True)

print(f"\n=== Side_mean heatmap (inc rows x freq cols) ===", flush=True)
print(f"{'':>5} | " + " | ".join(f"{f[1]:>8}" for f in freqs), flush=True)
for inc in incs:
    row = [f"{results[(inc, f[1])][0]['side_mean']:+.2f}" for f in freqs]
    print(f"{inc:>3}d | " + " | ".join(f"{v:>8}" for v in row), flush=True)

print(f"\n=== Flat-top compliance heatmap ===", flush=True)
print(f"{'':>5} | " + " | ".join(f"{f[1]:>8}" for f in freqs), flush=True)
for inc in incs:
    row = []
    for freq_hz, fl in freqs:
        b, f, p = results[(inc, fl)]
        row.append("OK" if f == n_restarts else f"{f}/{n_restarts}")
    print(f"{inc:>3}d | " + " | ".join(f"{v:>8}" for v in row), flush=True)
