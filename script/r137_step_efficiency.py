"""Round 137 -- Optimization step efficiency.

For surrogate-in-the-loop (R138+ patch transition), each forward pass costs
much more (~ms per surrogate eval vs sub-ms per analytical sim). So the
question matters: how many GD steps does R119 recipe really need?

Test: at the R119 sweet spot (n=51, inc=51, 38GHz, w=10), measure metric
quality vs gd_steps in {200, 400, 800, 1500, 3000}. 5 restarts each.

Goal: find smallest steps where metrics plateau, so surrogate workflow
can budget compute appropriately.
"""
import sys
sys.path.insert(0, "script")
import time
import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
n = 51; freq = 38e9; inc = 51.0
n_restarts = 5
width = 10  # narrow R119 sweet


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi


def deploy_with_steps(gd_steps):
    main_lo, main_hi = steer_to_indices(0, width)
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
            loss = -(mm - sx) + 2.0 * (mx - mm) + 1.0 * side.mean()
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
    return best, flats, seed_results


print("=" * 95, flush=True)
print(f"R137 -- GD step-count vs convergence (R119 recipe @ sweet spot)", flush=True)
print(f"  n={n}, inc={inc}, freq={freq/1e9}GHz, width={width}deg, 1-bit, {n_restarts} restarts", flush=True)
print("=" * 95, flush=True)

print(f"\n{'gd_steps':>9} | {'best_worst':>11} | {'mean_worst':>11} | {'side_mean':>10} | "
      f"{'ripple':>7} | {'flat':>5} | {'wall_sec':>8}", flush=True)
print("-" * 85, flush=True)

results_per_steps = {}
for steps in [200, 400, 800, 1500, 3000]:
    print(f"[runner] gd_steps={steps} ...", flush=True)
    t0 = time.time()
    b, f, seeds = deploy_with_steps(steps)
    elapsed = time.time() - t0
    mean_worst = float(np.mean([s["worst"] for s in seeds]))
    flat_str = "OK" if f == n_restarts else f"{f}/{n_restarts}"
    print(f"{steps:>9d} | {b['worst']:>+11.2f} | {mean_worst:>+11.2f} | {b['side_mean']:>+10.2f} | "
          f"{b['ripple']:>+7.2f} | {flat_str:>5} | {elapsed:>8.1f}", flush=True)
    results_per_steps[steps] = {"best_worst": b["worst"], "mean_worst": mean_worst, "elapsed": elapsed}

print("\n" + "=" * 70, flush=True)
print("Convergence analysis", flush=True)
print("=" * 70, flush=True)
ref = results_per_steps[1500]["best_worst"]
for steps, r in results_per_steps.items():
    delta = r["best_worst"] - ref
    speedup = results_per_steps[1500]["elapsed"] / r["elapsed"]
    print(f"  steps={steps:>5}: best_worst={r['best_worst']:+.2f} "
          f"(delta vs 1500: {delta:+.2f}, time-speedup: {speedup:.1f}x)", flush=True)
