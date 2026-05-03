"""Round 156 — Does bigger aperture rescue the 32% BW boundary?

R155 found 32% rel BW (32/38/44 GHz) FAILs flat-top at n=51.
R156 tests if n=71 aperture upgrade rescues, parallel to R127 which broke
the +45deg steering boundary.

Compute budget compromise: 1000 GD steps x 2 restarts (vs default 1500x3).
n=71 forward pass ~2x slower than n=51 due to 1.9x element count.
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
inc = 51.0
w_deg = 10
gd_steps = 1000        # reduced from 1500
n_restarts = 2          # reduced from 3
freqs_32pct = [32e9, 38e9, 44e9]


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi


main_lo, main_hi = steer_to_indices(0, w_deg)


def eval_metrics(resp_np):
    main = resp_np[main_lo:main_hi]
    side = np.delete(resp_np, np.arange(main_lo, main_hi))
    return {"worst": float(main.min() - side.max()),
            "side_mean": float(side.mean()),
            "ripple": float(main.max() - main.min()),
            "main_below_3": int(np.sum(main < -3)),
            "flat_top": int(np.sum(main < -3)) == 0}


def joint_optimize(n, sims, mean_w=1.0, ripple_w=2.0):
    seed_results = []
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)
        best_min_worst = -1e9; best_state = None
        for step in range(gd_steps):
            opt.zero_grad()
            total = 0.0
            for sim in sims:
                resp = sim(params)["response"]
                main = resp[main_lo:main_hi]
                side = torch.cat([resp[:main_lo], resp[main_hi:]])
                mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
                total = total + (-(mm - sx) + ripple_w * (mx - mm) + mean_w * side.mean())
            total.backward(); opt.step()
            if (step + 1) % 50 == 0:
                with torch.no_grad():
                    phase = (params * torch.pi) % (2 * torch.pi)
                    binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
                    metrics = [eval_metrics(sim(binary)["response"].cpu().numpy()) for sim in sims]
                if all(m["flat_top"] for m in metrics):
                    mw = min(m["worst"] for m in metrics)
                    if mw > best_min_worst:
                        best_min_worst = mw
                        best_state = params.detach().clone()
        eval_state = best_state if best_state is not None else params.detach()
        with torch.no_grad():
            phase = (eval_state * torch.pi) % (2 * torch.pi)
            binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
            per_freq = [eval_metrics(sim(binary)["response"].cpu().numpy()) for sim in sims]
        seed_results.append({"seed": seed, "per_freq": per_freq})
    return seed_results


def summarize(seed_results, freqs):
    out = {}
    for i, f in enumerate(freqs):
        worsts = [r["per_freq"][i]["worst"] for r in seed_results]
        flats = sum(1 for r in seed_results if r["per_freq"][i]["flat_top"])
        out[f] = {"best": max(worsts), "mean": float(np.mean(worsts)),
                  "min": min(worsts), "flat": flats, "n": len(seed_results)}
    return out


print("=" * 100, flush=True)
print(f"R156 -- Aperture rescue for 32% BW boundary", flush=True)
print(f"  freqs: {[f/1e9 for f in freqs_32pct]} GHz, inc={inc}, w={w_deg}", flush=True)
print(f"  Reduced budget: {gd_steps} steps x {n_restarts} restarts (n=71 is heavy)", flush=True)
print("=" * 100, flush=True)

# Reference: R155 result at n=51
print(f"\n[reference] R155 result at n=51 (3 restarts, 1500 steps):", flush=True)
print(f"  32GHz: best +2.44, mean +1.36, flat 1/3", flush=True)
print(f"  38GHz: best +3.34, mean +2.01, flat 1/3", flush=True)
print(f"  44GHz: best +2.91, mean +2.18, flat 2/3", flush=True)
print(f"  -> FAIL flat-top", flush=True)

# n=71 test
n = 71
print(f"\n[test] n={n} with same recipe...", flush=True)
sims = [RISSimulator(element_num=n, freq_hz=f, inc_theta_deg=inc) for f in freqs_32pct]
t0 = time.time()
results = joint_optimize(n, sims)
elapsed = time.time() - t0
s = summarize(results, freqs_32pct)
print(f"  done in {elapsed:.1f}s", flush=True)
for f in freqs_32pct:
    m = s[f]
    print(f"  {f/1e9:>5.0f}GHz: best {m['best']:>+.2f}, mean {m['mean']:>+.2f}, "
          f"min {m['min']:>+.2f}, flat {m['flat']}/{m['n']}", flush=True)

print("\n" + "=" * 70, flush=True)
all_flat = all(s[f]["flat"] >= n_restarts - 0 for f in freqs_32pct)
all_pos = all(s[f]["mean"] > 0 for f in freqs_32pct)
verdict = "PASS - aperture rescue WORKS" if (all_flat and all_pos) else \
          "PARTIAL - flat-top still struggles" if all_pos else "FAIL"
print(f"  {verdict}", flush=True)
print("=" * 70, flush=True)
