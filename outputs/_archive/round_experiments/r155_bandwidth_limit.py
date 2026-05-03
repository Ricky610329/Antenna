"""Round 155 — Find the bandwidth limit for multi-freq joint optimization.

R154 showed +/-5% (~10% rel BW around 38GHz) is universally better than
single-freq. R155 pushes BW: 32, 38, 44 GHz (~32% rel BW) to find where
the methodology breaks.

Compute budget: 3 freqs * 1500 steps * 3 restarts (reduced) ~= 12 min.
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
n = 51; inc = 51.0; w_deg = 10
gd_steps = 1500
n_restarts = 3  # reduced from 5 for compute budget


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
            "flat_top": int(np.sum(main < -3)) == 0}


def joint_optimize(sims, mean_w=1.0, ripple_w=2.0):
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


configs = [
    ("narrow ~10% BW", [36e9, 38e9, 40e9]),       # R154 reference
    ("wide ~32% BW",   [32e9, 38e9, 44e9]),       # NEW
    ("very wide ~53% BW", [28e9, 38e9, 48e9]),    # NEW push
]

print("=" * 100, flush=True)
print(f"R155 -- Bandwidth limit for multi-freq joint optimization", flush=True)
print(f"  n={n}, inc={inc}, w={w_deg}, broadside, 3 restarts x 1500 steps", flush=True)
print("=" * 100, flush=True)

results = {}
for label, freqs in configs:
    print(f"\n[{label}] freqs = {[f/1e9 for f in freqs]} GHz", flush=True)
    sims = [RISSimulator(element_num=n, freq_hz=f, inc_theta_deg=inc) for f in freqs]
    t0 = time.time()
    res = joint_optimize(sims)
    elapsed = time.time() - t0
    s = summarize(res, freqs)
    print(f"  done in {elapsed:.1f}s", flush=True)
    for f in freqs:
        m = s[f]
        print(f"    {f/1e9:>5.0f}GHz: best {m['best']:>+.2f}, mean {m['mean']:>+.2f}, "
              f"min {m['min']:>+.2f}, flat {m['flat']}/{m['n']}", flush=True)
    results[label] = s

# Compare across BW
print("\n" + "=" * 90, flush=True)
print("BANDWIDTH SUMMARY (mean worst across all in-band freqs):", flush=True)
print("=" * 90, flush=True)
for label, s in results.items():
    means = [s[f]["mean"] for f in s]
    flats = [s[f]["flat"] for f in s]
    all_pass = all(s[f]["mean"] > 0 and s[f]["flat"] >= n_restarts - 1 for f in s)
    print(f"  {label:<25}  per-freq means: [{', '.join(f'{m:+.2f}' for m in means)}]  "
          f"flat: [{', '.join(f'{f}/{n_restarts}' for f in flats)}]  "
          f"{'PASS' if all_pass else 'FAIL'}", flush=True)
