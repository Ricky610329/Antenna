"""Round 139 — verify early-stopping benefit generalizes across selector recipes.

R138 showed early-stopping fixes Adam drift at R119 sweet spot.
R139 tests whether the benefit is universal across selector recipes:
  A: R119 baseline (n=51, w=10, rw=2 lam=1)        — R138 reference
  B: R129 wide   (n=51, w=18, rw=3 lam=1)
  C: R131 28GHz rescue (n=51, inc=0, 28GHz, w=10, rw=2 lam=0.3)
  D: n=71 extrapolation (n=71, inc=51, 38GHz, w=10, rw=5 lam=0.5)

For each: compare (final-step) vs (early-stop) on the same 5 seeds.
If early-stop is universally non-worse, promote to default.
"""
import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
gd_steps = 1500
eval_every = 50


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi


def eval_binary(params, sim, main_lo, main_hi):
    with torch.no_grad():
        phase = (params * torch.pi) % (2 * torch.pi)
        binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
        resp = sim(binary)["response"].cpu().numpy()
    main_arr = resp[main_lo:main_hi]; side_arr = np.delete(resp, np.arange(main_lo, main_hi))
    return {
        "worst": float(main_arr.min() - side_arr.max()),
        "side_mean": float(side_arr.mean()),
        "flat_top": int(np.sum(main_arr < -3)) == 0,
    }


def run_pair(n, inc_deg, freq_hz, width_deg, mean_w, ripple_w, n_restarts):
    """Return (final_results, early_stop_results) lists across n_restarts seeds."""
    main_lo, main_hi = steer_to_indices(0, width_deg)
    sim = RISSimulator(element_num=n, freq_hz=freq_hz, inc_theta_deg=inc_deg)
    final_results = []; es_results = []

    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)

        best_es_worst = -1e9
        best_es_state = None

        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            main = resp[main_lo:main_hi]
            side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + ripple_w * (mx - mm) + mean_w * side.mean()
            loss.backward()
            opt.step()

            if (step + 1) % eval_every == 0:
                m = eval_binary(params, sim, main_lo, main_hi)
                if m["worst"] > best_es_worst:
                    best_es_worst = m["worst"]
                    best_es_state = params.detach().clone()

        # Final-step pattern
        final_results.append(eval_binary(params, sim, main_lo, main_hi))
        # Early-stop pattern
        es_results.append(eval_binary(best_es_state, sim, main_lo, main_hi))

    return final_results, es_results


def summarize(results, label):
    worsts = [r["worst"] for r in results]
    smeans = [r["side_mean"] for r in results]
    flats = sum(1 for r in results if r["flat_top"])
    return {
        "label": label,
        "best_worst": max(worsts),
        "mean_worst": float(np.mean(worsts)),
        "min_worst": min(worsts),
        "best_smean": min(smeans),
        "n_flats": flats,
        "n": len(results),
    }


configs = [
    ("A: R119 narrow",            51, 51, 38e9, 10, 1.0, 2.0, 5),
    ("B: R129 wide w=18",         51, 51, 38e9, 18, 1.0, 3.0, 5),
    ("C: R131 28GHz inc=0",       51,  0, 28e9, 10, 0.3, 2.0, 5),
    ("D: n=71 extrap w=10",       71, 51, 38e9, 10, 0.5, 5.0, 3),  # n=71 needs 3 restarts (VRAM)
]

print("=" * 110, flush=True)
print(f"R139 -- Early-stop benefit across 4 selector recipes", flush=True)
print(f"  Compare (final-step) vs (early-stop) for each recipe at 5 (or 3) seeds", flush=True)
print("=" * 110, flush=True)

for label, n, inc, freq, w, lam, rw, nr in configs:
    print(f"\n[runner] {label} (n={n}, inc={inc}, freq={freq/1e9}GHz, w={w}, rw={rw}, lam={lam}, {nr} restarts)", flush=True)
    final, es = run_pair(n, inc, freq, w, lam, rw, nr)
    sf = summarize(final, "final-step")
    se = summarize(es, "early-stop")

    delta_best  = se["best_worst"] - sf["best_worst"]
    delta_mean  = se["mean_worst"] - sf["mean_worst"]
    delta_min   = se["min_worst"]  - sf["min_worst"]

    print(f"\n  {label}", flush=True)
    print(f"  {'metric':<12} | {'final':>8} | {'early-stop':>10} | {'delta':>8}", flush=True)
    print(f"  {'-'*46}", flush=True)
    print(f"  {'best_worst':<12} | {sf['best_worst']:>+8.2f} | {se['best_worst']:>+10.2f} | {delta_best:>+8.2f}", flush=True)
    print(f"  {'mean_worst':<12} | {sf['mean_worst']:>+8.2f} | {se['mean_worst']:>+10.2f} | {delta_mean:>+8.2f}", flush=True)
    print(f"  {'min_worst':<12} | {sf['min_worst']:>+8.2f} | {se['min_worst']:>+10.2f} | {delta_min:>+8.2f}", flush=True)
    print(f"  {'best_smean':<12} | {sf['best_smean']:>+8.2f} | {se['best_smean']:>+10.2f} | {se['best_smean']-sf['best_smean']:>+8.2f}", flush=True)
    print(f"  {'flat':<12} | {sf['n_flats']}/{sf['n']:>5} | {se['n_flats']}/{se['n']:>8} | {se['n_flats']-sf['n_flats']:>+8d}", flush=True)
