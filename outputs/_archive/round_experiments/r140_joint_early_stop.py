"""Round 140 — Joint early-stop (worst AND flat-top) re-validation.

R139 found simple early-stop (track best worst alone) trades flat-top in
3/4 selector recipes. R140 implements joint criterion:

  Among trajectory snapshots where flat-top is satisfied,
  pick the one with the largest worst-case.
  If no snapshot satisfies flat-top, fall back to final-step.

Test: same 4 recipes as R139 (A/B/C/D). 3-way comparison:
  - final-step:        old default
  - simple early-stop: track max worst (R138/R139)
  - joint early-stop:  track max worst AMONG flat-top-valid snapshots (NEW)

Promotion criterion: joint must beat or match final-step on BOTH
mean_worst AND flat-top across all 4 recipes.
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


def run_three_strategies(n, inc_deg, freq_hz, width_deg, mean_w, ripple_w, n_restarts):
    """Per seed, return (final, simple_es, joint_es) metric dicts."""
    main_lo, main_hi = steer_to_indices(0, width_deg)
    sim = RISSimulator(element_num=n, freq_hz=freq_hz, inc_theta_deg=inc_deg)
    final_results, simple_results, joint_results = [], [], []

    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)

        # Track both criteria
        best_simple_worst = -1e9; best_simple_state = None
        best_joint_worst = -1e9;  best_joint_state = None

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
                # Simple: max worst alone
                if m["worst"] > best_simple_worst:
                    best_simple_worst = m["worst"]
                    best_simple_state = params.detach().clone()
                # Joint: max worst AMONG flat-top-valid
                if m["flat_top"] and m["worst"] > best_joint_worst:
                    best_joint_worst = m["worst"]
                    best_joint_state = params.detach().clone()

        # Evaluate three strategies
        final_results.append(eval_binary(params, sim, main_lo, main_hi))
        if best_simple_state is not None:
            simple_results.append(eval_binary(best_simple_state, sim, main_lo, main_hi))
        else:
            simple_results.append(eval_binary(params, sim, main_lo, main_hi))
        # Joint may have no valid snapshot -> fall back to final
        if best_joint_state is not None:
            joint_results.append(eval_binary(best_joint_state, sim, main_lo, main_hi))
        else:
            joint_results.append(eval_binary(params, sim, main_lo, main_hi))

    return final_results, simple_results, joint_results


def summarize(results):
    worsts = [r["worst"] for r in results]
    smeans = [r["side_mean"] for r in results]
    flats = sum(1 for r in results if r["flat_top"])
    return {
        "best": max(worsts),
        "mean": float(np.mean(worsts)),
        "min": min(worsts),
        "smean": min(smeans),
        "flat": flats,
        "n": len(results),
    }


configs = [
    ("A: R119 narrow",       51, 51, 38e9, 10, 1.0, 2.0, 5),
    ("B: R129 wide w=18",    51, 51, 38e9, 18, 1.0, 3.0, 5),
    ("C: R131 28GHz inc=0",  51,  0, 28e9, 10, 0.3, 2.0, 5),
    ("D: n=71 extrap w=10",  71, 51, 38e9, 10, 0.5, 5.0, 3),
]

print("=" * 110, flush=True)
print(f"R140 -- Joint early-stop (worst AND flat-top) re-validation", flush=True)
print(f"  3-way: final-step / simple ES (R138/R139) / joint ES (NEW)", flush=True)
print("=" * 110, flush=True)

promote_ok = True
for label, n, inc, freq, w, lam, rw, nr in configs:
    print(f"\n[runner] {label}", flush=True)
    fr, sr, jr = run_three_strategies(n, inc, freq, w, lam, rw, nr)
    sf, ss, sj = summarize(fr), summarize(sr), summarize(jr)

    print(f"\n  {label}", flush=True)
    print(f"  {'metric':<10} | {'final':>9} | {'simple-ES':>10} | {'joint-ES':>10}", flush=True)
    print(f"  {'-'*49}", flush=True)
    print(f"  {'best':<10} | {sf['best']:>+9.2f} | {ss['best']:>+10.2f} | {sj['best']:>+10.2f}", flush=True)
    print(f"  {'mean':<10} | {sf['mean']:>+9.2f} | {ss['mean']:>+10.2f} | {sj['mean']:>+10.2f}", flush=True)
    print(f"  {'min':<10} | {sf['min']:>+9.2f} | {ss['min']:>+10.2f} | {sj['min']:>+10.2f}", flush=True)
    print(f"  {'smean':<10} | {sf['smean']:>+9.2f} | {ss['smean']:>+10.2f} | {sj['smean']:>+10.2f}", flush=True)
    print(f"  {'flat':<10} | {sf['flat']}/{sf['n']:>7} | {ss['flat']}/{ss['n']:>8} | {sj['flat']}/{sj['n']:>8}", flush=True)

    # Promotion check: joint vs final
    if not (sj["mean"] >= sf["mean"] and sj["flat"] >= sf["flat"]):
        promote_ok = False
        print(f"  -> Joint FAILS promotion criterion at this config", flush=True)
    else:
        print(f"  -> Joint OK (mean dominates AND flat preserved/improved)", flush=True)

print("\n" + "=" * 70, flush=True)
print(f"PROMOTION VERDICT: {'PROMOTE joint-ES to default' if promote_ok else 'DO NOT promote (some config regresses)'}", flush=True)
print("=" * 70, flush=True)
