"""Round 138 — Early stopping + LR decay to fix R137 mean_worst dip.

R137 found mean_worst (across 5 seeds) dips negative at 1500 steps and
recovers at 3000. Suggests Adam at lr=0.05 drifts some seeds after they
find good solutions.

R138 tests two interventions:
  A: Early stopping — track best worst-case along trajectory, return that
     pattern at end (not the final-step pattern).
  B: LR decay — cosine schedule from 0.05 to 0.005 over the trajectory.
  C: Both A+B combined.

Compare vs the R137 baseline (no intervention).

Test config: n=51, inc=51, 38GHz, w=10, R119 recipe (rw=2, lam=1).
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
n_restarts = 5
gd_steps = 1500
width = 10


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi


def evaluate_binary(params, sim, main_lo, main_hi):
    """Quantize params to 1-bit and return metrics."""
    with torch.no_grad():
        phase = (params * torch.pi) % (2 * torch.pi)
        binary = ((phase > torch.pi/2) & (phase < 3*torch.pi/2)).float()
        resp = sim(binary)["response"].cpu().numpy()
    main_arr = resp[main_lo:main_hi]
    side_arr = np.delete(resp, np.arange(main_lo, main_hi))
    return {
        "worst": float(main_arr.min() - side_arr.max()),
        "side_mean": float(side_arr.mean()),
        "ripple": float(main_arr.max() - main_arr.min()),
        "flat_top": int(np.sum(main_arr < -3)) == 0,
    }


def run_recipe(use_early_stop, use_lr_decay, eval_every=50):
    """Run R119 optimization with optional early stopping and LR decay."""
    main_lo, main_hi = steer_to_indices(0, width)
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    seed_results = []

    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)

        best_worst = -1e9
        best_state = None

        for step in range(gd_steps):
            # LR decay: cosine from 0.05 to 0.005
            if use_lr_decay:
                progress = step / gd_steps
                lr = 0.005 + 0.5 * (0.05 - 0.005) * (1 + np.cos(np.pi * progress))
                for pg in opt.param_groups:
                    pg["lr"] = lr

            opt.zero_grad()
            resp = sim(params)["response"]
            main = resp[main_lo:main_hi]
            side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + 2.0 * (mx - mm) + 1.0 * side.mean()
            loss.backward()
            opt.step()

            # Early stopping: periodically evaluate quantized worst-case
            if use_early_stop and (step + 1) % eval_every == 0:
                m = evaluate_binary(params, sim, main_lo, main_hi)
                if m["worst"] > best_worst:
                    best_worst = m["worst"]
                    best_state = params.detach().clone()

        # Final evaluation
        if use_early_stop and best_state is not None:
            # Evaluate using the best-recorded params (not final-step)
            m = evaluate_binary(best_state, sim, main_lo, main_hi)
        else:
            m = evaluate_binary(params, sim, main_lo, main_hi)
        seed_results.append(m)

    return seed_results


def summarize(seed_results, label):
    worsts = [r["worst"] for r in seed_results]
    smeans = [r["side_mean"] for r in seed_results]
    flats = sum(1 for r in seed_results if r["flat_top"])
    return {
        "label": label,
        "best_worst": max(worsts),
        "mean_worst": float(np.mean(worsts)),
        "min_worst": min(worsts),
        "best_smean": min(smeans),
        "flat": f"{flats}/{n_restarts}",
        "all_worsts": worsts,
    }


print("=" * 100, flush=True)
print(f"R138 -- Early stopping + LR decay vs R137 baseline", flush=True)
print(f"  Goal: fix mean_worst dip at 1500 steps (Adam drift)", flush=True)
print(f"  n={n}, inc={inc}, freq={freq/1e9}GHz, width={width}d, gd_steps={gd_steps}, {n_restarts} restarts", flush=True)
print("=" * 100, flush=True)

variants = [
    ("baseline (R137)",   False, False),
    ("early-stop only",   True,  False),
    ("lr-decay only",     False, True),
    ("early-stop + decay", True, True),
]

print(f"\n{'variant':<22} | {'best':>7} | {'mean':>7} | {'min':>7} | {'side_mean':>10} | {'flat':>5}", flush=True)
print("-" * 70, flush=True)

results = []
for label, es, lrd in variants:
    print(f"[runner] {label} ...", flush=True)
    seed_results = run_recipe(es, lrd)
    s = summarize(seed_results, label)
    results.append(s)
    print(f"{s['label']:<22} | {s['best_worst']:>+7.2f} | {s['mean_worst']:>+7.2f} | "
          f"{s['min_worst']:>+7.2f} | {s['best_smean']:>+10.2f} | {s['flat']:>5}", flush=True)

print("\n" + "=" * 70, flush=True)
print("Per-seed worst-case (to see if interventions reduce variance):", flush=True)
print("=" * 70, flush=True)
for s in results:
    seed_str = " ".join(f"{w:+.2f}" for w in s["all_worsts"])
    print(f"  {s['label']:<22} | seeds: {seed_str}", flush=True)
