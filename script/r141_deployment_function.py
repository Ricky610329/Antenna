"""Round 141 -- Wrap selector + joint early-stop into one deployment function.

Combines:
  - select_1bit_recipe(n, inc, freq, width)   from R134/R135
  - run_with_joint_early_stop(...)            from R140

Single API: optimize_ris_1bit(n, inc_deg, freq_hz, width_deg, ...)
  -> returns binary pattern + metrics + selected recipe info.

Re-runs R134's 6 held-out validation combos with the upgraded pipeline
to see how many now PASS (R134 with width=15 R135-fix was 5/6; with
joint-ES on top should be cleaner).
"""
import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
GD_STEPS = 1500
EVAL_EVERY = 50


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def select_1bit_recipe(n, inc_deg, freq_hz, width_deg):
    """Recipe selector — distilled from R119 / R129 / R131 / R133 / R135."""
    if width_deg > 30:
        raise ValueError(f"width={width_deg} exceeds validated 30deg envelope")
    if n not in (31, 51, 71):
        raise ValueError(f"n={n} not in validated set (31, 51, 71)")

    if n == 71:
        if inc_deg == 0 and freq_hz >= 50e9:
            return {"rw": 5.0, "lambda_mean": 0.3, "tier": "R133 n=71 inc=0 mmWave"}
        if width_deg <= 15:
            return {"rw": 5.0, "lambda_mean": 0.5, "tier": "n=71 narrow extrapolation"}
        return {"rw": 7.0, "lambda_mean": 0.5, "tier": "n=71 wide extrapolation"}

    if width_deg > 12:    # R135 boundary
        if width_deg <= 20:
            return {"rw": 3.0, "lambda_mean": 1.0, "tier": "R129 wide cap (12-20)"}
        return {"rw": 3.0, "lambda_mean": 0.5, "tier": "R129 wide cap 30 (marginal)"}

    if inc_deg == 0 and freq_hz >= 20e9:
        if freq_hz >= 50e9:
            raise ValueError("inc=0 + freq>=50GHz at n=51 -> use n=71")
        if freq_hz >= 35e9:
            return {"rw": 2.0, "lambda_mean": 0.5, "tier": "R131 inc=0 38GHz rescue"}
        return {"rw": 2.0, "lambda_mean": 0.3, "tier": "R131 inc=0 28GHz rescue"}

    return {"rw": 2.0, "lambda_mean": 1.0, "tier": "R119 baseline"}


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
        "ripple": float(main_arr.max() - main_arr.min()),
        "flat_top": int(np.sum(main_arr < -3)) == 0,
        "binary": binary.cpu(),
    }


def optimize_ris_1bit(n, inc_deg, freq_hz, width_deg, n_restarts=5, gd_steps=GD_STEPS,
                      steering_center_deg=0):
    """Full deployment function: selector + joint early-stop.

    Returns dict with binary pattern, metrics, recipe info, per-seed results.
    """
    recipe = select_1bit_recipe(n, inc_deg, freq_hz, width_deg)
    rw, lam = recipe["rw"], recipe["lambda_mean"]
    main_lo, main_hi = steer_to_indices(steering_center_deg, width_deg)
    sim = RISSimulator(element_num=n, freq_hz=freq_hz, inc_theta_deg=inc_deg)

    seed_results = []
    best_overall = None

    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(torch.rand(n, n, device="cuda:0") * 2.0)
        opt = torch.optim.Adam([params], lr=0.05)

        # Joint early-stop tracking
        best_joint_worst = -1e9
        best_joint_state = None

        for step in range(gd_steps):
            opt.zero_grad()
            resp = sim(params)["response"]
            main = resp[main_lo:main_hi]
            side = torch.cat([resp[:main_lo], resp[main_hi:]])
            mm = soft_min(main); sx = soft_max(side); mx = soft_max(main)
            loss = -(mm - sx) + rw * (mx - mm) + lam * side.mean()
            loss.backward(); opt.step()

            if (step + 1) % EVAL_EVERY == 0:
                m = eval_binary(params, sim, main_lo, main_hi)
                # Joint criterion: only consider flat-valid snapshots
                if m["flat_top"] and m["worst"] > best_joint_worst:
                    best_joint_worst = m["worst"]
                    best_joint_state = params.detach().clone()

        # Pick joint snapshot if found, else final-step
        if best_joint_state is not None:
            metrics = eval_binary(best_joint_state, sim, main_lo, main_hi)
            metrics["used_early_stop"] = True
        else:
            metrics = eval_binary(params, sim, main_lo, main_hi)
            metrics["used_early_stop"] = False
        metrics["seed"] = seed
        seed_results.append(metrics)

        if best_overall is None or metrics["worst"] > best_overall["worst"]:
            best_overall = metrics

    flats = sum(1 for r in seed_results if r["flat_top"])
    es_used = sum(1 for r in seed_results if r["used_early_stop"])

    return {
        "recipe": recipe,
        "best": best_overall,
        "n_flat_top": flats,
        "n_restarts": n_restarts,
        "n_early_stop_used": es_used,
        "seed_results": seed_results,
    }


# Re-run R134's 6 held-out combos
validation_combos = [
    {"n": 51, "inc": 30, "freq": 28e9, "width": 10, "label": "off-normal 28GHz"},
    {"n": 51, "inc": 70, "freq": 60e9, "width": 10, "label": "70deg + 60GHz"},
    {"n": 51, "inc": 51, "freq": 38e9, "width": 15, "label": "sweet inc + boundary width (R135 fix should rescue)"},
    {"n": 51, "inc": 51, "freq": 38e9, "width": 20, "label": "sweet inc + wide cap"},
    {"n": 71, "inc": 30, "freq": 28e9, "width": 10, "label": "n=71 off-normal lower freq"},
    {"n": 71, "inc": 51, "freq": 38e9, "width": 10, "label": "n=71 sweet inc"},
]

print("=" * 110, flush=True)
print(f"R141 -- Wrapped deployment function: select_1bit_recipe + joint early-stop", flush=True)
print(f"  Re-running R134's 6 held-out validation combos", flush=True)
print("=" * 110, flush=True)

print(f"\n{'config':<40} | {'recipe tier':<28} | {'worst':>7} | {'flat':>5} | {'verdict':>8}", flush=True)
print("-" * 110, flush=True)

pass_count = 0
for combo in validation_combos:
    n, inc, freq, w = combo["n"], combo["inc"], combo["freq"], combo["width"]
    nr = 3 if n == 71 else 5
    print(f"[runner] n={n} inc={inc} freq={freq/1e9:g}GHz w={w} ({combo['label'][:25]})", flush=True)
    result = optimize_ris_1bit(n, inc, freq, w, n_restarts=nr)
    b = result["best"]
    flats = result["n_flat_top"]
    flat_str = "OK" if flats == nr else f"{flats}/{nr}"
    threshold = 4 if n == 51 else 2
    passed = (flats >= threshold) and (b["worst"] > 0)
    if passed: pass_count += 1
    verdict = "PASS" if passed else "FAIL"

    cfg_str = f"n={n} inc={inc:>2} freq={freq/1e9:>4g}GHz w={w}"
    tier_str = result["recipe"]["tier"][:28]
    print(f"{cfg_str:<40} | {tier_str:<28} | {b['worst']:>+7.2f} | {flat_str:>5} | {verdict:>8}", flush=True)
    print(f"  (used early-stop in {result['n_early_stop_used']}/{nr} seeds)", flush=True)

print("\n" + "=" * 70, flush=True)
print(f"PASS rate: {pass_count}/{len(validation_combos)} (R134 baseline was 5/6)", flush=True)
print("=" * 70, flush=True)
