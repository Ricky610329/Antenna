"""Round 134 -- Codify 1-bit recipe selector and validate at held-out combos.

Synthesizes R119, R128, R129, R131, R133 findings into one selector function
that picks (rw, lambda) given (n, inc_deg, freq_hz, width_deg).

Then validates the selector at 6+ unseen (n, inc, freq, width) combos to
confirm it picks the right recipe in regions not directly grid-searched.

Acceptance per validation point: worst > 0 AND flat-top >= 4/5 (or >=2/3 for n=71).
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


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def select_1bit_recipe(n, inc_deg, freq_hz, width_deg):
    """Pick (rw, lambda_mean) for 1-bit RIS optimization.

    Decision tree distilled from R119-R133:
      - rw scales with aperture and main width
      - lambda scales DOWN with optimization difficulty (inc=0 + mmWave)

    Returns dict with 'rw', 'lambda_mean', and 'tier' label.
    Raises if config is outside validated envelope.
    """
    if width_deg > 30:
        raise ValueError(f"width={width_deg} exceeds validated 30 deg envelope")
    if n not in (31, 51, 71):
        raise ValueError(f"n={n} not in validated set (31, 51, 71)")

    # n=71 branch (R133)
    if n == 71:
        if inc_deg == 0 and freq_hz >= 50e9:
            return {"rw": 5.0, "lambda_mean": 0.3, "tier": "R133 n=71 inc=0 mmWave rescue"}
        # Other n=71 configs: scale rw up from n=51 baselines
        if width_deg <= 15:
            return {"rw": 5.0, "lambda_mean": 0.5, "tier": "n=71 narrow extrapolation"}
        return {"rw": 7.0, "lambda_mean": 0.5, "tier": "n=71 wide extrapolation"}

    # n=51 wide cap (R129)
    if width_deg > 15:
        if width_deg <= 20:
            return {"rw": 3.0, "lambda_mean": 1.0, "tier": "R129 wide cap 20deg"}
        return {"rw": 3.0, "lambda_mean": 0.5, "tier": "R129 wide cap 30deg (marginal)"}

    # n=51 narrow cap, normal incidence + mmWave special cases (R131)
    if inc_deg == 0 and freq_hz >= 20e9:
        if freq_hz >= 50e9:
            raise ValueError("inc=0 + freq>=50GHz at n=51 has no clean recipe -> use n=71")
        if freq_hz >= 35e9:
            return {"rw": 2.0, "lambda_mean": 0.5, "tier": "R131 inc=0 38GHz rescue"}
        return {"rw": 2.0, "lambda_mean": 0.3, "tier": "R131 inc=0 28GHz rescue"}

    # n=51 default: R119 baseline
    return {"rw": 2.0, "lambda_mean": 1.0, "tier": "R119 baseline"}


def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi


def deploy_1bit(n, freq_hz, inc_deg, width_deg, mean_w, ripple_w, n_restarts=5):
    sim = RISSimulator(element_num=n, freq_hz=freq_hz, inc_theta_deg=inc_deg)
    main_lo, main_hi = steer_to_indices(0, width_deg)
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


# Held-out validation grid (none of these were used to fit the selector)
validation_combos = [
    # n=51 known-good zones
    {"n": 51, "inc": 30, "freq": 28e9, "width": 10, "label": "off-normal 28GHz"},
    {"n": 51, "inc": 70, "freq": 60e9, "width": 10, "label": "70deg + 60GHz mmWave"},
    {"n": 51, "inc": 51, "freq": 38e9, "width": 15, "label": "sweet inc + boundary width"},
    # n=51 wide
    {"n": 51, "inc": 51, "freq": 38e9, "width": 20, "label": "sweet inc + wide cap"},
    # n=71 extrapolations (no grid search done before)
    {"n": 71, "inc": 30, "freq": 28e9, "width": 10, "label": "n=71 off-normal lower freq"},
    {"n": 71, "inc": 51, "freq": 38e9, "width": 10, "label": "n=71 sweet inc + 38GHz"},
]

print("=" * 110, flush=True)
print(f"R134 -- 1-bit recipe selector validation at {len(validation_combos)} held-out combos", flush=True)
print(f"  Acceptance: worst > 0 AND flat-top >= 4/5 (n=51) or >= 2/3 (n=71 limited restarts)", flush=True)
print("=" * 110, flush=True)

print(f"\n{'config':<45} | {'recipe':<35} | {'worst':>7} | {'side_mean':>10} | {'flat':>5} | {'verdict':>8}", flush=True)
print("-" * 130, flush=True)

pass_count = 0
for combo in validation_combos:
    label = combo["label"]
    n, inc, freq, w = combo["n"], combo["inc"], combo["freq"], combo["width"]
    n_restarts = 3 if n == 71 else 5

    try:
        recipe = select_1bit_recipe(n, inc, freq, w)
    except ValueError as e:
        print(f"n={n} inc={inc} freq={freq/1e9:g}GHz w={w}d ({label[:18]:<18})  selector raised: {e}", flush=True)
        continue

    rw, lam = recipe["rw"], recipe["lambda_mean"]
    print(f"[runner] n={n} inc={inc} freq={freq/1e9:g}GHz w={w}d -> rw={rw} lam={lam} ({recipe['tier']})", flush=True)
    b, f = deploy_1bit(n, freq, inc, w, lam, rw, n_restarts=n_restarts)
    flat_str = "OK" if f == n_restarts else f"{f}/{n_restarts}"
    threshold = 4 if n == 51 else 2
    passed = (f >= threshold) and (b["worst"] > 0)
    if passed: pass_count += 1
    verdict = "PASS" if passed else "FAIL"

    cfg_str = f"n={n} inc={inc:>2}d freq={freq/1e9:>4g}GHz w={w}d"
    rec_str = f"rw={rw} lam={lam} ({recipe['tier'][:18]})"
    print(f"{cfg_str:<45} | {rec_str:<35} | {b['worst']:>+7.2f} | {b['side_mean']:>+10.2f} | {flat_str:>5} | {verdict:>8}", flush=True)

print("=" * 110, flush=True)
print(f"PASS rate: {pass_count}/{len(validation_combos)} held-out combos", flush=True)
print("=" * 110, flush=True)
