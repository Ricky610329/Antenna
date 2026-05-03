"""Round 136 -- Fab tolerance test for 1-bit recipe selector.

Real RIS hardware has phase tolerance: a "0" element produces phase ~0 +/- noise,
"pi" element produces phase ~pi +/- noise. R136 tests if optimized binary
patterns survive realistic phase perturbation.

Approach:
  1. Optimize 3 representative recipes from R134 selector.
  2. At evaluation, perturb binary phases by Gaussian noise (sigma in radians).
  3. Recompute metrics under noise.
  4. Sweep noise std: 0% / 1% / 2% / 5% (as fraction of pi).

Acceptance: recipe is fab-robust if metrics degrade gracefully (worst stays > 0,
flat-top stays >= 4/5) at sigma=2% level (typical commodity phase shifter).

Configurations:
  A: n=51, inc=51, 38GHz, w=10  (R119 baseline, sweet spot)
  B: n=51, inc=51, 38GHz, w=18  (R129 wide, recipe>R119)
  C: n=71, inc=51, 38GHz, w=10  (n=71 extrapolation)
"""
import sys
sys.path.insert(0, "script")
import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config

config.device = "cuda:0"
freq = 38e9
inc = 51.0
gd_steps = 1500


def soft_max(x, beta=20.0): return (1/beta) * torch.logsumexp(beta * x, dim=-1)
def soft_min(x, beta=20.0): return -(1/beta) * torch.logsumexp(-beta * x, dim=-1)


def steer_to_indices(center_deg, width_deg):
    lo = int(round((center_deg - width_deg/2 + 90) / 0.5))
    hi = int(round((center_deg + width_deg/2 + 90) / 0.5))
    return lo, hi


def optimize_1bit(n, width_deg, mean_w, ripple_w, n_restarts):
    """Optimize a 1-bit pattern under given recipe; return best binary pattern."""
    main_lo, main_hi = steer_to_indices(0, width_deg)
    sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)
    best = None
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
        worst = float(main_arr.min() - side_arr.max())
        if best is None or worst > best["worst"]:
            best = {"binary": binary.cpu(), "worst": worst, "seed": seed,
                    "main_lo": main_lo, "main_hi": main_hi, "n": n, "sim": sim}
    return best


def evaluate_with_noise(opt_result, noise_sigma_pi, n_trials):
    """Apply Gaussian phase noise (sigma as fraction of pi) and return metric stats."""
    binary = opt_result["binary"]   # 0 or 1, representing phase 0 or pi
    main_lo, main_hi = opt_result["main_lo"], opt_result["main_hi"]
    sim = opt_result["sim"]
    sigma_rad = noise_sigma_pi * np.pi  # convert to radians

    # Binary represents phase = binary * pi (so values 0 or pi)
    # After noise: phase = binary*pi + noise
    # Sim takes input as multiple of pi (phase = input * pi), so we need to convert back
    metrics = []
    for trial in range(n_trials):
        torch.manual_seed(1000 + trial)
        noise_rad = torch.randn_like(binary) * sigma_rad
        phase_rad = binary * np.pi + noise_rad
        phase_in_pi = phase_rad / np.pi  # what the sim expects
        with torch.no_grad():
            phase_in_pi = phase_in_pi.to("cuda:0")
            resp = sim(phase_in_pi)["response"].cpu().numpy()
        main_arr = resp[main_lo:main_hi]
        side_arr = np.delete(resp, np.arange(main_lo, main_hi))
        s = {
            "worst": float(main_arr.min() - side_arr.max()),
            "side_mean": float(side_arr.mean()),
            "ripple": float(main_arr.max() - main_arr.min()),
            "flat_top": int(np.sum(main_arr < -3)) == 0,
        }
        metrics.append(s)
    return metrics


configs = [
    {"label": "A: n=51 inc=51 38GHz w=10 (R119)",
     "n": 51, "width": 10, "rw": 2.0, "lam": 1.0, "n_restarts": 5},
    {"label": "B: n=51 inc=51 38GHz w=18 (R129 wide)",
     "n": 51, "width": 18, "rw": 3.0, "lam": 1.0, "n_restarts": 5},
    {"label": "C: n=71 inc=51 38GHz w=10 (n=71 extrap)",
     "n": 71, "width": 10, "rw": 5.0, "lam": 0.5, "n_restarts": 3},
]

noise_levels_pct = [0, 1, 2, 5]   # as % of pi
n_noise_trials = 20

print("=" * 110, flush=True)
print(f"R136 -- Fab tolerance test for 1-bit recipe selector", flush=True)
print(f"  3 recipes x 4 noise levels (0/1/2/5%-of-pi) x {n_noise_trials} trials", flush=True)
print("=" * 110, flush=True)

for cfg in configs:
    print(f"\n{'='*70}", flush=True)
    print(f"CONFIG {cfg['label']}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"[runner] optimizing pattern...", flush=True)
    opt = optimize_1bit(cfg["n"], cfg["width"], cfg["lam"], cfg["rw"], cfg["n_restarts"])
    print(f"  baseline (no noise) worst = {opt['worst']:+.2f} dB (seed={opt['seed']})", flush=True)

    print(f"\n{'noise%':>7} | {'worst_mean':>11} | {'worst_min':>10} | {'side_mean':>10} | "
          f"{'ripple':>7} | {'flat_pass':>10}", flush=True)
    print("-" * 75, flush=True)

    for noise_pct in noise_levels_pct:
        sigma_pi = noise_pct / 100.0
        metrics = evaluate_with_noise(opt, sigma_pi, n_noise_trials)
        worsts = [m["worst"] for m in metrics]
        smeans = [m["side_mean"] for m in metrics]
        ripples = [m["ripple"] for m in metrics]
        flats = sum(1 for m in metrics if m["flat_top"])
        print(f"{noise_pct:>5}%  | {np.mean(worsts):>+11.2f} | {min(worsts):>+10.2f} | "
              f"{np.mean(smeans):>+10.2f} | {np.mean(ripples):>+7.2f} | "
              f"{flats:>4}/{n_noise_trials}", flush=True)

print("\n" + "=" * 70, flush=True)
print("Verdict: recipe is fab-robust if at sigma=2% noise:", flush=True)
print("  - worst_min > 0 (no individual trial fails worst-case)", flush=True)
print("  - flat_pass >= 80% of trials", flush=True)
print("=" * 70, flush=True)
