"""Round 103 — Manufacturing tolerance test (random bit flips)."""

import sys
sys.path.insert(0, "script")
import numpy as np
import torch

from antenna.ris import RISSimulator
from antenna.utils.config import config
from methodology_demo import evaluate_metrics

config.device = "cuda:0"

# Load R92 best pattern (n=51)
pat_orig = np.load("outputs/r92_aperture_scaling/n51_best_pattern.npy")
n = 51
freq = 38e9
inc = 51.0
main_lo, main_hi = 162, 192

sim = RISSimulator(element_num=n, freq_hz=freq, inc_theta_deg=inc)

# Original metrics
orig_pat_t = torch.tensor(pat_orig, device="cuda:0", dtype=torch.float32)
with torch.no_grad():
    orig_resp = sim(orig_pat_t)["response"].cpu().numpy()
orig_metrics = evaluate_metrics(orig_resp, main_lo, main_hi)
print(f"Original: worst={orig_metrics['worst_supp']:+.2f}, "
      f"ripple={orig_metrics['main_ripple']:.2f}, "
      f"flat-top={'yes' if orig_metrics['flat_top_compliant'] else 'no'}")

# Test bit flip rates
print("\n=== Random bit flip robustness ===")
print(f"{'flip rate':>10} | {'mean worst':>10} | {'std worst':>9} | {'mean ripple':>11} | {'flat-top hit':>13}")
print("-" * 75)

flip_rates = [0.01, 0.02, 0.05, 0.10, 0.20]
n_trials = 30

for flip_rate in flip_rates:
    worsts, ripples, flats = [], [], []
    for trial in range(n_trials):
        rng = np.random.RandomState(trial)
        flip_mask = rng.rand(n, n) < flip_rate
        pat_perturbed = np.where(flip_mask, 1 - pat_orig, pat_orig)
        pat_t = torch.tensor(pat_perturbed, device="cuda:0", dtype=torch.float32)
        with torch.no_grad():
            resp = sim(pat_t)["response"].cpu().numpy()
        m = evaluate_metrics(resp, main_lo, main_hi)
        worsts.append(m["worst_supp"])
        ripples.append(m["main_ripple"])
        flats.append(m["flat_top_compliant"])
    
    print(f"{flip_rate*100:>9.0f}% | {np.mean(worsts):>+10.2f} | {np.std(worsts):>9.2f} | "
          f"{np.mean(ripples):>11.2f} | {sum(flats):>2}/{n_trials} ({100*sum(flats)/n_trials:.0f}%)")
print("=" * 75)

# Theoretical degradation
print("\nDegradation per flip rate:")
for flip_rate in flip_rates:
    n_flipped_avg = flip_rate * n * n
    print(f"  {flip_rate*100:.0f}% flip: avg ~{n_flipped_avg:.0f}/{n*n} elements perturbed")
