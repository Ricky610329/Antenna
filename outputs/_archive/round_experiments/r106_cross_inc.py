"""Round 106 — Cross-incidence test."""

import sys
sys.path.insert(0, "script")
import numpy as np
from methodology_demo import deploy_one_target

print("=" * 70)
print("R106 — Cross-incidence (n=51, 38GHz, 15° broadside flat-top, rw=2)")
print("=" * 70)

results = []
for inc in [30.0, 51.0, 60.0, 70.0]:
    spec = {
        "freq_ghz": 38.0, "n": 51, "inc_theta": inc,
        "main_lo": 162, "main_hi": 192,
        "name": f"inc_{inc}",
    }
    best, sr = deploy_one_target(spec, n_restarts=5, gd_steps=1500, ripple_weight=2.0)
    flats = sum(1 for r in sr if r['flat_top_compliant'])
    worsts = [r['worst_supp'] for r in sr]
    results.append({
        "inc": inc,
        "best_worst": best['metrics']['worst_supp'],
        "best_ripple": best['metrics']['main_ripple'],
        "best_flat_top": best['metrics']['flat_top_compliant'],
        "median_worst": float(np.median(worsts)),
        "flat_top_count": flats,
    })

print("\n" + "=" * 70)
print(f"{'inc':>5} | {'best worst':>10} | {'best ripple':>11} | {'flat-top':>9} | {'median':>8}")
print("-" * 70)
for s in results:
    flat_str = "yes" if s["best_flat_top"] else "no"
    print(f"{s['inc']:>4.0f}° | {s['best_worst']:>+10.2f} | {s['best_ripple']:>11.2f} | "
          f"{flat_str + ' ' + str(s['flat_top_count']) + '/5':>9} | {s['median_worst']:>+8.2f}")
print("=" * 70)
