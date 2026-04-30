"""Round 99 — GD steps scaling validation."""

import sys
sys.path.insert(0, "script")
from methodology_demo import deploy_one_target

base_spec = {
    "freq_ghz": 38.0, "n": 51, "inc_theta": 51.0,
    "main_lo": 162, "main_hi": 192,  # 15° broadside
    "name": "n51_rw2_steps_test",
}

print("=" * 70)
print("R99 — GD steps scaling (n=51, rw=2, 5 restarts)")
print("=" * 70)

results = []
for steps in [1500, 3000, 5000]:
    print(f"\n--- GD steps = {steps} ---")
    best, sr = deploy_one_target(
        {**base_spec, "name": f"steps_{steps}"},
        n_restarts=5, gd_steps=steps, ripple_weight=2.0,
    )
    flats = sum(1 for r in sr if r['flat_top_compliant'])
    import numpy as np
    worsts = [r['worst_supp'] for r in sr]
    results.append({
        "steps": steps,
        "best_worst": best['metrics']['worst_supp'],
        "best_ripple": best['metrics']['main_ripple'],
        "median_worst": float(np.median(worsts)),
        "flat_top_hit": flats,
    })

print("\n" + "=" * 70)
print(f"{'steps':>6} | {'best worst':>10} | {'best ripple':>11} | {'median worst':>12} | {'flat-top':>9}")
print("-" * 70)
for r in results:
    print(f"{r['steps']:>6} | {r['best_worst']:>+10.2f} | {r['best_ripple']:>11.2f} | "
          f"{r['median_worst']:>+12.2f} | {r['flat_top_hit']}/5")
print("=" * 70)
