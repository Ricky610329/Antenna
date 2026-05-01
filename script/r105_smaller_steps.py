"""Round 105 — Smaller GD steps (500/1000/1500) for compute budget."""

import sys, time
sys.path.insert(0, "script")
import numpy as np
from methodology_demo import deploy_one_target

base_spec = {
    "freq_ghz": 38.0, "n": 51, "inc_theta": 51.0,
    "main_lo": 162, "main_hi": 192,
}

print("=" * 70)
print("R105 — Smaller GD steps (n=51, rw=2, 5 restarts)")
print("=" * 70)

results = []
for steps in [500, 750, 1000, 1500]:
    print(f"\n--- GD steps = {steps} ---")
    t0 = time.time()
    best, sr = deploy_one_target(
        {**base_spec, "name": f"steps_{steps}"},
        n_restarts=5, gd_steps=steps, ripple_weight=2.0,
    )
    elapsed = time.time() - t0
    flats = sum(1 for r in sr if r['flat_top_compliant'])
    worsts = [r['worst_supp'] for r in sr]
    results.append({
        "steps": steps,
        "best_worst": best['metrics']['worst_supp'],
        "best_ripple": best['metrics']['main_ripple'],
        "median_worst": float(np.median(worsts)),
        "flat_top_hit": flats,
        "elapsed_sec": elapsed,
    })

print("\n" + "=" * 70)
print(f"{'steps':>6} | {'best worst':>10} | {'best ripple':>11} | {'median':>8} | {'flat-top':>9} | {'time':>6}")
print("-" * 70)
for r in results:
    print(f"{r['steps']:>6} | {r['best_worst']:>+10.2f} | {r['best_ripple']:>11.2f} | "
          f"{r['median_worst']:>+8.2f} | {r['flat_top_hit']}/5     | {r['elapsed_sec']:>5.0f}s")
print("=" * 70)

# Cost reduction analysis
ref = next(r for r in results if r['steps'] == 1500)
print(f"\nReference (1500 steps): worst {ref['best_worst']:+.2f}, time {ref['elapsed_sec']:.0f}s")
print("Cost reduction analysis:")
for r in results:
    if r['steps'] == 1500: continue
    cost_save = 1 - r['elapsed_sec'] / ref['elapsed_sec']
    quality_loss = ref['best_worst'] - r['best_worst']
    flat_loss = ref['flat_top_hit'] - r['flat_top_hit']
    print(f"  {r['steps']} steps: -{cost_save*100:.0f}% time, "
          f"-{quality_loss:.2f} dB worst, -{flat_loss}/5 flat-top hit")
