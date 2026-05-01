"""Round 110 — Sub-6G band + normal incidence."""

import sys
sys.path.insert(0, "script")
import numpy as np
from methodology_demo import deploy_one_target

print("=" * 75)
print("R110 — Sub-6G band + normal incidence test (n=51, rw=2, 5 restarts)")
print("=" * 75)

specs = [
    {"name": "5.6GHz_inc51",  "freq_ghz": 5.6,  "inc_theta": 51.0, "main_lo": 162, "main_hi": 192},
    {"name": "5.6GHz_inc0",   "freq_ghz": 5.6,  "inc_theta":  0.0, "main_lo": 162, "main_hi": 192},
    {"name": "38GHz_inc0",    "freq_ghz": 38.0, "inc_theta":  0.0, "main_lo": 162, "main_hi": 192},
    {"name": "38GHz_inc51_ref","freq_ghz":38.0, "inc_theta": 51.0, "main_lo": 162, "main_hi": 192},
]

results = []
for s in specs:
    spec = {"n": 51, "name": s["name"], **s}
    best, sr = deploy_one_target(spec, n_restarts=5, gd_steps=1500, ripple_weight=2.0)
    flats = sum(1 for r in sr if r['flat_top_compliant'])
    worsts = [r['worst_supp'] for r in sr]
    results.append({
        **s,
        "best_worst": best['metrics']['worst_supp'],
        "best_ripple": best['metrics']['main_ripple'],
        "best_flat_top": best['metrics']['flat_top_compliant'],
        "median_worst": float(np.median(worsts)),
        "flat_top_count": flats,
    })

print("\n" + "=" * 75)
print(f"{'config':<25} | {'worst':>8} | {'ripple':>8} | {'flat-top':>9} | {'median':>8}")
print("-" * 75)
for r in results:
    flat = "yes" if r["best_flat_top"] else "no"
    print(f"{r['name']:<25} | {r['best_worst']:>+8.2f} | {r['best_ripple']:>8.2f} | "
          f"{flat + ' ' + str(r['flat_top_count']) + '/5':>9} | {r['median_worst']:>+8.2f}")
print("=" * 75)
