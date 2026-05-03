"""Round 111 — Try to fix 38 GHz inc=0 with stricter recipe."""

import sys
sys.path.insert(0, "script")
import numpy as np
from methodology_demo import deploy_one_target

print("=" * 75)
print("R111 — Try to fix 38GHz inc=0 catastrophic (R110)")
print("=" * 75)

base = {"freq_ghz": 38.0, "n": 51, "main_lo": 162, "main_hi": 192}

trials = [
    {"name": "R110_baseline_inc0",  "inc_theta": 0.0,  "n_restarts": 5,  "rw": 2.0},
    {"name": "fix1_inc0_rw5",        "inc_theta": 0.0,  "n_restarts": 5,  "rw": 5.0},
    {"name": "fix2_inc0_10_restarts","inc_theta": 0.0,  "n_restarts": 10, "rw": 2.0},
    {"name": "fix3_inc0_rw5_10_restarts","inc_theta": 0.0, "n_restarts": 10, "rw": 5.0},
    {"name": "ref_inc51",            "inc_theta": 51.0, "n_restarts": 5,  "rw": 2.0},
]

results = []
for t in trials:
    spec = {**base, "inc_theta": t["inc_theta"], "name": t["name"]}
    best, sr = deploy_one_target(spec, n_restarts=t["n_restarts"], gd_steps=1500, ripple_weight=t["rw"])
    flats = sum(1 for r in sr if r['flat_top_compliant'])
    worsts = [r['worst_supp'] for r in sr]
    results.append({
        **t,
        "best_worst": best['metrics']['worst_supp'],
        "best_ripple": best['metrics']['main_ripple'],
        "best_flat_top": best['metrics']['flat_top_compliant'],
        "median_worst": float(np.median(worsts)),
        "flat_top_count": flats,
    })

print("\n" + "=" * 75)
print(f"{'config':<35} | {'worst':>8} | {'ripple':>8} | {'flat-top':>9}")
print("-" * 75)
for r in results:
    flat = "yes" if r["best_flat_top"] else "no"
    print(f"{r['name']:<35} | {r['best_worst']:>+8.2f} | {r['best_ripple']:>8.2f} | "
          f"{flat + ' ' + str(r['flat_top_count']) + '/' + str(r['n_restarts']):>9}")
print("=" * 75)
