"""Round 112 — Test if rw=5 also fixes inc=70 (R106 weak case)."""

import sys
sys.path.insert(0, "script")
import numpy as np
from methodology_demo import deploy_one_target

print("=" * 70)
print("R112 — rw=5 fix generalize to inc=70 (R106 weak case)?")
print("=" * 70)

base = {"freq_ghz": 38.0, "n": 51, "main_lo": 162, "main_hi": 192}

trials = [
    {"name": "R106_inc70_rw2",   "inc_theta": 70.0, "rw": 2.0},
    {"name": "R112_inc70_rw5",   "inc_theta": 70.0, "rw": 5.0},
    {"name": "R106_inc30_rw2",   "inc_theta": 30.0, "rw": 2.0},
    {"name": "R112_inc30_rw5",   "inc_theta": 30.0, "rw": 5.0},
    {"name": "ref_inc51_rw2",    "inc_theta": 51.0, "rw": 2.0},
]

for t in trials:
    spec = {**base, "inc_theta": t["inc_theta"], "name": t["name"]}
    best, sr = deploy_one_target(spec, n_restarts=5, gd_steps=1500, ripple_weight=t["rw"])
    flats = sum(1 for r in sr if r['flat_top_compliant'])
    print(f"\n{t['name']}: worst={best['metrics']['worst_supp']:+.2f}, "
          f"ripple={best['metrics']['main_ripple']:.2f}, flat-top={flats}/5")
