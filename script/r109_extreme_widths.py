"""Round 109 — Extreme target widths."""

import sys
sys.path.insert(0, "script")
import numpy as np
from methodology_demo import deploy_one_target

print("=" * 70)
print("R109 — Extreme target widths (n=51, 38GHz, broadside, rw=2)")
print("=" * 70)

# main_lo, main_hi for various widths centered at broadside (idx 180)
specs = [
    {"name": "5deg_narrow",  "main_lo": 175, "main_hi": 185, "width_deg": 5},
    {"name": "10deg",         "main_lo": 170, "main_hi": 190, "width_deg": 10},
    {"name": "15deg_baseline","main_lo": 162, "main_hi": 192, "width_deg": 15},
    {"name": "30deg",         "main_lo": 150, "main_hi": 210, "width_deg": 30},
    {"name": "45deg_wide",    "main_lo": 135, "main_hi": 225, "width_deg": 45},
]

results = []
for s in specs:
    spec = {
        "freq_ghz": 38.0, "n": 51, "inc_theta": 51.0,
        "main_lo": s["main_lo"], "main_hi": s["main_hi"],
        "name": s["name"],
    }
    best, sr = deploy_one_target(spec, n_restarts=5, gd_steps=1500, ripple_weight=2.0)
    flats = sum(1 for r in sr if r['flat_top_compliant'])
    results.append({
        **s,
        "best_worst": best['metrics']['worst_supp'],
        "best_ripple": best['metrics']['main_ripple'],
        "best_flat_top": best['metrics']['flat_top_compliant'],
        "flat_top_count": flats,
    })

print("\n" + "=" * 70)
print(f"{'width':>10} | {'samples':>7} | {'best worst':>10} | {'best ripple':>11} | {'flat-top':>9}")
print("-" * 70)
for r in results:
    samples = r["main_hi"] - r["main_lo"]
    flat = "yes" if r["best_flat_top"] else "no"
    print(f"{r['width_deg']:>8.0f}°  | {samples:>7} | {r['best_worst']:>+10.2f} | "
          f"{r['best_ripple']:>11.2f} | {flat + ' ' + str(r['flat_top_count']) + '/5':>9}")
print("=" * 70)
