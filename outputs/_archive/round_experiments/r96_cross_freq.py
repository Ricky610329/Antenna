"""Round 96 — Cross-frequency methodology validation."""

import sys
sys.path.insert(0, "script")
from methodology_demo import deploy_one_target

specs = [
    {"name": "28GHz_n51", "freq_ghz": 28.0, "n": 51, "inc_theta": 51.0,
     "main_lo": 162, "main_hi": 192},
    {"name": "38GHz_n51", "freq_ghz": 38.0, "n": 51, "inc_theta": 51.0,
     "main_lo": 162, "main_hi": 192},
    {"name": "60GHz_n51", "freq_ghz": 60.0, "n": 51, "inc_theta": 51.0,
     "main_lo": 162, "main_hi": 192},
]

print("=" * 75)
print("R96 — Cross-frequency methodology (n=51, rw=2, 5 restarts, 15° broadside)")
print("=" * 75)

results = []
for spec in specs:
    best, sr = deploy_one_target(spec, n_restarts=5, gd_steps=1500, ripple_weight=2.0)
    flats = sum(1 for r in sr if r['flat_top_compliant'])
    results.append({
        "name": spec["name"], "freq": spec["freq_ghz"],
        "best_worst": best['metrics']['worst_supp'],
        "best_ripple": best['metrics']['main_ripple'],
        "flat_top_hit": f"{flats}/{len(sr)}",
    })

print("\n" + "=" * 75)
print(f"{'freq':>6} | {'best worst':>10} | {'best ripple':>11} | {'flat-top':>9}")
print("-" * 75)
for r in results:
    print(f"{r['freq']:>5.0f}G | {r['best_worst']:>+10.2f} | {r['best_ripple']:>11.2f} | {r['flat_top_hit']:>9}")
print("=" * 75)
