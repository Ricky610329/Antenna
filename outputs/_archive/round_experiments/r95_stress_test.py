"""Round 95 — Stress test: off-axis + wide + flat-top combined."""

import sys
sys.path.insert(0, "script")
from methodology_demo import deploy_one_target

# Hard combined spec: off-axis (-30°), wide (25°), flat-top (rw=2), n=51
# main idx for θc=-30° (idx=120), w=25° (50 samples)
hard_specs = [
    {"name": "easy_baseline_R94",
     "freq_ghz": 38.0, "n": 51, "inc_theta": 51.0,
     "main_lo": 162, "main_hi": 192,  # 15° broadside
     },
    {"name": "off_axis_30",
     "freq_ghz": 38.0, "n": 51, "inc_theta": 51.0,
     "main_lo": 110, "main_hi": 140,  # 15° at -25°
     },
    {"name": "wide_25deg",
     "freq_ghz": 38.0, "n": 51, "inc_theta": 51.0,
     "main_lo": 155, "main_hi": 205,  # 25° broadside
     },
    {"name": "off_axis_wide",
     "freq_ghz": 38.0, "n": 51, "inc_theta": 51.0,
     "main_lo": 105, "main_hi": 155,  # 25° at -25°
     },
]

print("=" * 80)
print("R95 — Methodology stress test (n=51, rw=2, 5 restarts)")
print("=" * 80)

for spec in hard_specs:
    best, sr = deploy_one_target(spec, n_restarts=5, gd_steps=1500, ripple_weight=2.0)
    flats = sum(1 for r in sr if r['flat_top_compliant'])
    print(f"\n{spec['name']}: best worst={best['metrics']['worst_supp']:+.2f} dB, "
          f"ripple={best['metrics']['main_ripple']:.2f}, "
          f"flat-top hit={flats}/{len(sr)}")
