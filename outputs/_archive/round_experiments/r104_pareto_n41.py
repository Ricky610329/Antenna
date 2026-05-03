"""Round 104 — Pareto sweep at n=41 (cheap aperture)."""

import sys, json
sys.path.insert(0, "script")
import numpy as np
from pathlib import Path
from methodology_demo import deploy_one_target

base_spec = {
    "freq_ghz": 38.0, "n": 41, "inc_theta": 51.0,
    "main_lo": 162, "main_hi": 192,
}

ripple_weights = [0.0, 0.5, 1.0, 2.0, 5.0]
results = []

print("=" * 70)
print(f"Round 104 — Pareto sweep at n=41")
print("=" * 70)

for rw in ripple_weights:
    spec = {**base_spec, "name": f"n41_rw{rw}"}
    best, sr = deploy_one_target(spec, n_restarts=5, gd_steps=1500, ripple_weight=rw)
    flats = sum(1 for r in sr if r['flat_top_compliant'])
    worsts = [r['worst_supp'] for r in sr]
    ripples = [r['main_ripple'] for r in sr]
    results.append({
        "rw": rw,
        "best_worst": best['metrics']['worst_supp'],
        "best_ripple": best['metrics']['main_ripple'],
        "best_flat_top": best['metrics']['flat_top_compliant'],
        "median_worst": float(np.median(worsts)),
        "flat_top_count": flats,
    })

print("\n" + "=" * 70)
print(f"{'rw':>4} | {'best worst':>10} | {'best ripple':>11} | {'flat-top':>9} | {'median':>8}")
print("-" * 70)
for s in results:
    flat = "yes" if s['best_flat_top'] else "no"
    print(f"{s['rw']:>4.1f} | {s['best_worst']:>+10.2f} | {s['best_ripple']:>11.2f} | "
          f"{flat + ' ' + str(s['flat_top_count']) + '/5':>9} | {s['median_worst']:>+8.2f}")
print("=" * 70)

# Compare to R94 (n=51)
n51_results = [
    {"rw": 0.0, "best_worst": 7.35, "best_ripple": 10.98, "flat_top_count": 0},
    {"rw": 0.5, "best_worst": 6.69, "best_ripple": 6.72, "flat_top_count": 0},
    {"rw": 1.0, "best_worst": 5.44, "best_ripple": 5.31, "flat_top_count": 2},
    {"rw": 2.0, "best_worst": 1.92, "best_ripple": 2.59, "flat_top_count": 5},
    {"rw": 5.0, "best_worst": -0.17, "best_ripple": 2.02, "flat_top_count": 4},
]

print("\n=== n=41 vs n=51 comparison ===")
print(f"{'rw':>4} | {'n=41 worst':>10} | {'n=51 worst':>10} | {'Δ':>5} | {'n=41 ft':>8} | {'n=51 ft':>8}")
print("-" * 70)
for r41, r51 in zip(results, n51_results):
    delta = r51['best_worst'] - r41['best_worst']
    print(f"{r41['rw']:>4.1f} | {r41['best_worst']:>+10.2f} | {r51['best_worst']:>+10.2f} | "
          f"{delta:>+5.2f} | {r41['flat_top_count']}/5{'':>4} | {r51['flat_top_count']}/5")

# Save and visualize
out_dir = Path("outputs/r104_pareto_n41")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "summary.json", "w") as f:
    json.dump(results, f, indent=2)

# Render combined Pareto plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot([r['best_ripple'] for r in results], [r['best_worst'] for r in results],
        "o-", linewidth=2, markersize=10, color="darkblue", label="n=41 (cheaper)")
ax.plot([r['best_ripple'] for r in n51_results], [r['best_worst'] for r in n51_results],
        "s-", linewidth=2, markersize=10, color="darkgreen", label="n=51 (R94)")

# Annotate points
for r, mark in zip(results, ["o"] * 5):
    flat = "★" if r['best_flat_top'] else "x"
    ax.annotate(f"rw={r['rw']}\n{flat}", xy=(r['best_ripple'], r['best_worst']),
                xytext=(8, 5), textcoords="offset points", fontsize=8, color="darkblue")
for r, mark in zip(n51_results, ["s"] * 5):
    flat = "★" if r['flat_top_count'] >= 5 else "x"
    ax.annotate(f"rw={r['rw']}\n{flat}", xy=(r['best_ripple'], r['best_worst']),
                xytext=(8, -15), textcoords="offset points", fontsize=8, color="darkgreen")

ax.axvline(3, color="gray", linestyle="--", alpha=0.5, label="3 dB ripple (flat-top boundary)")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_xlabel("main ripple (dB)")
ax.set_ylabel("best worst suppression (dB)")
ax.set_title("Patch Deployment Design Space — n=41 vs n=51 Pareto Curves\n"
             "(★ = flat-top compliant, x = not)")
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("outputs/r104_n41_vs_n51_pareto.png", dpi=110, bbox_inches="tight")
print(f"\nSaved: outputs/r104_n41_vs_n51_pareto.png")
