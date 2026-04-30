"""
Round 94 — Pareto frontier (worst_supp vs ripple) at n=51

R65 已 sweep ripple_weight at n=41 width=80 (R63 max-max region).
R94 sweep ripple_weight at n=51 width=30 (current best deployable n).

Generate scatter plot + Pareto frontier curve showing trade-off.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "script")
from methodology_demo import deploy_one_target


def main() -> None:
    out_dir = Path("outputs/r94_pareto_n51")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_spec = {
        "freq_ghz": 38.0, "n": 51, "inc_theta": 51.0,
        "main_lo": 162, "main_hi": 192,  # 15° wide
    }

    ripple_weights = [0.0, 0.5, 1.0, 2.0, 5.0]
    n_restarts = 5  # save time
    gd_steps = 1500

    print("=" * 70)
    print(f"Round 94 — Pareto sweep at n=51 (15° wide flat-top)")
    print(f"  ripple_weights: {ripple_weights}, n_restarts={n_restarts}")
    print("=" * 70)

    all_results = []  # list of (rw, seed_results)
    pareto_summary = []

    for rw in ripple_weights:
        spec = {**base_spec, "name": f"n51_rw{rw}"}
        best, seed_results = deploy_one_target(
            spec, n_restarts=n_restarts, gd_steps=gd_steps,
            ripple_weight=rw, device="cuda:0",
        )
        all_results.append({"rw": rw, "results": seed_results, "best": best["metrics"]})
        pareto_summary.append({
            "rw": rw,
            "best_worst": best["metrics"]["worst_supp"],
            "best_ripple": best["metrics"]["main_ripple"],
            "best_flat_top": best["metrics"]["flat_top_compliant"],
            "median_worst": float(np.median([r["worst_supp"] for r in seed_results])),
            "median_ripple": float(np.median([r["main_ripple"] for r in seed_results])),
            "flat_top_count": sum(1 for r in seed_results if r["flat_top_compliant"]),
        })

    # Print summary
    print("\n" + "=" * 90)
    print(f"{'rw':>5} | {'best worst':>10} | {'best ripple':>11} | {'flat-top':>9} | "
          f"{'median worst':>12} | {'median ripple':>13}")
    print("-" * 90)
    for s in pareto_summary:
        flat_str = f"{s['flat_top_count']}/{n_restarts}"
        print(f"{s['rw']:>5.1f} | {s['best_worst']:>+10.2f} | {s['best_ripple']:>11.2f} | "
              f"{flat_str:>9} | {s['median_worst']:>+12.2f} | {s['median_ripple']:>13.2f}")
    print("=" * 90)

    # Render Pareto curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # All seed results scatter
    colors = ["red", "orange", "yellowgreen", "steelblue", "darkviolet"]
    for r, color in zip(all_results, colors):
        worsts = [s["worst_supp"] for s in r["results"]]
        ripples = [s["main_ripple"] for s in r["results"]]
        flat_top = [s["flat_top_compliant"] for s in r["results"]]
        markers = ["o" if ft else "x" for ft in flat_top]
        for w, rp, m in zip(worsts, ripples, markers):
            axes[0].scatter(rp, w, c=color, marker=m, s=80, alpha=0.7,
                            edgecolors="black", linewidths=0.5)
        # Best
        axes[0].scatter(r["best"]["main_ripple"], r["best"]["worst_supp"],
                        c=color, marker="*", s=200, edgecolors="black", linewidths=1.5,
                        label=f"rw={r['rw']}")

    axes[0].axvline(3, color="gray", linestyle="--", alpha=0.5, label="3 dB ripple ref")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("main ripple (dB)")
    axes[0].set_ylabel("worst suppression (dB)")
    axes[0].set_title("Pareto: worst_supp vs ripple\n(○ = flat-top compliant, × = not)")
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(alpha=0.3)

    # Best per rw
    rws = [s["rw"] for s in pareto_summary]
    bests = [s["best_worst"] for s in pareto_summary]
    ripples = [s["best_ripple"] for s in pareto_summary]
    flats = [s["best_flat_top"] for s in pareto_summary]

    ax2 = axes[1]
    ax2.plot(rws, bests, "bo-", linewidth=2, markersize=10, label="best worst_supp")
    ax2_b = ax2.twinx()
    ax2_b.plot(rws, ripples, "rs-", linewidth=2, markersize=10, label="best ripple")

    for i, (rw, b, r, ft) in enumerate(zip(rws, bests, ripples, flats)):
        marker = "★" if ft else "✗"
        ax2.annotate(f"{b:+.1f} dB\n{marker}", xy=(rw, b), xytext=(0, 10),
                     textcoords="offset points", ha="center", fontsize=9,
                     color="darkblue")

    ax2.set_xlabel("ripple_weight λ")
    ax2.set_ylabel("best worst_supp (dB)", color="blue")
    ax2_b.set_ylabel("best main ripple (dB)", color="red")
    ax2.set_title("Best worst_supp vs ripple_weight\n(★ = flat-top compliant, ✗ = not)")
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_b.tick_params(axis='y', labelcolor='red')
    ax2.grid(alpha=0.3)
    ax2.set_xticks(rws)

    fig.suptitle(
        "Patch Deployment Design Space — Pareto Frontier at n=51 (38 GHz, 15° wide flat-top)\n"
        "Trade-off: low rw → high worst_supp but high ripple; high rw → flat-top but lower worst",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out_path = "outputs/r94_pareto_n51.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Save summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(pareto_summary, f, indent=2)


if __name__ == "__main__":
    main()
