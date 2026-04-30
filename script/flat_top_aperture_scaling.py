"""
Round 92 — Flat-top deployment ceiling vs aperture

R91 n=41 flat-top: best worst +0.26 dB.
試 n={41, 51, 61, 71} (15° wide main, rw=2, 10 restarts each).
Larger aperture → 更多 phase DoF → 預期更好 flat-top + worst_supp.

對 patch: aperture vs surface area vs cost trade-off 也是 deployment decision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from antenna.ris import RISSimulator
from antenna.utils.config import config

sys.path.insert(0, "script")
from methodology_demo import (
    worst_case_loss, evaluate_metrics, deploy_one_target,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=int, nargs="+", default=[41, 51, 61, 71])
    p.add_argument("--n_restarts", type=int, default=10)
    p.add_argument("--gd_steps", type=int, default=1500)
    p.add_argument("--ripple_weight", type=float, default=2.0)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    out_dir = Path("outputs/r92_aperture_scaling")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed: 38 GHz, broadside, 15° wide flat-top, rw=2
    base_spec = {
        "freq_ghz": 38.0, "inc_theta": 51.0,
        "main_lo": 162, "main_hi": 192,  # 15° wide centered at broadside
    }

    print("=" * 70)
    print(f"Round 92 — Aperture scaling for flat-top deployment ceiling")
    print(f"  Pipeline: free-phase + worst-case (rw={args.ripple_weight}) + "
          f"{args.n_restarts} restarts + 1-bit quantize")
    print("=" * 70)

    summary = []
    for n in args.ns:
        spec = {**base_spec, "n": n, "name": f"flat_top_n{n}"}
        best, seed_results = deploy_one_target(
            spec, n_restarts=args.n_restarts, gd_steps=args.gd_steps,
            ripple_weight=args.ripple_weight, device=args.device,
        )
        worsts = [r["worst_supp"] for r in seed_results]
        ripples = [r["main_ripple"] for r in seed_results]
        flat_hits = sum(1 for r in seed_results if r["flat_top_compliant"])

        summary.append({
            "n": n,
            "aperture_lambda": n * 0.5,
            "elements": n * n,
            "best_worst": best["metrics"]["worst_supp"],
            "best_ripple": best["metrics"]["main_ripple"],
            "best_flat_top": best["metrics"]["flat_top_compliant"],
            "median_worst": float(np.median(worsts)),
            "mean_worst": float(np.mean(worsts)),
            "std_worst": float(np.std(worsts)),
            "flat_top_hit_rate": f"{flat_hits}/{len(seed_results)}",
        })

        # Save best
        np.save(out_dir / f"n{n}_best_pattern.npy", best["binary_pattern"])
        np.save(out_dir / f"n{n}_best_response.npy", best["response"])

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'n':>4} | {'aperture':>8} | {'elements':>9} | {'best worst':>10} | "
          f"{'ripple':>7} | {'flat-top':>9} | {'median':>8} | {'mean±std':>12}")
    print("-" * 90)
    for s in summary:
        flat_str = "YES" if s["best_flat_top"] else "no"
        print(f"{s['n']:>4} | {s['aperture_lambda']:>6.1f}λ | {s['elements']:>9} | "
              f"{s['best_worst']:>+10.2f} | {s['best_ripple']:>7.2f} | "
              f"{flat_str + ' ' + s['flat_top_hit_rate']:>9} | "
              f"{s['median_worst']:>+8.2f} | "
              f"{s['mean_worst']:+.2f}±{s['std_worst']:.2f}")
    print("=" * 90)

    # Theoretical peak gain
    print("\nTheoretical context:")
    for s in summary:
        peak_gain = 10 * np.log10(s["elements"])
        # Suppression isn't exactly array gain, but related:
        # spread over main width samples: peak_gain - 10*log10(main_width)
        main_w = base_spec["main_hi"] - base_spec["main_lo"]
        spread_gain = peak_gain - 10 * np.log10(main_w)
        print(f"  n={s['n']}: peak gain {peak_gain:+.1f} dB, "
              f"spread to {main_w} samples ≈ {spread_gain:+.1f} dB / sample")

    # Save summary
    import json
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir}/summary.json")


if __name__ == "__main__":
    main()
