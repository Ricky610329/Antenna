"""
Round 93 — R92 deployable vs R63 max-max 虛胖 視覺對比

R92 n=51: worst +1.92 dB (deployable, flat-top compliant)
R63 n=41 (recovered): worst -18.21 dB (max-max +30 虛胖, main 集中尖峰)

Side-by-side 視覺 + metrics 表，給 patch team 看 "max-max 虛胖" vs
"worst-case deployable" 真實差異。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, "script")
from methodology_demo import evaluate_metrics


THETA_DEG = np.arange(-90, 90.1, 0.5)


def main() -> None:
    # R92 n=51 (deployable best)
    pat_r92 = np.load("outputs/r92_aperture_scaling/n51_best_pattern.npy")
    resp_r92 = np.load("outputs/r92_aperture_scaling/n51_best_response.npy")
    main_lo, main_hi = 162, 192  # 15° wide

    # R63 n=41 best (max-max virtual record). 使用 R64 evaluation 跑出來的 -18.21 baseline.
    # R63 width=80 (40°). main_lo=137, main_hi=217.
    pat_r63 = np.load("outputs/r64_eval/r63_pattern.npy")
    resp_r63 = np.load("outputs/r64_eval/r63_response.npy")
    main_lo_r63, main_hi_r63 = 137, 217

    m_r92 = evaluate_metrics(resp_r92, main_lo, main_hi)
    m_r63 = evaluate_metrics(resp_r63, main_lo_r63, main_hi_r63)

    print(f"R92 n=51 deployable:")
    print(f"  worst: {m_r92['worst_supp']:+.2f}, ripple: {m_r92['main_ripple']:.2f}, "
          f"flat-top: {m_r92['flat_top_compliant']}")
    print(f"R63 n=41 max-max:")
    print(f"  headline: {m_r63['headline_supp']:+.2f}, worst: {m_r63['worst_supp']:+.2f}, "
          f"ripple: {m_r63['main_ripple']:.2f}, flat-top: {m_r63['flat_top_compliant']}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # R63 row (max-max)
    axes[0, 0].imshow(pat_r63, cmap="binary", vmin=0, vmax=1, aspect="equal")
    axes[0, 0].set_title(f"R63 binary pattern 41×41\nmax-max optimization")
    axes[0, 0].set_xlabel("element x")
    axes[0, 0].set_ylabel("element y")

    axes[0, 1].plot(THETA_DEG, resp_r63, "r-", linewidth=1.4)
    axes[0, 1].axvspan(-90 + main_lo_r63 * 0.5, -90 + main_hi_r63 * 0.5,
                        color="green", alpha=0.15, label="main beam region (40° wide)")
    axes[0, 1].axhline(0, color="black", linewidth=0.5)
    axes[0, 1].axhline(-3, color="red", linewidth=0.5, linestyle="--", label="-3 dB cap")
    axes[0, 1].set_ylim(-55, 5)
    axes[0, 1].set_xlabel("θ (deg)")
    axes[0, 1].set_ylabel("response (dB)")
    axes[0, 1].set_title(
        f"R63 response (max-max loss)\n"
        f"headline +{m_r63['headline_supp']:.1f} (虛胖)，worst {m_r63['worst_supp']:+.1f}",
        color="darkred",
    )
    axes[0, 1].legend(fontsize=9, loc="lower right")
    axes[0, 1].grid(alpha=0.3)

    side_r63 = np.delete(resp_r63, np.arange(main_lo_r63, main_hi_r63))
    main_r63 = resp_r63[main_lo_r63:main_hi_r63]
    axes[0, 2].hist(side_r63, bins=30, color="lightcoral", alpha=0.7, label="sidelobe", edgecolor="k")
    axes[0, 2].hist(main_r63, bins=30, color="lightgreen", alpha=0.7, label="main beam", edgecolor="k")
    axes[0, 2].axvline(-3, color="red", linewidth=1, linestyle="--", label="-3 dB cap")
    axes[0, 2].set_xlabel("response (dB)")
    axes[0, 2].set_title(f"R63 distribution: {m_r63['main_below_3dB']}/{m_r63['main_total']} main 違反帽蓋", color="darkred")
    axes[0, 2].legend(fontsize=9)

    # R92 row (worst-case deployable)
    axes[1, 0].imshow(pat_r92, cmap="binary", vmin=0, vmax=1, aspect="equal")
    axes[1, 0].set_title(f"R92 binary pattern 51×51\nworst-case + ripple penalty optimization")
    axes[1, 0].set_xlabel("element x")
    axes[1, 0].set_ylabel("element y")

    axes[1, 1].plot(THETA_DEG, resp_r92, "b-", linewidth=1.4)
    axes[1, 1].axvspan(-90 + main_lo * 0.5, -90 + main_hi * 0.5,
                        color="green", alpha=0.15, label="main beam region (15° wide)")
    axes[1, 1].axhline(0, color="black", linewidth=0.5)
    axes[1, 1].axhline(-3, color="red", linewidth=0.5, linestyle="--", label="-3 dB cap")
    axes[1, 1].set_ylim(-30, 5)
    axes[1, 1].set_xlabel("θ (deg)")
    axes[1, 1].set_ylabel("response (dB)")
    axes[1, 1].set_title(
        f"R92 response (worst-case loss, rw=2)\n"
        f"worst {m_r92['worst_supp']:+.1f} (real deployable)，flat-top compliant",
        color="darkgreen",
    )
    axes[1, 1].legend(fontsize=9, loc="lower right")
    axes[1, 1].grid(alpha=0.3)

    side_r92 = np.delete(resp_r92, np.arange(main_lo, main_hi))
    main_r92 = resp_r92[main_lo:main_hi]
    axes[1, 2].hist(side_r92, bins=30, color="lightblue", alpha=0.7, label="sidelobe", edgecolor="k")
    axes[1, 2].hist(main_r92, bins=30, color="lightgreen", alpha=0.7, label="main beam", edgecolor="k")
    axes[1, 2].axvline(-3, color="red", linewidth=1, linestyle="--", label="-3 dB cap")
    axes[1, 2].set_xlabel("response (dB)")
    axes[1, 2].set_title(f"R92 distribution: {m_r92['main_below_3dB']}/{m_r92['main_total']} main 違反帽蓋 ★", color="darkgreen")
    axes[1, 2].legend(fontsize=9)

    fig.suptitle(
        "max-max 虛胖紀錄 (R63) vs worst-case deployable (R92)\n"
        "兩個都是最佳 single-config result，但 main beam 形狀完全不同",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    out = "outputs/r93_max_max_vs_worst_case.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
