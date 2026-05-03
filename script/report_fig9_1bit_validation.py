"""Publication-quality figure: 1-bit RIS validation across 3 axes (POST-R128).

Replaces the outdated R121 2-bit "4-axis" figure. The R121 champion used 4
discrete phase levels (2-bit) which is NOT hardware-deployable per the
0/pi-only constraint. After R128 pivoted to 1-bit ONLY, the methodology was
re-validated. This figure shows the actual deployable 1-bit results from
R128 (width x steering grid), R130 (inc x freq universal grid) and R141
(6/6 held-out deployment configs).

Run:
    PYTHONIOENCODING=utf-8 C:/Users/Ricky/miniforge3/envs/ant/python.exe \
        script/report_fig9_1bit_validation.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle


# ---------------------------------------------------------------------------
# Hardcoded validation data (post-R128, 1-bit only, 0/pi phase)
# ---------------------------------------------------------------------------

# R128: 1-bit width x steering grid (n=51, inc=51 sweet, 38GHz, R119 recipe)
WIDTHS = [10, 20, 30]                  # rows: flat-top width cap (deg)
STEERINGS = ["0deg", "+30deg", "+45deg"]   # cols: steering offset
WORST_WS = np.array(
    [
        [+3.03, +1.41, +1.28],   # w=10
        [+3.02, +1.07, +0.69],   # w=20
        [+3.39, +0.99, +0.66],   # w=30
    ]
)
FLAT_WS = np.array(
    [
        [4, 4, 5],   # w=10  (cell (0,2) labelled "OK" in source -> 5/5)
        [3, 5, 5],   # w=20  ((1,1) "OK" -> 5/5, (1,2) "OK" -> 5/5)
        [2, 3, 4],   # w=30
    ]
)

# R130: 1-bit narrow-cap (w=10) inc x freq grid (n=51, R119 recipe)
INCS = [0, 30, 51, 70]                                # rows
FREQS = ["5.8GHz", "28GHz", "38GHz", "60GHz"]         # cols
WORST_IF = np.array(
    [
        [+3.12, +1.63, +0.54, +0.34],   # inc=0   (mmWave failures here)
        [+2.79, +2.52, +2.54, +2.04],   # inc=30
        [+2.53, +2.50, +3.03, +2.62],   # inc=51 (sweet)
        [+2.79, +2.21, +1.63, +2.36],   # inc=70
    ]
)
FLAT_IF = np.array(
    [
        [5, 2, 1, 1],   # inc=0  (mmWave fails)
        [5, 4, 5, 5],   # inc=30
        [5, 5, 4, 4],   # inc=51
        [4, 5, 5, 5],   # inc=70
    ]
)

# R141: 6/6 deployment validation (selector + joint early-stop)
CONFIGS_141 = [
    # (config label, recipe key, worst dB, was_R134_fail flag)
    ("n=51 inc=30 28GHz w=10",   "R119",         +3.13, False),
    ("n=51 inc=70 60GHz w=10",   "R119",         +2.72, False),
    ("n=51 inc=51 38GHz w=15",   "R129 wide",    +1.74, True),
    ("n=51 inc=51 38GHz w=20",   "R129 wide",    +1.72, False),
    ("n=71 inc=30 28GHz w=10",   "n=71 extrap",  +4.19, False),
    ("n=71 inc=51 38GHz w=10",   "n=71 extrap",  +5.46, False),
]

RECIPE_COLORS = {
    "R119":         "#2CA02C",   # green (R119 baseline)
    "R129 wide":    "#1F77B4",   # blue
    "n=71 extrap":  "#FF7F0E",   # orange
}


# ---------------------------------------------------------------------------
# ROW 1 - R128: width x steering heatmap
# ---------------------------------------------------------------------------

def panel_a_width_steering(ax: plt.Axes) -> None:
    """3x3 heatmap, worst-case dB with flat-top compliance annotations.

    Bold red border on cells where flat-top fails (< 5/5).
    """
    data = WORST_WS
    vmax = float(np.max(np.abs(data)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(STEERINGS)))
    ax.set_xticklabels(STEERINGS, fontsize=11)
    ax.set_yticks(np.arange(len(WIDTHS)))
    ax.set_yticklabels([f"w={w}deg" for w in WIDTHS], fontsize=11)
    ax.set_xlabel("steering offset", fontsize=11)
    ax.set_ylabel("flat-top width (cap)", fontsize=11)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            flat = int(FLAT_WS[i, j])
            ax.text(
                j, i - 0.18, f"{val:+.2f} dB",
                ha="center", va="center",
                color="black", fontsize=12, fontweight="bold",
            )
            ax.text(
                j, i + 0.22, f"flat: {flat}/5",
                ha="center", va="center",
                color="black", fontsize=10, style="italic",
            )
            # Bold red border on cells failing flat-top criterion (< 5/5)
            if flat < 5:
                rect = Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    linewidth=2.8, edgecolor="#C8102E",
                    facecolor="none", zorder=10,
                )
                ax.add_patch(rect)

    ax.set_title(
        "(a) R128: 1-bit width x steering grid (n=51, 38GHz, R119 recipe)",
        fontsize=13, fontweight="bold", pad=10,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("worst-case (dB, higher = better)", fontsize=10)


# ---------------------------------------------------------------------------
# ROW 2 - R130: two side-by-side panels (worst dB + flat-top compliance)
# ---------------------------------------------------------------------------

def panel_b1_inc_freq_worst(ax: plt.Axes) -> None:
    """4x4 worst-case heatmap: green if mean > 0, red if < 0."""
    data = WORST_IF
    vmax = float(np.max(np.abs(data)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(FREQS)))
    ax.set_xticklabels(FREQS, fontsize=11)
    ax.set_yticks(np.arange(len(INCS)))
    ax.set_yticklabels([f"inc={i}deg" for i in INCS], fontsize=11)
    ax.set_xlabel("frequency", fontsize=11)
    ax.set_ylabel("incidence angle", fontsize=11)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            ax.text(
                j, i, f"{val:+.2f}",
                ha="center", va="center",
                color="black", fontsize=12, fontweight="bold",
            )

    # Red dashed box around the mmWave failure cluster (inc=0, 28/38/60 GHz)
    fail_row = 0
    box_x = 0.5    # left edge between j=0 and j=1
    box_y = fail_row - 0.5
    rect = Rectangle(
        (box_x, box_y), 3, 1,
        linewidth=2.6, edgecolor="#C8102E",
        facecolor="none", linestyle=(0, (5, 3)), zorder=11,
    )
    ax.add_patch(rect)
    ax.annotate(
        "R131 rescues\nfound",
        xy=(2.0, fail_row - 0.5), xytext=(0.05, -1.15),
        fontsize=9, color="#C8102E", fontweight="bold",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#C8102E", lw=1.2),
        annotation_clip=False,
    )

    ax.set_title(
        "worst-case (dB)", fontsize=11, fontweight="bold", pad=6,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("worst-case (dB)", fontsize=10)


def panel_b2_inc_freq_flat(ax: plt.Axes) -> None:
    """4x4 flat-top compliance heatmap.

    Color rule:
      - 5/5 ("OK")  -> green
      - 2..4/5      -> yellow
      - 0..1/5      -> red
    """
    data = FLAT_IF.astype(float)

    # Custom 3-step colormap: red, yellow, green (mapped via discrete bins)
    color_grid = np.empty(data.shape + (3,), dtype=float)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data[i, j])
            if v == 5:
                color_grid[i, j] = (0.18, 0.63, 0.18)   # green
            elif 2 <= v <= 4:
                color_grid[i, j] = (1.00, 0.85, 0.20)   # yellow
            else:
                color_grid[i, j] = (0.84, 0.10, 0.18)   # red

    ax.imshow(color_grid, aspect="auto")

    ax.set_xticks(np.arange(len(FREQS)))
    ax.set_xticklabels(FREQS, fontsize=11)
    ax.set_yticks(np.arange(len(INCS)))
    ax.set_yticklabels([f"inc={i}deg" for i in INCS], fontsize=11)
    ax.set_xlabel("frequency", fontsize=11)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = int(data[i, j])
            label = "OK" if v == 5 else f"{v}/5"
            ax.text(
                j, i, label,
                ha="center", va="center",
                color="black", fontsize=12, fontweight="bold",
            )

    # Red dashed box around mmWave failure cluster
    fail_row = 0
    rect = Rectangle(
        (0.5, fail_row - 0.5), 3, 1,
        linewidth=2.6, edgecolor="#C8102E",
        facecolor="none", linestyle=(0, (5, 3)), zorder=11,
    )
    ax.add_patch(rect)
    ax.annotate(
        "R131 rescues\nfound",
        xy=(2.0, fail_row - 0.5), xytext=(0.05, -1.15),
        fontsize=9, color="#C8102E", fontweight="bold",
        ha="left",
        arrowprops=dict(arrowstyle="->", color="#C8102E", lw=1.2),
        annotation_clip=False,
    )

    ax.set_title(
        "flat-top compliance (5 seeds)", fontsize=11, fontweight="bold", pad=6,
    )

    # Manual legend for the 3-color scheme
    legend_handles = [
        Patch(facecolor=(0.18, 0.63, 0.18), edgecolor="black", label="OK (5/5)"),
        Patch(facecolor=(1.00, 0.85, 0.20), edgecolor="black", label="partial (2-4/5)"),
        Patch(facecolor=(0.84, 0.10, 0.18), edgecolor="black", label="fail (<= 1/5)"),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right",
        bbox_to_anchor=(1.32, 1.02),
        fontsize=9, framealpha=0.95,
    )


# ---------------------------------------------------------------------------
# ROW 3 - R141: 6/6 deployment bar chart
# ---------------------------------------------------------------------------

def panel_c_deployment(ax: plt.Axes) -> None:
    """Horizontal bar chart of worst-case for the 6 R141 configs.

    Color-coded by recipe tier. PASS badge on each bar. Reference line at 0.
    Highlight the n=51 inc=51 38GHz w=15 entry (was R134 fail).
    """
    labels = [c[0] for c in CONFIGS_141]
    recipes = [c[1] for c in CONFIGS_141]
    values = [c[2] for c in CONFIGS_141]
    flags = [c[3] for c in CONFIGS_141]

    # Reverse so first config sits on top
    labels = labels[::-1]
    recipes = recipes[::-1]
    values = values[::-1]
    flags = flags[::-1]

    colors = [RECIPE_COLORS[r] for r in recipes]
    y_positions = np.arange(len(labels))

    bars = ax.barh(
        y_positions, values, color=colors,
        edgecolor="black", linewidth=0.8, height=0.6,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("worst-case (dB, higher = better)", fontsize=11)
    ax.set_xlim(-0.5, max(values) * 1.45)
    ax.axvline(0, color="black", lw=1.2, label="acceptance threshold (worst > 0)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Annotate value + PASS badge per bar
    for bar, val, flag in zip(bars, values, flags):
        x = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            x + 0.08, y, f"{val:+.2f} dB",
            va="center", ha="left", fontsize=10, fontweight="bold",
        )
        ax.text(
            x + 1.15, y, "PASS",
            va="center", ha="left", fontsize=10, fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#2CA02C", edgecolor="#1B6F1B", linewidth=1.0,
            ),
        )
        # Highlight previously-failing config (R134 fail -> fixed)
        if flag:
            note_x = ax.get_xlim()[1] * 0.55
            note_y = y + 0.55
            ax.annotate(
                "was R134 fail (fixed by R135 boundary + R140 joint ES)",
                xy=(x * 0.5, y + bar.get_height() / 2),
                xytext=(note_x, note_y),
                fontsize=9, color="#B22222", fontweight="bold",
                va="center", ha="left",
                arrowprops=dict(
                    arrowstyle="->", color="#B22222", lw=1.2,
                    connectionstyle="arc3,rad=-0.25",
                ),
            )

    # Recipe color legend (left side)
    legend_handles = [
        Patch(facecolor=RECIPE_COLORS["R119"],         edgecolor="black",
              label="R119 baseline"),
        Patch(facecolor=RECIPE_COLORS["R129 wide"],    edgecolor="black",
              label="R129 wide cap (w=12-20)"),
        Patch(facecolor=RECIPE_COLORS["n=71 extrap"],  edgecolor="black",
              label="n=71 extrapolation"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", fontsize=9,
        title="recipe selected", title_fontsize=9, framealpha=0.95,
    )

    ax.set_title(
        "(c) R141: deployment API 6/6 PASS (selector + joint early-stop)",
        fontsize=13, fontweight="bold", pad=10,
    )


# ---------------------------------------------------------------------------
# Compose figure
# ---------------------------------------------------------------------------

def build_figure(out_path: Path) -> None:
    fig = plt.figure(figsize=(16, 14), dpi=120)

    # 3 rows. Row 2 has two heatmap panels side-by-side.
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[1.0, 1.10, 1.05],
        width_ratios=[1.0, 1.0],
        hspace=0.78, wspace=0.32,
        left=0.07, right=0.96, top=0.90, bottom=0.07,
    )

    # Row 1 spans both columns
    ax_a = fig.add_subplot(gs[0, :])
    panel_a_width_steering(ax_a)

    # Row 2: two side-by-side panels
    ax_b1 = fig.add_subplot(gs[1, 0])
    ax_b2 = fig.add_subplot(gs[1, 1])
    panel_b1_inc_freq_worst(ax_b1)
    panel_b2_inc_freq_flat(ax_b2)

    # Row 2 shared title spanning both panels (placed above sub-panel titles)
    row2_top = ax_b1.get_position().y1
    fig.text(
        0.5, row2_top + 0.022,
        "(b) R130: 1-bit narrow-cap (w=10) inc x freq universal validation "
        "-> 13/16 PASS",
        ha="center", va="bottom",
        fontsize=13, fontweight="bold",
    )

    # Row 3 spans both columns
    ax_c = fig.add_subplot(gs[2, :])
    panel_c_deployment(ax_c)

    # Overall titles
    fig.suptitle(
        "1-bit RIS validation across 3 axes -- POST R128 hardware-realistic results",
        fontsize=17, fontweight="bold", y=0.975,
    )
    fig.text(
        0.5, 0.945,
        "Replacing R121 (2-bit) with R128 / R130 / R141 (1-bit, deployment-spec)",
        ha="center", va="top",
        fontsize=12, style="italic", color="#333333",
    )

    # Footnote
    fig.text(
        0.5, 0.012,
        "For comparison, the previous 4-axis figure used R121 (2-bit, 4 phase "
        "levels) which is NOT hardware-deployable per the user's 0/pi-only "
        "constraint.",
        ha="center", va="bottom",
        fontsize=10, style="italic", color="#555555",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
        glyph_warnings = [
            w for w in caught
            if "Glyph" in str(w.message) or "missing from" in str(w.message)
        ]
        if glyph_warnings:
            print("WARNING: glyph rendering issues detected:")
            for w in glyph_warnings:
                print(f"   - {w.message}")
        else:
            print("OK: no glyph warnings")

    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "outputs" / "report_fig9_1bit_validation.png"
    build_figure(out_path)


if __name__ == "__main__":
    main()
