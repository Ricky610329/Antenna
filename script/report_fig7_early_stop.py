"""Report figure 7: trajectory-selection comparison (R138-R140 story).

Generates a 2-row publication-quality figure comparing three trajectory-selection
strategies across 4 RIS recipes:
  - final-step  (vanilla)
  - simple-ES   (R138, max-worst along trajectory)
  - joint-ES    (R140, max-worst AMONG flat-top-valid snapshots)

Top row:    4 grouped bar charts (mean worst-case dB), per-config, with
            min->best error bars showing seed spread.
Bottom row: 4 horizontal flat-top compliance bars; simple-ES failures in B, C
            highlighted with cross-hatch + red callout boxes.

ASCII labels only (Agg backend), no CJK / Greek glyphs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch  # noqa: F401  (kept for reference)

# ---------------------------------------------------------------------------
# Hardcoded data (R140 results, 3-way comparison)
# ---------------------------------------------------------------------------
configs = [
    "Config A\nR119 narrow\n(n=51, w=10, rw=2, lam=1)",
    "Config B\nR129 wide w=18\n(n=51, rw=3, lam=1)",
    "Config C\nR131 inc=0 28GHz\n(n=51, w=10, rw=2, lam=0.3)",
    "Config D\nn=71 extrap w=10\n(n=71, rw=5, lam=0.5)",
]

# Mean worst-case margin (dB) across seeds
final_mean = [-0.38, +0.64, +1.21, +2.21]
simple_mean = [+2.79, +1.93, +2.71, +4.27]
joint_mean = [+2.47, +1.56, +2.00, +3.76]

# Per-seed min worst (lower whisker)
final_min = [-8.02, -0.32, -0.13, +0.55]
simple_min = [+1.32, +1.41, +1.34, +3.61]
joint_min = [+1.32, +1.21, +1.05, +2.20]

# Per-seed best worst (upper whisker)
final_best = [+3.03, +1.13, +2.85, +3.48]
simple_best = [+3.48, +2.50, +3.44, +5.46]
joint_best = [+3.48, +1.94, +3.07, +5.46]

# Flat-top compliance (numerator / denominator)
final_flat = [4, 5, 5, 3]
simple_flat = [4, 1, 1, 2]
joint_flat = [5, 5, 5, 3]
denoms = [5, 5, 5, 3]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
COLOR_FINAL = "#7f8c8d"   # gray
COLOR_SIMPLE = "#e67e22"  # orange
COLOR_JOINT = "#27ae60"   # green

EDGE = "black"
LW = 0.9

# ---------------------------------------------------------------------------
# Figure layout: 2 rows x 4 cols
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 12), dpi=120)
gs = fig.add_gridspec(
    nrows=2,
    ncols=4,
    height_ratios=[1.55, 1.0],
    hspace=0.85,
    wspace=0.38,
    left=0.05,
    right=0.985,
    top=0.82,
    bottom=0.09,
)

# ---------------------------------------------------------------------------
# Top row: 4 grouped bar charts (mean worst across 3 strategies, with min/best
# error bars)
# ---------------------------------------------------------------------------
strategies = ["final-step", "simple-ES\n(R138)", "joint-ES\n(R140)"]
colors = [COLOR_FINAL, COLOR_SIMPLE, COLOR_JOINT]

# Determine a shared y-range for the top row so eye can compare across configs.
# Add extra headroom so callout annotations don't overlap bar labels.
all_min = min(min(final_min), min(simple_min), min(joint_min))
all_best = max(max(final_best), max(simple_best), max(joint_best))
y_lo = min(all_min - 1.5, -10.0)
y_hi = all_best + 5.0

top_axes = []
for ci in range(4):
    ax = fig.add_subplot(gs[0, ci])
    top_axes.append(ax)

    means = [final_mean[ci], simple_mean[ci], joint_mean[ci]]
    mins = [final_min[ci], simple_min[ci], joint_min[ci]]
    bests = [final_best[ci], simple_best[ci], joint_best[ci]]

    # Asymmetric error bars: lower = mean - min, upper = best - mean
    err_lo = [m - mn for m, mn in zip(means, mins)]
    err_hi = [b - m for m, b in zip(means, bests)]

    xs = np.arange(3)
    bars = ax.bar(
        xs,
        means,
        width=0.62,
        color=colors,
        edgecolor=EDGE,
        linewidth=LW,
    )

    # Cross-hatch on simple-ES bar in configs B (ci==1) and C (ci==2) to flag
    # the flat-top crash.
    if ci in (1, 2):
        bars[1].set_hatch("xxxx")
        bars[1].set_edgecolor("#a93226")
        bars[1].set_linewidth(1.6)

    ax.errorbar(
        xs,
        means,
        yerr=[err_lo, err_hi],
        fmt="none",
        ecolor="black",
        elinewidth=1.1,
        capsize=5,
        capthick=1.1,
        zorder=5,
    )

    # Annotate bar height (centered inside the bar when room, otherwise above)
    for bx, m in zip(xs, means):
        if m >= 0.6:
            ax.text(
                bx,
                m / 2,
                f"{m:+.2f}",
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                bx,
                m + (0.35 if m >= 0 else -0.55),
                f"{m:+.2f}",
                ha="center",
                va="bottom" if m >= 0 else "top",
                fontsize=10.5,
                fontweight="bold",
                color="black",
            )

    # Whisker label (best at top of upper cap, min at bottom). Place to the
    # right of each bar without overlapping neighbouring bar.
    for bx, b in zip(xs, bests):
        ax.text(
            bx,
            b + 0.12,
            f"{b:+.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#222222",
        )
    for bx, mn in zip(xs, mins):
        ax.text(
            bx,
            mn - 0.12,
            f"{mn:+.2f}",
            ha="center",
            va="top",
            fontsize=7.5,
            color="#222222",
        )

    # Zero line
    ax.axhline(0.0, color="black", linewidth=1.1, linestyle="--", alpha=0.7)
    ax.axhspan(y_lo, 0.0, facecolor="red", alpha=0.04, zorder=0)

    ax.set_xticks(xs)
    ax.set_xticklabels(strategies, fontsize=9)
    ax.set_ylim(y_lo, y_hi)
    ax.set_title(configs[ci], fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    if ci == 0:
        ax.set_ylabel("Worst-case margin (dB)\nbars=mean, whiskers=[min, best]", fontsize=10)

    # ---- per-config callouts -------------------------------------------------
    if ci == 0:
        # Config A: joint-ES IMPROVES flat 4/5 -> 5/5 (highlight on joint bar)
        ax.annotate(
            "flat 4/5 -> 5/5\n(joint IMPROVES)",
            xy=(2, joint_best[0] + 0.2),
            xytext=(1.0, y_hi - 0.6),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor=COLOR_JOINT,
                edgecolor="black",
                linewidth=0.8,
            ),
            arrowprops=dict(arrowstyle="->", color=COLOR_JOINT, lw=1.4),
        )

    if ci == 1:
        # Config B: simple-ES crash callout
        ax.annotate(
            "simple-ES CRASH\nflat 5/5 -> 1/5",
            xy=(1, simple_best[1] + 0.2),
            xytext=(0.0, y_hi - 0.6),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="#c0392b",
                edgecolor="black",
                linewidth=0.8,
            ),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4),
        )

    if ci == 2:
        # Config C: simple-ES crash callout
        ax.annotate(
            "simple-ES CRASH\nflat 5/5 -> 1/5",
            xy=(1, simple_best[2] + 0.2),
            xytext=(0.0, y_hi - 0.6),
            ha="center",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="#c0392b",
                edgecolor="black",
                linewidth=0.8,
            ),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.4),
        )

    if ci == 3:
        # Config D: VRAM-limited note
        ax.text(
            0.5,
            0.97,
            "3 seeds only (VRAM-limited)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.5,
            style="italic",
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="#fdf6e3",
                edgecolor="#666666",
                linewidth=0.7,
            ),
        )

# Row label for the top row (left side, outside first axes)
top_axes[0].text(
    -0.32,
    1.18,
    "(a) Mean worst-case margin (dB) per recipe   --   bars=mean, whiskers=[min, best] across seeds",
    transform=top_axes[0].transAxes,
    fontsize=12.5,
    fontweight="bold",
    ha="left",
    va="bottom",
)

# ---------------------------------------------------------------------------
# Bottom row: 4 horizontal flat-top compliance bars (one subplot per config)
# ---------------------------------------------------------------------------
bot_axes = []
labels = ["final-step", "simple-ES", "joint-ES"]
for ci in range(4):
    ax = fig.add_subplot(gs[1, ci])
    bot_axes.append(ax)

    nums = [final_flat[ci], simple_flat[ci], joint_flat[ci]]
    d = denoms[ci]
    pcts = [100 * n / d for n in nums]

    ys = np.arange(3)[::-1]  # final on top, joint on bottom
    bar_colors = [COLOR_FINAL, COLOR_SIMPLE, COLOR_JOINT]

    bars = ax.barh(
        ys,
        pcts,
        height=0.62,
        color=bar_colors,
        edgecolor=EDGE,
        linewidth=LW,
    )

    # Hatch the crashing simple-ES bar (B and C)
    if ci in (1, 2):
        bars[1].set_hatch("xxxx")
        bars[1].set_edgecolor("#a93226")
        bars[1].set_linewidth(1.6)

    # Annotate N/total inside the bar (right-aligned at 100% reference if room)
    for y, n, pct in zip(ys, nums, pcts):
        # Place text at the end of bar
        if pct >= 25:
            ax.text(
                pct - 2,
                y,
                f"{n}/{d}  ({pct:.0f}%)",
                ha="right",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                pct + 3,
                y,
                f"{n}/{d}  ({pct:.0f}%)",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black",
            )

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Flat-top compliance (%)", fontsize=9)
    ax.axvline(100.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_title(f"Config {chr(ord('A') + ci)}", fontsize=10, fontweight="bold")

    # Bold red CRASH annotation for B and C
    if ci == 1:
        ax.text(
            0.5,
            -0.55,
            "CRASH 5/5 -> 1/5",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="#c0392b",
                edgecolor="black",
                linewidth=0.8,
            ),
        )
    if ci == 2:
        ax.text(
            0.5,
            -0.55,
            "CRASH 5/5 -> 1/5",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="#c0392b",
                edgecolor="black",
                linewidth=0.8,
            ),
        )
    if ci == 0:
        # Highlight A's improvement
        ax.text(
            0.5,
            -0.55,
            "IMPROVED 4/5 -> 5/5",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor=COLOR_JOINT,
                edgecolor="black",
                linewidth=0.8,
            ),
        )
    if ci == 3:
        ax.text(
            0.5,
            -0.55,
            "preserved 3/3 (VRAM-limited)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="#2c3e50",
                edgecolor="black",
                linewidth=0.8,
            ),
        )

bot_axes[0].text(
    -0.32,
    1.24,
    "(b) Flat-top compliance (N seeds passing flat-top constraint / total)",
    transform=bot_axes[0].transAxes,
    fontsize=12.5,
    fontweight="bold",
    ha="left",
    va="bottom",
)

# ---------------------------------------------------------------------------
# Title + subtitle + legend
# ---------------------------------------------------------------------------
fig.suptitle(
    "Joint early-stop (R140) -- beats final-step AND avoids simple-ES flat-top crash",
    fontsize=16.5,
    fontweight="bold",
    y=0.975,
)
fig.text(
    0.5,
    0.945,
    "Trajectory selection: 'best worst' alone sacrifices flat-top in 3/4 recipes; "
    "joint criterion preserves both",
    ha="center",
    va="top",
    fontsize=12,
    style="italic",
    color="#222222",
)

# Shared legend at the very top
from matplotlib.patches import Patch  # local import for clarity

legend_handles = [
    Patch(facecolor=COLOR_FINAL, edgecolor=EDGE, label="final-step (last trajectory point)"),
    Patch(facecolor=COLOR_SIMPLE, edgecolor=EDGE, label="simple-ES (R138): max-worst over trajectory"),
    Patch(facecolor=COLOR_JOINT, edgecolor=EDGE, label="joint-ES (R140): max-worst AMONG flat-top valid"),
    Patch(
        facecolor=COLOR_SIMPLE,
        edgecolor="#a93226",
        hatch="xxxx",
        label="simple-ES flat-top CRASH",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="upper center",
    ncol=4,
    fontsize=10.5,
    framealpha=0.95,
    bbox_to_anchor=(0.5, 0.915),
)

# Footer note
fig.text(
    0.5,
    0.012,
    "PROMOTION verdict: joint-ES wins all 4 -- mean dominates final-step (Configs A-D) AND flat-top preserved (5/5, 5/5, 5/5, 3/3).",
    ha="center",
    va="bottom",
    fontsize=10,
    style="italic",
    color="#222222",
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = Path("outputs/report_fig7_early_stop.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_path.resolve()}")
