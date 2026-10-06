"""Draw the manuscript's modular search-and-evaluation architecture."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BASE = Path(__file__).resolve().parent
OUT = BASE / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",
        "svg.hashsalt": "antenna-paper-architecture-v4",
    }
)

INK = "#17324A"
TEAL = "#087F83"
TEAL_DARK = "#07666A"
TEAL_PALE = "#EAF5F3"
BLUE_PALE = "#EEF3F8"
CONTROL = "#6C638A"
BOUNDARY = "#AFCFCC"
PAPER = "#FFFFFF"

fig, ax = plt.subplots(figsize=(13.8, 8.0))
fig.subplots_adjust(left=0.018, right=0.988, top=0.98, bottom=0.025)
ax.set(xlim=(0, 14), ylim=(0, 8))
ax.axis("off")


def module(x, y, w, h, number, title, subtitle, *, shared=False, title_size=13.0):
    """Draw one compact, numbered module."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.035,rounding_size=0.12",
            facecolor=BLUE_PALE if shared else TEAL_PALE,
            edgecolor=INK if shared else TEAL,
            linewidth=1.5,
            zorder=3,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.18, y + h - 0.53),
            0.48,
            0.34,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=INK if shared else TEAL_DARK,
            edgecolor="none",
            zorder=4,
        )
    )
    ax.text(
        x + 0.42,
        y + h - 0.36,
        f"M{number}",
        ha="center",
        va="center",
        color=PAPER,
        fontsize=10.2,
        weight="bold",
        zorder=5,
    )
    ax.text(
        x + 0.78,
        y + h - 0.35,
        title,
        ha="left",
        va="center",
        color=INK,
        fontsize=title_size,
        weight="bold",
        zorder=5,
    )
    ax.text(
        x + 0.22,
        y + 0.31,
        subtitle,
        ha="left",
        va="center",
        color=INK,
        fontsize=10.8,
        zorder=5,
    )


def path(points, *, control=False, width=1.8):
    """Draw an orthogonal polyline with one arrowhead at the destination."""
    color = CONTROL if control else TEAL
    style = (0, (5, 3)) if control else "-"
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=width,
            linestyle=style,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=2,
        )
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=width,
            color=color,
            linestyle=style,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def label(x, y, text, *, control=False, rotation=0):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        rotation=rotation,
        color=CONTROL if control else TEAL_DARK,
        fontsize=10.2,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.18", "fc": PAPER, "ec": "none"},
        zorder=6,
    )


# Strategy-owned state is intentionally enclosed; evaluation and storage are shared.
ax.add_patch(
    FancyBboxPatch(
        (0.45, 0.78),
        7.95,
        6.60,
        boxstyle="round,pad=0.025,rounding_size=0.15",
        facecolor="#F8FBFA",
        edgecolor=BOUNDARY,
        linewidth=1.25,
        zorder=0,
    )
)
ax.text(
    0.72,
    7.58,
    "PER-STRATEGY SEARCH STATE",
    color=TEAL_DARK,
    fontsize=12.5,
    weight="bold",
)
ax.text(
    9.20,
    7.58,
    "SHARED EVALUATION",
    color=INK,
    fontsize=12.5,
    weight="bold",
)

# Six numbered modules.
module(
    1.05,
    5.43,
    3.20,
    1.20,
    1,
    "Candidate construction",
    "Seeds · structural variants",
    title_size=11.8,
)
module(1.05, 3.12, 3.20, 1.20, 2, "Learned surrogate", "Training · inference")
module(4.90, 5.43, 3.00, 1.20, 3, "Batch selection", "Performance · exploration")
module(4.90, 1.48, 2.80, 1.20, 6, "Strategy control", "State · update timing")
module(9.25, 5.03, 3.55, 1.52, 4, "Shared EM evaluation", "HFSS · validation · dispatch", shared=True)
module(9.25, 2.25, 3.55, 1.52, 5, "Results & data", "Responses · scores · labels", shared=True)

# Solid data flow.
path([(4.25, 6.03), (4.90, 6.03)])
label(4.58, 6.82, "candidates")

# Candidate patterns are the surrogate's inference input.
path([(2.65, 5.43), (2.65, 4.32)])
label(3.13, 4.88, "patterns")

path([(4.25, 3.72), (4.52, 3.72), (4.52, 4.91), (5.35, 4.91), (5.35, 5.43)])
label(4.94, 4.91, "predictions")

path([(7.90, 6.03), (9.25, 6.03)])
label(8.58, 6.25, "selected")

path([(11.03, 5.03), (11.03, 3.77)])
label(11.67, 4.40, "responses")

path([(9.25, 2.78), (8.72, 2.78), (8.72, 2.08), (7.70, 2.08)])
label(8.20, 2.08, "results")

# M5 returns historical pattern-response pairs for surrogate training.
path([(9.25, 2.48), (8.93, 2.48), (8.93, 0.47), (2.46, 0.47), (2.46, 3.12)])
label(5.60, 0.47, "training data")

# Dashed strategy control fans out to construction, model, and selection.
ax.plot(
    [4.90, 0.72, 0.72],
    [1.90, 1.90, 6.03],
    color=CONTROL,
    linewidth=1.65,
    linestyle=(0, (5, 3)),
    dash_capstyle="round",
    zorder=1,
)
path([(0.72, 6.03), (1.05, 6.03)], control=True, width=1.65)
path([(0.72, 3.72), (1.05, 3.72)], control=True, width=1.65)
path([(6.86, 2.68), (6.86, 5.43)], control=True, width=1.65)
label(0.72, 2.78, "control", control=True, rotation=90)

# Minimal legend.
ax.plot([9.30, 9.90], [7.12, 7.12], color=TEAL, linewidth=1.8)
ax.add_patch(
    FancyArrowPatch(
        (9.72, 7.12),
        (9.90, 7.12),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.8,
        color=TEAL,
    )
)
ax.text(10.02, 7.12, "data", va="center", color=INK, fontsize=10.2)
ax.plot(
    [11.00, 11.60],
    [7.12, 7.12],
    color=CONTROL,
    linewidth=1.65,
    linestyle=(0, (5, 3)),
)
ax.add_patch(
    FancyArrowPatch(
        (11.42, 7.12),
        (11.60, 7.12),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.65,
        color=CONTROL,
        linestyle=(0, (5, 3)),
    )
)
ax.text(11.72, 7.12, "control", va="center", color=INK, fontsize=10.2)

svg = OUT / "architecture.svg"
fig.savefig(svg, metadata={"Date": None})
fig.savefig(OUT / "architecture.png", dpi=180)
plt.close(fig)

svg.write_text(
    "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
    encoding="utf-8",
)
print("Wrote architecture.svg and architecture.png (six-module mechanism).")
