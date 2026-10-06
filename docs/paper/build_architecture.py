"""Draw the learning cycle and the emforge execution support beneath it."""

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
        "svg.hashsalt": "antenna-paper-architecture-v5",
    }
)

INK = "#17324A"
TEAL = "#087F83"
TEAL_DARK = "#07666A"
TEAL_PALE = "#EAF5F3"
BLUE_PALE = "#EEF3F8"
SUPPORT = "#6C638A"
BOUNDARY = "#AFCFCC"
SUPPORT_BOUNDARY = "#C6C0D8"
PAPER = "#FFFFFF"
TEXT_BOUNDS = []

fig, ax = plt.subplots(figsize=(13.8, 8.2))
fig.subplots_adjust(left=0.018, right=0.988, top=0.985, bottom=0.025)
ax.set(xlim=(0, 14), ylim=(0, 8.4))
ax.axis("off")


def panel(x, y, w, h, *, face, edge):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.15",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.25,
            zorder=0,
        )
    )


def module(x, y, w, h, prefix, number, title, subtitle, *, execution=False):
    """Draw one compact learning or execution module."""
    edge = INK if execution else TEAL
    face = BLUE_PALE if execution else TEAL_PALE
    badge = INK if execution else TEAL_DARK
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.035,rounding_size=0.12",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.5,
            zorder=3,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.18, y + h - 0.52),
            0.50,
            0.34,
            boxstyle="round,pad=0.02,rounding_size=0.07",
            facecolor=badge,
            edgecolor="none",
            zorder=4,
        )
    )
    ax.text(
        x + 0.43,
        y + h - 0.35,
        f"{prefix}{number}",
        ha="center",
        va="center",
        color=PAPER,
        fontsize=10.0,
        weight="bold",
        zorder=5,
    )
    title_artist = ax.text(
        x + 0.80,
        y + h - 0.35,
        title,
        ha="left",
        va="center",
        color=INK,
        fontsize=12.4,
        weight="bold",
        zorder=5,
    )
    subtitle_artist = ax.text(
        x + 0.22,
        y + 0.29,
        subtitle,
        ha="left",
        va="center",
        color=INK,
        fontsize=10.4,
        zorder=5,
    )
    TEXT_BOUNDS.append(((x, y, w, h), title_artist, subtitle_artist))


def path(points, *, support=False, width=1.8):
    """Draw an orthogonal polyline with one arrowhead at the destination."""
    color = SUPPORT if support else TEAL
    style = (0, (5, 3)) if support else "-"
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


def label(x, y, text, *, support=False, size=10.0):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=SUPPORT if support else TEAL_DARK,
        fontsize=size,
        weight="bold",
        linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.18", "fc": PAPER, "ec": "none"},
        zorder=6,
    )


# The research method is the primary mechanism.
panel(0.42, 3.10, 13.15, 4.62, face="#F8FBFA", edge=BOUNDARY)
ax.text(0.70, 7.96, "LEARNING CYCLE", color=TEAL_DARK, fontsize=12.8, weight="bold")

# Top row continues left to right; the loop returns along the lower row.
module(0.85, 5.96, 2.75, 1.22, "L", 5, "SM update", "Fit accumulated data")
module(4.25, 5.96, 2.75, 1.22, "L", 1, "SM model", "Performance · disagreement")
module(8.02, 5.96, 4.12, 1.22, "L", 2, "Candidate selection", "Construct pool · select batch")
module(8.02, 3.58, 4.12, 1.22, "L", 3, "Batch EM acquisition", "Evaluate selected patterns")
module(0.85, 3.58, 3.20, 1.22, "L", 4, "Accumulated dataset", "Historical + new pairs")

# Solid learning-cycle data flow.
path([(7.00, 6.57), (8.02, 6.57)])
label(7.51, 7.42, "predictions")

path([(8.02, 6.18), (7.00, 6.18)])
label(7.51, 5.73, "candidate pool")

path([(10.08, 5.96), (10.08, 4.80)])
label(11.15, 5.38, "selected patterns")

path([(8.02, 4.19), (4.05, 4.19)])
label(6.04, 4.44, "completed pairs")

path([(2.45, 4.80), (2.45, 5.96)])
label(3.43, 5.38, "training data (old + new)\nstrategy trigger", size=9.5)

path([(3.60, 6.57), (4.25, 6.57)])
label(3.92, 7.42, "updated model")

# emforge provides the execution data plane beneath the learning method.
panel(0.42, 0.35, 13.15, 2.02, face="#FAF9FC", edge=SUPPORT_BOUNDARY)
ax.text(5.05, 2.61, "EMFORGE EXECUTION SUPPORT", color=SUPPORT, fontsize=12.8, weight="bold")

# Right-to-left order follows the acquisition path from L3 back toward L4.
module(8.70, 0.72, 3.05, 1.16, "E", 1, "Dispatch", "Validate · schedule", execution=True)
module(4.72, 0.72, 3.05, 1.16, "E", 2, "HFSS workers", "Build · solve", execution=True)
module(0.85, 0.72, 3.05, 1.16, "E", 3, "Collect & store", "Responses · identity", execution=True)

path([(8.70, 1.30), (7.77, 1.30)])
label(8.23, 1.54, "jobs")
path([(4.72, 1.30), (3.90, 1.30)])
label(4.31, 1.54, "responses")

# Dashed links map learning responsibilities to emforge support; they are not data flow.
path([(10.08, 3.58), (10.08, 1.88)], support=True, width=1.6)
path([(2.45, 3.58), (2.45, 1.88)], support=True, width=1.6)

# Minimal legend.
ax.plot([9.30, 9.90], [7.98, 7.98], color=TEAL, linewidth=1.8)
ax.add_patch(
    FancyArrowPatch(
        (9.72, 7.98),
        (9.90, 7.98),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.8,
        color=TEAL,
    )
)
ax.text(10.02, 7.98, "data", va="center", color=INK, fontsize=10.0)
ax.plot(
    [10.98, 11.58],
    [7.98, 7.98],
    color=SUPPORT,
    linewidth=1.6,
    linestyle=(0, (5, 3)),
)
ax.add_patch(
    FancyArrowPatch(
        (11.40, 7.98),
        (11.58, 7.98),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.6,
        color=SUPPORT,
        linestyle=(0, (5, 3)),
    )
)
ax.text(11.70, 7.98, "implementation support", va="center", color=INK, fontsize=10.0)

svg = OUT / "architecture.svg"
fig.canvas.draw()
for (x, y, w, h), *artists in TEXT_BOUNDS:
    left, bottom = ax.transData.transform((x, y))
    right, top = ax.transData.transform((x + w, y + h))
    for artist in artists:
        bounds = artist.get_window_extent(renderer=fig.canvas.get_renderer())
        if not (
            left <= bounds.x0
            and bounds.x1 <= right
            and bottom <= bounds.y0
            and bounds.y1 <= top
        ):
            raise RuntimeError(f"Text exceeds module bounds: {artist.get_text()}")
fig.savefig(svg, metadata={"Date": None})
fig.savefig(OUT / "architecture.png", dpi=180)
plt.close(fig)

svg.write_text(
    "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
    encoding="utf-8",
)
print("Wrote architecture.svg and architecture.png (learning cycle + execution support).")
