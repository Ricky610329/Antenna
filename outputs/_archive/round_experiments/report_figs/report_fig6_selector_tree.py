"""Render the 1-bit RIS recipe selector decision tree as a flowchart.

Output: outputs/report_fig6_selector_tree.png
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
COLOR_INPUT = "#dbe9f5"          # pale steel blue
COLOR_DECISION = "#fff4cf"       # cream / pale gold
COLOR_RECIPE = "#bfe6d4"         # mint / teal
COLOR_RECIPE_EDGE = "#1f6f52"
COLOR_ERROR = "#fff7c2"          # pale yellow
COLOR_ERROR_TEXT = "#b3261e"     # red

ZONE_71 = "#fbe3e1"              # light coral
ZONE_71_EDGE = "#c44a3f"
ZONE_51 = "#dceefb"              # light blue
ZONE_51_EDGE = "#2f6fa6"

ARROW_COLOR = "#3a3a3a"
EDGE_COLOR = "#444444"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = EDGE_COLOR,
    fontsize: int = 11,
    fontweight: str = "normal",
    textcolor: str = "black",
):
    """Draw a rounded box centred at (x, y) and return its anchor points."""
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.10,rounding_size=0.18",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor,
        zorder=4,
    )
    return {
        "center": (x, y),
        "top": (x, y + h / 2),
        "bottom": (x, y - h / 2),
        "left": (x - w / 2, y),
        "right": (x + w / 2, y),
        "w": w,
        "h": h,
    }


def add_arrow(ax, p_from, p_to, label: str | None = None, *, label_dx=0.0, label_dy=0.0,
              label_color="#222222", label_bg="white"):
    arrow = FancyArrowPatch(
        p_from,
        p_to,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.3,
        color=ARROW_COLOR,
        zorder=2,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)
    if label:
        mx = (p_from[0] + p_to[0]) / 2 + label_dx
        my = (p_from[1] + p_to[1]) / 2 + label_dy
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=label_color,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=label_bg, edgecolor="none", alpha=0.92),
            zorder=5,
        )


def add_zone(ax, x0, y0, x1, y1, *, facecolor, edgecolor, label, label_color):
    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6,
        linestyle="--",
        alpha=0.55,
        zorder=1,
    )
    ax.add_patch(rect)
    ax.text(
        x0 + 0.25,
        y1 - 0.35,
        label,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color=label_color,
        zorder=2,
    )


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def main() -> Path:
    fig, ax = plt.subplots(figsize=(20, 13), dpi=120)
    ax.set_xlim(0, 32)
    ax.set_ylim(-1.2, 22)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Title / subtitle ---------------------------------------------------
    fig.suptitle(
        "1-bit Recipe Selector  --  select_1bit_recipe(n, inc, freq, width)",
        fontsize=18,
        fontweight="bold",
        y=0.975,
    )
    ax.text(
        16,
        21.3,
        "decision tree distilled from R119, R129, R131, R133, R135",
        ha="center",
        va="center",
        fontsize=12,
        style="italic",
        color="#444",
    )

    # ---- Coloured zones (drawn first, sit behind boxes) --------------------
    add_zone(
        ax, 0.2, 5.4, 12.0, 14.8,
        facecolor=ZONE_71, edgecolor=ZONE_71_EDGE,
        label="n = 71  (large aperture)", label_color=ZONE_71_EDGE,
    )
    add_zone(
        ax, 12.4, 2.4, 32.6, 14.8,
        facecolor=ZONE_51, edgecolor=ZONE_51_EDGE,
        label="n = 51 or 31  (small / medium aperture)", label_color=ZONE_51_EDGE,
    )

    # ---- Top: input + global guards ----------------------------------------
    inp = add_box(
        ax, 16, 19.8, 11.0, 1.1,
        "INPUT  (n, inc_deg, freq_hz, width_deg)",
        facecolor=COLOR_INPUT, fontsize=13, fontweight="bold",
    )

    guard_n = add_box(
        ax, 9.5, 17.6, 8.4, 1.1,
        "n in {31, 51, 71} ?",
        facecolor=COLOR_DECISION, fontsize=11, fontweight="bold",
    )
    guard_w = add_box(
        ax, 22.5, 17.6, 8.4, 1.1,
        "width_deg <= 30 ?",
        facecolor=COLOR_DECISION, fontsize=11, fontweight="bold",
    )

    err_n = add_box(
        ax, 3.6, 17.6, 5.4, 0.95,
        "ERROR\n\"not validated\"",
        facecolor=COLOR_ERROR, edgecolor=COLOR_ERROR_TEXT,
        fontsize=10, fontweight="bold", textcolor=COLOR_ERROR_TEXT,
    )
    err_w = add_box(
        ax, 28.4, 17.6, 5.4, 0.95,
        "ERROR\n\"out of envelope\"",
        facecolor=COLOR_ERROR, edgecolor=COLOR_ERROR_TEXT,
        fontsize=10, fontweight="bold", textcolor=COLOR_ERROR_TEXT,
    )

    add_arrow(ax, inp["bottom"], (guard_n["center"][0], guard_n["top"][1]))
    add_arrow(ax, inp["bottom"], (guard_w["center"][0], guard_w["top"][1]))
    add_arrow(ax, guard_n["left"], err_n["right"], label="no",
              label_dy=0.32, label_color=COLOR_ERROR_TEXT)
    add_arrow(ax, guard_w["right"], err_w["left"], label="no",
              label_dy=0.32, label_color=COLOR_ERROR_TEXT)

    # split node: n branch
    split = add_box(
        ax, 16, 15.6, 8.4, 1.0,
        "yes  -->  branch on n",
        facecolor=COLOR_DECISION, fontsize=11, fontweight="bold",
    )
    add_arrow(ax, guard_n["bottom"], (split["left"][0] + 0.2, split["top"][1]),
              label="yes", label_dx=-0.3, label_dy=0.25)
    add_arrow(ax, guard_w["bottom"], (split["right"][0] - 0.2, split["top"][1]),
              label="yes", label_dx=0.3, label_dy=0.25)

    # =========================================================================
    # LEFT subtree:  n == 71
    # =========================================================================
    n71_root = add_box(
        ax, 6.5, 13.6, 7.4, 1.0,
        "n == 71",
        facecolor="#f6c7c2", edgecolor=ZONE_71_EDGE,
        fontsize=12, fontweight="bold",
    )
    add_arrow(ax, split["left"], n71_root["right"], label="n = 71",
              label_dy=0.32, label_color=ZONE_71_EDGE)

    n71_q1 = add_box(
        ax, 6.5, 11.5, 7.4, 1.1,
        "inc == 0  AND  freq >= 50 GHz ?",
        facecolor=COLOR_DECISION, fontsize=10.5,
    )
    add_arrow(ax, n71_root["bottom"], n71_q1["top"])

    n71_leaf1 = add_box(
        ax, 2.6, 9.0, 4.4, 1.5,
        "rw = 5.0\nlambda = 0.3\n(R133 inc=0 + 60 GHz)",
        facecolor=COLOR_RECIPE, edgecolor=COLOR_RECIPE_EDGE,
        fontsize=10, fontweight="bold",
    )
    add_arrow(ax, n71_q1["left"], n71_leaf1["top"], label="yes",
              label_dx=-0.3, label_dy=0.05)

    n71_q2 = add_box(
        ax, 8.6, 9.0, 5.0, 1.1,
        "width_deg <= 15 ?",
        facecolor=COLOR_DECISION, fontsize=10.5,
    )
    add_arrow(ax, n71_q1["bottom"], n71_q2["top"], label="no",
              label_dx=0.65, label_dy=-0.15)

    n71_leaf2 = add_box(
        ax, 4.0, 6.4, 4.4, 1.5,
        "rw = 5.0\nlambda = 0.5\n(n=71 narrow extrap.)",
        facecolor=COLOR_RECIPE, edgecolor=COLOR_RECIPE_EDGE,
        fontsize=10, fontweight="bold",
    )
    n71_leaf3 = add_box(
        ax, 9.0, 6.4, 4.4, 1.5,
        "rw = 7.0\nlambda = 0.5\n(n=71 wide extrap.)",
        facecolor=COLOR_RECIPE, edgecolor=COLOR_RECIPE_EDGE,
        fontsize=10, fontweight="bold",
    )
    add_arrow(ax, n71_q2["bottom"], n71_leaf2["top"], label="yes",
              label_dx=-0.55, label_dy=0.10)
    add_arrow(ax, n71_q2["bottom"], n71_leaf3["top"], label="no  (w > 15)",
              label_dx=0.55, label_dy=0.10)

    # =========================================================================
    # RIGHT subtree:  n == 51 (or 31)
    # =========================================================================
    n51_root = add_box(
        ax, 22.5, 13.6, 9.0, 1.0,
        "n == 51  (or 31)",
        facecolor="#bedaf2", edgecolor=ZONE_51_EDGE,
        fontsize=12, fontweight="bold",
    )
    add_arrow(ax, split["right"], n51_root["left"], label="n in {31, 51}",
              label_dy=0.32, label_color=ZONE_51_EDGE)

    # ---- First check: width > 12 -------------------------------------------
    n51_qw = add_box(
        ax, 17.5, 11.5, 6.4, 1.1,
        "width_deg > 12 ?",
        facecolor=COLOR_DECISION, fontsize=10.5,
    )
    n51_qinc = add_box(
        ax, 27.5, 11.5, 7.4, 1.1,
        "inc == 0  AND  freq >= 20 GHz ?",
        facecolor=COLOR_DECISION, fontsize=10.5,
    )
    add_arrow(ax, n51_root["bottom"], n51_qw["top"])
    add_arrow(ax, n51_root["bottom"], n51_qinc["top"])

    # width > 12 yes branch -> sub-decision width <= 20
    n51_qw_yes = add_box(
        ax, 15.0, 9.0, 5.2, 1.1,
        "width_deg <= 20 ?",
        facecolor=COLOR_DECISION, fontsize=10.5,
    )
    add_arrow(ax, n51_qw["bottom"], n51_qw_yes["top"], label="yes",
              label_dx=-0.5, label_dy=0.05)

    n51_leaf_w20 = add_box(
        ax, 13.0, 6.4, 4.4, 1.5,
        "rw = 3.0\nlambda = 1.0\n(R129 wide 12-20 deg)",
        facecolor=COLOR_RECIPE, edgecolor=COLOR_RECIPE_EDGE,
        fontsize=10, fontweight="bold",
    )
    n51_leaf_w30 = add_box(
        ax, 17.8, 6.4, 4.4, 1.5,
        "rw = 3.0\nlambda = 0.5\n(R129 wide 20-30 deg)",
        facecolor=COLOR_RECIPE, edgecolor=COLOR_RECIPE_EDGE,
        fontsize=10, fontweight="bold",
    )
    add_arrow(ax, n51_qw_yes["bottom"], n51_leaf_w20["top"], label="yes",
              label_dx=-0.55, label_dy=0.10)
    add_arrow(ax, n51_qw_yes["bottom"], n51_leaf_w30["top"],
              label="no  (20 < w <= 30)",
              label_dx=0.85, label_dy=0.25)

    # inc==0 & freq>=20 branch
    n51_qinc_yes = add_box(
        ax, 27.5, 9.0, 7.4, 1.1,
        "freq tier ?",
        facecolor=COLOR_DECISION, fontsize=10.5,
    )
    add_arrow(ax, n51_qinc["bottom"], n51_qinc_yes["top"], label="yes",
              label_dx=0.45, label_dy=0.05)

    n51_err_50 = add_box(
        ax, 23.6, 6.4, 4.0, 1.5,
        "ERROR\n\"use n = 71\"\n(freq >= 50 GHz)",
        facecolor=COLOR_ERROR, edgecolor=COLOR_ERROR_TEXT,
        fontsize=10, fontweight="bold", textcolor=COLOR_ERROR_TEXT,
    )
    n51_leaf_38 = add_box(
        ax, 28.0, 6.4, 4.4, 1.5,
        "rw = 2.0\nlambda = 0.5\n(R131 inc=0 38 GHz)",
        facecolor=COLOR_RECIPE, edgecolor=COLOR_RECIPE_EDGE,
        fontsize=10, fontweight="bold",
    )
    n51_leaf_28 = add_box(
        ax, 31.6, 4.0, 4.4, 1.5,
        "rw = 2.0\nlambda = 0.3\n(R131 inc=0 28 GHz)",
        facecolor=COLOR_RECIPE, edgecolor=COLOR_RECIPE_EDGE,
        fontsize=10, fontweight="bold",
    )
    add_arrow(ax, n51_qinc_yes["bottom"], n51_err_50["top"],
              label="freq >= 50",
              label_dx=-0.55, label_dy=0.15, label_color=COLOR_ERROR_TEXT)
    add_arrow(ax, n51_qinc_yes["bottom"], n51_leaf_38["top"],
              label="35 <= f < 50",
              label_dx=0.40, label_dy=0.25)
    add_arrow(ax, n51_qinc_yes["bottom"], n51_leaf_28["top"],
              label="20 <= f < 35",
              label_dx=0.40, label_dy=-0.55)

    # default leaf (R119 baseline)
    n51_leaf_default = add_box(
        ax, 22.0, 3.6, 7.2, 1.6,
        "DEFAULT\nrw = 2.0,  lambda = 1.0\n(R119 baseline, narrow off-normal)",
        facecolor=COLOR_RECIPE, edgecolor=COLOR_RECIPE_EDGE,
        fontsize=10.5, fontweight="bold",
    )
    # default arrow from n51_qw "no" and from n51_qinc "no" both feed default
    add_arrow(ax, n51_qw["bottom"], (n51_leaf_default["center"][0] - 1.8, n51_leaf_default["top"][1]),
              label="no", label_dx=1.5, label_dy=-3.2)
    add_arrow(ax, n51_qinc["bottom"], (n51_leaf_default["center"][0] + 1.8, n51_leaf_default["top"][1]),
              label="no", label_dx=-1.6, label_dy=-3.2)

    # ---- Legend (above the envelope strip) ---------------------------------
    legend_y = 0.40
    legend_w, legend_h = 0.55, 0.40
    legend_entries = [
        (COLOR_RECIPE, COLOR_RECIPE_EDGE, "recipe leaf (rw, lambda)"),
        (COLOR_ERROR, COLOR_ERROR_TEXT, "ERROR leaf"),
        (COLOR_DECISION, EDGE_COLOR, "decision node"),
        (COLOR_INPUT, EDGE_COLOR, "input"),
    ]
    legend_x = 1.5
    legend_step = 7.2
    for i, (fc, ec, lbl) in enumerate(legend_entries):
        x = legend_x + i * legend_step
        ax.add_patch(FancyBboxPatch(
            (x, legend_y - legend_h / 2), legend_w, legend_h,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=4,
        ))
        ax.text(x + legend_w + 0.20, legend_y, lbl, ha="left", va="center",
                fontsize=10, color="#222", zorder=5)

    # ---- Bottom envelope strip (below the legend) --------------------------
    ax.text(
        16,
        -0.65,
        "Validated envelope:  n in {15, 31, 51, 71},   inc in [0, 70] deg,   "
        "freq in [5.8, 60] GHz,   width in [10, 30] deg",
        ha="center",
        va="center",
        fontsize=11,
        style="italic",
        color="#222",
        bbox=dict(boxstyle="round,pad=0.40", facecolor="#f3f3f3",
                  edgecolor="#888", linewidth=1.0),
    )

    plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.04)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "report_fig6_selector_tree.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    p = main()
    print(f"saved: {p.resolve()}")
