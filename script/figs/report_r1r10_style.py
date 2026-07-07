# -*- coding: utf-8 -*-
"""report_r1r10_style.py — R1-R10 成果報告圖的共用風格（docs/report/ 專用,三支 report_r1r10_*.py 共 import）。
沿用 repo 慣例:Microsoft JhengHei、淺色底、色票對齊 champ_figs/expected_best。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(REPO, "docs", "report", "assets")
CACHE = os.path.join(REPO, "tmp", "report_r1r10")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURF, RED = "#e1e0d9", "#fcfcfb", "#d03b3b"
DBLUE, AQUA, ORANGE, GREEN, GOLD, PURPLE = "#1c5cab", "#0e7a55", "#eb6834", "#1baf7a", "#b8860b", "#7b5ac2"

plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "sans-serif"],
    "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10.5,
})


def style_ax(ax, xl="", yl="", title="", tfs=11.5):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.7, alpha=0.85)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)
    if xl:
        ax.set_xlabel(xl, color=INK2, fontsize=10)
    if yl:
        ax.set_ylabel(yl, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=tfs)


def save(fig, name, dpi=140):
    os.makedirs(ASSETS, exist_ok=True)
    out = os.path.join(ASSETS, name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("→", out)


def show_pattern(ax, p, title="", feed=(24, 12), color=DBLUE, tfs=10):
    """25×25 pattern 底圖（金屬=color）＋feed 三角。"""
    from matplotlib.colors import ListedColormap
    import numpy as np
    p = np.asarray(p).reshape(25, 25) > 0.5
    ax.imshow(p.astype(int), cmap=ListedColormap([SURF, color]), vmin=0, vmax=1,
              origin="upper", interpolation="nearest")
    ax.scatter([feed[1]], [feed[0]], marker="^", s=46, color=AQUA, zorder=5,
               edgecolor=SURF, lw=0.8)
    if title:
        ax.set_title(title, color=INK, fontsize=tfs)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
