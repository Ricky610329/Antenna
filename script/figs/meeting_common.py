# -*- coding: utf-8 -*-
"""meeting_common.py — 2026-10 組會簡報補圖的共用設定（投影用大字版）。
沿用 report_r1r10_style 的色票與字型；輸出到 docs/進度報告/組會_2026-10/figs/，dpi 200。
只讀 repo 內 JSON/CSV 與本機 tmp 快取，不碰 NAS。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402,F401
    REPO, INK, INK2, MUTED, GRID, SURF, RED, DBLUE, AQUA, ORANGE, GREEN, GOLD, PURPLE,
    plt, show_pattern)

OUT = os.path.join(REPO, "docs", "進度報告", "組會_2026-10", "figs")
FULL = (12, 5.2)
HALF = (6.2, 5.0)
TICK, LABEL, NOTE = 12.5, 13.5, 12

__all__ = ["REPO", "INK", "INK2", "MUTED", "GRID", "SURF", "RED", "DBLUE", "AQUA", "ORANGE",
           "GREEN", "GOLD", "PURPLE", "plt", "show_pattern", "OUT", "FULL", "HALF",
           "TICK", "LABEL", "NOTE", "style_ax", "save"]

plt.rcParams.update({"font.size": NOTE})


def style_ax(ax, xl="", yl="", sub=""):
    """投影版 style_ax：刻度 ≥12pt、軸標 ≥13pt；sub＝小副標（不放大標題）。"""
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.7, alpha=0.85)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=TICK)
    if xl:
        ax.set_xlabel(xl, color=INK2, fontsize=LABEL)
    if yl:
        ax.set_ylabel(yl, color=INK2, fontsize=LABEL)
    if sub:
        ax.set_title(sub, color=INK2, fontsize=NOTE, loc="left")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("→", out)
