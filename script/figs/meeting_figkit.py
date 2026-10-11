# -*- coding: utf-8 -*-
"""meeting_figkit.py — 2026-10 組會補圖（粉塵雙向、翼瀑布、對角文法、菱形 vs 切斷、網格收斂、
diffsim 對照、emforge 接入）共用的投影字級與存檔。色票與底層風格沿用 report_r1r10_style。
（另有 meeting_common.py 屬同場另一組補圖；兩者獨立，避免互踩。）
輸出：docs/進度報告/組會_2026-10/figs/<名>.png（dpi 200）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402,F401
    INK, INK2, MUTED, GRID, SURF, RED, DBLUE, AQUA, ORANGE, GREEN, GOLD, PURPLE,
    REPO, style_ax, plt)

OUT = os.path.join(REPO, "docs", "進度報告", "組會_2026-10", "figs")
FULL = (12, 5.2)
HALF = (6.2, 5.0)
TICK, LABEL, NOTE = 12.5, 13.5, 12.5
plt.rcParams["axes.unicode_minus"] = True   # JhengHei 有 U+2212，刻度用真負號


def f2(v, nd=2):
    """帶號數字，用真負號。"""
    return f"{v:+.{nd}f}".replace("-", "−")


def sax(ax, xl="", yl=""):
    """style_ax ＋ 投影字級（刻度 12.5、軸標 13.5）。"""
    style_ax(ax, xl, yl)
    ax.tick_params(labelsize=TICK)
    if xl:
        ax.set_xlabel(xl, color=INK2, fontsize=LABEL)
    if yl:
        ax.set_ylabel(yl, color=INK2, fontsize=LABEL)


def save(fig, name, dpi=200):
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("->", out)


__all__ = ["INK", "INK2", "MUTED", "GRID", "SURF", "RED", "DBLUE", "AQUA", "ORANGE", "GREEN", "GOLD",
           "PURPLE", "REPO", "plt", "OUT", "FULL", "HALF", "TICK", "LABEL", "NOTE", "f2", "sax", "save"]
