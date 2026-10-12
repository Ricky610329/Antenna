# -*- coding: utf-8 -*-
"""meeting_schematic.py — 組會 2026-10：講解用示意圖（不是實際設計、不帶數字）。

  schem_dust.png     移除 1–3 像素懸浮碎片的前後示意
  schem_wm.png       WM 定義：每個頻點的餘裕，跨規格、跨頻點取最小
  schem_antenna.png  單埠天線示意：像素金屬、底部饋入、往上輻射
  schem_filter.png   雙埠濾波器示意：P1 進、P2 出
曲線是平滑的假資料，只用來說明；規格門檻與頻帶照 docs/paper/extended_abstract_figures.py。
配色與 meeting_clean.py 相同。零 HFSS、零 NAS。執行：python -m script.figs.meeting_schematic
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "docs", "進度報告", "組會_2026-10", "figs")

BLUE, AMBER, TEAL = "#234c68", "#bd6532", "#3b8a7e"
INK, MUTED, GRID, FORBID = "#18212b", "#9eabb5", "#e3e8ec", "#f6e0d8"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"], "font.size": 13,
                     "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": .8, "axes.edgecolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK})


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    print("→", out)


def grid(ax, bits, color=BLUE, hi=None, ghost=None):
    """bits[r][c]=1 畫金屬；hi 的格子改用琥珀色；ghost 的格子畫虛線框。第 0 列在圖上方。"""
    n = len(bits)
    for r in range(n):
        for c in range(n):
            if (r, c) in (ghost or ()):
                ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, fill=False, ec=AMBER, lw=1.6, ls=(0, (2, 2))))
            elif bits[r][c]:
                ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, color=AMBER if (r, c) in (hi or ()) else color, lw=0))
    ax.add_patch(Rectangle((0, 0), n, n, fill=False, ec=MUTED, lw=.8))
    ax.set_xlim(-.3, n + .3)
    ax.set_ylim(-.9, n + .3)
    ax.set_aspect("equal")
    ax.axis("off")


BODY = ["11100111",
        "11100111",
        "00111000",
        "00011000",
        "00011000",
        "00011000",
        "11111111",
        "00011000"]


def dust():
    base = [[int(ch) for ch in row] for row in BODY]
    frag = [(3, 6), (4, 6)]
    for r, c in frag:
        base[r][c] = 1
    fig, (a, b) = plt.subplots(1, 2, figsize=(4.35, 2.25))
    fig.subplots_adjust(left=.02, right=.98, top=.86, bottom=.02, wspace=.45)
    grid(a, base, hi=frag)
    after = [row[:] for row in base]
    for r, c in frag:
        after[r][c] = 0
    grid(b, after, ghost=frag)
    a.set_title("移除前", fontsize=14, color=INK)
    b.set_title("移除後", fontsize=14, color=INK)
    fig.add_artist(FancyArrowPatch((.47, .45), (.55, .45), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=16, color=INK, lw=1.4))
    save(fig, "schem_dust.png")


def wm():
    f = np.linspace(24, 32, 161)
    s11 = -4 - 17 * np.exp(-((f - 27.9) / 1.7) ** 2)
    gain = 6 - .3 * (f - 27.8) ** 2
    pts = np.arange(26.5, 29.51, .5)
    m11 = [-10 - np.interp(x, f, s11) for x in pts]
    mg = [np.interp(x, f, gain) - 4 for x in pts]
    worst = ("s11", int(np.argmin(m11))) if min(m11) < min(mg) else ("gain", int(np.argmin(mg)))

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 4.0))
    fig.subplots_adjust(left=.1, right=.98, top=.88, bottom=.15, wspace=.38)
    for ax, y, thr, lo, hi, title, key, up in (
            (a, s11, -10, -25, 0, "S11 (dB)", "s11", True),
            (b, gain, 4, -2, 8, "增益 (dBi)", "gain", False)):
        ax.set_box_aspect(1)
        ax.add_patch(Rectangle((26.5, thr), 3, (hi - thr) if up else (lo - thr), color=FORBID, lw=0, zorder=0))
        ax.plot([26.5, 29.5], [thr, thr], color=INK, lw=.9, ls=(0, (4, 3)))
        ax.plot(f, y, color=BLUE, lw=2)
        for i, x in enumerate(pts):
            v = np.interp(x, f, y)
            w = (key, i) == worst
            ax.plot([x, x], [v, thr], color=AMBER if w else TEAL, lw=3.2 if w else 2, solid_capstyle="butt")
            ax.plot(x, v, "o", ms=4.5, color=AMBER if w else BLUE, zorder=4)
            if w:
                ax.annotate("最差點", (x, (v + thr) / 2), xytext=(x + .5, (v + thr) / 2 - (3 if up else -1.2)),
                            fontsize=13, color=AMBER, fontweight="bold")
        ax.set_xlim(24, 32)
        ax.set_ylim(lo, hi)
        ax.set_xticks([24, 26, 28, 30, 32])
        ax.set_xlabel("頻率 (GHz)")
        ax.set_title(title, fontsize=14, color=INK)
        ax.grid(color=GRID, lw=.6)
    save(fig, "schem_wm.png")


def antenna():
    rng = np.random.default_rng(7)
    half = (rng.random((10, 5)) < .55).astype(int)
    bits = np.hstack([half, half[:, ::-1]]).tolist()
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    grid(ax, bits)
    ax.plot(5, -.45, marker="^", ms=9, color=INK)
    for k, r in enumerate((1.2, 2.0, 2.8)):
        t = np.linspace(np.pi * .3, np.pi * .7, 30)
        ax.plot(5 + r * np.cos(t), 10.1 + r * np.sin(t) - r * .55, color=AMBER, lw=1.6, alpha=1 - k * .25)
    ax.set_ylim(-.9, 12.6)
    ax.set_xlim(-1.3, 11.3)
    save(fig, "schem_antenna.png")


def filt():
    rng = np.random.default_rng(11)
    bits = (rng.random((10, 10)) < .5).astype(int)
    bits[:, 4:6] = 1
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    grid(ax, bits.tolist())
    ax.plot(5, -.45, marker="^", ms=9, color=INK)
    ax.plot(5, 10.45, marker="v", ms=9, color=INK)
    ax.text(5.8, -.75, "P1", fontsize=12, color=INK)
    ax.text(5.8, 10.35, "P2", fontsize=12, color=INK)
    ax.set_ylim(-1.3, 11.3)
    ax.set_xlim(-1.3, 11.3)
    save(fig, "schem_filter.png")


if __name__ == "__main__":
    dust()
    wm()
    antenna()
    filt()
