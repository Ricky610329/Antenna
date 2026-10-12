# -*- coding: utf-8 -*-
"""meeting_schematic.py — 組會 2026-10：講解用示意圖（不是實際設計、不帶數字）。

  schem_dust.png     移除 1–3 像素懸浮碎片的前後示意
  schem_wm.png       WM 定義：每個頻點的餘裕，跨規格、跨頻點取最小
  schem_antenna.png  單埠天線示意：像素金屬、底部饋入、往上輻射
  schem_filter.png   雙埠濾波器示意：P1 進、P2 出
  schem_wm1.png      v6：WM 定義只畫 S11 一格（併進輸出頁）
  schem_flow_senior.png／schem_flow_mine.png  v6：學長的線上迴圈與批次線上學習的時間軸示意
  schem_diamond.png  v6：對角接觸三種幾何（0.01 mm 重疊／挖掉 0.10 mm 菱形／加 0.10 mm 菱形，照實際比例）
  schem_point.png    v6：濾波器舊幾何（逐欄合併、欄間零厚度貼合）對照 8/13 起的單一導體加菱形橋
曲線是平滑的假資料，只用來說明；規格門檻與頻帶照 docs/paper/extended_abstract_figures.py。
配色與 meeting_clean.py 相同。零 HFSS、零 NAS。執行：python -m script.figs.meeting_schematic
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

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


def wm1():
    f = np.linspace(24, 32, 161)
    s11 = -4 - 17 * np.exp(-((f - 27.9) / 1.7) ** 2)
    pts = np.arange(26.5, 29.51, .5)
    vals = [np.interp(x, f, s11) for x in pts]
    worst = int(np.argmax(vals))
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    fig.subplots_adjust(left=.2, right=.97, top=.9, bottom=.17)
    ax.set_box_aspect(1)
    ax.add_patch(Rectangle((26.5, -10), 3, 10, color=FORBID, lw=0, zorder=0))
    ax.plot([26.5, 29.5], [-10, -10], color=INK, lw=.9, ls=(0, (4, 3)))
    ax.plot(f, s11, color=BLUE, lw=2)
    for i, (x, v) in enumerate(zip(pts, vals)):
        w = i == worst
        ax.plot([x, x], [v, -10], color=AMBER if w else TEAL, lw=3.4 if w else 2, solid_capstyle="butt")
        ax.plot(x, v, "o", ms=4.5, color=AMBER if w else BLUE, zorder=4)
    ax.annotate("最差點", (pts[worst], vals[worst]), xytext=(29.9, -16), fontsize=13, color=AMBER,
                fontweight="bold", arrowprops=dict(arrowstyle="-", color=AMBER, lw=1))
    ax.set_xlim(24, 32)
    ax.set_ylim(-25, 0)
    ax.set_xticks([24, 26, 28, 30, 32])
    ax.set_xlabel("頻率 (GHz)")
    ax.set_title("S11 (dB)", fontsize=14, color=INK)
    ax.grid(color=GRID, lw=.6)
    save(fig, "schem_wm1.png")


GEN, SIM, FIT, HUMAN = TEAL, BLUE, AMBER, "#c9d2d9"


def lanes(ax, names, t_end):
    ax.set_xlim(0, t_end)
    ax.set_ylim(-.6, len(names) - .4)
    ax.set_yticks(range(len(names)), names[::-1])
    ax.tick_params(axis="y", length=0, labelsize=13)
    ax.set_xticks([])
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)


def block(ax, lane, t0, w, color, label=None):
    ax.add_patch(Rectangle((t0, lane - .32), w, .64, color=color, lw=0))
    if label:
        ax.text(t0 + w / 2, lane, label, ha="center", va="center", fontsize=12,
                color=INK if color == HUMAN else "white")


def legend(ax):
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (GEN, SIM, FIT, HUMAN)]
    ax.legend(handles, ["產生候選", "HFSS 模擬", "訓練 SM", "人工調整"], ncol=4, frameon=False,
              loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=12, handlelength=1.2, borderaxespad=.2)


def flow_senior():
    """單機逐筆：產生→模擬→訓練一輪接一輪；中間人工調整算法時模擬機停下。"""
    fig, ax = plt.subplots(figsize=(12.1, 1.5))
    fig.subplots_adjust(left=.075, right=.99, top=.72, bottom=.12)
    lanes(ax, ["模擬機"], 40)
    t, k = 0.0, 0
    while t < 40:
        if k == 4:
            block(ax, 0, t, 9.0, HUMAN, "人工調整算法，HFSS 停下等待")
            t += 9.0
        else:
            for w, c in ((.7, GEN), (1.9, SIM), (1.3, FIT)):
                block(ax, 0, t, min(w, max(40 - t, 0)), c)
                t += w
        k += 1
    legend(ax)
    save(fig, "schem_flow_senior.png")


def flow_mine():
    """三台模擬機只跑 HFSS、不停；選批、重訓 SM、人工調整都在開發機上並行。"""
    fig, ax = plt.subplots(figsize=(12.1, 2.1))
    fig.subplots_adjust(left=.075, right=.99, top=.97, bottom=.08)
    lanes(ax, ["開發機", "HFSS 1", "HFSS 2", "HFSS 3"], 40)
    for lane, off in ((2, 0), (1, .6), (0, 1.1)):
        t = off
        while t < 40:
            block(ax, lane, t, min(1.85, 40 - t), SIM)
            t += 1.95
    t = 0.0
    for w, c, lab in ((1.6, GEN, "選批"), (2.6, FIT, "重訓 SM"), (7.0, HUMAN, "人工判讀與調整"),
                      (1.6, GEN, "選批"), (2.6, FIT, "重訓 SM"), (7.0, HUMAN, "人工判讀與調整"),
                      (1.6, GEN, "選批"), (2.6, FIT, "重訓 SM")):
        block(ax, 3, t, w, c, lab)
        t += w + 1.2
    save(fig, "schem_flow_mine.png")


def _sq(ax, x, y, side, color=BLUE):
    ax.add_patch(Rectangle((x - side / 2, y - side / 2), side, side, color=color, lw=0))


def _contact_panel(ax, title):
    ax.set_xlim(-.06, .46)
    ax.set_ylim(-.1, .46)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, color=INK, pad=6)
    for k in (0, .2, .4):                              # 像素格線（每格 0.2 mm）
        ax.plot([k, k], [0, .4], color=GRID, lw=.8, zorder=0)
        ax.plot([0, .4], [k, k], color=GRID, lw=.8, zorder=0)


RED = "#d23c3c"


def _diamond(ax, cx, cy, side, color):
    h = side / np.sqrt(2)                              # 邊長 side 的正方形轉 45°
    ax.add_patch(Polygon([(cx, cy + h), (cx + h, cy), (cx, cy - h), (cx - h, cy)], color=color, lw=0, zorder=3))


def diamond():
    """單埠（R54）：原始每格畫 0.21 mm，對角只靠角上 0.01 mm 重疊相連；
    切斷＝在角點 Subtract 一顆 0.10 mm 的 45° 菱形（diag_bridge_w=−0.10），加橋＝同形狀 Unite。尺寸照實際比例。"""
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 3.0))
    fig.subplots_adjust(left=.01, right=.99, top=.88, bottom=.02, wspace=.12)
    a, b, c = axs
    for ax, title in ((a, "原始幾何"), (b, "直接切斷"), (c, "加菱形橋")):
        _contact_panel(ax, title)
        for x, y in ((.1, .3), (.3, .1)):
            _sq(ax, x, y, .21)
    a.add_patch(Circle((.2, .2), .045, fill=False, ec=RED, lw=2))
    a.text(.2, -.05, "只靠 0.01 mm 重疊相連", ha="center", va="center", fontsize=12, color=RED)
    _diamond(b, .2, .2, .1, "white")
    b.text(.2, -.05, "挖掉 0.10 mm 菱形", ha="center", va="center", fontsize=12, color=INK)
    _diamond(c, .2, .2, .1, AMBER)
    c.text(.2, -.05, "加上 0.10 mm 菱形", ha="center", va="center", fontsize=12, color=AMBER)
    save(fig, "schem_diamond.png")


def point():
    """濾波器舊幾何（p00，學長原版 dual_port）：每格剛好 0.2 mm、逐欄 Unite，欄與欄是不同物件，
    只是零厚度貼合，對角只在一條邊上相接；8/13 起（p01）＋0.01 mm 重疊、全部聯集成一個導體、對角加 0.075 mm 菱形橋。"""
    bits = [(0, 0), (1, 0), (1, 1), (0, 2), (2, 2)]        # (列, 欄)，第 0 列在上
    cols = {0: BLUE, 1: TEAL, 2: "#6f8ea6"}                # 舊幾何：一欄一個物件，各上一色
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 3.4))
    fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.02, wspace=.2)
    a, b = axs
    for ax, title in ((a, "舊幾何"), (b, "8/13 起")):
        ax.set_xlim(-.06, .66)
        ax.set_ylim(-.14, .66)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=14, color=INK, pad=6)
        for k in (0, .2, .4, .6):
            ax.plot([k, k], [0, .6], color=GRID, lw=.8, zorder=0)
            ax.plot([0, .6], [k, k], color=GRID, lw=.8, zorder=0)
    for r, c in bits:
        x, y = .1 + .2 * c, .5 - .2 * r
        _sq(a, x, y, .2, color=cols[c])
        _sq(b, x, y, .21)
    for x, y in ((.4, .4), (.4, .2)):                      # 第 1 欄與第 2 欄的兩個對角接點
        a.add_patch(Circle((x, y), .035, fill=False, ec=RED, lw=2, zorder=4))
        _diamond(b, x, y, .075, AMBER)
    a.text(.3, -.07, "每欄各成一塊，欄與欄只是貼著", ha="center", va="center", fontsize=12, color=RED)
    b.text(.3, -.07, "合成一塊，對角加 0.075 mm 橋", ha="center", va="center", fontsize=12, color=AMBER)
    save(fig, "schem_point.png")

if __name__ == "__main__":
    dust()
    wm()
    antenna()
    filt()
    wm1()
    flow_senior()
    flow_mine()
    diamond()
    point()
