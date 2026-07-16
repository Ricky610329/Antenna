# -*- coding: utf-8 -*-
"""report_dip_schematic.py — 報告 §3.1：生成器＝Deep Image Prior 的順向/反向示意。
教學點：梯度直接改 pattern；靠近 pattern 的層梯度才強、傳回可學習輸入時幾乎消失（所以可學習輸入沒用）；
網路權重＝存起來的先驗知識。用法: python -m script.figs.report_dip_schematic --out <path>"""
import argparse
import os
import sys

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    INK, INK2, MUTED, SURF, DBLUE, ORANGE, plt)

PALE = "#e9f1fb"
LGREY = "#f0efe9"
YC = 56


def rbox(ax, x0, x1, y0, y1, edge, face, lw=1.8):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                 boxstyle="round,pad=0.35,rounding_size=1.8",
                 linewidth=lw, edgecolor=edge, facecolor=face, zorder=3))


def farrow(ax, x0, x1, color=DBLUE, y=YC):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=15, lw=1.9, color=color, shrinkA=0, shrinkB=0, zorder=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(12.8, 5.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 95, "生成器就是一個 Deep Image Prior", ha="center", va="center",
            fontsize=15.5, fontweight="bold", color=INK)
    ax.text(50, 88.3, "梯度直接改 pattern；網路權重存的是「好天線長什麼樣」的先驗知識",
            ha="center", va="center", fontsize=10.5, color=INK2)

    # 規格 spec
    rbox(ax, 2, 14, YC - 6, YC + 6, MUTED, LGREY)
    ax.text(8, YC + 1.6, "規格 spec", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(8, YC - 3.3, "固定不變", ha="center", va="center", fontsize=8.4, color=INK2)

    # 生成器網路：梯形（窄輸入 → 寬輸出）
    tx0, tx1 = 22, 46
    hl, hr = 5.0, 9.0
    tri = [(tx0, YC - hl), (tx0, YC + hl), (tx1, YC + hr), (tx1, YC - hr)]
    ax.add_patch(Polygon(tri, closed=True, facecolor=PALE, edgecolor=DBLUE, lw=2, zorder=2))
    for f in (0.3, 0.55, 0.8):
        xx = tx0 + (tx1 - tx0) * f
        hh = hl + (hr - hl) * f
        ax.plot([xx, xx], [YC - hh, YC + hh], color=DBLUE, lw=0.8, alpha=0.32, zorder=2)
    ax.text((tx0 + tx1) / 2, YC + 13.5, "生成器網路", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=DBLUE)
    ax.text((tx0 + tx1) / 2, YC - 13.2, "結構＝先驗　·　權重＝存起來的知識",
            ha="center", va="center", fontsize=8.7, color=INK2)

    # pattern 25×25（清爽網格，左右對稱像天線）
    px0, py0, n = 51, YC - 8, 16
    cell = 16.0 / n
    rng = np.random.default_rng(5)
    m = rng.random((n, n)) > 0.5
    m = m | m[:, ::-1]
    ax.add_patch(Rectangle((px0, py0), 16, 16, facecolor=SURF, edgecolor=INK2,
                 lw=1.4, zorder=2))
    for i in range(n):
        for j in range(n):
            if m[i, j]:
                ax.add_patch(Rectangle((px0 + j * cell, py0 + (n - 1 - i) * cell),
                             cell, cell, facecolor=DBLUE, edgecolor=SURF, lw=0.3, zorder=3))
    ax.text(px0 + 8, YC + 11.2, "★ 真正被優化的對象", ha="center", va="center",
            fontsize=10, fontweight="bold", color=ORANGE)
    ax.text(px0 + 8, YC - 11.4, "天線圖案（25×25 金屬像素）", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=INK)

    # 代理模型 / loss
    rbox(ax, 71, 84, YC - 6, YC + 6, DBLUE, SURF)
    ax.text(77.5, YC + 1.6, "代理模型 SM", ha="center", va="center", fontsize=10,
            fontweight="bold", color=INK)
    ax.text(77.5, YC - 3.3, "預測 S11／增益", ha="center", va="center", fontsize=8.2, color=INK2)
    rbox(ax, 87, 98, YC - 6, YC + 6, DBLUE, SURF)
    ax.text(92.5, YC + 1.4, "loss", ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
    ax.text(92.5, YC - 3.4, "與規格差距", ha="center", va="center", fontsize=8.2, color=INK2)

    # 順向箭頭
    farrow(ax, 14, 22)
    farrow(ax, 46, 51)
    farrow(ax, 67, 71)
    farrow(ax, 84, 87)
    ax.text(18, YC + 4.2, "順向", ha="center", color=DBLUE, fontsize=8.4)

    # 反向梯度＝楔形（pattern 端粗 → 輸入端縮成尖點：梯度消失）
    wy = 30
    wedge = [(64, wy - 4.2), (64, wy + 4.2), (24, wy + 0.35), (24, wy - 0.35)]
    ax.add_patch(Polygon(wedge, closed=True, facecolor=ORANGE, edgecolor="none",
                 alpha=0.9, zorder=3))
    # loss/pattern → 楔形粗端 的細連接線
    ax.add_patch(FancyArrowPatch((92.5, YC - 6.5), (65, wy + 3.5),
                 connectionstyle="arc3,rad=-0.25", arrowstyle="-|>", mutation_scale=14,
                 lw=1.6, ls="--", color=ORANGE, zorder=4))
    ax.text(59, wy + 7.3, "梯度在這端最強", ha="center", color=ORANGE, fontsize=8.8,
            fontweight="bold")
    ax.text(27.5, wy + 3.6, "傳到輸入 ≈ 0", ha="center", color=MUTED, fontsize=8.6,
            fontweight="bold")
    ax.text(44, wy - 7.6, "反向傳播：越靠近 pattern 梯度越強，傳回輸入端幾乎消失——所以「可學習輸入」沒用",
            ha="center", color=INK2, fontsize=9.6)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=145, facecolor=SURF)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
