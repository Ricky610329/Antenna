# -*- coding: utf-8 -*-
"""meeting_gain_waterfall.py — 組會 2026-10 第 29 頁：可製造尺上的演算法增益瀑布＋橋三態。

同一把尺 wm_mfg＝min(m1+2, m2+2, m3, m4+5)，新幾何 p01、0.075 mm 菱形橋、規格 v2。
(a) docs/log/round-71-dual-dynasty-mining.md §4 b1：hc 對照臂（均勻隨機抽前代資料＋標準橋，15 筆）
    best −9.53 → A 臂選拔 best −5.90（R70 開帳）→ 開採 −3.82（R70 第三跳）；
    之後 −3.82 → −2.39 取自 docs/records_dual.json history（8/14–8/24，14 次刷新，
    含批次開採、SM 補池與無人值守迴圈 grind_loop）。
(b) round-71 §4 x1 探針：同一設計切掉橋 −31.03／0.01 mm 微橋 −6.86／0.075 mm 橋 −3.82。
零讀檔。執行：python -m script.figs.meeting_gain_waterfall
"""
import os

from script.figs.report_r1r10_style import (INK, INK2, MUTED, SURF, RED, DBLUE, GREEN, ORANGE,
                                            plt, style_ax)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "docs", "進度報告", "組會_2026-10", "figs")


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("→", out)


def big(ax, xl="", yl=""):
    style_ax(ax, xl, yl)
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.grid(axis="x", visible=False)


def waterfall(ax):
    big(ax, "", "最差餘裕 (dB)")
    levels = [-9.53, -5.90, -3.82, -2.39]
    names = ["隨機前代資料\n＋橋，15 筆", "選拔\n前代資料", "開採\n鄰域微調", "持續開採＋\n自動補池迴圈"]
    w = 0.56
    # 起點：從圖底畫到 −9.53 的灰柱
    floor = -11.5
    ax.bar(0, levels[0] - floor, bottom=floor, width=w, color=MUTED, zorder=3)
    ax.text(0, levels[0] + 0.15, f"{levels[0]:.2f}", ha="center", va="bottom", fontsize=12.5,
            color=INK)
    cols = [DBLUE, DBLUE, GREEN]
    for i in range(1, 4):
        a, b = levels[i - 1], levels[i]
        ax.bar(i, b - a, bottom=a, width=w, color=cols[i - 1], zorder=3)
        ax.plot([i - 1 + w / 2, i - w / 2], [a, a], color=INK2, lw=1, ls=":", zorder=2)
        ax.text(i, b + 0.15, f"{b:.2f}", ha="center", va="bottom", fontsize=12.5, color=INK)
        ax.text(i, (a + b) / 2, f"+{b - a:.2f}", ha="center", va="center", fontsize=12.5,
                color=SURF, fontweight="bold")
    ax.axhline(0, color=RED, lw=1.4, ls="--")
    ax.text(3.45, -0.2, "規格線 0", ha="right", va="top", fontsize=12, color=RED)
    ax.set_xticks(range(4), names)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(floor, 0.6)
    ax.text(3, -4.05, "8/14–8/24\n14 次刷新", ha="center", va="top", fontsize=11.5, color=INK2)
    ax.text(0.97, 0.05, "隨機組只有 15 筆，各段增益\n不可相加、不可外推", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12, color=INK)
    ax.text(0, 1.03, "(a) 同一把可製造尺上的進展", transform=ax.transAxes, fontsize=13,
            color=INK)


def bridge(ax):
    big(ax, "", "最差餘裕 (dB)")
    names = ["切掉橋", "0.01 mm\n微橋", "0.075 mm\n菱形橋"]
    vals = [-31.03, -6.86, -3.82]
    ax.bar(range(3), vals, width=0.6, color=[RED, ORANGE, DBLUE], zorder=3)
    for x, v in enumerate(vals):
        ax.text(x, v - 0.6, f"{v:.2f}", ha="center", va="top", fontsize=12.5, color=INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(3), names)
    ax.set_ylim(-36, 1)
    ax.text(0, 1.03, "(b) 同一設計的橋三態", transform=ax.transAxes, fontsize=13, color=INK)


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2),
                               gridspec_kw=dict(width_ratios=[1.75, 1], wspace=0.28))
    waterfall(a)
    bridge(b)
    fig.text(0.5, -0.04, "皆為新幾何、規格 v2 的可製造尺；(a) 出處 round-71 與 records_dual，"
             "(b) 出處 round-71 探針", ha="center", fontsize=12, color=INK2)
    save(fig, "gain_waterfall.png")


if __name__ == "__main__":
    main()
