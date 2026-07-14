# -*- coding: utf-8 -*-
"""report_dist_vs_strategy.py — 報告 §2 用：「分布 ≫ 策略」概念圖。
左＝把搜尋做到極聰明但候選分布沒好解（搆不到達標線）；右＝把候選分布做好（隨機抽就中）。
概念示意（非實資料），用來把「我們輸 random，輸的是分布不是搜尋」講到秒懂。
用法: python -m script.figs.report_dist_vs_strategy --out <path>"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    INK, INK2, GRID, SURF, RED, DBLUE, GREEN, ORANGE, MUTED, plt)


def _density(x, peaks):
    y = np.zeros_like(x)
    for mu, sig, amp in peaks:
        y += amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    x = np.linspace(-8, 2, 400)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)

    # ── 左：薄尾分布（線上能生成的）＋ 聰明策略搆不到
    yL = _density(x, [(-4.2, 1.1, 1.0), (-1.5, 0.7, 0.06)])   # 好解區(>0)幾乎空
    axL.fill_between(x, yL, color=ORANGE, alpha=0.22)
    axL.plot(x, yL, color=ORANGE, lw=2)
    axL.axvline(0, color=RED, ls="--", lw=1.6)
    axL.text(0.15, axL.get_ylim()[1] * 0.02 + 0.9, "達標線", color=RED, fontsize=10, rotation=90, va="bottom")
    # 聰明搜尋路徑：從 −6 爬到 −0.4 就到頂（越不過）
    px = np.array([-6.0, -5.0, -4.2, -3.3, -2.4, -1.5, -0.8, -0.4, -0.35])
    py = 0.16 + 0.02 * np.arange(len(px))
    axL.plot(px, py, "-o", color=INK, lw=1.6, ms=4, zorder=5)
    axL.annotate("聰明搜尋：搜遍了，卡在 −0.35 就到頂", xy=(-0.4, py[-1]), xytext=(-6.6, 0.62),
                 fontsize=9.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    axL.set_title("① 把搜尋做到極聰明（分布沒變）", color=INK, fontsize=12.5)
    axL.text(-4.0, -0.14, "好解幾乎不存在 → 策略再聰明也搆不到", ha="center",
             color=ORANGE, fontsize=10.5, fontweight="bold")

    # ── 右：厚尾分布（學長池）＋ 隨機抽就中
    yR = _density(x, [(-3.0, 1.5, 0.66), (-0.4, 1.2, 0.46), (0.7, 0.7, 0.24)])
    axR.fill_between(x, yR, color=DBLUE, alpha=0.18)
    axR.plot(x, yR, color=DBLUE, lw=2)
    axR.axvline(0, color=RED, ls="--", lw=1.6)
    axR.text(0.15, 0.9, "達標線", color=RED, fontsize=10, rotation=90, va="bottom")
    # 隨機抽的點：從右分布抽，>0 的標綠（中）
    sx = rng.choice(x, size=16, p=yR / yR.sum())
    sy = 0.14 + rng.uniform(0, 0.30, size=16)
    hit = sx > 0
    axR.scatter(sx[~hit], sy[~hit], color=MUTED, s=42, zorder=5, edgecolor=SURF, lw=0.6)
    axR.scatter(sx[hit], sy[hit], color=GREEN, s=70, zorder=6, edgecolor=SURF, lw=0.8, marker="*")
    axR.annotate(f"隨機抽 16 個，{int(hit.sum())} 個直接中", xy=(sx[hit][0], sy[hit][0]),
                 xytext=(-7.4, 0.62), fontsize=9.5, color=GREEN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))
    axR.set_title("② 把候選分布做好（隨便抽）", color=INK, fontsize=12.5)
    axR.text(-3.0, -0.14, "好解夠多 → 隨機抽就中（等效預算領先 200–450×）", ha="center",
             color=DBLUE, fontsize=10.5, fontweight="bold")

    for ax in (axL, axR):
        ax.set_xlim(-8, 2)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        ax.set_xlabel("候選的 worst-margin（越右越好，0＝達標）", color=INK2, fontsize=10)
        ax.grid(axis="x", color=GRID, lw=0.6, alpha=0.6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2)

    fig.suptitle("分布 ≫ 策略：我們輸給 random，輸的是「候選分布」不是「搜尋策略」",
                 color=INK, fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.015, "→ 別再優化「怎麼選」，去優化「從哪個分布選」——這就是我們轉去用既有池當起點、餵飛輪讓分布持續變好的原因。",
             ha="center", color=INK, fontsize=10.5)
    fig.subplots_adjust(left=0.03, right=0.985, top=0.87, bottom=0.16, wspace=0.06)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
