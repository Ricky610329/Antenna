# -*- coding: utf-8 -*-
"""report_online_profile.py — 報告 §8.1 用：線上迴圈 278s/ep 每階段佔比 vs 批次線 160s/筆。
一眼看出「線上迴圈近四成不在產資料（陪跑單樣本過擬合）；批次線更短且 100% 在產資料」。
數字來源＝正式機 profiling（commit ec774de, 2026-06-21）。用法: python -m script.figs.report_online_profile --out <path>"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    INK, INK2, GRID, SURF, DBLUE, ORANGE, GOLD, MUTED, plt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # 線上迴圈 278s：HFSS 61% / rad 24% / S11 12% / 其他 3%
    online = [("HFSS 求解＋讀回", 170, DBLUE), ("rad 單筆擬合", 67, ORANGE),
              ("S11／Gain 單筆擬合", 33, GOLD), ("開專案/收尾/推論", 8, MUTED)]
    fig, ax = plt.subplots(figsize=(12.6, 4.6))

    # Bar 1（上）＝線上迴圈
    x = 0
    for lab, w, c in online:
        ax.barh(1, w, left=x, color=c, edgecolor=SURF, height=0.52)
        if w >= 25:
            pct = round(100 * w / 278)
            ax.text(x + w / 2, 1, f"{lab}\n{w}s（{pct}%）", ha="center", va="center",
                    color=SURF if c in (DBLUE, ORANGE) else INK, fontsize=9, fontweight="bold")
        x += w
    # Bar 2（下）＝批次線 160s 純 HFSS
    ax.barh(0, 160, color=DBLUE, edgecolor=SURF, height=0.52)
    ax.text(80, 0, "HFSS 求解＋讀回\n160s（100%）", ha="center", va="center",
            color=SURF, fontsize=9.5, fontweight="bold")

    # 「虧掉的近四成」括號（線上 bar 的 HFSS 之後那段）
    ax.annotate("", xy=(278, 1.42), xytext=(170, 1.42),
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.4))
    ax.plot([170, 170], [1.40, 1.44], color=ORANGE, lw=1.4)
    ax.plot([278, 278], [1.40, 1.44], color=ORANGE, lw=1.4)
    ax.text(224, 1.55, "≈ 39% 虧在「單樣本過擬合」——不是在產資料\n（rad 那 24% 撞滿 20,000 迭代純浪費）",
            ha="center", va="bottom", color=ORANGE, fontsize=9.5, fontweight="bold")
    # 「真正產資料」標註
    ax.text(85, 0.62, "↑ 只有這 61% 真正在產資料", ha="center", color=DBLUE, fontsize=9)
    ax.text(80, -0.5, "批次線：把訓練解耦 → HFSS 只產資料，更短、100% 有效",
            ha="center", color=DBLUE, fontsize=9.5, fontweight="bold")

    ax.set_yticks([1, 0])
    ax.set_yticklabels(["線上迴圈\n278s / epoch", "批次線\n160s / 筆"], fontsize=11, color=INK)
    ax.set_xlabel("每筆資料耗時（秒）", color=INK2, fontsize=10.5)
    ax.set_xlim(0, 300)
    ax.set_ylim(-0.75, 1.95)
    ax.set_title("同一台 HFSS，線上迴圈近四成時間在陪跑訓練；批次線解耦後只產資料",
                 color=INK, fontsize=13, pad=14)
    ax.grid(axis="x", color=GRID, lw=0.7, alpha=0.7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.14)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
