# -*- coding: utf-8 -*-
"""meeting_ga_vs_random.py — 組會第 13 頁：R15 三組對照＋R20 三代 GA vs 隨機。

數字出處：
- docs/log/round-15-pushbutton-vs-toolbox.md §4 表：三項全過 G 11/30、I 7/30、N 5/20。
- docs/log/round-20-evolution-loop.md §4 gen1–gen3 表與「三代終審」表：
  GA 12/60、10/58、6/54（累計 28/172）；隨機 14/60、11/57、14/59（終審表寫 39/175；
  逐代分母加總為 176，此處照原表標 39/175 並在腳本註記）；三項全過且帶外 <10：GA 19、隨機 10。
三項全過＝帶內最差餘裕 ≥0、方向圖餘裕 ≥0、金屬碎片皆 ≥4 px（S0 網格、無橋、單次量測）。
"""
from script.figs.meeting_common import (
    INK, INK2, MUTED, DBLUE, ORANGE, plt, style_ax, save, NOTE, TICK, LABEL)

R15 = [("GA\n自動演化", 11, 30, DBLUE), ("人挑\n知情選點", 7, 30, ORANGE), ("隨機\n同空間", 5, 20, MUTED)]
R20_GA = [(12, 60), (10, 58), (6, 54)]
R20_RN = [(14, 60), (11, 57), (14, 59)]


def main():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1, 1.35]))

    # ---- 左：R15 ----
    for i, (lab, k, n, c) in enumerate(R15):
        p = 100 * k / n
        a.bar(i, p, width=0.62, color=c, zorder=3)
        a.text(i, p + 1.2, f"{p:.0f}%", ha="center", va="bottom", fontsize=LABEL + 1, color=INK)
        a.text(i, p / 2, f"{k}/{n}", ha="center", va="center", fontsize=NOTE, color="white")
    a.set_xticks(range(3))
    a.set_xticklabels([r[0] for r in R15], fontsize=TICK)
    a.set_ylim(0, 48)
    style_ax(a, "", "三項規格全過率 (%)", "R15（7/9）：同一組件空間，每組 30 筆 HFSS，隨機組 20 筆")
    a.grid(axis="x", visible=False)

    # ---- 右：R20 三代 ----
    w = 0.36
    for g in range(3):
        for j, ((k, n), c, lab) in enumerate(((R20_GA[g], DBLUE, "GA（代理模型粗篩）"),
                                              (R20_RN[g], MUTED, "隨機（同子代池）"))):
            x = g + (j - 0.5) * w
            b.bar(x, k, width=w * 0.92, color=c, zorder=3, label=lab if g == 0 else None)
            b.text(x, k + 0.3, f"{k}/{n}", ha="center", va="bottom", fontsize=NOTE, color=INK)
    b.set_xticks(range(3))
    b.set_xticklabels(["第 1 代", "第 2 代", "第 3 代"], fontsize=TICK)
    b.set_ylim(0, 21.5)
    b.set_yticks(range(0, 21, 4))
    style_ax(b, "", "三項全過筆數", "R20（7/11）：每代約 60 筆 HFSS／組")
    b.grid(axis="x", visible=False)
    b.legend(fontsize=NOTE, loc="upper right", frameon=False)
    b.text(-0.42, 21.0, "累計  GA 28/172　隨機 39/175\n三項全過且帶外 <10：GA 19　隨機 10",
           fontsize=NOTE, color=INK2, va="top")
    fig.tight_layout(w_pad=2.5)
    save(fig, "ga_vs_random.png")


if __name__ == "__main__":
    main()
