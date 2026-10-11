# -*- coding: utf-8 -*-
"""meeting_diamond_vs_cut.py — R54 菱形橋 vs 切斷（組會 2026-10 第 23 頁）。
資料：docs/log/assets/round-54/r54_close_analysis.md
  §2 菱形直方圖（0.5 dB 分箱，合格層＋近合格層相加＝全體 n=3,318）、|Δ|≤0.3 帶內 86.5%、中位 −0.07；
  §3 挖空（切斷）Δwm 直方（不等寬分箱，全體 n=3,317）、中位 −4.84、P10 −12.38；
  §2「稅 vs n_sites」分組表與二分帳（<17：91.4%，n=3,012；≥17：38.6%，n=306）。
條件：S0 網格、菱形橋寬 0.10 mm、每親一次（單次）；S0 對幾何微變自帶 ±0.3 dB 重網格抖動。"""
import numpy as np

from script.figs.meeting_figkit import (FULL, NOTE, INK, INK2, MUTED, RED, DBLUE, ORANGE,
                                        plt, sax, save)

# 菱形（全體＝valq＋valn）
D_EDGES = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]
D_CNT = [1 + 1, 0 + 5, 6 + 14, 4 + 20, 43 + 104, 806 + 1252, 348 + 713, 0 + 1]
# 切斷（全體）
C_EDGES = [-30, -25, -20, -15, -12, -10, -8, -6, -4, -2, -1, 0, 2]
C_CNT = [1, 13, 139, 217, 384, 422, 321, 330, 608, 263, 422, 197]
assert sum(D_CNT) == 3318 and sum(C_CNT) == 3317

# 帶內保持率 vs 橋座數（略去 1–4 座 n=2、25 座以上 n=4）
SITES = [("5–8", 573, 92.1), ("9–12", 1913, 91.9), ("13–16", 524, 88.4), ("17–24", 302, 38.7)]

fig, (a1, a2) = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1.75, 1]))


def bars(ax, edges, cnt, n, **kw):
    e = np.asarray(edges, float)
    pct = 100 * np.asarray(cnt) / n
    ax.bar(e[:-1], pct, width=np.diff(e), align="edge", **kw)


bars(a1, C_EDGES, C_CNT, 3317, color=RED, alpha=0.55, edgecolor=RED, lw=0.8, zorder=2,
     label="切斷對角（n=3,317）")
bars(a1, D_EDGES, D_CNT, 3318, color=DBLUE, alpha=0.9, edgecolor="white", lw=0.6, zorder=3,
     label="0.10 mm 菱形橋（n=3,318）")
a1.axvline(-4.84, color=RED, lw=1.6, ls="--", zorder=4)
a1.axvline(-12.38, color=RED, lw=1.2, ls=":", zorder=4)
a1.text(-4.6, 50, "切斷中位\n−4.84", ha="left", va="center", fontsize=NOTE, color=RED)
a1.text(-12.1, 28, "切斷 P10\n−12.38", ha="left", va="center", fontsize=NOTE, color=RED)
a1.annotate("菱形中位 −0.07\n|Δ| ≤ 0.3 dB 佔 86.5%", xy=(-0.5, 56), xytext=(-15.5, 45),
            fontsize=NOTE, color=DBLUE, ha="right", va="center", weight="bold",
            arrowprops=dict(arrowstyle="->", color=DBLUE, lw=1.3))
a1.set_xlim(-30.5, 2.5)
a1.set_ylim(0, 68)
sax(a1, "Δ帶內最差餘裕 (dB)，改幾何後 − 原樣", "每個分箱的佔比 (%)")
a1.legend(loc="upper left", fontsize=12, frameon=False)
a1.text(0.015, 0.80, "菱形 0.5 dB 分箱；切斷 1–5 dB 不等寬分箱", transform=a1.transAxes,
        fontsize=12, color=MUTED, va="top")

# ── 右：橋座數 ──
x = np.arange(len(SITES))
cols = [DBLUE if s[0] != "17–24" else ORANGE for s in SITES]
a2.bar(x, [s[2] for s in SITES], color=cols, width=0.74, zorder=3)
for i, (lab, n, r) in enumerate(SITES):
    a2.text(i, r + 1.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=NOTE, color=INK, weight="bold")
    a2.text(i, 4, f"n={n:,}", ha="center", va="bottom", fontsize=12, color="white")
a2.axvline(2.5, color=INK2, lw=1, ls="--")
a2.text(1.0, 108, "< 17 座合計 91.4%（n=3,012）", ha="center", fontsize=NOTE, color=DBLUE, weight="bold")
a2.text(3.0, 48, "≥ 17 座\n合計 38.6%\nn=306", ha="center", va="bottom", linespacing=1.15, fontsize=NOTE, color=ORANGE, weight="bold")
a2.set_xticks(x, [s[0] for s in SITES])
a2.set_ylim(0, 118)
a2.set_yticks([0, 25, 50, 75, 100])
sax(a2, "每個設計的菱形橋座數", "|Δ| ≤ 0.3 dB 的比例 (%)")

fig.text(0.5, -0.02, "R54 全母體 3,319 個設計各改兩種幾何，S0 網格、各單次；S0 對幾何微變自帶 ±0.3 dB 抖動，"
         "只做母體統計、不判個案", ha="center", va="top", fontsize=12, color=MUTED)
fig.tight_layout(w_pad=2.5)
save(fig, "diamond_vs_cut.png")
