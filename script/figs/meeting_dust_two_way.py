# -*- coding: utf-8 -*-
"""meeting_dust_two_way.py — 粉塵／懸浮件的雙向因果（組會 2026-10 第 8 頁）。
左：R7 拔掉 1–3 px 粉塵（d3 變體，全碎片 ≥4 px）後的 Δ帶內最差餘裕（docs/log/round-07-dedust.md §4 表）。
右：R11 把懸浮翼用 1 px 金屬接到主件（docs/report/status-2026-07-08.md §3.2 表；round-11 §5 probes P2）。
數字寫死自原表，零 HFSS、零 NAS。全部 S0 網格、單次量測。"""
import numpy as np

from script.figs.meeting_figkit import (FULL, NOTE, INK, INK2, MUTED, RED, DBLUE,
                                        f2, plt, sax, save)


# R7：d3＝拔 <4 px（1–3 px 全拔）；Δworst vs orig
R7 = [("p00", -6.70, 26), ("p01", -16.93, 35), ("p02", -5.02, 17), ("p03", -0.02, 19), ("p04", -8.84, 27)]
# R11：原樣 wm → 兩個搭橋變體 wm（status-2026-07-08 §3.2 表列的是絕對值，不是 Δ）
R11 = [("c21", 0.20, (-5.45, -5.94)), ("b11", 0.13, (-5.47, -5.64))]

fig, (axl, axr) = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1.25, 1]))

# ── 左：拔掉 ──
x = np.arange(len(R7))
vals = [v for _, v, _ in R7]
axl.bar(x, vals, color=[MUTED if v > -1 else RED for v in vals], width=0.62, zorder=3)
for i, (name, v, px) in enumerate(R7):
    axl.text(i, v - 0.45, f2(v), ha="center", va="top", fontsize=NOTE, color=INK)
axl.axhline(0, color=INK2, lw=1)
axl.set_xticks(x, [f"{n}\n拔 {px} px" for n, _, px in R7])
axl.set_ylim(-20, 1.5)
sax(axl, "", "Δ帶內最差餘裕 (dB)")
axl.text(0.5, 1.035, "拔掉 1–3 px 粉塵：5 筆崩 4 筆", transform=axl.transAxes, ha="center",
         va="bottom", fontsize=14, color=INK, weight="bold")
axl.annotate("p03 是整塊型設計\n粉塵才真的是粉塵", xy=(3, -1.6), xytext=(2.35, -13.2),
             fontsize=NOTE, color=INK2, ha="left",
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
axl.text(0.01, 0.02, "R7，S0 網格，各單次", transform=axl.transAxes, fontsize=12, color=MUTED)

# ── 右：接上 ──
xs = []
for i, (name, orig, (a, b)) in enumerate(R11):
    xc = i * 1.0
    xs.append(xc)
    axr.bar(xc - 0.22, orig, width=0.36, color=DBLUE, zorder=3, label="原樣（懸浮翼）" if i == 0 else None)
    axr.bar([xc + 0.14, xc + 0.34], [a, b], width=0.18, color=RED, zorder=3,
            label="翼用 1 px 接到主件" if i == 0 else None)
    axr.text(xc - 0.22, orig + 0.25, f2(orig), ha="center", va="bottom", fontsize=NOTE, color=INK)
    axr.text(xc + 0.24, min(a, b) - 0.35, f"{f2(a)}\n{f2(b)}", ha="center", va="top",
             fontsize=NOTE, color=INK, linespacing=1.1)
axr.axhline(0, color=INK2, lw=1)
axr.set_xticks(xs, [n for n, *_ in R11])
axr.set_xlim(-0.6, 1.6)
axr.set_ylim(-10, 1.5)
sax(axr, "", "帶內最差餘裕 (dB)")
axr.legend(loc="lower right", fontsize=12, frameon=False, ncol=1)
axr.text(0.5, 1.035, "把懸浮翼接上：材料只增不減，照樣崩 ~5.6 dB", transform=axr.transAxes,
         ha="center", va="bottom", fontsize=14, color=INK, weight="bold")
axr.text(0.01, 0.02, "R11，S0 網格\n各兩種接法、單次", transform=axr.transAxes, ha="left", va="bottom",
         fontsize=12, color=MUTED)

fig.text(0.5, -0.04, "拔掉會崩，接上也會崩 → 懸浮金屬有寄生功能，不是雜訊",
         ha="center", va="top", fontsize=15, color=RED, weight="bold")
fig.tight_layout(w_pad=3)
save(fig, "dust_two_way.png")
