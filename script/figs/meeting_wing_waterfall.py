# -*- coding: utf-8 -*-
"""meeting_wing_waterfall.py — R14 翼消融瀑布（組會 2026-10 第 12 頁）。
數字取自 docs/log/round-14-component-axis.md §4 ablate 表（c21 欄；rad／oob 取 c21 那一側）。
full 基準公證 ×2 逐位一致；消融變體各單次。S0 網格。"""
from script.figs.meeting_figkit import (HALF, NOTE, INK, INK2, MUTED, RED, GREEN,
                                        f2, plt, sax, save)

MAIN, W0, W1, FULL_ = -5.79, -1.33, -1.71, 0.20     # c21 wm
RAD = (1.41, 1.24, 0.12)                             # 只留主件／＋單翼 w0／完整
OOB = (7.6, 9.9, 10.4)

fig = plt.figure(figsize=HALF)
ax = fig.add_axes([0.16, 0.37, 0.80, 0.56])
xs = [0, 1, 2]
# 只留主件（實心）
ax.bar(0, MAIN, color=MUTED, width=0.6, zorder=3)
# 首翼增量（浮動）
ax.bar(1, W0 - MAIN, bottom=MAIN, color=GREEN, width=0.6, zorder=3)
# 次翼增量
ax.bar(2, FULL_ - W0, bottom=W0, color=GREEN, width=0.6, zorder=3, alpha=0.75)
ax.plot([0.3, 0.7], [MAIN, MAIN], color=INK2, lw=1, ls=":")
ax.plot([1.3, 1.7], [W0, W0], color=INK2, lw=1, ls=":")
ax.axhline(0, color=INK2, lw=1)
ax.text(0, MAIN - 0.25, f2(MAIN), ha="center", va="top", fontsize=NOTE, color=INK)
ax.text(1, W0 + 0.15, f2(W0), ha="center", va="bottom", fontsize=NOTE, color=INK)
ax.text(1, (W0 + MAIN) / 2, f"首翼\n{f2(W0 - MAIN, 1)}", ha="center", va="center", fontsize=NOTE,
        color="white", weight="bold")
ax.text(2, FULL_ + 0.15, f2(FULL_), ha="center", va="bottom", fontsize=NOTE, color=INK)
ax.text(2.36, (FULL_ + W0) / 2, f"次翼\n{f2(FULL_ - W0, 1)}", ha="left", va="center",
        fontsize=NOTE, color=INK2)
ax.text(1, -6.75, f"另一翼單獨 {f2(W1)}", ha="center", va="center", fontsize=12, color=MUTED)
ax.set_xticks(xs, ["只留主件", "＋一翼", "＋兩翼＝完整"])
ax.set_xlim(-0.5, 2.9)
ax.set_ylim(-7.3, 1.0)
sax(ax, "", "帶內最差餘裕 (dB)")

# 代價列：rad 與帶外
tx = fig.add_axes([0.16, 0.07, 0.80, 0.20])
tx.set_xlim(-0.5, 2.9)
tx.set_ylim(0, 2)
tx.axis("off")
tx.text(-0.62, 1.5, "方向圖餘裕", ha="right", va="center", fontsize=NOTE, color=INK2)
tx.text(-0.62, 0.5, "帶外洩漏", ha="right", va="center", fontsize=NOTE, color=INK2)
for i in range(3):
    c = RED if i == 2 else INK
    tx.text(i, 1.5, f2(RAD[i]), ha="center", va="center", fontsize=13, color=c,
            weight="bold" if i == 2 else None)
    tx.text(i, 0.5, f"{OOB[i]:.1f}", ha="center", va="center", fontsize=13, color=c,
            weight="bold" if i == 2 else None)
tx.text(2.36, 1.0, "翼的代價：\n方向圖變差\n帶外變髒", ha="left", va="center", fontsize=12, color=RED)
fig.text(0.16, 0.0, "帶外洩漏＝帶外 Gain 峰值 − 帶外 S11 最低值，越低越乾淨", fontsize=12, color=MUTED)
fig.text(0.96, 0.955, "R14 c21，S0 網格；完整版重測兩次逐位一致，其餘單次", ha="right", fontsize=12, color=MUTED)
save(fig, "wing_waterfall.png")
