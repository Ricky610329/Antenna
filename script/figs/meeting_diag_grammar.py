# -*- coding: utf-8 -*-
"""meeting_diag_grammar.py — R43–44 組文法：同骨架只差對角（組會 2026-10 第 18 頁）。
數字取自 docs/log/round-44-grammar-2.md §4「文法帳六批總表（R43+R44）」：
GD n=6 lo 中位 +2.27、lo≤−2 1/6；GDd n=6 −3.34、4/6；舊隨機（old）n=12 −4.46、9/12。
lo＝左側帶外 Gain 峰值（越負＝左側壓得越好）。S0 網格、各單次；此批全部未達標（best wm ≤ −6）。"""
from script.figs.meeting_figkit import (HALF, NOTE, INK, INK2, MUTED, RED, DBLUE, ORANGE,
                                        f2, plt, sax, save)

ROWS = [  # 名稱, lo 中位, 左傾數, n, 顏色
    ("組文法\n無對角", 2.27, 1, 6, ORANGE),
    ("組文法\n有對角", -3.34, 4, 6, DBLUE),
    ("舊隨機塊\n（天然多對角）", -4.46, 9, 12, MUTED),
]

fig, ax = plt.subplots(figsize=HALF)
for i, (name, v, k, n, c) in enumerate(ROWS):
    ax.bar(i, v, color=c, width=0.6, zorder=3, alpha=0.6 if i == 2 else 1)
    off = 0.25 if v > 0 else -0.25
    ax.text(i, v + off, f2(v), ha="center", va="bottom" if v > 0 else "top", fontsize=13,
            color=INK, weight="bold")
    ax.text(i, 3.85, f"左傾 {k}/{n}", ha="center", va="center", fontsize=NOTE, color=INK2)
ax.axhline(0, color=INK2, lw=1)
ax.axhline(-2, color=RED, lw=1, ls="--")
ax.text(2.45, -2.12, "左傾門檻 −2", ha="right", va="top", fontsize=12, color=RED)
ax.plot([0.3, 0.55], [2.27, 2.27], color=INK2, lw=1, ls=":")
ax.plot([0.55, 0.7], [-3.34, -3.34], color=INK2, lw=1, ls=":")
ax.annotate("", xy=(0.55, 2.27), xytext=(0.55, -3.34),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.3))
ax.text(0.5, -1.25, "只差對角\n左側差 5.6 dB", ha="right", va="center", fontsize=NOTE, color=INK)
ax.set_xticks(range(3), [r[0] for r in ROWS])
ax.set_xlim(-0.55, 2.55)
ax.set_ylim(-5.8, 4.4)
sax(ax, "", "左側帶外 Gain 峰值中位 (dBi)\n越低＝左側壓得越好")
ax.get_xticklabels()[2].set_color(MUTED)
fig.text(0.98, 0.005, "R43–44 六批，S0 網格，各單次；左傾＝峰值 ≤ −2 dBi\n"
         "有無對角兩組的金屬量不同，R43 已註記此混雜",
         ha="right", va="bottom", fontsize=12, color=MUTED)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, "diag_grammar.png")
