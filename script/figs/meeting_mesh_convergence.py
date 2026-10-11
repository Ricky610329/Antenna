# -*- coding: utf-8 -*-
"""meeting_mesh_convergence.py — R52 HFSS 網格收斂（組會 2026-10 第 22 頁）。
數字：docs/log/round-52-anchor-assault.md §4「網格收斂定讞」與「S0↔S1 初判」——
  S0→S1 |Δf_res| 中位 4.066 GHz、S1→S2 0.003 GHz（A 組配對 n=5）；全體初判 S0→S1 中位 0.985 GHz（n=18）；
  ρ(wm_S0, wm_S1)=+0.837（n=18）。
王 c48nq1p05_16 S0 +0.79（三次一致）→ S1 −0.25：出自 docs/log/round-54-diamond-bridge.md §4 s1base（單次），
  round-52 本身沒有這個數字。"""
from script.figs.meeting_figkit import (HALF, NOTE, INK, INK2, MUTED, RED, GREEN,
                                        plt, sax, save)

BARS = [  # 標籤, 值 GHz, 顏色, 註
    ("S0 → S1\n同 5 筆", 4.066, RED),
    ("S1 → S2\n同 5 筆", 0.003, GREEN),
    ("S0 → S1\n全體 18 筆", 0.985, RED),
]

fig = plt.figure(figsize=HALF)
ax = fig.add_axes([0.17, 0.40, 0.79, 0.55])
for i, (lab, v, c) in enumerate(BARS):
    ax.bar(i, v, color=c, width=0.6, zorder=3, alpha=0.45 if i == 2 else 1, log=True)
    ax.text(i, v * 1.35, f"{v:.3f} GHz", ha="center", va="bottom", fontsize=NOTE, color=INK,
            weight="bold" if i < 2 else None)
ax.set_yscale("log")
ax.set_ylim(1e-3, 120)
ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10], ["0.001", "0.01", "0.1", "1", "10"])
ax.minorticks_off()
ax.set_xticks(range(3), [b[0] for b in BARS])
ax.get_xticklabels()[2].set_color(MUTED)
sax(ax, "", "共振頻率位移中位 (GHz)\n對數軸")
ax.text(0.5, 0.36, "細到 S1 已收斂\n現行 S0 還沒有", transform=ax.transAxes, ha="center",
        fontsize=NOTE, color=INK2)
ax.text(0.5, 0.97, "S0＝現行網格設定，S1、S2 逐級加細", transform=ax.transAxes, ha="center",
        va="top", fontsize=12, color=MUTED)

tx = fig.add_axes([0.02, 0.0, 0.96, 0.25])
tx.axis("off")
tx.text(0.0, 0.85, "• 紀錄王：S0 +0.79 → S1 −0.25", fontsize=13, color=INK, weight="bold",
        va="center")
tx.text(0.04, 0.58, "S0 值三次一致；S1 值單次，出自 R54", fontsize=12, color=MUTED, va="center")
tx.text(0.0, 0.30, "• S0 與 S1 排序一致度 ρ = 0.837（n=18）", fontsize=13, color=INK, weight="bold",
        va="center")
tx.text(0.04, 0.03, "＝所有代理模型排序的天花板；紀錄只在 S0 內互比", fontsize=12, color=MUTED, va="center")
save(fig, "mesh_convergence.png")
