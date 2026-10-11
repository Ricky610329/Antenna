# -*- coding: utf-8 -*-
"""meeting_diffsim_final.py — 可微物理模擬器 vs 代理模型最終對照（組會 2026-10 第 21 頁）。
上：正片 clean_OOS（n=817，三方都沒見過）同池同尺，docs/log/analysis-10-sm-vs-physics.md §51.6 表
   （取代 §48.2 的跨池數字）：top-60 挑中真 top10% 的命中率、P(挑出批的最佳值勝過隨機批)。
下：負片 neg_OOS（n=1200，SM-mlp/cnn 沒見過）層內 ρ(wm)，§3.1 表（物理為 8/3 Stage 1 前的 L3 精確 MoM；
   SM-two 該域 in-sample）。§4.1 結論：資料沒覆蓋的域物理有用，一覆蓋就輸。"""
import numpy as np

from script.figs.meeting_figkit import (HALF, INK, INK2, MUTED, DBLUE, ORANGE,
                                        f2, plt, sax, save)

GROUPS = [  # 標籤, 命中率 %, P(勝隨機) %, 顏色
    ("隨機", 10.0, 50.0, MUTED),
    ("物理\n修正前", 10.0, 17.9, ORANGE),
    ("物理\n修正後", 41.7, 61.7, ORANGE),
    ("SM\ncnn", 41.7, 80.0, DBLUE),
    ("SM\ntwo", 45.0, 86.4, DBLUE),
    ("SM\nmlp", 51.7, 87.5, DBLUE),
]
NEG = [("物理", 0.433, ORANGE), ("SM-mlp", 0.069, DBLUE), ("SM-cnn", 0.078, DBLUE)]

fig = plt.figure(figsize=HALF)
ax = fig.add_axes([0.13, 0.47, 0.85, 0.46])
x = np.arange(len(GROUPS))
w = 0.38
for i, (lab, hit, pw, c) in enumerate(GROUPS):
    ax.bar(i - w / 2, hit, width=w, color=c, zorder=3)
    ax.bar(i + w / 2, pw, width=w, color=c, alpha=0.35, zorder=3, hatch="//", edgecolor=c, lw=0)
    ax.text(i - w / 2, hit + 1.5, f"{hit:.0f}" if hit == 10 else f"{hit:.1f}", ha="center",
            va="bottom", fontsize=11.5 + 0.5, color=INK, rotation=90)
    ax.text(i + w / 2, pw + 1.5, f"{pw:.1f}", ha="center", va="bottom", fontsize=12, color=INK2,
            rotation=90)
ax.set_xticks(x, [g[0] for g in GROUPS])
ax.set_ylim(0, 118)
ax.set_xlim(-0.6, len(GROUPS) - 0.4)
sax(ax, "", "%")
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=INK2, label="命中率：挑 60 筆中真前 10% 的比例"),
                   Patch(facecolor=INK2, alpha=0.35, hatch="//", label="P(挑出批的最佳勝過隨機批)")],
          loc="upper left", fontsize=11.5 + 0.5, frameon=False, borderaxespad=0.1)

# 負片小面板
bx = fig.add_axes([0.30, 0.04, 0.55, 0.20])
for i, (lab, v, c) in enumerate(NEG):
    bx.barh(i, v, color=c, height=0.62, zorder=3)
    bx.text(v + 0.01, i, f2(v, 3), va="center", ha="left", fontsize=12, color=INK)
bx.set_yticks(range(len(NEG)), [n[0] for n in NEG])
bx.invert_yaxis()
bx.set_xlim(0, 0.55)
sax(bx, "排序相關 ρ", "")
bx.tick_params(axis="y", labelsize=12.5)
bx.text(0.98, 0.08, "物理為 8/3 早期版\nSM 吃過負片後可達 +0.92", transform=bx.transAxes, ha="right",
        va="bottom", fontsize=12, color=MUTED)
fig.text(0.02, 0.305, "資料沒覆蓋的形態（負片 n=1,200）：物理贏", fontsize=13, color=INK, weight="bold")
fig.text(0.98, 0.955, "正片 n=817，三方都沒見過；S0 真值", ha="right", fontsize=12, color=MUTED)
save(fig, "diffsim_final.png")
