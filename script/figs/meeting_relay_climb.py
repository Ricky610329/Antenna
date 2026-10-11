# -*- coding: utf-8 -*-
"""meeting_relay_climb.py — 組會第 19 頁：深水右爬接力（R45–R47）＋嫁接當錨爬山（R48）。

資料：docs/chains/{c45g1..g4, c45d1, c45d2, c47d2..d4, c48nq1}.jsonl（每包 best 與是否勝錨）。
y＝每包結束後的錨 wm：勝錨就換成該包 best 的 wm，否則沿用。
c45／c47 鏈的 score＝wm − 0.15（例 c45g1 p1 best_score −7.94 ↔ wm −7.79），故起點 wm＝anchor_score＋0.15；
c48nq1 鏈的 score＝wm。數字對 round-45 §4、round-46 §4（g 系 −8.04→−2.72,+5.32）、
round-47 §4（c47d2 起點 −8.10→−6.21,+1.89,19 包）、round-48 §4（−0.15→+0.79）。
"""
import json
import os

from script.figs.meeting_common import (
    REPO, INK, MUTED, RED, DBLUE, ORANGE, GREEN, GOLD, PURPLE,
    plt, style_ax, save, NOTE, FULL)

CH = os.path.join(REPO, "docs", "chains")


def chain(names, offset):
    """串接接力段 → (包序, 錨 wm, 是否勝錨)。"""
    rows = []
    for n in names:
        rows += [json.loads(l) for l in open(os.path.join(CH, n + ".jsonl"), encoding="utf-8")]
    xs, ys, wins = [0], [rows[0]["anchor_score"] + offset], [False]
    cur = ys[0]
    for i, r in enumerate(rows, 1):
        if r["win"]:
            cur = r["wm"]
        xs.append(i)
        ys.append(cur)
        wins.append(bool(r["win"]))
    return xs, ys, wins


SERIES = [
    # (段名, 偏移, 顏色, 標籤)
    (["c45g1", "c45g2", "c45g3", "c45g4"], 0.15, DBLUE, "g 系"),
    (["c45d1", "c45d2"], 0.15, GOLD, "d 系"),
    (["c47d2", "c47d3", "c47d4"], 0.15, PURPLE, "c47 系"),
    (["c48nq1"], 0.0, ORANGE, "嫁接體當錨"),
]


def main():
    fig, ax = plt.subplots(figsize=FULL)
    data = {}
    for names, off, c, lab in SERIES:
        xs, ys, wins = chain(names, off)
        data[lab] = (xs, ys)
        ax.step(xs, ys, where="post", color=c, lw=2.6, zorder=3)
        ax.scatter([x for x, w in zip(xs, wins) if w], [y for y, w in zip(ys, wins) if w],
                   s=44, color=c, edgecolor="white", lw=1.0, zorder=4)
        ax.scatter([xs[0]], [ys[0]], s=70, marker="s", color=c, edgecolor="white", lw=1.0, zorder=4)
        print(lab, f"{ys[0]:+.2f} → {ys[-1]:+.2f}（{ys[-1] - ys[0]:+.2f}，{len(xs) - 1} 包）")

    g, d, c47, gr = (data[k] for k in ("g 系", "d 系", "c47 系", "嫁接體當錨"))
    assert round(g[1][0], 2) == -8.04 and g[1][-1] == -2.72
    assert round(d[1][0], 2) == -9.07 and d[1][-1] == -7.0
    assert round(c47[1][0], 2) == -8.10 and c47[1][-1] == -6.21
    assert gr[1][0] == -0.15 and gr[1][-1] == 0.79

    ax.axhline(0, color=RED, ls=":", lw=1.5, zorder=1)
    ax.axhline(-1, color=GREEN, ls="--", lw=1.3, zorder=1)
    ax.text(23.4, 0.08, "規格線 0", color=RED, fontsize=NOTE, ha="right", va="bottom")
    ax.text(23.4, -1.08, "−1：接近合格", color=GREEN, fontsize=NOTE, ha="right", va="top")

    # 終點標註
    ax.annotate("深水 g 系　−8.04 → −2.72\n23 包，累計 +5.32 dB", (23, -2.72), (16.6, -4.6),
                fontsize=NOTE, color=DBLUE, ha="left",
                arrowprops=dict(arrowstyle="-", color=DBLUE, lw=0.9))
    ax.annotate("單包 +2.91", (12, -3.79), (12.4, -5.3), fontsize=NOTE, color=DBLUE,
                arrowprops=dict(arrowstyle="-", color=DBLUE, lw=0.9))
    ax.annotate("d 系　−9.07 → −7.00\n5 包 +2.07 後停在 −7 帶", (5, -7.0), (2.6, -8.95),
                fontsize=NOTE, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.9))
    ax.annotate("c47 系　−8.10 → −6.21\n穿過 −7 帶，19 包 +1.89", (19, -6.21), (18.3, -7.75),
                fontsize=NOTE, color=PURPLE, ha="left",
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.9))
    ax.annotate("嫁接體當錨　−0.15 → +0.79\n5 包 +0.94，現任紀錄（三次重測一致）", (5, 0.79), (7.6, 0.55),
                fontsize=NOTE, color=ORANGE, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))
    ax.text(0.2, -1.55, "■＝起點錨　●＝該包勝錨、換錨　平段＝該包沒贏",
            fontsize=NOTE - 0.5, color=MUTED, va="bottom")

    ax.set_xlim(-0.6, 23.8)
    ax.set_ylim(-9.9, 1.6)
    ax.set_xticks(range(0, 24, 2))
    style_ax(ax, "累計包數（每包約 25 筆 HFSS）", "錨的帶內最差餘裕 (dB)",
             "S0 網格、無橋；每包錨值為單次量測，+0.79 經三次重測")
    save(fig, "relay_climb.png")


if __name__ == "__main__":
    main()
