# -*- coding: utf-8 -*-
"""report_senior_trajs.py — 報告 §4（對外版）：學長 41 條線上學習軌跡疊圖。
階梯狀進步＋run 間大變異＋逐輪中位慢爬（個體靠躍遷、期望對數爬）一張講完。
資料:tmp/expected_best/senior_curves.npz（R6 快取,丟了用 script.expected_best collect-senior 重建）。
用法: python -m script.figs.report_senior_trajs --out <path>"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    REPO, INK2, RED, SURF, DBLUE, ORANGE, plt, style_ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = np.load(os.path.join(REPO, "tmp", "expected_best", "senior_curves.npz"),
                allow_pickle=True)
    curves = [np.asarray(d[k], float)[1] for k in d.files]      # 已是 best-so-far（單調）
    best_i = int(np.argmax([c.max() for c in curves]))

    fig, ax = plt.subplots(figsize=(12.6, 6.2))
    for i, c in enumerate(curves):
        if i == best_i:
            continue
        ax.plot(np.arange(1, len(c) + 1), c, color="#9aa8bd", lw=0.9, alpha=0.55)
    cb = curves[best_i]
    ax.plot(np.arange(1, len(cb) + 1), cb, color=ORANGE, lw=2.4,
            label=f"41 條裡最好的一條（+{cb.max():.2f}，達標）")

    #? 逐輪中位（僅計仍存活的 run;剩 <15 條後截斷,避免存活集合變動的組成假象——
    #  個別曲線單調,但存活集合的中位可能回落,R6 表 1 的 n=2@k=1000 假象同因）
    kmax = max(len(c) for c in curves)
    med_x, med_y = [], []
    for k in range(1, kmax + 1):
        alive = [c[k - 1] for c in curves if len(c) >= k]
        if len(alive) < 15:
            break
        med_x.append(k)
        med_y.append(float(np.median(alive)))
    ax.plot(med_x, med_y, color=DBLUE, lw=2.6, label="41 條的逐輪中位數")

    ax.axhline(0, color=RED, ls=":", lw=1.6)
    ax.text(20, 0.25, "spec 達標線", color=RED, fontsize=10)
    ax.annotate("跑 41 次只有 2 次達標", xy=(278, 0.38), xytext=(500, 1.6),
                color=ORANGE, fontsize=10.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    ax.text(920, -5.6, "個別軌跡＝階梯狀，長時間水平、偶爾跳一階\n中位數（藍）＝越爬越慢，到後段幾乎平掉",
            color=INK2, fontsize=10)

    style_ax(ax, "HFSS 模擬次數（輪）", "目前為止最好成績（距全達標 dB）",
             "同一套線上學習方法的 41 條歷史軌跡：進步靠罕見躍遷、彼此差異巨大", tfs=13)
    ax.set_xlim(0, 1250)
    ax.set_ylim(-13, 2.6)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.94)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
