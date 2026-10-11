# -*- coding: utf-8 -*-
"""meeting_f04.py — 組會第 7 頁：R6 離線期望基準 f04 的投影改寫版（圖例「前代線上方法」）。

計算與 report_r1r10_batch.fig4 相同，只讀本機快取 tmp/expected_best/
（harvest_margins.npy、ours_curves.npz、senior_curves.npz；2026-07-03 產），不碰 NAS。
改動：圖例措辭、字級放大、去掉大標題。
"""
import os

import numpy as np

from script.figs.meeting_common import (
    REPO, GRID, RED, DBLUE, ORANGE, GREEN, MUTED, plt, style_ax, save, NOTE)

EB = os.path.join(REPO, "tmp", "expected_best")


def _ebest_pool(margins, ns):
    """池經驗分布 best-of-N 期望（iid 有放回閉式解）。"""
    v = np.sort(margins)
    n = len(v)
    i = np.arange(1, n + 1)
    return np.asarray([float((v * ((i / n) ** N - ((i - 1) / n) ** N)).sum()) for N in ns])


def _mean_curve(npz, ks):
    """多 run best-so-far 曲線 → E[best@k]（k 超出長度取末值;僅平均涵蓋該 k 的 run）。"""
    d = np.load(npz, allow_pickle=True)
    curves = [np.asarray(d[k], float) for k in d.files]
    curves = [c[1] if c.ndim == 2 else c for c in curves]
    out = []
    for k in ks:
        vals = [c[min(k, len(c)) - 1] for c in curves if len(c) >= min(k, 30)]
        out.append(np.mean(vals))
    return np.asarray(out), len(curves)


def main():
    h = np.load(os.path.join(EB, "harvest_margins.npy"))
    ks = np.unique(np.geomspace(1, 1000, 40).astype(int))
    ours, n_ours = _mean_curve(os.path.join(EB, "ours_curves.npz"), ks)
    sen, n_sen = _mean_curve(os.path.join(EB, "senior_curves.npz"), ks)
    pool = _ebest_pool(h, ks)
    print(f"runs: ours={n_ours} senior={n_sen} pool n={len(h)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]
    ax.plot(ks, ours, color=DBLUE, lw=2.6, label=f"我們的線上方法（{n_ours} runs 平均）")
    ax.plot(ks, sen, color=GREEN, lw=2.6, label=f"前代線上方法（{n_sen} runs 平均）")
    ax.plot(ks, pool, color=MUTED, lw=2.2, ls="--", label="從前代資料池抽樣")
    ax.plot(ks, -9.18 + 0.75 * np.log(ks), color=DBLUE, lw=1.1, ls=":", label="擬合 −9.18 + 0.75·ln k")
    ax.axhline(0, color=RED, ls=":", lw=1.5)
    ax.text(1.2, -0.25, "規格線 0", color=RED, fontsize=NOTE, va="top")
    ax.set_xscale("log")
    style_ax(ax, "HFSS 模擬預算 k（log）", "期望最佳 worst-margin (dB)", "(a) 期望最佳值隨預算爬升")
    ax.legend(fontsize=NOTE - 0.5, loc="lower right", framealpha=0.94).get_frame().set_edgecolor(GRID)

    ax = axes[1]
    v = np.sort(h)
    n = len(v)
    for T, c in ((0.0, RED), (-1.0, ORANGE), (-3.0, DBLUE)):
        FT = np.searchsorted(v, T) / n
        ax.plot(ks, 1 - FT ** ks, color=c, lw=2.4, label=f"達 {T:+.0f} dB")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    style_ax(ax, "HFSS 模擬預算 k（log）", "達到門檻的機率", "(b) 從前代資料池抽樣的達標機率")
    ax.legend(fontsize=NOTE, loc="upper left", framealpha=0.94).get_frame().set_edgecolor(GRID)
    fig.tight_layout(w_pad=2.5)
    save(fig, "f04_expected_best_v2.png")


if __name__ == "__main__":
    main()
