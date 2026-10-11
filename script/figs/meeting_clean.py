# -*- coding: utf-8 -*-
"""meeting_clean.py — 組會 2026-10 v4：五張分析圖的乾淨版（投影片 1:1 尺寸）。

圖內只留資料與數值；標題、註腳、結論都移到投影片的描述行、右欄與底部藍帶。
配色與 meeting_square.py 相同。數字與原 meeting_* 腳本同源：
  clean_f04       ← meeting_f04.py（tmp/expected_best/ 本機快取，2026-07-03 產）
  clean_dust      ← meeting_dust_two_way.py（round-07 §4、status-2026-07-08 §3.2；S0、單次）
  clean_diag      ← meeting_diag_grammar.py（round-44 §4 六批總表；S0、單次）
  clean_mesh      ← meeting_mesh_convergence.py（round-52 §4）
  clean_phantom   ← meeting_phantom_evidence.py (a)（round-69 §4；舊幾何、單次）
零 HFSS、零 NAS。執行：python -m script.figs.meeting_clean
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "docs", "進度報告", "組會_2026-10", "figs")
EB = os.path.join(REPO, "tmp", "expected_best")

BLUE, AMBER, TEAL = "#234c68", "#bd6532", "#3b8a7e"
INK, MUTED, GRID = "#18212b", "#9eabb5", "#e3e8ec"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"], "font.size": 13,
                     "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": .8, "axes.edgecolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
                     "legend.frameon": False, "legend.fontsize": 12})


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    print("→", out)


def bars(ax, x, vals, colors, fmt="{:+.2f}", pad=.3, width=.6):
    ax.bar(x, vals, color=colors, width=width, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (pad if v >= 0 else -pad), fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=13, color=INK)
    ax.axhline(0, color=INK, lw=.8)
    ax.grid(axis="y", color=GRID, lw=.6, zorder=0)


def f04():
    def pool_best(margins, ks):
        v = np.sort(margins)
        n = len(v)
        i = np.arange(1, n + 1)
        return np.asarray([float((v * ((i / n) ** k - ((i - 1) / n) ** k)).sum()) for k in ks])

    def mean_curve(path, ks):
        d = np.load(path, allow_pickle=True)
        curves = [np.asarray(d[k], float) for k in d.files]
        curves = [c[1] if c.ndim == 2 else c for c in curves]
        return np.asarray([np.mean([c[min(k, len(c)) - 1] for c in curves if len(c) >= min(k, 30)])
                           for k in ks])

    ks = np.unique(np.geomspace(1, 1000, 40).astype(int))
    pool = pool_best(np.load(os.path.join(EB, "harvest_margins.npy")), ks)
    ours = mean_curve(os.path.join(EB, "ours_curves.npz"), ks)
    senior = mean_curve(os.path.join(EB, "senior_curves.npz"), ks)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    fig.subplots_adjust(left=.12, right=.97, top=.95, bottom=.15)
    ax.plot(ks, pool, color=AMBER, lw=2.6, label="從前代資料集抽樣")
    ax.plot(ks, senior, color=TEAL, lw=2.2, label="前代線上方法")
    ax.plot(ks, ours, color=BLUE, lw=2.2, label="我們的線上方法")
    ax.axhline(0, color=INK, lw=.9, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("HFSS 模擬次數")
    ax.set_ylabel("期望最佳 WM (dB)")
    ax.grid(color=GRID, lw=.6)
    ax.legend(loc="lower right")
    save(fig, "clean_f04.png")


def dust():
    removed = [-6.70, -16.93, -5.02, -0.02, -8.84]
    wings = [("設計 A", 0.20, (-5.45, -5.94)), ("設計 B", 0.13, (-5.47, -5.64))]
    fig, (al, ar) = plt.subplots(1, 2, figsize=(12.0, 4.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    fig.subplots_adjust(left=.07, right=.99, top=.9, bottom=.13, wspace=.28)
    bars(al, range(5), removed, [MUTED if v > -1 else AMBER for v in removed], pad=.5)
    al.set_xticks(range(5), [f"設計 {i}" for i in range(1, 6)])
    al.set_ylim(-20, 2)
    al.set_yticks([-20, -15, -10, -5, 0])
    al.set_ylabel("WM 變化 (dB)")
    al.set_title("移除 1–3 像素碎片", fontsize=14, color=INK)
    x = np.arange(len(wings))
    ar.bar(x - .22, [w[1] for w in wings], width=.3, color=BLUE, zorder=3, label="原設計")
    ar.bar(x + .12, [w[2][0] for w in wings], width=.18, color=AMBER, zorder=3, label="懸浮翼接到主體")
    ar.bar(x + .32, [w[2][1] for w in wings], width=.18, color=AMBER, zorder=3)
    for xi, (_, o, (a, b)) in zip(x, wings):
        ar.text(xi - .22, o + .2, f"{o:+.2f}", ha="center", va="bottom", fontsize=13)
        ar.text(xi + .22, min(a, b) - .3, f"{a:.2f}\n{b:.2f}", ha="center", va="top", fontsize=12,
                linespacing=1.1)
    ar.axhline(0, color=INK, lw=.8)
    ar.grid(axis="y", color=GRID, lw=.6, zorder=0)
    ar.set_xticks(x, [w[0] for w in wings])
    ar.set_xlim(-.6, 1.6)
    ar.set_ylim(-9, 1.5)
    ar.set_ylabel("WM (dB)")
    ar.set_title("懸浮翼用 1 像素接到主體", fontsize=14, color=INK)
    ar.legend(loc="lower right")
    save(fig, "clean_dust.png")


def diag():
    rows = [("有對角接觸", -3.34, BLUE), ("無對角接觸", 2.27, AMBER)]
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    fig.subplots_adjust(left=.17, right=.97, top=.95, bottom=.12)
    bars(ax, range(2), [r[1] for r in rows], [r[2] for r in rows], pad=.25, width=.5)
    ax.set_xticks(range(2), [r[0] for r in rows])
    ax.set_xlim(-.6, 1.6)
    ax.set_ylim(-5, 4)
    ax.set_ylabel("左側帶外增益中位 (dBi)")
    save(fig, "clean_diag.png")


def mesh():
    rows = [("S0 → S1", 4.066, AMBER), ("S1 → S2", 0.003, TEAL)]
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    fig.subplots_adjust(left=.17, right=.97, top=.95, bottom=.12)
    for i, (lab, v, c) in enumerate(rows):
        ax.bar(i, v, color=c, width=.5, zorder=3)
        ax.text(i, v * 1.4, f"{v:g} GHz", ha="center", va="bottom", fontsize=13)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 60)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10], ["0.001", "0.01", "0.1", "1", "10"])
    ax.minorticks_off()
    ax.set_xticks(range(2), [r[0] for r in rows])
    ax.set_xlim(-.6, 1.6)
    ax.set_ylabel("共振頻移中位 (GHz)")
    ax.grid(axis="y", color=GRID, lw=.6, zorder=0)
    save(fig, "clean_mesh.png")


def phantom():
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    fig.subplots_adjust(left=.17, right=.97, top=.95, bottom=.12)
    bars(ax, range(2), [-3.05, -23.83], [MUTED, AMBER], fmt="{:.2f}", pad=.6, width=.5)
    ax.set_xticks(range(2), ["原設計", "點接觸補成實際連接"])
    ax.set_xlim(-.6, 1.6)
    ax.set_ylim(-28, 2)
    ax.set_ylabel("WM (dB)")
    save(fig, "clean_phantom.png")


if __name__ == "__main__":
    f04()
    dust()
    diag()
    mesh()
    phantom()
