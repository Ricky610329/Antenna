# -*- coding: utf-8 -*-
"""report_r1r10_online.py — 成果報告圖 F1-F3（docs/report/assets/）。
F1 可製造紀錄推進時間軸（批次線 R7→R10）
F2 批次假設迴圈系統示意
F3 R1-R5 線上學習線多臂 best-wm vs HFSS calls 疊圖（讀 NAS metrics.csv,快取 tmp/report_r1r10/）
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    CACHE, INK, INK2, MUTED, GRID, SURF, RED, DBLUE, ORANGE, GREEN, GOLD,
    plt, style_ax, save)

from antenna.utils import config as _config  # noqa: E402
_config.device = "cpu"

# ==== F1 可製造紀錄推進（批次驗證線）====
# (round, 累計批次 HFSS, 紀錄 wm, 紀錄保持者, 附註)——數字對 docs/log/README.md 時間軸
MILESTONES = [
    ("R7",  15,  -2.68, "p03_d3", "整塊型除塵例外\n（可製造紀錄起點）"),
    ("R8",  112, -2.68, "—",      "測繪 round:規則產出\n紀錄未動"),
    ("R9",  274, -0.29, "s05",    "F2×10-5-10 對稱化\n（+2.39）"),
    ("R10", 624,  0.20, "c21_sm", "八冠軍 certified\n（首批三標全過）"),
]


def fig1():
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    xs = [0] + [m[1] for m in MILESTONES]
    ys = [-2.68] + [m[2] for m in MILESTONES]     # R7 前無可製造紀錄,以 R7 值起步
    ax.step(xs, ys, where="post", color=DBLUE, lw=2.6, zorder=3)
    ax.axhline(0, color=RED, ls=":", lw=1.6)
    ax.text(8, 0.08, "spec 達標線（margin = 0）", color=RED, fontsize=9.5, va="bottom")
    off = {"R7": (10, -0.72), "R8": (10, 0.45), "R9": (12, -0.75), "R10": (-245, 0.28)}
    for name, x, y, holder, note in MILESTONES:
        ax.scatter([x], [y], s=64, color=ORANGE if y >= 0 else DBLUE, zorder=4,
                   edgecolor=SURF, lw=1.2)
        dx, dy = off[name]
        ax.annotate(f"{name}｜{holder}  {y:+.2f}\n{note}", (x, y), (x + dx, y + dy),
                    fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.plot([0, 640], [-2.89, -2.89], color=MUTED, ls="--", lw=1.4)
    ax.text(620, -2.83, "線上學習線最佳（R4 E+D −2.89,含粉塵不可製造）", color=MUTED,
            fontsize=9, ha="right", va="bottom")
    ax.set_xlim(-15, 660)
    ax.set_ylim(-3.6, 1.2)
    style_ax(ax, "累計批次 HFSS 模擬筆數（R7 起）", "可製造紀錄 worst-margin (dB,越高越好)",
             "可製造紀錄推進：一週 −2.68 → +0.20（批次假設迴圈,R7→R10）", tfs=12.5)
    save(fig, "f01_record_timeline.png")


# ==== F2 批次假設迴圈示意 ====
def fig2():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, fc, tc=INK, fs=10.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1.1",
                                    facecolor=fc, edgecolor=GRID, lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3, linespacing=1.55)

    def arrow(x1, y1, x2, y2, text="", color=INK2, rad=0.0, fs=8.6, toff=(0, 1.6)):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                     color=color, lw=1.6, zorder=1,
                                     connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((x1 + x2) / 2 + toff[0], (y1 + y2) / 2 + toff[1], text,
                    ha="center", fontsize=fs, color=color)

    y0 = 26
    box(0, y0, 15, 12, "假設＋判準\n（發車前寫死）", "#eef2f8")
    box(22.5, y0, 16, 12, "決定性算子生成\n（seed 全可重現）", "#eef2f8")
    box(45.5, y0, 13, 12, "SM 篩選\n（導航儀）", "#e9f4ef")
    box(65.5, y0, 13, 12, "HFSS 批次\n（NAS 續跑）", "#fdf1e7")
    box(85.5, y0, 16, 12, "公證\n（紀錄級多維驗證）", "#fbecec")
    arrow(16.2, y0 + 6, 21.3, y0 + 6)
    arrow(39.7, y0 + 6, 44.3, y0 + 6, "候選池", toff=(0, 2.0))
    arrow(59.7, y0 + 6, 64.3, y0 + 6, "top-N", toff=(0, 2.0))
    arrow(79.7, y0 + 6, 84.3, y0 + 6, "真值", toff=(0, 2.0))
    box(23, 3, 18, 10, "規則目錄\n（因果過關才升級）", "#f4f0fa")
    box(47, 3, 14, 10, "SM 重錨\n（真值回灌）", "#f4f0fa")
    box(67, 3, 16, 10, "冠軍＝新錨點\n（血統可追溯）", "#f4f0fa")
    arrow(96, y0 - 1.0, 77, 14.4)
    arrow(92, y0 - 1.0, 55, 14.2)
    arrow(88, y0 - 1.0, 34, 14.0)
    arrow(22.4, 8.5, 7, y0 - 1.0, "先驗 →\n下一批假設", rad=0.12, toff=(-8.5, -3.2))
    ax.text(53, 43.5, "批次假設迴圈：一批＝一個判準預註冊的對照實驗（搜尋同時生產知識）",
            ha="center", fontsize=12.5, color=INK)
    save(fig, "f02_loop_schematic.png")


# ==== F3 R1-R5 線上線 ====
ROUNDS = {   # round → (色, [(run 夾名, 臂名)])
    "R1": ("#8f8577", [("pixel_single_guided_harvest", "dlf"),
                       ("pixel_single_guided_dlffit_harvest", "dlf_fit"),
                       ("pixel_single_guided_refit_harvest", "refit")]),
    "R2": (GREEN, [("pixel_single_r2_ens_harvest", "ens"),
                   ("pixel_single_r2_enstrust_harvest", "ens+trust"),
                   ("pixel_single_r2_refit_enstrust_harvest", "refit+e+t")]),
    "R3": (GOLD, [("pixel_single_r3_explore", "E"),
                  ("pixel_single_r3_dip", "D"),
                  ("pixel_single_r3_dip_explore", "E+D")]),
    "R4": (DBLUE, [("pixel_single_r4_explore", "E"),
                   ("pixel_single_r4_dip", "D"),
                   ("pixel_single_r4_dip_explore", "E+D")]),
    "R5": (ORANGE, [("pixel_single_r5_explore", "E"),
                    ("pixel_single_r5_dip", "D"),
                    ("pixel_single_r5_dip_explore", "E+D")]),
}


def _curves():
    """15 run 的 best-so-far 曲線,NAS 掃一次後快取。"""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "online_curves.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d}
    from script.benchmark_vs_random import _resolve_run, run_curve
    out = {}
    for rname, (_c, runs) in ROUNDS.items():
        for folder, arm in runs:
            try:
                x, y = run_curve(_resolve_run(folder))
                out[f"{rname}|{arm}|x"] = np.asarray(x, float)
                out[f"{rname}|{arm}|y"] = np.asarray(y, float)
                print(f"  {rname} {arm}: {len(x)} 點 best={max(y):+.2f}")
            except SystemExit as e:
                print(f"  ⚠ {rname} {arm} 跳過:{e}")
    np.savez(cache, **out)
    return out


def fig3():
    d = _curves()
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    finals = {}
    for rname, (color, runs) in ROUNDS.items():
        best_arm, best_v = None, -1e9
        for _f, arm in runs:
            if f"{rname}|{arm}|y" not in d:
                continue
            v = float(np.max(d[f"{rname}|{arm}|y"]))
            if v > best_v:
                best_arm, best_v = arm, v
        for _f, arm in runs:
            kx, ky = f"{rname}|{arm}|x", f"{rname}|{arm}|y"
            if kx not in d:
                continue
            if arm == best_arm:
                ax.plot(d[kx], d[ky], color=color, lw=2.4,
                        label=f"{rname} 最佳臂 {arm}（{best_v:+.2f}）")
            else:
                ax.plot(d[kx], d[ky], color=color, lw=1.0, alpha=0.30)
        finals[rname] = (best_arm, best_v)
    ax.axhline(0, color=RED, ls=":", lw=1.6)
    ax.text(6, 0.15, "spec 達標線", color=RED, fontsize=9.5)
    if "R4" in finals:
        ax.annotate(f"R4 E+D 破紀錄 {finals['R4'][1]:+.2f}\n（探索躍遷,線上線最佳）",
                    (154, -2.89), (210, -1.9), fontsize=9.2, color=DBLUE,
                    arrowprops=dict(arrowstyle="-|>", color=DBLUE, lw=1.1))
    ax.set_ylim(-13, 1.2)
    style_ax(ax, "HFSS 模擬次數（單臂累計）", "best worst-margin so far (dB)",
             "線上學習線 R1→R5：五輪診斷史（細線=同輪其他臂;粗線=該輪最佳臂）", tfs=12.5)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.94).get_frame().set_edgecolor(GRID)
    save(fig, "f03_online_line.png")
    print("\n各輪最佳臂:", {k: f"{a} {v:+.2f}" for k, (a, v) in finals.items()})


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
