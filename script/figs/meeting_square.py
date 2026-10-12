# -*- coding: utf-8 -*-
"""meeting_square.py — 組會 2026-10 v3：pattern／響應曲線一律正方形子圖。

資料與畫法沿用 docs/paper/extended_abstract_figures.py 的凍結 JSON 與 helper，
只改成投影片尺寸、方格子圖與較大字級。零 HFSS。
執行：python -m script.figs.meeting_square
輸出：docs/進度報告/組會_2026-10/figs/sq_{spec,antenna,filter,leftband,spec_ant,spec_flt,leftband_pat}.png
"""
import importlib.util
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location(
    "ea_figs", os.path.join(REPO, "docs", "paper", "extended_abstract_figures.py"))
ea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ea)
plt = ea.plt

OUT_DIR = os.path.join(REPO, "docs", "進度報告", "組會_2026-10", "figs")
BAD = "#f6e0d8"           # 規格不允許的區域
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.titlepad": 6,
                     "legend.fontsize": 10, "axes.linewidth": .8,
                     "xtick.major.size": 3, "ytick.major.size": 3, "axes.labelpad": 3})


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, dpi=220, facecolor="white")
    plt.close(fig)
    print("→", out)


def square(ax):
    ax.set_box_aspect(1)


def freq(ax, title):
    ea.freq_axis(ax, title, band=None)       # 只畫淡紅禁止區，不疊灰底
    ax.set_xlabel("頻率 (GHz)")
    square(ax)


def pattern(ax, bits, title, ports=("feed",)):
    ea.pattern_axis(ax, bits, ea.BLUE, title, ports=ports)
    for c in ax.collections:          # 饋入／埠標記放大
        c.set_sizes([40])
    if "P2" in ports:                 # 上緣有埠標記，標題往上讓
        ax.set_title(title, pad=18)
    square(ax)


def antenna_data():
    free = ea.load(ea.FREE)
    bits = np.asarray(free["candidate"]["bits"], dtype=int)
    f = np.asarray(free["frequency_ghz"])
    s11 = np.asarray(free["response_db"]["S11"])
    gain = np.asarray(free["response_db"]["RealizedGainTotal"])
    return bits, f, s11, gain, free["radiation"]


def filter_data():
    resp = ea.load(ea.FILTER)
    bits = np.asarray(ea.load(ea.HISTORY)["candidate"]["bits"], dtype=int)
    f = np.asarray(resp["frequency_ghz"])
    s11, s21, s22 = (np.asarray(resp["response_db"][k]) for k in ("S11", "S21", "S22"))
    return bits, f, s11, s21, s22


def ant_s11(ax, f, s11):
    freq(ax, "S11 (dB)")
    ax.fill_between([26.5, 29.5], -10, 0, color=BAD, lw=0)
    ea.limit(ax, 26.5, 29.5, -10)
    ax.plot(f, s11, color=ea.BLUE, lw=1.6, marker="o", ms=3)
    ax.set_ylim(-30, 0)


def ant_gain(ax, f, gain):
    freq(ax, "增益 (dBi)")
    ax.fill_between([26.5, 29.5], -2, 4, color=BAD, lw=0)
    ea.limit(ax, 26.5, 29.5, 4)
    ax.plot(f, gain, color=ea.BLUE, lw=1.6, marker="o", ms=3)
    ax.set_ylim(-2, 8)


def flt_s11(ax, f, s11, s22):
    freq(ax, "S11、S22 (dB)")
    ax.fill_between([26.5, 29.5], -10, 0, color=BAD, lw=0)
    ea.limit(ax, 26.5, 29.5, -10)
    ax.plot(f, s11, color=ea.BLUE, lw=1.6, marker="o", ms=3, label="S11")
    ax.plot(f, s22, color=ea.TEAL, lw=1.6, marker="s", ms=2.6, label="S22")
    ax.set_ylim(-35, 0)
    ax.legend(loc="lower right")


def flt_s21(ax, f, s21):
    freq(ax, "S21 (dB)")
    ax.fill_between([25.5, 30.5], -3, 0, color=BAD, lw=0)
    for x0, x1 in ((24, 25), (31, 32)):
        ax.fill_between([x0, x1], -15, 0, color=BAD, lw=0)
        ea.limit(ax, x0, x1, -15)
    ea.limit(ax, 25.5, 30.5, -3)
    ax.plot(f, s21, color=ea.BLUE, lw=1.6, marker="o", ms=3)
    ax.set_ylim(-40, 0)


def spec():
    """規格頁：上排天線、下排濾波器；淡紅＝規格不允許的區域。"""
    bits, f, s11, gain, _ = antenna_data()
    fb, ff, fs11, fs21, fs22 = filter_data()
    fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.2))     # 投影片上 1:1，字級即實際大小
    fig.subplots_adjust(left=.07, right=.99, top=.92, bottom=.1, wspace=.42, hspace=.62)
    pattern(axs[0, 0], bits, "單埠天線")
    ant_s11(axs[0, 1], f, s11)
    ant_gain(axs[0, 2], f, gain)
    pattern(axs[1, 0], fb, "雙埠濾波器", ports=("P1", "P2"))
    flt_s11(axs[1, 1], ff, fs11, fs22)
    flt_s21(axs[1, 2], ff, fs21)
    save(fig, "sq_spec.png")


def antenna():
    bits, f, s11, gain, rad = antenna_data()
    fig = plt.figure(figsize=(11.6, 3.5))
    gs = fig.add_gridspec(1, 4, left=.03, right=.96, top=.88, bottom=.17, wspace=.45)
    pattern(fig.add_subplot(gs[0, 0]), bits, "金屬圖形，0.10 mm 橋")
    ant_s11(fig.add_subplot(gs[0, 1]), f, s11)
    ant_gain(fig.add_subplot(gs[0, 2]), f, gain)
    ax = fig.add_subplot(gs[0, 3], projection="polar")
    ea.polar_axis(ax, np.asarray(rad["theta_deg"]),
                  [("φ = 0°", rad["phi0_db"], ea.BLUE), ("φ = 90°", rad["phi90_db"], ea.AMBER)],
                  "28 GHz 方向圖")
    ax.set_xticklabels(["0°", "45°", "90°", "", "180°", "", "-90°", "-45°"], fontsize=9)
    save(fig, "sq_antenna.png")


def filter_():
    bits, f, s11, s21, s22 = filter_data()
    fig = plt.figure(figsize=(9.0, 3.5))
    gs = fig.add_gridspec(1, 3, left=.03, right=.99, top=.88, bottom=.17, wspace=.4)
    pattern(fig.add_subplot(gs[0, 0]), bits, "金屬圖形，0.075 mm 橋", ports=("P1", "P2"))
    flt_s11(fig.add_subplot(gs[0, 1]), f, s11, s22)
    flt_s21(fig.add_subplot(gs[0, 2]), f, s21)
    save(fig, "sq_filter.png")


def leftband():
    """左側帶外：代表設計（0.10 mm 橋）對照左側帶外紀錄設計（無橋，−3.46 dBi 三次重測一致）。"""
    f = np.asarray(ea.load(ea.FREE)["frequency_ghz"])
    gain = np.asarray(ea.load(ea.FREE)["response_db"]["RealizedGainTotal"])
    rec = np.asarray(ea.load(ea.FROZEN)["snapshot"]["oob_record"]["RealizedGainTotal_dbi"])
    assert abs(rec[f <= 25.5].max() - (-3.46)) < .01
    with plt.rc_context({"font.size": 13, "axes.titlesize": 14, "legend.fontsize": 12}):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        fig.subplots_adjust(left=.14, right=.97, top=.9, bottom=.14)
        freq(ax, "增益 (dBi)")
        ax.axvspan(24, 25.5, color="#efe4f0", lw=0)
        ax.fill_between([26.5, 29.5], -12, 4, color=BAD, lw=0)
        ea.limit(ax, 26.5, 29.5, 4)
        ax.plot(f, gain, color=ea.MUTED, lw=1.8, marker="o", ms=3.5, label="代表設計")
        ax.plot(f, rec, color=ea.BLUE, lw=2, marker="o", ms=3.5, label="左側帶外紀錄")
        ax.text(24.75, 6.6, "左側帶外", ha="center", fontsize=12, color=ea.INK)
        ax.set_ylim(-12, 8)
        ax.set_yticks([-10, -5, 0, 5])
        ax.legend(loc="lower right")
        save(fig, "sq_leftband.png")


EXTRA = "#efe4f0"         # 左側帶外：規格之外的額外要求


def _note(ax, x, y, s, color=None, fontsize=12):
    ax.text(x, y, s, ha="center", va="center", fontsize=fontsize, color=color or ea.INK)


def spec_ant():
    """v7 輸出頁（天線）：規格直接畫在圖上；曲線為代表設計，淡紅＝規格不允許，淡紫＝左側帶外額外要求。"""
    _, f, s11, gain, rad = antenna_data()
    with plt.rc_context({"font.size": 12, "axes.titlesize": 14}):
        fig = plt.figure(figsize=(11.6, 4.3))
        gs = fig.add_gridspec(1, 3, left=.05, right=.95, top=.9, bottom=.13, wspace=.32)
        ax = fig.add_subplot(gs[0, 0])
        ant_s11(ax, f, s11)
        _note(ax, 28, -5, "≤ −10 dB", ea.INK)
        ax = fig.add_subplot(gs[0, 1])
        ant_gain(ax, f, gain)
        ax.set_ylim(-12, 8)
        ax.set_yticks([-10, -5, 0, 5])
        ax.fill_between([26.5, 29.5], -12, 4, color=BAD, lw=0, zorder=0)
        ax.axvspan(24, 25.5, color=EXTRA, lw=0, zorder=0)
        _note(ax, 28, -2, "≥ 4 dBi")
        _note(ax, 24.75, -9, "額外\n要求")
        ax = fig.add_subplot(gs[0, 2], projection="polar")
        theta = np.asarray(rad["theta_deg"])
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        w = np.radians(np.linspace(-45, 45, 91))
        ax.fill_between(w, -30, -3, color=BAD, lw=0)
        ax.plot(w, np.full_like(w, -3), color=ea.INK, lw=.9, ls=(0, (3, 2)))
        for label, cut, color in (("φ = 0°", rad["phi0_db"], ea.BLUE), ("φ = 90°", rad["phi90_db"], ea.AMBER)):
            norm = np.asarray(cut) - np.asarray(cut)[theta == 0][0]
            ax.plot(np.radians(theta), np.clip(norm, -30, 3), color=color, lw=1.6, label=label)
        ax.set_ylim(-30, 3)
        ax.set_yticks([-20, -10, -3])
        ax.set_yticklabels([])
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["0°", "45°", "90°", "", "180°", "", "-90°", "-45°"], fontsize=11)
        ax.grid(color=ea.GRID, lw=.6)
        ax.spines["polar"].set(linewidth=.6, color=ea.MUTED)
        ax.text(0, -16, "±45° 內\n下降 ≤ 3 dB", ha="center", va="center", fontsize=11, color=ea.INK)
        ax.set_title("28 GHz 方向圖", pad=10)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.04), ncol=2, fontsize=11)
        save(fig, "sq_spec_ant.png")


def spec_flt():
    """v7 輸出頁（濾波器）：規格直接畫在圖上；曲線為代表設計。"""
    _, f, s11, s21, s22 = filter_data()
    with plt.rc_context({"font.size": 12, "axes.titlesize": 14, "legend.fontsize": 11}):
        fig, axs = plt.subplots(1, 2, figsize=(7.8, 4.3))
        fig.subplots_adjust(left=.08, right=.98, top=.9, bottom=.13, wspace=.3)
        flt_s11(axs[0], f, s11, s22)
        _note(axs[0], 28, -5, "≤ −10 dB")
        flt_s21(axs[1], f, s21)
        _note(axs[1], 28, -6.5, "通帶 ≥ −3 dB")
        _note(axs[1], 28, -30, "兩側阻帶 ≤ −15 dB")
        save(fig, "sq_spec_flt.png")


def leftband_pat():
    """v7 左側帶外頁：兩個設計的 pattern 上下疊在左，增益對照在右。"""
    bits, f, _, gain, _ = antenna_data()
    rec = ea.load(ea.FROZEN)["snapshot"]["oob_record"]
    rbits = np.asarray(rec["bits"], dtype=int)
    rgain = np.asarray(rec["RealizedGainTotal_dbi"])
    assert rbits.shape == (25, 25) and abs(rgain[f <= 25.5].max() - (-3.46)) < .01
    with plt.rc_context({"font.size": 13, "axes.titlesize": 14, "legend.fontsize": 12}):
        fig = plt.figure(figsize=(7.9, 5.0))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.25], left=.02, right=.98, top=.92, bottom=.11,
                              wspace=.18, hspace=.3)
        pattern(fig.add_subplot(gs[0, 0]), bits, "代表設計")
        a = fig.add_subplot(gs[1, 0])
        ea.pattern_axis(a, rbits, ea.BLUE, "左側帶外紀錄")
        for c in a.collections:
            c.set_sizes([40])
        square(a)
        ax = fig.add_subplot(gs[:, 1])
        freq(ax, "增益 (dBi)")
        ax.axvspan(24, 25.5, color=EXTRA, lw=0)
        ax.fill_between([26.5, 29.5], -12, 4, color=BAD, lw=0)
        ea.limit(ax, 26.5, 29.5, 4)
        ax.plot(f, gain, color=ea.MUTED, lw=1.8, marker="o", ms=3.5, label="代表設計")
        ax.plot(f, rgain, color=ea.BLUE, lw=2, marker="o", ms=3.5, label="左側帶外紀錄")
        ax.text(24.75, 6.6, "左側帶外", ha="center", fontsize=12, color=ea.INK)
        ax.set_ylim(-12, 8)
        ax.set_yticks([-10, -5, 0, 5])
        ax.legend(loc="lower right")
        save(fig, "sq_leftband_pat.png")


if __name__ == "__main__":
    spec()
    antenna()
    filter_()
    leftband()
    spec_ant()
    spec_flt()
    leftband_pat()
