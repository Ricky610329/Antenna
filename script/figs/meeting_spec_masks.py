# -*- coding: utf-8 -*-
"""meeting_spec_masks.py — 組會 2026-10 第 2／26 頁：天線與濾波器的規格遮罩示意。

左：單埠天線 S11 ≤ −10 dB、Gain ≥ 4 dBi（帶內 26.5–29.5 GHz），疊現況最佳解 c48nq1p05_16
    的 17 點曲線（application/pattern_browser/data/resp.npz，S0 網格、無橋）。
右：雙埠濾波器 v1（實線）與 v2（虛線）。頻帶＝`antenna/losses.py::worst_margin_dual` 的 mask：
    S11/S22 帶內 idx 5-11＝26.5–29.5 GHz、S21 通帶 idx 3-13＝25.5–30.5、阻帶 idx 0-2 ∪ 14-16。
    門檻：v1＝論文 p.44（configs/dual_r1_eval.yaml）；v2＝docs/discuss/decisions.md「dual 規格 v2」。
只讀本機快取，零 HFSS。執行：python -m script.figs.meeting_spec_masks
"""
import os

import numpy as np

from script.figs.report_r1r10_style import (INK, INK2, MUTED, SURF, RED, DBLUE, ORANGE, AQUA,
                                            plt, style_ax)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "docs", "進度報告", "組會_2026-10", "figs")
F = np.linspace(24, 32, 17)
KING = "c48nq1p05_16"
FORBID = "#f3d9d6"


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("→", out)


def big(ax, xl="", yl=""):
    style_ax(ax, xl, yl)
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)


def load_king():
    p = os.path.join(REPO, "application", "pattern_browser", "data", "resp.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    ids = list(z["ids"])
    if KING not in ids:
        return None
    k = ids.index(KING)
    if not bool(z["has_resp"][k]):
        return None
    return z["resp"][k].astype(float)          # (2,17)：S11, Gain


def antenna(ax_s, ax_g, king):
    lo, hi = 26.5, 29.5
    for ax, thr, low, yl, ylim in ((ax_s, -10, True, "S11 (dB)", (-30, 0)),
                                    (ax_g, 4, False, "Gain (dBi)", (-1, 8))):
        big(ax, "", yl)
        if low:
            ax.fill_between([lo, hi], thr, ylim[1], color=FORBID, lw=0, zorder=0)
        else:
            ax.fill_between([lo, hi], ylim[0], thr, color=FORBID, lw=0, zorder=0)
        ax.plot([lo, hi], [thr, thr], color=RED, lw=2.4, zorder=3)
        ax.set_ylim(*ylim)
        ax.set_xlim(24, 32)
    ax_s.text(28, -8.4, "S11 ≤ −10 dB", ha="center", va="bottom", fontsize=12, color=RED)
    ax_g.text(28, 2.6, "Gain ≥ 4 dBi", ha="center", va="top", fontsize=12, color=RED)
    ax_s.text(28, -1.2, "帶內 26.5–29.5 GHz", ha="center", va="top", fontsize=12, color=INK2)
    if king is not None:
        lab = "現況最佳解 +0.79 dB：S0 網格、無橋、三次重測一致"
        ax_s.plot(F, king[0], color=DBLUE, lw=2, marker="o", ms=4, label=lab)
        ax_g.plot(F, king[1], color=DBLUE, lw=2, marker="o", ms=4)
        band = (F >= lo - 1e-9) & (F <= hi + 1e-9)
        gb = np.where(band, king[1], np.inf)
        j = int(np.argmin(gb))
        ax_g.annotate(f"最差點 +{king[1][j] - 4:.2f} dB", xy=(F[j], king[1][j]),
                      xytext=(24.3, 6.6), fontsize=12, color=INK,
                      arrowprops=dict(arrowstyle="->", color=INK2, lw=1.1))
        ax_s.legend(loc="lower center", fontsize=11, frameon=False, borderaxespad=0.2)
    ax_s.tick_params(labelbottom=False)
    ax_g.set_xlabel("頻率 (GHz)", fontsize=13, color=INK2)
    ax_s.text(0.0, 1.03, "(a) 單埠天線", transform=ax_s.transAxes, fontsize=13, color=INK)


def seg(ax, x0, x1, y, ls, color, lw=2.4):
    ax.plot([x0, x1], [y, y], ls=ls, color=color, lw=lw, zorder=3, solid_capstyle="butt")


def filt(ax_21, ax_11):
    # S21：v2 不合格區淡紅底
    big(ax_21, "", "S21 (dB)")
    ax_21.set_xlim(24, 32)
    ax_21.set_ylim(-30, 2)
    ax_21.fill_between([25.5, 30.5], -30, -3, color=FORBID, lw=0, zorder=0)
    for a, b in ((24, 25), (31, 32)):
        ax_21.fill_between([a, b], -15, 2, color=FORBID, lw=0, zorder=0)
    seg(ax_21, 25.5, 30.5, -3, "-", DBLUE)               # 通帶 v1=v2
    for a, b in ((24, 25), (31, 32)):
        seg(ax_21, a, b, -20, "-", ORANGE)               # v1 阻帶 −20
        seg(ax_21, a, b, -15, "--", ORANGE)              # v2 阻帶 −15
    ax_21.text(28, -5.2, "通帶 25.5–30.5 GHz：≥ −3 dB，兩版相同", ha="center", va="top",
               fontsize=12, color=DBLUE)
    ax_21.text(24.5, -21.3, "阻帶 ≤ −20\n→ v2 ≤ −15", ha="center", va="top", fontsize=11.5,
               color=ORANGE)
    ax_21.text(31.5, -21.3, "阻帶 ≤ −20\n→ v2 ≤ −15", ha="center", va="top", fontsize=11.5,
               color=ORANGE)
    ax_21.annotate("", xy=(25.5, -12), xytext=(25.0, -12),
                   arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.1))
    ax_21.text(25.25, -11.2, "0.5 GHz", ha="center", va="bottom", fontsize=11, color=INK2)
    ax_21.tick_params(labelbottom=False)
    ax_21.text(0.0, 1.03, "(b) 雙埠濾波器", transform=ax_21.transAxes, fontsize=13, color=INK)

    big(ax_11, "頻率 (GHz)", "S11、S22 (dB)")
    ax_11.set_xlim(24, 32)
    ax_11.set_ylim(-25, 0)
    ax_11.fill_between([26.5, 29.5], -10, 0, color=FORBID, lw=0, zorder=0)
    seg(ax_11, 26.5, 29.5, -12, "-", AQUA)
    seg(ax_11, 26.5, 29.5, -10, "--", AQUA)
    ax_11.text(28, -12.8, "帶內 26.5–29.5 GHz：≤ −12 → v2 ≤ −10", ha="center", va="top",
               fontsize=12, color=AQUA)
    ax_11.text(24.2, -23.6, "帶外：v2 起不計分", fontsize=12, color=INK2)
    # 圖例：線型代表版本
    h1, = ax_11.plot([], [], color=MUTED, lw=2.4, ls="-", label="v1 論文規格")
    h2, = ax_11.plot([], [], color=MUTED, lw=2.4, ls="--", label="v2 現行，8/12 起")
    ax_11.legend(handles=[h1, h2], loc="lower right", fontsize=11.5, frameon=False, ncol=1)


def main():
    king = load_king()
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.2, left=0.07, right=0.99,
                          top=0.93, bottom=0.16)
    ax_s, ax_g = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_21, ax_11 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    antenna(ax_s, ax_g, king)
    filt(ax_21, ax_11)
    for ax in (ax_s, ax_g, ax_21, ax_11):
        ax.set_xticks(range(24, 33))
    fig.text(0.99, 0.0, "淡紅底＝不合格區；天線依現行規格，濾波器依 v2",
             ha="right", va="bottom", fontsize=11.5, color=INK2)
    save(fig, "spec_masks.png")


if __name__ == "__main__":
    main()
