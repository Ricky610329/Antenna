# -*- coding: utf-8 -*-
"""report_r1r10_style.py — R1-R10 成果報告圖的共用風格（docs/report/ 專用,三支 report_r1r10_*.py 共 import）。
沿用 repo 慣例:Microsoft JhengHei、淺色底、色票對齊 champ_figs/expected_best。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(REPO, "docs", "report", "assets")
CACHE = os.path.join(REPO, "tmp", "report_r1r10")

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURF, RED = "#e1e0d9", "#fcfcfb", "#d03b3b"
DBLUE, AQUA, ORANGE, GREEN, GOLD, PURPLE = "#1c5cab", "#0e7a55", "#eb6834", "#1baf7a", "#b8860b", "#7b5ac2"

plt.rcParams.update({
    "font.family": ["Microsoft JhengHei", "sans-serif"],
    "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10.5,
})


def style_ax(ax, xl="", yl="", title="", tfs=11.5):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.7, alpha=0.85)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)
    if xl:
        ax.set_xlabel(xl, color=INK2, fontsize=10)
    if yl:
        ax.set_ylabel(yl, color=INK2, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=tfs)


def save(fig, name, dpi=140):
    os.makedirs(ASSETS, exist_ok=True)
    out = os.path.join(ASSETS, name)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("→", out)


def show_pattern(ax, p, title="", feed=(24, 12), color=DBLUE, tfs=10):
    """25×25 pattern 底圖（金屬=color）＋feed 三角。"""
    from matplotlib.colors import ListedColormap
    import numpy as np
    p = np.asarray(p).reshape(25, 25) > 0.5
    ax.imshow(p.astype(int), cmap=ListedColormap([SURF, color]), vmin=0, vmax=1,
              origin="upper", interpolation="nearest")
    ax.scatter([feed[1]], [feed[0]], marker="^", s=46, color=AQUA, zorder=5,
               edgecolor=SURF, lw=0.8)
    if title:
        ax.set_title(title, color=INK, fontsize=tfs)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def diff_pattern(ax, p_new, p_old, title="", feed=(24, 12), tfs=9.6, show_counts=True):
    """25×25 pattern 差異底圖：白＝雙方空、藍＝雙方金屬、綠＝加銅（新有舊無）、紅＝去銅（新無舊有）。
    p_new/p_old 任意輸入。回傳 (n_add, n_remove)。"""
    from matplotlib.colors import ListedColormap
    import numpy as np
    pn = np.asarray(p_new).reshape(25, 25) > 0.5
    po = np.asarray(p_old).reshape(25, 25) > 0.5
    img = pn.astype(int)                       # 0=空 1=金屬
    added = pn & ~po
    removed = (~pn) & po
    img[added] = 2                             # 綠＝加銅
    img[removed] = 3                           # 紅＝去銅
    ax.imshow(img, cmap=ListedColormap([SURF, DBLUE, GREEN, RED]), vmin=0, vmax=3,
              origin="upper", interpolation="nearest")
    ax.scatter([feed[1]], [feed[0]], marker="^", s=46, color=AQUA, zorder=5,
               edgecolor=SURF, lw=0.8)
    na, nr = int(added.sum()), int(removed.sum())
    if title:
        sub = f"\n綠＝加銅 {na}px・紅＝去銅 {nr}px" if show_counts else ""
        ax.set_title(title + sub, color=INK, fontsize=tfs)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    return na, nr


def polar_rad_ax(ax, theta, series, window=45, floor_db=3, rmax=None, rmin=None,
                 show_window=True, g0_ref=None):
    """在 polar ax 上畫方向圖切面（主波束朝上、半徑=gain dB、每環 5dB）。
    theta：角度(度)。series：list of (gain陣列, 顏色, 標籤, 線寬)。
    畫 ±window° 覆蓋窗（金）＋兩邊界（橘）＋G0−floor_db 圈（紅虛）。回傳 (rmin, rmax, g0)。"""
    import numpy as np
    th = np.asarray(theta)
    o = th.argsort()
    th = th[o]
    ser = [(np.asarray(g).reshape(-1)[o], c, lab, lw) for (g, c, lab, lw) in series]
    bi = int(np.abs(th).argmin())
    g0 = g0_ref if g0_ref is not None else max(float(g[bi]) for g, *_ in ser)
    gmax = max(float(g.max()) for g, *_ in ser)
    gmin = min(float(g.min()) for g, *_ in ser)
    if rmax is None:
        rmax = int(np.ceil((gmax + 0.5) / 5.0) * 5)
    if rmin is None:
        rmin = int(max(np.floor(gmin / 5.0) * 5, rmax - 30))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, rmax - rmin)
    ax.set_rlabel_position(0)
    if show_window:
        ax.fill_between(np.deg2rad(np.linspace(-window, window, 60)), 0, rmax - rmin,
                        color="gold", alpha=0.14)
        for a in (-window, window):
            ax.plot([np.deg2rad(a)] * 2, [0, rmax - rmin], color=ORANGE, ls="--", lw=1.3)
    ax.plot(np.deg2rad(np.linspace(-180, 180, 361)), np.full(361, (g0 - floor_db) - rmin),
            color=RED, ls="--", lw=1.1)
    for g, c, lab, lw in ser:
        ax.plot(np.deg2rad(th), np.clip(g, rmin, None) - rmin, color=c, lw=lw, label=lab)
    rt = list(range(rmin, rmax + 1, 5))
    ax.set_rticks([t - rmin for t in rt])
    ax.set_yticklabels([str(t) for t in rt], fontsize=7, color=INK2)
    ax.set_thetagrids(range(0, 360, 30),
                      ["0", "30", "60", "90", "120", "150", "180",
                       "-150", "-120", "-90", "-60", "-30"], fontsize=7.5, color=INK2)
    return rmin, rmax, g0
