# -*- coding: utf-8 -*-
"""meeting_budget_scatter.py — 組會 2026-10 第 30 頁：通帶 × 阻帶散點＋耦合律。

資料：本機快取 tmp/handoff_index.json（8/31 交付索引），只取 port=dual ∧ geom=p01 ∧ bridge_w=0.075。
  m[0..5]＝m1..m6（量測口徑凍結在 configs/dual_r1_eval.yaml）；規格 v2 尺：
  m1'=m1+2、m2'=m2+2、m4'=m4+5，wm_mfg=min(m1',m2',m3,m4')（decisions「dual 規格 v2」；
  index 的 score 欄＝此式，腳本會自檢）。
  和＝m3+m4'、失衡＝|m3−m4'|，(和−失衡)/2＝min(m3,m4')（analysis-15 ★修正的恆等式）。
左：x＝通帶餘裕 m3、y＝阻帶餘裕 m4′，hexbin（對數色階）＋等 wm 的 L 形線＋等和斜線＋經驗前緣。
右：按和分箱（0.5 dB）的最佳 (和−失衡)/2，對照完美均衡上限 和/2。
只讀本機快取，零 HFSS。執行：python -m script.figs.meeting_budget_scatter
"""
import json
import os

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from script.figs.report_r1r10_style import (INK, INK2, MUTED, SURF, RED, DBLUE, ORANGE, AQUA,
                                            PURPLE, GREEN, plt, style_ax)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "docs", "進度報告", "組會_2026-10", "figs")
INDEX = os.path.join(REPO, "tmp", "handoff_index.json")
XL, YL = (-14.5, 3.0), (-13.0, 4.5)


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


def load():
    with open(INDEX, encoding="utf-8") as f:
        rows = [r for r in json.load(f)
                if r.get("port") == "dual" and r.get("geom") == "p01" and r.get("bridge_w") == 0.075]
    M = np.array([r["m"] for r in rows], dtype=float)
    ids = [r["id"] for r in rows]
    m3, m4p = M[:, 2], M[:, 3] + 5
    wm = np.minimum.reduce([M[:, 0] + 2, M[:, 1] + 2, m3, m4p])
    sc = np.array([r["score"] for r in rows])
    assert np.abs(sc - wm).max() < 1e-6, "score 欄≠wm_mfg 公式"
    return ids, m3, m4p, wm


def pareto(x, y):
    o = np.argsort(-x)
    px, py, best = [], [], -np.inf
    for i in o:
        if y[i] > best:
            px.append(x[i])
            py.append(y[i])
            best = y[i]
    return np.array(px), np.array(py)


def scatter(ax, ids, m3, m4p, wm):
    big(ax, "通帶餘裕 m3 (dB)", "阻帶餘裕 m4′ (dB)")
    ax.set_xlim(*XL)
    ax.set_ylim(*YL)
    cmap = LinearSegmentedColormap.from_list("b", ["#dfe8f3", "#8fb0d8", DBLUE, "#0d2f5c"])
    inside = (m3 >= XL[0]) & (m3 <= XL[1]) & (m4p >= YL[0]) & (m4p <= YL[1])
    hb = ax.hexbin(m3[inside], m4p[inside], gridsize=(46, 40), extent=(*XL, *YL),
                   cmap=cmap, norm=LogNorm(), mincnt=1, linewidths=0.2, edgecolors=SURF, zorder=2)
    # 合格區
    ax.fill_between([0, XL[1]], 0, YL[1], color=GREEN, alpha=0.12, lw=0, zorder=1)
    ax.text(1.5, 0.35, "合格區\n零筆", ha="center", va="bottom", fontsize=12, color=GREEN)
    # 等和斜線（文字角度依實際顯示比例計算）
    xx = np.linspace(*XL, 50)
    for s_ in (0.0, -4.6):
        ax.plot(xx, s_ - xx, color=MUTED, lw=1.2, ls="--", zorder=3)
        y0 = 3.75
        lab = "總和 = 0" if s_ == 0 else "總和 = −4.6"
        ax.text(s_ - y0 - 0.35, y0, lab, fontsize=12, color=INK2, ha="right", va="center",
                zorder=6)
    # 等最差餘裕的 L 形線
    c = -2.39
    ax.plot([c, XL[1]], [c, c], color=ORANGE, lw=1.6, zorder=4)
    ax.plot([c, c], [c, YL[1]], color=ORANGE, lw=1.6, zorder=4)
    ax.text(c + 0.15, YL[1] - 0.15, "最差 = −2.39", ha="left", va="top", fontsize=12,
            color=ORANGE, zorder=6)
    # 經驗前緣
    px, py = pareto(m3, m4p)
    ax.step(px, py, where="post", color=INK, lw=1.3, zorder=5)
    # 標記
    pick = {
        "smp073_d_040": ("目前最佳 −2.39", ORANGE, "*", 260, (-1.1, -3.6)),
        "smp021_L_010": ("前任最佳 −2.54", PURPLE, "D", 70, (-1.1, -5.0)),
        "d77b1_S2g4_36": ("第二款式 −3.36", AQUA, "s", 80, (-9.0, -1.2)),
        "d72b3_W_25": ("通帶標本\nm3 −1.01", RED, "^", 90, (0.2, -9.6)),
        "d72b2_c_84": ("阻帶標本 m4′ +2.32", RED, "v", 90, (-12.6, 0.4)),
    }
    for pid, (lab, col, mk, sz, txy) in pick.items():
        k = ids.index(pid)
        x, y = m3[k], m4p[k]
        assert XL[0] <= x <= XL[1] and YL[0] <= y <= YL[1], pid
        ax.scatter([x], [y], marker=mk, s=sz, color=col, edgecolor=SURF, lw=1.0, zorder=7)
        ax.annotate(lab, xy=(x, y), xytext=txy, fontsize=12, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9), zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.85))
    n_out = int((~inside).sum())
    return hb, n_out


def binned(ax, m3, m4p):
    big(ax, "通帶＋阻帶 總和 (dB)", "該箱最佳 (總和−失衡)/2 (dB)")
    S = m3 + m4p
    H = (S - np.abs(m3 - m4p)) / 2
    edges = np.arange(-10.0, -1.0 + 1e-9, 0.5)
    cx, best, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (S > lo) & (S <= hi)
        if k.sum():
            i = np.flatnonzero(k)[np.argmax(H[k])]
            cx.append(S[i])
            best.append(H[i])
            ns.append(int(k.sum()))
    cx, best = np.array(cx), np.array(best)
    xx = np.array([-10, -1])
    ax.plot(xx, xx / 2, color=MUTED, ls="--", lw=1.3, zorder=2)
    ax.text(-5.7, -4.3, "虛線＝完美均衡上限 總和/2", fontsize=12, color=INK2)
    ax.plot(cx, best, color=DBLUE, lw=1.8, marker="o", ms=6, zorder=4)
    j = int(np.argmax(best))
    ax.scatter([cx[j]], [best[j]], marker="*", s=240, color=ORANGE, edgecolor=SURF, zorder=5)
    ax.annotate(f"膝點：總和 −5～−4.5\n最佳 {best[j]:.2f}", xy=(cx[j], best[j]),
                xytext=(-8.6, -1.3), fontsize=12, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))
    ax.text(-1.2, -2.75, "總和再高\n失衡變大", ha="right", va="top", fontsize=12,
            color=INK2)
    ax.set_xlim(-10, -1)
    ax.set_ylim(-5.2, -0.6)
    return list(zip(cx.tolist(), best.tolist(), ns))


def main():
    ids, m3, m4p, wm = load()
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.22, left=0.06, right=0.99,
                          top=0.93, bottom=0.13)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    hb, n_out = scatter(a, ids, m3, m4p, wm)
    bins = binned(b, m3, m4p)
    a.text(0, 1.03, f"(a) 可製造代 {len(m3):,} 筆", transform=a.transAxes, fontsize=13, color=INK)
    b.text(0, 1.03, "(b) 按總和分箱", transform=b.transAxes, fontsize=13, color=INK)
    b.text(0.98, 0.04, "S21 為瓶頸時：最差餘裕＝(總和−失衡)/2", transform=b.transAxes,
           ha="right", va="bottom", fontsize=12, color=INK,
           bbox=dict(boxstyle="round,pad=0.3", fc=SURF, ec=MUTED, lw=0.8))
    fig.text(0.06, 0.0, f"新幾何＋0.075 mm 橋、規格 v2；藍越深＝樣本越密，黑階梯＝經驗前緣；視窗外另有 {n_out:,} 筆；"
             "8/31 交付索引，含重測列", ha="left", fontsize=11.5, color=INK2)
    share = float(np.mean(np.isclose(wm, np.minimum(m3, m4p))))
    print(f"n={len(m3)} 視窗外={n_out} S21 為約束軸比例={share:.3f} 和max={np.max(m3 + m4p):.2f}")
    for c, v, n in bins:
        print(f"  最佳點總和 {c:+.2f}  n={n:5d}  最佳(和−失衡)/2={v:.2f}")
    save(fig, "budget_scatter.png")


if __name__ == "__main__":
    main()
