# -*- coding: utf-8 -*-
"""meeting_records_overview.py — 組會 N1：四個月紀錄總覽（single 可製造 margin 紀錄＋dual wm_mfg）。

資料出處（只讀 repo）：
- single：docs/champions.md 現任榜／血統、docs/records.json history（7/26、7/31 事件）、
  早期 R7–R29 日期對 docs/log/README.md 收檔日與 round 檔公證時間戳
  （R28 s28b3_005 公證 07-15 10:40；R29 o29b2_011 公證 20:37 於 07-15；R23 k23b1_021 07-12 16:14）。
  ⚠ records.json history 有一列 "2026-07-13 wm 0.5 R28"，與 round-28（07-15 10:40 公證）衝突，此處以 round 檔為準，
  07-13 的紀錄改用 R23b4 m23b4_030 +0.49。
- dual：docs/records_dual.json history 中 key=wm_mfg 的 17 跳（8/13 −5.90 → 8/24 −2.39）。
  8/13 前的舊幾何主尺（wm_dual／wm_full／wm_r2）不畫：零厚度對角接觸為數值幻影，與 wm_mfg 不可比。
"""
import datetime as dt
import json
import os

import matplotlib.dates as mdates

from script.figs.meeting_common import (
    REPO, INK, INK2, MUTED, GRID, RED, DBLUE, ORANGE, GREEN, PURPLE,
    plt, style_ax, save, NOTE, TICK)


def D(m, d, h=12.0):
    return dt.datetime(2026, m, d) + dt.timedelta(hours=h)


# (時間, wm, 標籤, 是否標註, 備註)
SINGLE = [
    (D(7, 3), -2.68, "R7 除塵例外", True, "單次"),
    (D(7, 6), -0.29, "R9 對稱化", True, ""),
    (D(7, 7), 0.20, "R10 首批三項全過", True, ""),
    (D(7, 8), 0.22, "R11", False, ""),
    (D(7, 9, 9), 0.29, "R15", False, ""),
    (D(7, 9, 18), 0.35, "R17", False, ""),
    (D(7, 11), 0.39, "R20", False, ""),
    (D(7, 12, 16), 0.41, "R23", False, ""),
    (D(7, 13), 0.49, "R23", False, ""),
    (D(7, 15, 10.7), 0.50, "R28", False, ""),
    (D(7, 15, 20.6), 0.56, "R29", True, ""),
    (D(7, 26), 0.73, "R42 嫁接", True, ""),
    (D(7, 31), 0.79, "R48 嫁接＋爬山", True, ""),
]

X0, X1 = D(6, 27, 0), D(9, 13, 0)
END = D(9, 12, 18)

EVENTS = [
    (D(7, 24), "7/24 首例左側合格解", GREEN),
    (D(8, 13), "8/13 dual 幾何換代", PURPLE),
    (D(8, 31), "8/31 送板 39 件", ORANGE),
    (D(9, 12), "9/12 收尾", INK2),
]


def load_dual():
    d = json.load(open(os.path.join(REPO, "docs", "records_dual.json"), encoding="utf-8"))
    rows = [h for h in d["history"] if h.get("key") == "wm_mfg"]
    # 同日多跳：日內均分時間，讓階梯看得出來
    out, byday = [], {}
    for h in rows:
        byday.setdefault(h["date"], []).append(h)
    for day, hs in byday.items():
        y, m, dd = map(int, day.split("-"))
        for i, h in enumerate(hs):
            out.append((dt.datetime(y, m, dd) + dt.timedelta(hours=24 * (i + 1) / (len(hs) + 1)),
                        float(h["value"]), h["id"]))
    out.sort()
    return out


def step(ax, pts, color, end):
    xs = [p[0] for p in pts] + [end]
    ys = [p[1] for p in pts] + [pts[-1][1]]
    ax.step(xs, ys, where="post", color=color, lw=2.6, zorder=3)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=46, color=color, zorder=4,
               edgecolor="white", lw=1.0)


def main():
    dual = load_dual()
    assert len(dual) == 17 and dual[0][1] == -5.9 and dual[-1][1] == -2.39, dual

    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.85, 0.13], hspace=0.30)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # ---- single ----
    step(ax1, [(t, v) for t, v, *_ in SINGLE], DBLUE, END)
    ax1.axhline(0, color=RED, ls=":", lw=1.5, zorder=1)
    ax1.axhline(0.15, color=GREEN, ls="--", lw=1.3, zorder=1)
    ax1.text(D(9, 12, 12), 0.01, "規格線 0", color=RED, fontsize=NOTE - 1, ha="right", va="top")
    ax1.text(D(9, 12, 12), 0.18, "合格線 +0.15", color=GREEN, fontsize=NOTE - 1,
             ha="right", va="bottom")
    lab = {
        -2.68: ("R7 除塵例外 −2.68（單次）", (D(7, 8), -2.55)),
        -0.29: ("R9 對稱化 −0.29", (D(7, 9), -0.9)),
        0.20: ("R10 首批三項全過 +0.20", (D(6, 27, 6), 0.95)),
        0.56: ("R29 +0.56", (D(7, 12, 18), 1.25)),
        0.73: ("R42 嫁接 +0.73", (D(7, 20, 12), 1.25)),
        0.79: ("R48 嫁接＋爬山 +0.79，至今未破", (D(8, 4), 1.25)),
    }
    for t, v, _l, show, _n in SINGLE:
        if v in lab:
            txt, xy = lab[v]
            ax1.annotate(txt, (t, v), xy, fontsize=NOTE, color=INK, va="center",
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    ax1.text(D(6, 27, 6), -1.45, "6/27–7/3\n線上學習\n無可製造紀錄",
             color=MUTED, fontsize=NOTE - 0.5, va="center")
    ax1.set_ylim(-3.1, 1.65)
    style_ax(ax1, "", "帶內最差餘裕 (dB)",
             "天線 single：可製造 margin 紀錄（零粉塵；S0 網格、無橋；R7 為單次，其餘皆三次重測一致）")
    ax1.tick_params(labelbottom=False)

    # ---- dual ----
    step(ax2, [(t, v) for t, v, _ in dual], PURPLE, END)
    ax2.axhline(0, color=RED, ls=":", lw=1.5, zorder=1)
    ax2.text(D(9, 12, 12), 0.05, "規格線 0", color=RED, fontsize=NOTE - 1, ha="right", va="bottom")
    dlab = {
        "d70b1_A_11": ("8/13 開帳 −5.90", (D(7, 26), -5.5)),
        "d70b3_A_11_d2_00_d2_01": ("同日 −3.82", (D(7, 29), -3.55)),
        "smp021_L_010": ("8/19 −2.54", (D(8, 22), -1.15)),
        "smp073_d_040": ("8/24 −2.39\n自動迴圈第 17 跳", (D(8, 30), -1.25)),
    }
    for t, v, pid in dual:
        if pid in dlab:
            txt, xy = dlab[pid]
            ax2.annotate(txt, (t, v), xy, fontsize=NOTE, color=INK, va="center",
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    ax2.text(D(6, 27, 12), -3.2,
             "8/13 前為舊幾何，零厚度對角接觸\n數字不可比，不畫",
             color=MUTED, fontsize=NOTE - 0.5, va="center")
    ax2.set_ylim(-6.6, 0.8)
    style_ax(ax2, "", "最差餘裕 (dB)",
             "濾波器 dual：可製造尺 wm_mfg（0.075 mm 對角橋；每跳皆經重測公證，跨機取保守值）")

    # ---- 事件帶 ----
    for t, txt, c in EVENTS:
        for ax in (ax1, ax2):
            ax.axvline(t, color=c, lw=1.1, ls="-", alpha=0.35, zorder=0)
        ax3.scatter([t], [0.75], marker="^", s=80, color=c, zorder=3)
        ha = "right" if t.month == 9 and t.day > 10 else "center"
        ax3.text(t + (dt.timedelta(hours=14) if ha == "right" else dt.timedelta(0)), 0.05, txt,
                 color=INK, fontsize=NOTE, ha=ha, va="top")
    ax3.set_ylim(-1.2, 1.0)
    ax3.set_yticks([])
    for s in ax3.spines.values():
        s.set_visible(False)
    ax3.spines["top"].set_visible(True)
    ax3.spines["top"].set_color(GRID)
    ax3.tick_params(colors=INK2, labelsize=TICK, labelbottom=False, bottom=False)
    ax3.spines["top"].set_visible(False)
    ax2.set_xlim(X0, X1)
    ax2.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 15]))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d" if os.name != "nt" else "%#m/%#d"))
    ax3.set_facecolor("white")
    ax3.patch.set_alpha(0)
    save(fig, "records_overview.png")


if __name__ == "__main__":
    main()
