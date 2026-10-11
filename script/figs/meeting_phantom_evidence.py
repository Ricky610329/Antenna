# -*- coding: utf-8 -*-
"""meeting_phantom_evidence.py — 組會 2026-10 第 28 頁：零厚度對角接觸＝數值幻影的三步證據。

數字全部抄自 docs/log/round-69-dual-hd50.md（皆為舊幾何 p00、規格 v2 尺 wm_r2、單次）：
  (a) §4 mfg 探針：kn_16 原值 −3.05 → 47 個點接觸補成真 0.2 mm 連接 −23.83（Δ −20.78）。
  (b) §4 煙測：同幾何 25×25 → 50×50 上採樣，Δm1 +0.51／Δm2 −0.40／Δm3 −2.93／Δm4 −3.50。
  (c) §4 開代實驗 4.：p2 存活地圖 30 筆分層，舊幾何 → 新幾何 p01 重測 |Δ| 中位：
      乾淨 ≤4 接點 0.16／低接點 ≤10 0.04／湯頂 1.54／湯中 3.81。
零 HFSS、零讀檔。執行：python -m script.figs.meeting_phantom_evidence
"""
import os

from script.figs.report_r1r10_style import (INK, INK2, MUTED, SURF, RED, DBLUE, ORANGE, AQUA,
                                            plt, style_ax)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "docs", "進度報告", "組會_2026-10", "figs")


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
    ax.grid(axis="x", visible=False)


def vlabel(ax, x, v, fmt="{:+.2f}", pad=0.4):
    ax.text(x, v + (pad if v >= 0 else -pad), fmt.format(v), ha="center",
            va="bottom" if v >= 0 else "top", fontsize=12.5, color=INK)


def main():
    fig, axs = plt.subplots(1, 3, figsize=(12, 5.2), gridspec_kw=dict(wspace=0.38))

    # (a) 補接點探針
    ax = axs[0]
    big(ax, "", "最差餘裕 (dB)")
    xs, vals = [0, 1], [-3.05, -23.83]
    ax.bar(xs, vals, width=0.6, color=[MUTED, RED], zorder=3)
    for x, v in zip(xs, vals):
        vlabel(ax, x, v, "{:.2f}")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(xs, ["原圖樣\n47 個點接觸", "點接觸補成\n真連接"])
    ax.set_ylim(-28, 2)
    ax.text(0, -14, "Δ −20.78", fontsize=14, color=RED, ha="center", va="center")
    ax.text(0, 1.03, "(a) 補上 47 個對角接點", transform=ax.transAxes, fontsize=13, color=INK)

    # (b) 同幾何換表示
    ax = axs[1]
    big(ax, "", "50×50 減 25×25 的差 (dB)")
    labs = ["S11\n帶內", "S22\n帶內", "S21\n通帶", "S21\n阻帶"]
    vals = [0.51, -0.40, -2.93, -3.50]
    cols = [MUTED, MUTED, DBLUE, ORANGE]
    ax.bar(range(4), vals, width=0.62, color=cols, zorder=3)
    for x, v in enumerate(vals):
        vlabel(ax, x, v, pad=0.12)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(4), labs)
    ax.set_ylim(-4.6, 1.3)
    ax.text(0, 1.03, "(b) 同一幾何、換離散解析度", transform=ax.transAxes, fontsize=13,
            color=INK)

    # (c) 分層存活
    ax = axs[2]
    big(ax, "", "新舊幾何重測 |Δ| 中位 (dB)")
    labs = ["乾淨\n≤4 接點", "低接點\n≤10", "隨機湯\n頂層", "隨機湯\n中段"]
    vals = [0.16, 0.04, 1.54, 3.81]
    cols = [AQUA, AQUA, ORANGE, RED]
    ax.bar(range(4), vals, width=0.62, color=cols, zorder=3)
    for x, v in enumerate(vals):
        vlabel(ax, x, v, "{:.2f}", pad=0.06)
    ax.set_xticks(range(4), labs)
    ax.set_ylim(0, 4.4)
    ax.text(0.03, 0.62, "隨機湯＝靠 46–49 個\n點接觸的隨機圖樣", transform=ax.transAxes,
            fontsize=12, color=INK2, va="top")
    ax.text(0, 1.03, "(c) 30 筆按接點數分層重測", transform=ax.transAxes, fontsize=13, color=INK)

    fig.text(0.5, -0.04, "(a)(b) 舊幾何；(c) 舊幾何與新幾何重測之差；規格 v2 尺，皆為單次量測；出處 round-69 §4",
             ha="center", fontsize=12, color=INK2)
    save(fig, "phantom_evidence.png")


if __name__ == "__main__":
    main()
