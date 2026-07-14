# -*- coding: utf-8 -*-
"""report_paradigm_compare.py — 報告 §10 用：主流文獻（離線攤提·模型驅動）vs 我們（基於現有框架·agent 調度）
兩欄流程對照示意圖。用法: python -m script.figs.report_paradigm_compare --out <path>"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    INK, INK2, MUTED, GRID, SURF, DBLUE, GREEN, ORANGE, plt)
from matplotlib.patches import FancyBboxPatch  # noqa: E402


def _box(ax, cx, cy, w, h, text, fc, ec, tc, fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.018",
                                facecolor=fc, edgecolor=ec, lw=1.4, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.35)


def _arrow(ax, cx, y0, y1, color):
    ax.annotate("", xy=(cx, y1), xytext=(cx, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.7), zorder=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig = plt.figure(figsize=(13.2, 8.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.5, 0.965, "主流 AI 逆設計 vs 我們：兩種範式的區別",
             ha="center", fontsize=15, color=INK, fontweight="bold")
    ax.axvline(0.5, 0.06, 0.9, color=GRID, lw=1.2)

    LX, RX = 0.265, 0.735
    W = 0.4
    # ── 左欄：他們（離線攤提·模型驅動）
    _box(ax, LX, 0.885, W, 0.075, "主流文獻（Sengupta 三篇·日月光提供）\n離線攤提 · 模型驅動",
         MUTED, INK2, SURF, fs=11.5, bold=True)
    steps_L = [
        (0.75, "① 一次付清 10⁵–10⁶ 次模擬\n（HPC 天級離線）"),
        (0.585, "② 訓一個「全域準」的\n代理 CNN／RL 策略"),
        (0.42, "③ 之後每個新設計\n「分鐘級」攤提產出"),
    ]
    for cy, t in steps_L:
        _box(ax, LX, cy, W, 0.088, t, SURF, INK2, INK, fs=10.5)
    _arrow(ax, LX, 0.845, 0.795, INK2)
    _arrow(ax, LX, 0.706, 0.629, INK2)
    _arrow(ax, LX, 0.541, 0.464, INK2)
    ax.text(LX, 0.30, "agent ＝ RL 策略（非 LLM）\nhuman ＝ 事前約束 ＋ 事後挑選\n前提：付得起海量離線算力",
            ha="center", va="center", fontsize=10, color=INK2, linespacing=1.5)
    ax.text(LX, 0.145, "※ 少樣本下這套我們付不起\n（也是我們 Part A 線上學習追不上的原因）",
            ha="center", va="center", fontsize=9.6, color=ORANGE, linespacing=1.4, style="italic")

    # ── 右欄：我們（基於現有框架·agent 調度）
    _box(ax, RX, 0.885, W, 0.075, "我們\n基於現有框架 · agent 調度",
         DBLUE, DBLUE, SURF, fs=11.5, bold=True)
    steps_R = [
        (0.765, "① 既有池當起點分布\n（學長 24k，不從零）"),
        (0.62, "② agent ＋ 人 調度工具\n結構規則 · SM 只當初篩"),
        (0.475, "③ HFSS 在迴圈當真值\n（數百次／批，不信純代理）"),
        (0.33, "④ 可製造冠軍 → 回灌池\n（分布變好，飛輪）"),
    ]
    cols_R = [INK, INK, INK, GREEN]
    for (cy, t), tc in zip(steps_R, cols_R):
        _box(ax, RX, cy, W, 0.082, t, SURF, DBLUE, tc, fs=10.5,
             bold=(tc == GREEN))
    _arrow(ax, RX, 0.845, 0.808, DBLUE)
    _arrow(ax, RX, 0.722, 0.663, DBLUE)
    _arrow(ax, RX, 0.577, 0.518, DBLUE)
    _arrow(ax, RX, 0.432, 0.373, DBLUE)
    # 回灌迴圈箭頭（④ → ①）
    ax.annotate("", xy=(RX - W / 2 - 0.005, 0.765), xytext=(RX - W / 2 - 0.005, 0.33),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.5,
                                connectionstyle="arc3,rad=-0.45"), zorder=1)
    ax.text(RX, 0.205, "agent ＝ LLM 編排（互動迭代）\nhuman ＝ 迴圈中共同調度、非只事前/事後\n少樣本 ~2–3k 真值 · 不訓大模型",
            ha="center", va="center", fontsize=10, color=INK2, linespacing=1.5)

    # ── 底部一句話
    fig.text(0.5, 0.035,
             "他們：海量離線算力 → 全域準攤提　│　我們：少樣本·真值在迴圈 → 可製造·可衍生　"
             "（兩條路互補，非取代；他們的方法論骨架與物理正當性我們照借，資料量假設不搬）",
             ha="center", fontsize=10.2, color=INK, fontweight="bold")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
