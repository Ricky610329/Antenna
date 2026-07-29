# -*- coding: utf-8 -*-
"""methodology_figs.py — docs/methodology.md 的兩張示意圖（純 schematic,無外部資料,決定性可重跑）。

  fig1 methodology_loop.png       §3.1 三角色一制度的研究迴圈 ＋ §3.2 三層時間尺度
  fig2 methodology_authority.png  §8.1 決策權分佈 ＋ 原則級定案的署名歸屬

用法：python script/figs/methodology_figs.py   → docs/assets/
署名分布的數字來自 docs/discuss/decisions.md 的現場統計（標題含 "Ricky" / "Claude|Opus"）,
不寫死;其餘皆為概念示意。
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_r1r10_style import (  # noqa: E402
    plt, REPO, INK, INK2, MUTED, GRID, SURF, DBLUE, AQUA, ORANGE, GOLD, PURPLE,
)
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

OUT = os.path.join(REPO, "docs", "assets")
os.makedirs(OUT, exist_ok=True)


def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def _box(ax, x, y, w, h, title, sub, face, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=face, ec=edge, lw=1.6, zorder=3))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
            fontsize=12.5, color=INK, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            fontsize=9.2, color=INK2, zorder=4, linespacing=1.45)


def _arrow(ax, p0, p1, label, color, rad=0.0, lx=0.0, ly=0.0, fs=9.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                                 lw=1.5, color=color, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))
    mx, my = (p0[0] + p1[0]) / 2 + lx, (p0[1] + p1[1]) / 2 + ly
    ax.text(mx, my, label, ha="center", va="center", fontsize=fs, color=color,
            zorder=5, bbox=dict(fc=SURF, ec="none", pad=1.6))


def fig_loop():
    fig = plt.figure(figsize=(11.6, 8.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.35, 1.0], hspace=0.24)

    # ── 上：四角色一條環 ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("三個角色與一層制度", fontsize=13.5, color=INK, pad=12)

    W, H = 0.30, 0.20
    _box(ax, 0.05, 0.62, W, H, "研究者", "定價值軸・分配資源・行使否決權\n判定「這個推進有沒有意義」",
         "#eef4fc", DBLUE)
    _box(ax, 0.63, 0.62, W, H, "代理人（LLM agent）", "起草假設與判準・分析批次\n更新帳目・提出下一輪",
         "#fdf1ea", ORANGE)
    _box(ax, 0.63, 0.14, W, H, "驗證機具", "決定性生成・零模擬成本預篩\n計算指標・產生圖表",
         "#eaf6f1", AQUA)
    _box(ax, 0.05, 0.14, W, H, "全波模擬（唯一真值）", "只產生真值\n不參與任何決策",
         "#f3eefa", PURPLE)

    _arrow(ax, (0.355, 0.72), (0.625, 0.72), "價值軸 / 資源 / 否決", DBLUE, ly=0.045)
    _arrow(ax, (0.79, 0.615), (0.79, 0.345), "決定測什麼", ORANGE, lx=0.075)
    _arrow(ax, (0.625, 0.21), (0.355, 0.21), "候選批次", AQUA, ly=-0.048)
    ax.annotate("", xy=(0.20, 0.615), xytext=(0.20, 0.345),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=PURPLE))
    ax.text(0.20, 0.48, "評估指標\n＋紀錄真相源", ha="center", va="center",
            fontsize=9.2, color=PURPLE, linespacing=1.4,
            bbox=dict(fc=SURF, ec="none", pad=2.0))

    ax.text(0.50, 0.505, "「什麼算數」＝制度", ha="center", va="center",
            fontsize=10.8, color=INK, fontweight="bold")
    ax.text(0.50, 0.415, "判準預註冊・重複量測認證\n結論唯讀\n繞過會留下紀錄",
            ha="center", va="center", fontsize=9.0, color=INK2, linespacing=1.6)
    ax.add_patch(FancyBboxPatch((0.395, 0.355), 0.21, 0.21,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#fffbe8", ec=GOLD, lw=1.4, zorder=-1))

    # ── 下：三層時間尺度 ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.set_title("三層時間尺度", fontsize=13.5, color=INK, pad=8)

    rows = [
        ("期", "1–2 週", "一個戰略段落", "里程碑紀錄", 0.70, 1, PURPLE, "#f3eefa"),
        ("輪次", "1–2 天", "一個假設　—　上限 3 個批次", "一份輪次紀錄", 0.40, 3, DBLUE, "#eef4fc"),
        ("批次", "數小時", "一次對照實驗（30–75 個候選）", "分析→認證→重錨→下一批發出", 0.10, 9, ORANGE, "#fdf1ea"),
    ]
    X0, BW = 0.235, 0.45
    for name, dur, span, art, y, nseg, col, face in rows:
        ax2.add_patch(FancyBboxPatch((X0, y), BW, 0.20,
                                     boxstyle="round,pad=0.005,rounding_size=0.018",
                                     fc=face, ec=col, lw=1.5))
        for k in range(1, nseg):          # 巢狀切分：一期含多輪、一輪含多批
            ax2.plot([X0 + BW * k / nseg] * 2, [y + 0.010, y + 0.042],
                     color=col, lw=1.0, alpha=0.55)
        ax2.text(0.215, y + 0.125, name, ha="right", va="center",
                 fontsize=11.2, color=col, fontweight="bold")
        ax2.text(0.215, y + 0.055, dur, ha="right", va="center", fontsize=9.0, color=MUTED)
        ax2.text(X0 + BW / 2, y + 0.152, span, ha="center", va="center", fontsize=9.2, color=INK)
        ax2.text(X0 + BW / 2, y + 0.072, art, ha="center", va="center", fontsize=8.2, color=MUTED)

    ax2.text(0.72, 0.42, "觀察到的迭代節奏（單日五次判讀）\n01:47・06:45・12:46・17:54・22:13\n"
                         "→ 迭代週期由模擬耗時決定，\n　 不由研究者的作息決定",
             ha="left", va="center", fontsize=9.4, color=INK2, linespacing=1.7,
             bbox=dict(boxstyle="round,pad=0.45", fc=SURF, ec=GRID, lw=1.2))

    _save(fig, "methodology_loop.png")


def _count_decisions():
    """現場統計 decisions.md 標題的署名分布（不寫死）。"""
    p = os.path.join(REPO, "docs", "discuss", "decisions.md")
    heads = [h for h in open(p, encoding="utf-8").read().splitlines() if h.startswith("## ")]
    ricky = sum(1 for h in heads if "Ricky" in h)
    claude = sum(1 for h in heads if re.search(r"Claude|Opus", h))
    return len(heads), ricky, claude, len(heads) - ricky - claude


def fig_authority():
    total, ricky, claude, none = _count_decisions()

    fig = plt.figure(figsize=(11.6, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28)

    # ── 左：六類決策歸屬 ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("決策權的實際分佈", fontsize=13, color=INK, pad=10)

    items = [
        ("價值軸／戰略方向", "研究者", DBLUE, "判定一項推進「有沒有意義」；\n一次此類判定直接催生了價值軸的拆分"),
        ("資源力度", "研究者", DBLUE, "每輪批次上限；多樣性約束的強度"),
        ("否決權（隨時）", "研究者", DBLUE, "每份輪次紀錄的判準欄均載明可被否決"),
        ("提出／延續輪次、設計實驗分支", "代理人", ORANGE, "連續二十餘輪由代理人自主提出，不逐輪核准"),
        ("判準草擬・分析・記帳・收分支", "代理人", ORANGE, "發車前檢查表；連兩批未達門檻即自動收掉該分支"),
        ("「什麼算數」", "制度", GOLD, "判準預註冊・重複量測認證・結論唯讀\n繞過會留下紀錄"),
    ]
    y = 0.90
    for label, who, col, ev in items:
        ax.add_patch(FancyBboxPatch((0.015, y - 0.075), 0.115, 0.085,
                                    boxstyle="round,pad=0.004,rounding_size=0.02",
                                    fc=col, ec="none"))
        ax.text(0.0725, y - 0.033, who, ha="center", va="center",
                fontsize=10.5, color="white", fontweight="bold")
        ax.text(0.155, y - 0.008, label, ha="left", va="center", fontsize=10.8, color=INK)
        ax.text(0.155, y - 0.058, ev, ha="left", va="center", fontsize=8.4,
                color=MUTED, linespacing=1.45)
        y -= 0.155

    # ── 右：decisions.md 署名分布 ────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    labels = ["研究者署名", "代理人署名", "未署名"]
    vals = [ricky, claude, none]
    cols = [DBLUE, ORANGE, GRID]
    bars = ax2.barh(range(3)[::-1], vals, color=cols, height=0.55, ec="none")
    for b, v in zip(bars, vals):
        ax2.text(v + total * 0.02, b.get_y() + b.get_height() / 2, str(v),
                 va="center", fontsize=11, color=INK, fontweight="bold")
    ax2.set_yticks(range(3)[::-1]); ax2.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax2.set_xlim(0, total * 1.15)
    ax2.set_xlabel(f"原則級定案條目（共 {total} 條）", color=INK2, fontsize=9.6)
    ax2.set_title("原則級定案的歸屬", fontsize=13, color=INK, pad=10)
    ax2.grid(axis="x", color=GRID, lw=0.7, alpha=0.85)
    ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)
    ax2.tick_params(colors=INK2, labelsize=9.5, left=False)
    ax2.text(total * 0.02, -0.72,
             "⚠ 這是「原則級」定案的署名統計，\n不是總體控制權——代理人每天做出的\n數十個戰術性判斷不進入該紀錄。",
             fontsize=8.6, color=MUTED, va="top", linespacing=1.6)

    _save(fig, "methodology_authority.png")


if __name__ == "__main__":
    fig_loop()
    fig_authority()
