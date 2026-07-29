# -*- coding: utf-8 -*-
"""methodology_figs.py — docs/methodology.md 的兩張示意圖（純 schematic,無外部資料,決定性可重跑）。

  fig1 methodology_loop.png       §0.1 四角色一條環 ＋ §0.2 三層時間尺度
  fig2 methodology_authority.png  §0.3 決策權分配 ＋ decisions.md 署名分布

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
    ax.set_title("四個角色，一條環　—　每個角色只做一件事", fontsize=13.5, color=INK, pad=12)

    W, H = 0.30, 0.20
    _box(ax, 0.05, 0.62, W, H, "人（Ricky）", "定價值軸・給資源・隨時否決\n判定「這個推進有沒有意義」",
         "#eef4fc", DBLUE)
    _box(ax, 0.63, 0.62, W, H, "Agent（Claude）", "起草假設與判準・判讀批次\n記帳・自主續輪（宣告制）",
         "#fdf1ea", ORANGE)
    _box(ax, 0.63, 0.14, W, H, "工具箱（script/）", "決定性生成・零成本初篩\n算指標・出圖",
         "#eaf6f1", AQUA)
    _box(ax, 0.05, 0.14, W, H, "HFSS（唯一真值）", "只產真值\n不參與任何決策",
         "#f3eefa", PURPLE)

    _arrow(ax, (0.355, 0.72), (0.625, 0.72), "定軸 / 資源 / 否決", DBLUE, ly=0.045)
    _arrow(ax, (0.79, 0.615), (0.79, 0.345), "決定跑什麼", ORANGE, lx=0.075)
    _arrow(ax, (0.625, 0.21), (0.355, 0.21), "候選批次", AQUA, ly=-0.048)
    ax.annotate("", xy=(0.20, 0.615), xytext=(0.20, 0.345),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=PURPLE))
    ax.text(0.20, 0.48, "五軸 KPI\n＋ records.json", ha="center", va="center",
            fontsize=9.2, color=PURPLE, linespacing=1.4,
            bbox=dict(fc=SURF, ec="none", pad=2.0))

    ax.text(0.50, 0.505, "「什麼算數」＝制度", ha="center", va="center",
            fontsize=10.8, color=INK, fontweight="bold")
    ax.text(0.50, 0.415, "公證 3/3・判準發車前寫死\nappend-only\n人與 agent 都不能繞過",
            ha="center", va="center", fontsize=9.0, color=INK2, linespacing=1.6)
    ax.add_patch(FancyBboxPatch((0.395, 0.355), 0.21, 0.21,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc="#fffbe8", ec=GOLD, lw=1.4, zorder=-1))

    # ── 下：三層時間尺度 ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.axis("off")
    ax2.set_title("三層時間尺度", fontsize=13.5, color=INK, pad=8)

    rows = [
        ("期 chapter", "1–2 週", "一個戰略段落（至今九期）", "MILESTONES.md ／ 聯合回顧", 0.70, 1, PURPLE, "#f3eefa"),
        ("輪 round", "1–2 天", "一個假設　—　硬上限 3 批", "round-NN.md ／ /new-round → /close-round", 0.40, 3, DBLUE, "#eef4fc"),
        ("批 batch", "4–6 小時", "一次對照（30–75 筆 HFSS）", "判讀→公證→重錨→發車 ／ /batch-cycle", 0.10, 9, ORANGE, "#fdf1ea"),
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

    ax2.text(0.72, 0.42, "實測節奏（2026-07-28 單日判讀）\n01:47・06:45・12:46・17:54・22:13\n"
                         "→ 研究迴圈已與人的作息解耦",
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
    ax.set_title("決策權分配：誰決定什麼", fontsize=13, color=INK, pad=10)

    items = [
        ("價值軸／戰略方向", "人", DBLUE, "2026-07-23 判「同型邊際」→ 否決紀錄慶祝\n→ 催生左右側拆帳制"),
        ("資源力度", "人", DBLUE, "每輪 ≤3 批；相似度稅同日兩次加壓"),
        ("否決權（隨時）", "人", DBLUE, "round §1 固定寫「Ricky 可隨時否決」"),
        ("開輪／續輪／建新臂", "Agent", ORANGE, "R22→R46 連續 25 輪自主開輪（宣告制）"),
        ("判準草擬・判讀・記帳・收臂", "Agent", ORANGE, "/new-round §1 六條檢查表；連兩批 <6% 自動收臂"),
        ("「什麼算數」", "制度", GOLD, "公證 3/3・判準寫死・append-only\n人與 agent 都不能繞過"),
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
    labels = ["Ricky 署名", "Claude 署名", "未署名"]
    vals = [ricky, claude, none]
    cols = [DBLUE, ORANGE, GRID]
    bars = ax2.barh(range(3)[::-1], vals, color=cols, height=0.55, ec="none")
    for b, v in zip(bars, vals):
        ax2.text(v + total * 0.02, b.get_y() + b.get_height() / 2, str(v),
                 va="center", fontsize=11, color=INK, fontweight="bold")
    ax2.set_yticks(range(3)[::-1]); ax2.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax2.set_xlim(0, total * 1.15)
    ax2.set_xlabel(f"decisions.md 原則級定案條目（共 {total} 條）", color=INK2, fontsize=9.6)
    ax2.set_title("原則級決策的歸屬", fontsize=13, color=INK, pad=10)
    ax2.grid(axis="x", color=GRID, lw=0.7, alpha=0.85)
    ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(GRID)
    ax2.tick_params(colors=INK2, labelsize=9.5, left=False)
    ax2.text(total * 0.02, -0.72,
             "⚠ 這是「原則級」決策的署名統計，\n不是總體控制權——agent 每天做的\n幾十個微決策不會進 decisions.md。",
             fontsize=8.6, color=MUTED, va="top", linespacing=1.6)

    _save(fig, "methodology_authority.png")


if __name__ == "__main__":
    fig_loop()
    fig_authority()
