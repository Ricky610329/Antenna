"""四代世代時間軸（2023/07 → 2026/10）：每代一條帶，標代表方法與里程碑；上方一條日月光要求與學長傳達。

用法（repo 根目錄）：
    python script/figs/report_generations.py
輸出：docs/進度報告/10月_歷來實驗總整理/figs/timeline_generations.png（寬圖，dpi 200）

日期與數字全部抄自同夾 `實驗總表.md`（§1 世代一覽、§2.10 規格演進、§2.11 量測與交付、§3 全實驗清單）；
本圖不讀 NAS、不跑模擬。2026/06 起第四代事件密集，時間軸在 2026/06/01 之後放大 3.5 倍，圖上以淺色底與註記標明。
"""
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "進度報告" / "10月_歷來實驗總整理" / "figs" / "timeline_generations.png"

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

INK, MUTED, RULE, SURFACE = "#0b0b0b", "#52514e", "#c9ccd1", "#ffffff"
GEN_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # dataviz 預設類別色 1–4，依世代固定順序
ZOOM_FILL = "#f3f5f8"

# ---------------------------------------------------------------- 時間 → 圖上位置（英吋）
T0 = date(2023, 6, 1)          # 左端
BREAK = date(2026, 6, 1)       # 之後放大
T1 = date(2026, 11, 15)        # 右端
S1 = 0.285                     # 英吋／月，放大前
S2 = S1 * 3.5                  # 英吋／月，放大後
X0 = 0.35                      # 左邊界


def months(a: date, b: date) -> float:
    return (b.year - a.year) * 12 + (b.month - a.month) + (b.day - a.day) / 30.4


def X(d: date) -> float:
    if d <= BREAK:
        return X0 + months(T0, d) * S1
    return X0 + months(T0, BREAK) * S1 + months(BREAK, d) * S2


W = X(T1) + 0.35
H = 7.30
AXIS_Y = 0.70
ROW_Y = [5.40, 4.12, 2.86, 1.58]     # 第一代 → 第四代
LANE_Y = 6.32
BAR_H = 0.30

GENS = [
    dict(name="第一代　陳廷茂", span="2023/07–2024/08", method="MPGN 生成器＋GEN 代理模型的線上學習",
         start=date(2023, 7, 25), end=date(2024, 8, 7), side="right"),
    dict(name="第二代　錢鵬予", span="2024/08–2025/07", method="BiScaleNorm＋變異與回滾的二階段生成",
         start=date(2024, 8, 1), end=date(2025, 7, 8), side="right"),
    dict(name="第三代　吳維文", span="2025/07–2026/06", method="ACP 單階段＋熱啟動＋DLF＋SC 連通 loss",
         start=date(2025, 7, 8), end=date(2026, 6, 30), side="left"),
    dict(name="第四代　我們", span="2026/06–2026/10", method="批次線上學習＋可製造：菱形橋與公證",
         start=date(2026, 6, 8), end=date(2026, 10, 23), side="left"),
]

# 里程碑：(世代, 日期, 文字, 類型, 層)；層 +1/+2 在帶上方、−1/−2 在帶下方；文字預設靠右對齊到標記
#   類型 report＝日月光會議或報告、defense＝口試、deliver＝交付、measure＝量測、result＝代表成果
MILESTONES = [
    (0, date(2023, 11, 15), "2023/11 線上學習成形", "result", -1),
    (0, date(2024, 6, 25), "2024/06/25 口試", "defense", -1),
    (1, date(2024, 12, 5), "2024/12/05 首份標明日月光的簡報", "report", 1),
    (1, date(2025, 1, 15), "2025/01 變異＋回滾", "result", -2),
    (1, date(2025, 4, 29), "2025/04/29 ASE CCU 聯合會議", "report", 2),
    (1, date(2025, 6, 17), "2025/06/17 口試", "defense", -1),
    (2, date(2025, 7, 27), "2025/07 首份 ASE 月報", "report", -1),
    (2, date(2026, 3, 24), "2026/03/24 ASE-CCU：5 個樣品實測", "measure", 1),
    (2, date(2026, 5, 28), "2026/05/28 ASE-CCU：pixel 量測", "measure", -1),
    (2, date(2026, 6, 30), "6/30 口試", "defense", 1),
    (3, date(2026, 6, 24), "6/24 六月報告", "report", 2),
    (3, date(2026, 7, 16), "7/16 七月報告", "report", 1),
    (3, date(2026, 7, 31), "7/31 單埠 +0.79 公證", "result", -1),
    (3, date(2026, 8, 24), "8/24 雙埠 −2.39 公證", "result", 2),
    (3, date(2026, 8, 31), "8/31 送板交付 39 個", "deliver", -2),
    (3, date(2026, 9, 13), "9/13 九月報告", "report", 2, "left"),
    (3, date(2026, 10, 15), "10/15 對稱天線 +0.599 公證", "result", 1),
    (3, date(2026, 10, 23), "10/23 寬頻 v6a +0.4104 公證", "result", -1),
]

# 日月光要求與學長傳達：(日期, 文字, 層)；層 >0 在線上方、<0 在線下方；文字靠右對齊到標記。
# 層的安排保證每條引線不穿過別的標籤：靠右對齊的標籤只會蓋到比它早、且層更低的標記上方
REQUESTS = [
    (date(2025, 5, 1), "2025/05 濾波天線改架構、看不同截面 → 雙埠、n260", 1),
    (date(2025, 10, 2), "2025/10 先做天線、改 loss、至少一軸對稱、S21 改 −3 dB", 2),
    (date(2026, 5, 28), "2026/05 場型規格納入 loss", 1),
    (date(2026, 8, 7), "8/07 單埠收尾、帶外交後級濾波器；8/12 雙埠規格 v2", 2),
    (date(2026, 9, 12), "9 月 平台：沒有 AI 也能整晚跑、不綁策略", -1),
    (date(2026, 10, 7), "10/07 起 對稱天線、寬頻濾波器規格 v1→v6a", -2),
]

MARK = {"report": ("^", 9.5), "defense": ("D", 8), "deliver": ("*", 15), "measure": ("o", 9), "result": ("o", 8)}
LABEL_FS, NAME_FS, METHOD_FS = 12.5, 14, 12.5


def label(ax, x, y, s, ha="right", fs=LABEL_FS, color=INK, weight="normal"):
    return ax.text(x, y, s, ha=ha, va="center", fontsize=fs, color=color, fontweight=weight, zorder=6,
                   bbox=dict(boxstyle="square,pad=0.08", fc=SURFACE, ec="none"))


def main():
    fig = plt.figure(figsize=(W, H), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)

    # 放大區底色與註記
    xb = X(BREAK)
    ax.add_patch(Rectangle((xb, AXIS_Y), X(T1) - xb, LANE_Y + 0.88 - AXIS_Y, fc=ZOOM_FILL, ec="none", zorder=0))
    ax.text(xb + 0.08, AXIS_Y - 0.50, "2026/06 起時間軸放大 3.5 倍", ha="left", va="center", fontsize=11.5,
            color=MUTED)

    # 時間軸
    ax.plot([X0, X(T1)], [AXIS_Y, AXIS_Y], color=MUTED, lw=1.2, zorder=2)
    ticks = [date(y, m, 1) for y in (2023, 2024, 2025, 2026) for m in (1, 4, 7, 10)
             if T0 <= date(y, m, 1) and months(date(y, m, 1), BREAK) > 3.5]
    for d in ticks:
        x = X(d)
        ax.plot([x, x], [AXIS_Y - 0.06, AXIS_Y + 0.06], color=MUTED, lw=1, zorder=2)
        ax.text(x, AXIS_Y - 0.22, f"{d.year}/{d.month:02d}", ha="center", va="center", fontsize=11.5,
                color=MUTED, fontweight="bold" if d.month == 1 else "normal")
    for m in (6, 7, 8, 9, 10, 11):
        x = X(date(2026, m, 1))
        ax.plot([x, x], [AXIS_Y - 0.06, AXIS_Y + 0.06], color=MUTED, lw=1, zorder=2)
        ax.text(x, AXIS_Y - 0.22, f"2026/{m:02d}" if m == 6 else f"{m:02d}", ha="center", va="center",
                fontsize=11.5, color=MUTED)

    # 日月光要求與學長傳達
    ax.plot([X0, X(T1)], [LANE_Y, LANE_Y], color=INK, lw=1.0, zorder=2)
    ax.text(X0, LANE_Y - 0.24, "日月光要求與學長傳達", ha="left", va="center", fontsize=NAME_FS,
            fontweight="bold", color=INK)
    ax.text(X0, LANE_Y - 0.50, "合作從第一代的天線系列就開始", ha="left", va="center", fontsize=METHOD_FS, color=MUTED)
    for d, s, tier in REQUESTS:
        x = X(d)
        ax.plot([x, x], [AXIS_Y, LANE_Y], color=RULE, lw=0.9, ls=(0, (3, 3)), zorder=1)
        y = LANE_Y + 0.08 + 0.27 * tier if tier > 0 else LANE_Y - 0.04 + 0.28 * tier
        ax.plot([x, x], [LANE_Y, y], color=INK, lw=0.8, zorder=1.5)
        ax.plot(x, LANE_Y, marker="v", ms=9, color=INK, zorder=5, mec=SURFACE, mew=1.2)
        label(ax, x + 0.03, y, s)

    # 世代帶
    for gi, g in enumerate(GENS):
        y, c = ROW_Y[gi], GEN_COLOR[gi]
        x1, x2 = X(g["start"]), X(g["end"])
        ax.add_patch(FancyBboxPatch((x1, y - BAR_H / 2), x2 - x1, BAR_H, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=c, ec="none", alpha=0.28, zorder=2))
        ax.plot([x1, x2], [y, y], color=c, lw=2.2, solid_capstyle="round", zorder=3)
        if g["side"] == "right":
            tx, ha = x2 + 0.25, "left"
        else:
            tx, ha = x1 - 0.25, "right"
        ax.text(tx, y + 0.13, f"{g['name']}　{g['span']}", ha=ha, va="center", fontsize=NAME_FS,
                fontweight="bold", color=INK, zorder=6)
        ax.text(tx, y - 0.15, g["method"], ha=ha, va="center", fontsize=METHOD_FS, color=MUTED, zorder=6)

    for gi, d, s, kind, tier, *opt in MILESTONES:
        y0, c = ROW_Y[gi], GEN_COLOR[gi]
        x = X(d)
        sgn = 1 if tier > 0 else -1
        ly = y0 + sgn * (0.36 + 0.24 * (abs(tier) - 1))
        ax.plot([x, x], [y0, ly], color=MUTED, lw=0.8, zorder=3)
        m, ms = MARK[kind]
        face = SURFACE if kind == "result" else c
        ax.plot(x, y0, marker=m, ms=ms, mfc=face, mec=c if kind == "result" else INK, mew=1.6 if kind == "result" else 0.8,
                zorder=5)
        ha = opt[0] if opt else "right"
        label(ax, x + (0.03 if ha == "right" else -0.03), ly, s, ha=ha)

    # 圖例
    handles = [
        Line2D([], [], ls="none", marker="^", ms=9, mfc="#9aa3ad", mec=INK, mew=0.8, label="日月光會議或報告"),
        Line2D([], [], ls="none", marker="D", ms=7.5, mfc="#9aa3ad", mec=INK, mew=0.8, label="口試"),
        Line2D([], [], ls="none", marker="o", ms=8.5, mfc="#9aa3ad", mec=INK, mew=0.8, label="實作量測"),
        Line2D([], [], ls="none", marker="*", ms=14, mfc="#9aa3ad", mec=INK, mew=0.8, label="交付"),
        Line2D([], [], ls="none", marker="o", ms=8, mfc=SURFACE, mec="#6b7480", mew=1.6, label="方法或成果"),
        Line2D([], [], ls="none", marker="v", ms=9, color=INK, label="要求或傳達"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(X0 / W, 0.005), ncol=6, frameon=False,
                    fontsize=11.5, handletextpad=0.3, columnspacing=1.4)
    for t in leg.get_texts():
        t.set_color(MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"saved {OUT}  ({W:.2f} x {H:.2f} in)")


if __name__ == "__main__":
    main()
