"""日月光「歷來實驗總整理」第三版第四代：目前的系統架構圖＋新版代理模型兩張圖。

用法（repo 根目錄）：
    python -m script.figs.report_ase_arch [--only arch,surrogate_idea,surrogate_rank]
輸出：docs/進度報告/10月_歷來實驗總整理/figs/gen4_ase/
    arch.png            系統架構：元件方塊＋資料流箭頭，依開發機／NAS／三台模擬機分區（16:7，dpi 200）
    surrogate_idea.png  代理模型做法示意：前一版「一張圖形→整條曲線」對照新版「已模擬的相近設計＋改了哪幾格→變化量」
                        （示意圖：格子與曲線都是畫出來的，不是實際設計，圖上不放數字；16:7）
    surrogate_rank.png  新版代理模型上線後逐輪的排序相關（預測 vs HFSS 的 Spearman ρ），三條線（約 4:3，給左圖右文）

零 NAS、零模型：架構圖照 10 月寬頻濾波器任務實際在跑的程式畫（對應程式與出處見
`docs/進度報告/10月_歷來實驗總整理/notes/gen4_架構與代理模型.md`）；surrogate_rank 的 ρ 值抄自研究日誌
round-82 §4（第 1–3 輪＝同批 10 筆爬山候選）、round-83 §4 自問檢查（第 4–7 輪＝全批 23–24 筆）、
round-84 §4 自問檢查（第 8–10 輪＝全批，改以阻帶 −15 dB 版計分），數字不在這裡重算。
圖面只放白話名稱，不放內部代號、輪次編號與圖下小字；條件寫在 notes 與投影片備忘稿。
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "進度報告" / "10月_歷來實驗總整理" / "figs" / "gen4_ase"

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # 參考調色盤前三格（已過 CVD 檢查）
NAVY = "#245b85"                                                  # 投影片小標色
ZONE_DEV, ZONE_NAS, ZONE_HFSS, ZONE_HUMAN = "#eef4f8", "#f4f3ef", "#eaf3fb", "#fdf1ea"
EDGE = "#9fb3c4"


def _style():
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def _box(ax, x, y, w, h, text, fc="white", ec=EDGE, fs=17, color=INK, bold=False, lw=1.6, r=0.08):
    """圓角方塊（data 座標；x,y＝左下角），文字置中。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", zorder=4, linespacing=1.35)


def _arrow(ax, p, q, color=MUTED, lw=2.0, rad=0.0, ls="-", ms=18):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=2,
                                 shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------- 架構圖

# 功能色沿用第 16 頁流程圖的圖例（挑選或產生候選／HFSS 模擬／訓練代理模型／人工），兩頁可對照。
F_GEN, F_HFSS, F_SM, F_HUMAN, F_DATA = "#3a8a7c", "#244c6a", "#c0622f", "#7d8a96", "#8a8578"


def _tint(hex_color, a):
    """把顏色和白色混合，a＝原色比例（做淡底）。"""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], float)
    m = c * a + 255 * (1 - a)
    return "#" + "".join(f"{int(round(v)):02x}" for v in m)


def _cbox(ax, x, y, w, h, title, body, col, fs=17, bfs=15):
    """元件方塊：粗體名稱＋一到兩行說明；邊框用功能色、底色用淡色。"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=_tint(col, 0.10), ec=col, lw=2.2, zorder=3))
    if body:
        nb = body.count("\n") + 1
        ty = y + h / 2 + 0.17 * nb + 0.02
        ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=INK,
                zorder=4)
        ax.text(x + w / 2, ty - 0.27 - 0.17 * nb, body, ha="center", va="center", fontsize=bfs, color=INK,
                zorder=4, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold",
                color=INK, zorder=4)


def _zone(ax, x, y, w, h, name, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15",
                                fc=fc, ec="none", zorder=0))
    ax.text(x + 0.22, y + h - 0.3, name, ha="left", va="center", fontsize=18, fontweight="bold", color=NAVY,
            zorder=1)


def _label(ax, x, y, text, ha="center", fs=14, color=MUTED):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, zorder=5, linespacing=1.3)


def arch(out):
    """10 月寬頻濾波器任務實際在跑的元件與資料流（對應程式見 notes/gen4_架構與代理模型.md）。"""
    _style()
    fig, ax = plt.subplots(figsize=(16, 7), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    _zone(ax, 0.2, 0.3, 7.85, 6.45, "開發機", "#f3f6f9")
    _zone(ax, 8.25, 0.3, 3.55, 6.45, "共享儲存", "#f5f4f1")
    _zone(ax, 12.0, 0.3, 3.8, 6.45, "3 台 HFSS 模擬機", "#f0f5fa")

    # 開發機：人與 AI 協作（上）、候選產生 → 代理模型篩選（中）
    _cbox(ax, 0.55, 4.95, 7.15, 1.1, "人與 AI 協作", "判讀結果、事先寫下成功標準、決定下一步方向", F_HUMAN)
    _cbox(ax, 0.55, 1.9, 3.15, 1.95, "候選產生", "從好設計局部修改、\n區塊移植、變異交叉", F_GEN)
    _cbox(ax, 4.55, 1.9, 3.15, 1.95, "代理模型篩選", "代理模型或模擬結果統計\n預估表現，排序挑出一輪", F_SM)
    # 共享儲存：資料庫（上）、模擬佇列（下）
    _cbox(ax, 8.6, 3.95, 2.85, 2.1, "模擬結果資料庫", "歷史累積的設計＋\n每筆新模擬", F_DATA)
    _cbox(ax, 8.6, 0.75, 2.85, 1.95, "模擬佇列", "查重複、分成三份，\n三台依優先順序取件", F_DATA)
    # 模擬機：評分（上）、三台模擬機（下）
    _cbox(ax, 12.35, 4.95, 3.1, 1.1, "評分", "曲線換成最差餘裕", F_HFSS)
    for k, yy in enumerate((3.05, 2.05, 1.05)):
        _cbox(ax, 12.35, yy, 3.1, 0.75, f"HFSS 模擬機 {k + 1}", "", F_HFSS, fs=16)

    # 資料流
    _arrow(ax, (3.72, 2.88), (4.53, 2.88))                                  # 候選 → 代理模型
    _arrow(ax, (7.72, 2.3), (8.58, 1.75))                                   # 代理模型 → 佇列
    _label(ax, 7.95, 1.55, "每輪約 24 筆", ha="right")
    for yy in (3.42, 2.42, 1.42):                                           # 佇列 → 三台各自取件
        _arrow(ax, (11.47, 1.72), (12.33, yy))
    _arrow(ax, (13.2, 3.82), (13.2, 4.93))                                  # 模擬機 → 評分
    _label(ax, 13.35, 4.38, "S11、S21、S22 曲線", ha="left", fs=13)
    _arrow(ax, (12.33, 5.5), (11.47, 5.5))                                  # 評分 → 資料庫
    _arrow(ax, (8.58, 5.5), (7.72, 5.5))                                    # 資料庫 → 人與 AI
    _label(ax, 8.15, 5.78, "判讀")
    # 資料庫 → 候選產生與代理模型（一條匯流線分兩支）
    bus_y = 4.42
    ax.plot([8.58, 2.6], [bus_y, bus_y], color=MUTED, lw=2.0, zorder=2, solid_capstyle="butt")
    _arrow(ax, (2.6, bus_y + 0.01), (2.6, 3.87))
    _arrow(ax, (6.1, bus_y + 0.01), (6.1, 3.87))
    _label(ax, 8.4, bus_y + 0.25, "全部結果：好設計當起點，代理模型每輪重訓", ha="right")
    # 人與 AI → 候選產生（決定方向）
    _arrow(ax, (0.95, 4.93), (0.95, 3.87))
    _label(ax, 1.08, 4.42, "決定方向", ha="left")
    # 好設計換一台機器重跑（資料庫 → 佇列，虛線）
    _arrow(ax, (9.35, 3.93), (9.35, 2.72), ls="--", color=F_HFSS)
    _label(ax, 9.55, 3.32, "好設計換一台\n機器重跑", ha="left", fs=13, color=F_HFSS)

    fig.savefig(out, dpi=200, facecolor=SURF, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


# ---------------------------------------------------------------- 代理模型做法示意

_RNG = np.random.default_rng(7)
_GRID_BITS = _RNG.random((7, 7)) < 0.45        # 示意用的固定格子（不是實際設計）


def _grid_icon(ax, x, y, s, flips=(), bits=_GRID_BITS, flip_only=False):
    """s×s 的 7×7 格子示意；flips＝標成橘色的格子（候選改動的格）。flip_only＝只畫框與改動格。"""
    n = bits.shape[0]
    c = s / n
    ax.add_patch(Rectangle((x, y), s, s, fc="white", ec=MUTED, lw=1.2, zorder=4))
    for i in range(n):
        for j in range(n):
            on = bits[i, j] and not flip_only
            fc = ORANGE if (i, j) in flips else ("#5c6f80" if on else None)
            if fc:
                ax.add_patch(Rectangle((x + j * c, y + (n - 1 - i) * c), c, c, fc=fc, ec="white", lw=0.6,
                                       zorder=5))


def _curve_icon(ax, x, y, w, h, kind="band", color=NAVY):
    """小曲線示意：band＝帶通形狀；delta＝在 0 附近的變化量（虛線為 0）。不畫刻度、不放數字。"""
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=MUTED, lw=1.2, zorder=4))
    t = np.linspace(0, 1, 200)
    band = 0.12 + 0.72 * np.exp(-((t - 0.55) / 0.16) ** 4)
    wave = np.sin(2 * np.pi * (t * 1.6 + 0.1)) * np.exp(-((t - 0.55) / 0.35) ** 2)
    if kind == "band":
        v = band
    elif kind == "sum":                       # 鄰居曲線＋變化量：形狀略有不同，示意「加回去」
        v = np.clip(band + 0.12 * wave, 0.02, 0.98)
    else:
        ax.plot([x + 0.04 * w, x + 0.96 * w], [y + 0.5 * h] * 2, color=MUTED, lw=1.0, ls="--", zorder=5)
        v = 0.5 + 0.28 * wave
    ax.plot(x + (0.06 + 0.88 * t) * w, y + (0.08 + 0.84 * v) * h, color=color, lw=2.4, zorder=5)


def surrogate_idea(out):
    _style()
    fig, ax = plt.subplots(figsize=(16, 7), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # 左側列名
    ax.text(0.25, 5.55, "前一版", fontsize=22, fontweight="bold", color=NAVY, va="center")
    ax.text(0.25, 2.25, "新版", fontsize=22, fontweight="bold", color=NAVY, va="center")
    ax.plot([0.25, 15.75], [3.95, 3.95], color=GRID, lw=1.5)

    # 前一版：一張設計圖形 → 神經網路 → 整條曲線（輸出與新版的預測曲線上下對齊）
    _grid_icon(ax, 2.0, 4.85, 1.4)
    ax.text(2.7, 4.62, "設計圖形", ha="center", va="top", fontsize=17, color=INK)
    _arrow(ax, (3.65, 5.55), (6.5, 5.55))
    _box(ax, 6.6, 5.0, 2.8, 1.1, "神經網路", fc=ZONE_DEV, fs=19)
    _arrow(ax, (9.5, 5.55), (13.95, 5.55))
    _curve_icon(ax, 14.05, 5.0, 1.5, 1.1)
    ax.text(14.8, 4.62, "預測整條曲線", ha="center", va="top", fontsize=17, color=INK)

    # 新版：已模擬的相近設計（圖形＋HFSS 曲線，1–4 個）＋候選改了哪幾格 → 神經網路 → 變化量 ＋ 鄰居曲線 ＝ 預測曲線
    for k, (dx, dy) in enumerate(((0.3, 0.3), (0.15, 0.15), (0.0, 0.0))):
        x0, y0 = 1.2 + dx, 1.65 + dy
        ax.add_patch(FancyBboxPatch((x0, y0), 2.65, 1.25, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc="white", ec=EDGE, lw=1.4, zorder=3 + k * 0.1))
        if k == 2:
            _grid_icon(ax, x0 + 0.12, y0 + 0.14, 0.97)
            _curve_icon(ax, x0 + 1.25, y0 + 0.14, 1.28, 0.97)
    ax.text(2.65, 1.45, "已模擬的相近設計\n圖形＋HFSS 曲線，1–4 個", ha="center", va="top", fontsize=16, color=INK,
            linespacing=1.35)
    ax.text(4.45, 2.4, "＋", ha="center", va="center", fontsize=22, color=MUTED)
    _grid_icon(ax, 4.75, 1.9, 1.0, flips={(1, 4), (2, 4), (5, 2)}, flip_only=True)
    ax.text(5.25, 1.45, "候選改了\n哪幾格", ha="center", va="top", fontsize=16, color=INK, linespacing=1.35)
    _arrow(ax, (5.9, 2.4), (6.5, 2.4))
    _box(ax, 6.6, 1.6, 2.8, 1.6, "神經網路\n估計曲線變化量；\n多個鄰居互相參考", fc=ZONE_DEV, fs=16)
    _arrow(ax, (9.5, 2.4), (9.95, 2.4))
    _curve_icon(ax, 10.05, 1.85, 1.5, 1.1, kind="delta", color=BLUE)
    ax.text(10.8, 1.6, "變化量", ha="center", va="top", fontsize=16, color=INK)
    ax.text(11.8, 2.4, "＋", ha="center", va="center", fontsize=22, color=MUTED)
    _curve_icon(ax, 12.05, 1.85, 1.5, 1.1)
    ax.text(12.8, 1.6, "相近設計的\nHFSS 曲線", ha="center", va="top", fontsize=16, color=INK, linespacing=1.35)
    ax.text(13.8, 2.4, "＝", ha="center", va="center", fontsize=22, color=MUTED)
    _curve_icon(ax, 14.05, 1.85, 1.5, 1.1, kind="sum", color=BLUE)
    ax.text(14.8, 1.6, "預測曲線", ha="center", va="top", fontsize=17, color=INK, fontweight="bold")

    fig.savefig(out, dpi=200, facecolor=SURF, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------- 逐輪排序相關

# 預測（選樣當下，結果回來前）與 HFSS 最差餘裕的 Spearman ρ；出處見檔頭與 notes。
RANK_ROUNDS = list(range(1, 11))
RANK_OLD = [-0.27, -0.39, -0.30, -0.25, -0.63, +0.10, -0.23, -0.17, -0.07, +0.43]
RANK_DELTA = [+0.47, +0.50, +0.24, +0.71, +0.69, +0.69, +0.55, +0.29, +0.41, +0.60]
RANK_CTX = {7: +0.42, 8: +0.48, 9: +0.45, 10: +0.42}   # 第 7 輪起以旁觀方式同步預測


def surrogate_rank(out):
    _style()
    fig, ax = plt.subplots(figsize=(8.4, 6.4), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    ax.axhline(0, color=INK, lw=1.3, zorder=1)
    kw = dict(lw=2.2, ms=9, mec="white", mew=1.6, zorder=3)
    x = np.asarray(RANK_ROUNDS, float)
    ax.plot(x, RANK_DELTA, "o-", color=BLUE, label="變化量模型", **kw)
    cx = sorted(RANK_CTX)
    ax.plot(np.asarray(cx) + 0.14, [RANK_CTX[k] for k in cx], "D-", color=AQUA, label="上下文模型", **kw)
    ax.plot(x - 0.14, RANK_OLD, "s-", color=ORANGE, label="前一版代理模型", **kw)   # 左右錯開，同輪數值相近時不互蓋
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(-0.8, 0.9)
    ax.set_xticks(RANK_ROUNDS)
    ax.set_yticks([-0.6, -0.3, 0, 0.3, 0.6, 0.9])
    ax.set_xlabel("上線後的模擬輪次", fontsize=16, color=MUTED)
    ax.set_ylabel("預測與 HFSS 的排序相關", fontsize=16, color=MUTED)
    ax.tick_params(labelsize=14, colors=MUTED)
    ax.grid(axis="y", color=GRID, lw=1.0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=3, frameon=False, fontsize=15,
              handlelength=1.8, columnspacing=1.4)
    fig.savefig(out, dpi=200, facecolor=SURF, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


FIGS = {"arch": arch, "surrogate_idea": surrogate_idea, "surrogate_rank": surrogate_rank}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=",".join(FIGS), help="逗號分隔：" + ",".join(FIGS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.only.split(","):
        name = name.strip()
        FIGS[name](OUT / f"{name}.png")
        print("寫出", OUT / f"{name}.png")


if __name__ == "__main__":
    main()
