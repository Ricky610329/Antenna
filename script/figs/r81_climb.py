"""R81 性能爬升圖：x＝累計 HFSS 模擬筆數，y＝WM；每筆一點、累計最佳一條線、標出關鍵事件。

用法：python -m script.figs.r81_climb <輸出資料夾>
輸出：r81_climb_v4a.png（階段 spec）、r81_climb_v4.png（最終 spec），各一張正方形圖。
資料：r81 寬頻真值（首批 survey＋v2_v001 各批），依批號排序；重測（~rep）不計入。
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from script import r81_curve_sm as sm

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DOT, LINE, MARK = "#9ec5f4", "#2a78d6", "#eb6834"

# 關鍵事件：(批號, 文字)；放在第一筆屬於該批的模擬上
EVENTS = [(2, "小步變異 1–8 格"), (3, "加爬山臂"), (20, "改用總不足量挑起點"),
          (25, "v4a 首次過關（公證）"), (30, "v4 改用總不足量挑起點")]


def batch_of(store):
    m = re.search(r"b(\d+)s\d$", store)
    return int(m.group(1)) if m else 0          # 首批 survey／工程掃描＝0


def ordered_rows(cfg, grid):
    rows = [r for r in sm.load_wide(sm.WIDE_ROOT, cfg, grid) if "~rep" not in r["id"]]
    rows = [r for r in rows if sm.true_wm(r, cfg, grid)]
    for r in rows:
        r["batch"] = batch_of(r["store"])
    rows.sort(key=lambda r: (r["batch"], r["store"], r["id"]))
    return rows


def plot(out, cfg_path, title, ylim):
    cfg = sm.load_cfg(cfg_path)
    grid = sm.frequency_grid(cfg.measurement)
    rows = ordered_rows(cfg, grid)
    wm = np.array([sm.true_wm(r, cfg, grid)["worst_margin"] for r in rows])
    best = np.maximum.accumulate(wm)
    x = np.arange(1, len(wm) + 1)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=160)
    fig.patch.set_facecolor(SURF)
    ax.scatter(x, np.clip(wm, ylim[0], None), s=7, color=DOT, lw=0, label="每筆模擬")
    ax.step(x, best, where="post", color=LINE, lw=2.2, label="累計最佳")
    ax.axhline(0, color=INK, lw=1.2, ls="--")
    ax.text(5, 0.15, "達標 WM = 0", ha="left", fontsize=9, color=INK)
    batches = [r["batch"] for r in rows]
    for b, text in EVENTS:
        if b in batches:
            i = batches.index(b)
            ax.axvline(x[i], color=MARK, lw=0.8, ls=":")
            ax.text(x[i] - 6, ylim[0] + 0.3, text, rotation=90, ha="right", va="bottom", fontsize=8.5, color=MARK,
                    bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5))
    ax.annotate(f"{best[-1]:+.2f}", xy=(x[-1], best[-1]), xytext=(-6, -16), textcoords="offset points",
                ha="right", fontsize=10, color=LINE)
    ax.set_xlim(0, x[-1] + 5)
    ax.set_ylim(*ylim)
    ax.set_xlabel("累計 HFSS 模擬筆數", color=MUTED)
    ax.set_ylabel("WM (dB)", color=MUTED)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.93), frameon=False, fontsize=9)
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED)
    ax.set_box_aspect(1)
    fig.text(0.02, 0.01, f"共 {len(wm)} 筆單次 HFSS（Discrete 22–40 GHz，舊 16–40 掃描切片沿用）；低於下限的點壓在底線。",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)
    print("saved", out, len(wm), round(best[-1], 3))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    args.outdir.mkdir(parents=True, exist_ok=True)
    plot(args.outdir / "r81_climb_v4a.png", sm.REPO / "configs" / "dual_r81_v4a.yaml",
         "性能爬升：階段 spec（S21 阻帶 ≤ −15 dB＋S11/S22 匹配）", (-12, 1))
    plot(args.outdir / "r81_climb_v4.png", sm.REPO / "configs" / "dual_r81_v4.yaml",
         "性能爬升：最終 spec（S21 阻帶 ≤ −20 dB＋S11/S22 匹配）", (-14, 1))


if __name__ == "__main__":
    main()
