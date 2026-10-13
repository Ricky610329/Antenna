"""R81 spec v2 進度圖（給學長）：最佳設計 S21 對 spec、各批最佳 WM、最佳 pattern。

用法：python -m script.figs.r81_v2_progress <out.png>
資料：NAS v2_v001 資料根全部 dedust_r81v2b* store（單次 HFSS，未公證）＋首批 survey 最佳作基準。
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from script import r81_curve_sm as sm

INK, MUTED, BLUE, BAD, GRID = "#0b0b0b", "#52514e", "#2a78d6", "#e34948", "#e4e3df"
SURVEY_BEST = -4.28          # 10/12 首批 16–40 GHz 舊好設計，v2 重算最佳


def main(out):
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    cfg = sm.load_cfg()
    grid = sm.frequency_grid(cfg.measurement)
    root = sm.WIDE_ROOT / "v2_v001" / "dataset"
    rows = []
    for store in sorted(root.glob("dedust_r81v2b??s?")):
        if (store / "results.json").exists() and (store / "measurement.json").exists():
            for r in sm.load_store(store, grid):
                s = sm.true_wm(r, cfg, grid)
                if s:
                    r["wm"], r["batch"] = s["worst_margin"], int(re.search(r"b(\d\d)s", store.name).group(1))
                    rows.append(r)
    best = max(rows, key=lambda r: r["wm"])
    batches = sorted({r["batch"] for r in rows})
    per = [max(r["wm"] for r in rows if r["batch"] == b) for b in batches]
    run = np.maximum.accumulate([SURVEY_BEST] + per)
    near = sum(r["wm"] > -1 for r in rows)

    fig = plt.figure(figsize=(13, 5.2), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 1.4, 1.0], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    for a, b, lo, hi in [(22, 24.5, -15, 2), (36, 40, -15, 2), (26, 30, -12, -3)]:
        ax.fill_between([a, b], lo, hi, color=BAD, alpha=0.10, lw=0)
    for a, b, y in [(22, 24.5, -15), (36, 40, -15), (26, 30, -3)]:
        ax.plot([a, b], [y, y], color=INK, lw=1.5, ls="--")
    s21 = best["y"][1]
    ax.plot(grid, s21, color=BLUE, lw=2.2)
    i26, i30 = int(np.argmin(abs(grid - 26))), int(np.argmin(abs(grid - 30)))
    ax.annotate(f"26 GHz {s21[i26]:.2f} dB", xy=(26, s21[i26]), xytext=(25.2, -40), fontsize=9.5, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD))
    ax.annotate(f"30 GHz {s21[i30]:.2f} dB", xy=(30, s21[i30]), xytext=(29.4, -35), fontsize=9.5, color=BAD,
                arrowprops=dict(arrowstyle="->", color=BAD))
    ax.text(23.25, -12.5, "≤ −15 dB", ha="center", fontsize=9.5, color=INK)
    ax.text(38, -12.5, "≤ −15 dB", ha="center", fontsize=9.5, color=INK)
    ax.text(28, -1.0, "≥ −3 dB", ha="center", fontsize=9.5, color=INK)
    ax.set_xlim(22, 40); ax.set_ylim(-45, 2)
    ax.set_xticks([22, 24.5, 26, 28, 30, 32, 34, 36, 38, 40])
    ax.set_xlabel("頻率 (GHz)", color=MUTED); ax.set_ylabel("S21 (dB)", color=MUTED)
    ax.set_title(f"最佳設計 S21：WM {best['wm']:+.2f} dB", loc="left", fontsize=11.5, color=INK)
    fig.suptitle("R81 濾波器 spec v2 進度：兩阻帶已過，只差通帶兩端約 0.25 dB", x=0.01, ha="left", fontsize=13.5, color=INK)

    bx = fig.add_subplot(gs[1])
    xs = np.arange(len(run))
    bx.step(xs, run, where="post", color=BLUE, lw=2.2)
    bx.plot(xs, run, "o", color=BLUE, ms=5)
    bx.axhline(0, color=INK, lw=1.2, ls="--")
    bx.text(len(run) - 1, 0.15, "達標線 WM = 0", ha="right", fontsize=9.5, color=INK)
    bx.set_xticks(xs, ["首批"] + [f"b{b}" for b in batches], fontsize=8.5)
    bx.set_ylim(min(run) - 0.5, 0.8)
    bx.set_ylabel("累計最佳 WM (dB)", color=MUTED)
    bx.set_title(f"各批累計最佳 WM：−4.28 → {run[-1]:+.2f}", loc="left", fontsize=11.5, color=INK)

    cx = fig.add_subplot(gs[2])
    cx.imshow(best["pattern"].reshape(25, 25), cmap="Greys", origin="upper", interpolation="nearest")
    cx.set_xticks([]); cx.set_yticks([])
    cx.set_title("最佳設計 pattern（25×25）", loc="left", fontsize=11.5, color=INK)
    cx.set_xlabel(f"{best['id']}\n{best['lineage']} 家族", fontsize=8.5, color=MUTED)

    for a in (ax, bx):
        a.set_facecolor("#fcfcfb")
        a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color("#bdbcb6")
        a.tick_params(colors=MUTED)
    fig.text(0.01, -0.03, f"spec v2：S21 26–30 GHz ≥ −3 dB；22–24.5、36–40 GHz ≤ −15 dB。HFSS Discrete 22–40 GHz，"
             f"共 {len(rows)} 筆，均為單次模擬未重測；WM 差距 < 1 dB 的設計 {near} 筆。紅色區＝不允許的範圍。",
             fontsize=8.5, color=MUTED)
    fig.savefig(out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("saved", out, best["id"], round(best["wm"], 3), len(rows), near)


if __name__ == "__main__":
    main(sys.argv[1])
