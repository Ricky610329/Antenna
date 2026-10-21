"""R81 v6a（−15 版）最佳設計演進 GIF：累計最佳每刷新一次就是一格。
2×2：pattern、S21、S11、S22（各含規格線）；標題＝第幾筆模擬、v6a WM、設計 id（WM ≥ 0 轉綠）。

用法：python -m script.figs.r81_best_gif <輸出資料夾> [--ms 350]
輸出：r81v6a_best_evolution.gif
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from script import r81_curve_sm as sm
from script.figs import r81_progress as rp
from script.figs.r81_v6 import LINE, V6, V6A, rows_v6

PANELS = (("S21", 1, "s21"), ("S11", 0, "s11_match"), ("S22", 2, "s22_match"))


def curve_panel(ax, grid, y, label, spec, margins):
    for b in spec.score_spec["bands"]:
        if b["label"] != label:
            continue
        lo, hi, thr = b["start_ghz"], b["stop_ghz"], b["threshold"]
        ax.axvspan(lo, hi, color="#f6dede" if b["relation"] == "le" else "#dff0e3", alpha=0.6, lw=0)
        ax.plot([lo, hi], [thr, thr], color=rp.INK, lw=1.3, ls="--")
    ax.plot(grid, y, color=LINE, lw=2.0)
    ax.set_xlim(11, 40)
    ax.set_ylim(-45, 2)
    ax.set_xticks([11, 15, 19, 22, 26, 30, 36, 40])
    ax.set_ylabel(f"{label} (dB)", color=rp.MUTED, fontsize=9)
    if label == "S21":
        t = f"S21：通帶 {margins['s21_pass']:+.2f}、低阻 {margins['s21_low_stop']:+.2f}、高阻 {margins['s21_high_stop']:+.2f}"
    else:
        t = f"{label}：匹配 {margins[label.lower() + '_match']:+.2f}"
    ax.set_title(t, loc="left", fontsize=9.5, color=rp.INK)
    rp.style(ax)
    ax.set_box_aspect(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--ms", type=int, default=350, help="每格毫秒")
    ap.add_argument("--hold", type=int, default=8, help="最後一格多停幾格")
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    v6, v6a = sm.load_cfg(V6), sm.load_cfg(V6A)
    grid = np.asarray(sm.frequency_grid(v6.measurement))
    rows = rows_v6(v6, grid)
    frames, best = [], -np.inf
    for n, r in enumerate(rows, 1):
        sc = sm.true_wm(r, v6a, grid)
        if sc["worst_margin"] > best:
            best = sc["worst_margin"]
            frames.append((n, sc, r))
    print("刷新次數", len(frames), "；最後", round(frames[-1][1]["worst_margin"], 3))
    seq = list(range(len(frames))) + [len(frames) - 1] * args.hold
    fig, axes = plt.subplots(2, 2, figsize=(9, 9.2), dpi=90)
    fig.patch.set_facecolor(rp.SURF)
    ax_p = axes[1, 1]
    curve_axes = {"S21": axes[0, 0], "S11": axes[0, 1], "S22": axes[1, 0]}

    def draw(k):
        n, sc, r = frames[seq[k]]
        for label, li, _ in PANELS:
            ax = curve_axes[label]
            ax.clear()
            curve_panel(ax, grid, r["y"][li], label, v6a, sc["margins"])
        axes[1, 0].set_xlabel("頻率 (GHz)", color=rp.MUTED, fontsize=9)
        ax_p.clear()
        ax_p.imshow(np.asarray(r["pattern"]).reshape(25, 25) > 0.5, cmap="Greys", origin="upper",
                    interpolation="nearest", vmin=0, vmax=1)
        ax_p.set_xticks([]); ax_p.set_yticks([])
        ax_p.set_title("pattern（25×25，上下緣中央為兩埠饋墊）", loc="left", fontsize=9.5, color=rp.INK)
        ax_p.set_box_aspect(1)
        w = sc["worst_margin"]
        fig.suptitle(f"−15 版最佳演進　第 {n} 筆模擬　WM {w:+.3f} dB　（{seq[k] + 1}／{len(frames)}）\n{r['id']}",
                     x=0.03, ha="left", fontsize=12, color="#1a7f37" if w >= 0 else rp.INK)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.06, wspace=0.22, hspace=0.25)

    anim = FuncAnimation(fig, draw, frames=len(seq), interval=args.ms)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "r81v6a_best_evolution.gif"
    anim.save(out, writer=PillowWriter(fps=1000 / args.ms))
    plt.close(fig)
    print("saved", out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
