"""R81 v6a（−15 版）最佳設計演進 GIF：累計最佳每刷新一次就是一格。
左 2×2：S21、S11、S22（各含規格線）、pattern；右：爬坡（到該格為止的每筆模擬＋累計最佳，y 軸 symlog 讓精修段看得清楚）。
標題＝第幾筆模擬、v6a WM、設計 id（WM ≥ 0 轉綠）。

用法：python -m script.figs.r81_best_gif <輸出資料夾> [--ms 280]
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
from script.figs.r81_v6 import DOT, LINE, V6, V6A, rows_v6

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
    ap.add_argument("--ms", type=int, default=280, help="每格毫秒")
    ap.add_argument("--hold", type=int, default=10, help="最後一格多停幾格")
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    v6, v6a = sm.load_cfg(V6), sm.load_cfg(V6A)
    grid = np.asarray(sm.frequency_grid(v6.measurement))
    rows = rows_v6(v6, grid)
    frames, best, allw = [], -np.inf, []
    for n, r in enumerate(rows, 1):
        sc = sm.true_wm(r, v6a, grid)
        allw.append(sc["worst_margin"])
        if sc["worst_margin"] > best:
            best = sc["worst_margin"]
            frames.append((n, sc, r))
    allw = np.array(allw)
    cum = np.maximum.accumulate(allw)
    xs = np.arange(1, len(allw) + 1)
    print("刷新次數", len(frames), "；最後", round(frames[-1][1]["worst_margin"], 3))
    seq = list(range(len(frames))) + [len(frames) - 1] * args.hold
    fig = plt.figure(figsize=(17, 9.2), dpi=80)
    fig.patch.set_facecolor(rp.SURF)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 2])
    axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                     [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
    ax_c = fig.add_subplot(gs[:, 2])
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
        ax_c.clear()
        ax_c.scatter(xs[:n], np.clip(allw[:n], -15, None), s=7, color=DOT, lw=0)
        ax_c.step(xs[:n], cum[:n], where="post", color=LINE, lw=2.4)
        ax_c.scatter([n], [cum[n - 1]], s=70, color="#e0322b", zorder=5)
        ax_c.axhline(0, color=rp.INK, lw=1.2, ls="--")
        ax_c.set_yscale("symlog", linthresh=0.3, linscale=1.2)
        ax_c.set_ylim(-15, max(0.06, cum.max() + 0.04))
        ax_c.set_yticks([-10, -3, -1, -0.3, -0.1, 0])
        ax_c.set_yticklabels(["−10", "−3", "−1", "−0.3", "−0.1", "0"])
        ax_c.set_xlim(0, xs[-1] + 10)
        ax_c.set_xlabel("累計 HFSS 模擬筆數", color=rp.MUTED)
        ax_c.set_ylabel("v6a WM (dB)", color=rp.MUTED)
        ax_c.set_title(f"爬坡：累計最佳 {cum[n - 1]:+.3f} dB（y 軸 −0.3～0 線性、以下對數壓縮）", loc="left",
                       fontsize=11, color=rp.INK)
        rp.style(ax_c)
        ax_c.set_box_aspect(1)
        w = sc["worst_margin"]
        fig.suptitle(f"−15 版最佳演進　第 {n} 筆模擬　WM {w:+.3f} dB　（{seq[k] + 1}／{len(frames)}）\n{r['id']}",
                     x=0.03, ha="left", fontsize=12, color="#1a7f37" if w >= 0 else rp.INK)
        fig.subplots_adjust(left=0.05, right=0.98, top=0.89, bottom=0.06, wspace=0.25, hspace=0.25)

    anim = FuncAnimation(fig, draw, frames=len(seq), interval=args.ms)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "r81v6a_best_evolution.gif"
    anim.save(out, writer=PillowWriter(fps=1000 / args.ms))
    plt.close(fig)
    print("saved", out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
