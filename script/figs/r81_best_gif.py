"""R81 v6a（−15 版）最佳設計演進 GIF：累計最佳每刷新一次就是一格。
左＝pattern（紅框＝與上一格最佳不同的格），右＝S21 曲線與規格線；標題＝第幾筆模擬、v6a WM、設計 id。

用法：python -m script.figs.r81_best_gif <輸出資料夾> [--ms 600]
輸出：r81v6a_best_evolution.gif
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from script import r81_curve_sm as sm
from script.figs import r81_progress as rp
from script.figs.r81_v6 import LINE, V6, V6A, rows_v6


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--ms", type=int, default=600, help="每格毫秒")
    ap.add_argument("--hold", type=int, default=6, help="最後一格多停幾格")
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    v6, v6a = sm.load_cfg(V6), sm.load_cfg(V6A)
    grid = np.asarray(sm.frequency_grid(v6.measurement))
    rows = rows_v6(v6, grid)
    frames, best = [], -np.inf
    for n, r in enumerate(rows, 1):
        w = sm.true_wm(r, v6a, grid)["worst_margin"]
        if w > best:
            best = w
            frames.append((n, w, r))
    print("刷新次數", len(frames), "；最後", round(frames[-1][1], 3))
    seq = list(range(len(frames))) + [len(frames) - 1] * args.hold
    fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(10.5, 5.4), dpi=100)
    fig.patch.set_facecolor(rp.SURF)

    def draw(k):
        n, w, r = frames[seq[k]]
        prev = frames[seq[k] - 1][2] if seq[k] > 0 else None
        pat = np.asarray(r["pattern"]).reshape(25, 25) > 0.5
        ax_p.clear()
        ax_p.imshow(pat, cmap="Greys", origin="upper", interpolation="nearest", vmin=0, vmax=1)
        if prev is not None:
            diff = np.argwhere(pat != (np.asarray(prev["pattern"]).reshape(25, 25) > 0.5))
            for i, j in diff[:60]:
                ax_p.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#e0322b", lw=1.6))
            ax_p.set_title(f"pattern（紅框＝比上一個最佳改了 {len(diff)} 格）", loc="left", fontsize=10, color=rp.INK)
        else:
            ax_p.set_title("pattern（起點）", loc="left", fontsize=10, color=rp.INK)
        ax_p.set_xticks([]); ax_p.set_yticks([])
        ax_p.set_box_aspect(1)
        ax_s.clear()
        for b in v6a.score_spec["bands"]:
            if b["label"] != "S21":
                continue
            lo, hi, thr = b["start_ghz"], b["stop_ghz"], b["threshold"]
            ax_s.axvspan(lo, hi, color="#f6dede" if b["relation"] == "le" else "#dff0e3", alpha=0.6, lw=0)
            ax_s.plot([lo, hi], [thr, thr], color=rp.INK, lw=1.3, ls="--")
        ax_s.plot(grid, r["y"][1], color=LINE, lw=2.2)
        ax_s.set_xlim(11, 40); ax_s.set_ylim(-45, 2)
        ax_s.set_xlabel("頻率 (GHz)", color=rp.MUTED); ax_s.set_ylabel("S21 (dB)", color=rp.MUTED)
        mg = sm.true_wm(r, v6a, grid)["margins"]
        ax_s.set_title(f"通帶 {mg['s21_pass']:+.2f}｜低阻 {mg['s21_low_stop']:+.2f}｜高阻 {mg['s21_high_stop']:+.2f}｜"
                       f"S11 {mg['s11_match']:+.2f}｜S22 {mg['s22_match']:+.2f}", loc="left", fontsize=9, color=rp.INK)
        rp.style(ax_s)
        ax_s.set_box_aspect(1)
        color = "#1a7f37" if w >= 0 else rp.INK
        fig.suptitle(f"−15 版最佳演進　第 {n} 筆模擬　WM {w:+.3f} dB　（{seq[k] + 1}／{len(frames)}）　{r['id']}",
                     x=0.02, ha="left", fontsize=12, color=color)
        fig.subplots_adjust(left=0.03, right=0.98, top=0.86, bottom=0.11, wspace=0.18)

    anim = FuncAnimation(fig, draw, frames=len(seq), interval=args.ms)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "r81v6a_best_evolution.gif"
    anim.save(out, writer=PillowWriter(fps=1000 / args.ms))
    plt.close(fig)
    print("saved", out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
