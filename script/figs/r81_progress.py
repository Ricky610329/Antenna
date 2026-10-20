"""R81 給學長的兩張圖：最佳設計（S21／S11／S22 各一格＋pattern）、各批累計最佳 WM。

用法：python -m script.figs.r81_progress <輸出資料夾> [--primary configs/dual_r81_v4a.yaml] [--final configs/dual_r81_v4.yaml]
輸出：<資料夾>/r81_best_design.png、<資料夾>/r81_progress_wm.png
primary＝現階段搜尋目標（選最佳設計用）；final＝最終 spec（並列評估）。
資料：NAS v2_v001 資料根全部 r81 store（單次 HFSS，未公證）。
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from script import r81_curve_sm as sm

INK, MUTED, GRID, BAD, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#e34948", "#fcfcfb"
CURVE = "#2a78d6"
STAGE, FINAL = "#2a78d6", "#6250d6"


def load_rows(primary, final, grid):
    root = sm.WIDE_ROOT / "v2_v001" / "dataset"
    rows = []
    for store in sorted(root.glob("dedust_r81v*b??s?")):
        if not ((store / "results.json").exists() and (store / "measurement.json").exists()):
            continue
        batch = int(re.search(r"b(\d+)s\d$", store.name).group(1))
        for r in sm.load_store(store, grid):
            a, b = sm.true_wm(r, primary, grid), sm.true_wm(r, final, grid)
            if a and b:
                r.update(batch=batch, wm=a["worst_margin"], wm_final=b["worst_margin"], margins=a["margins"])
                rows.append(r)
    return rows


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbcb6")
    ax.tick_params(colors=MUTED, labelsize=9)


def curve_panel(ax, grid, y, label, primary, final, margins):
    """一條曲線一格：階段 spec 虛線、最終 spec 點線、禁區淡紅。"""
    for spec, ls in ((final, ":"), (primary, "--")):
        for b in spec.score_spec["bands"]:
            if b["label"] != label:
                continue
            lo, hi, thr = b["start_ghz"], b["stop_ghz"], b["threshold"]
            ax.plot([lo, hi], [thr, thr], color=INK, lw=1.4, ls=ls)
            if spec is primary:
                y0, y1 = (thr, 2) if b["relation"] == "le" else (-45, thr)
                ax.fill_between([lo, hi], y0, y1, color=BAD, alpha=0.08, lw=0)
    ax.plot(grid, y, color=CURVE, lw=2.0)
    ax.set_xlim(22, 40)
    ax.set_ylim(-45, 2)
    ax.set_xticks([22, 26, 30, 34, 38, 40])
    ax.set_xlabel("頻率 (GHz)", color=MUTED, fontsize=9.5)
    ax.set_ylabel(f"{label} (dB)", color=MUTED, fontsize=9.5)
    keys = [k for k in margins if k.startswith(label.lower())]
    text = "、".join(f"{k.split('_', 1)[1]} {margins[k]:+.2f}" for k in keys)
    ax.set_title(f"{label}：{text} dB", loc="left", fontsize=10.5, color=INK)
    style(ax)
    ax.set_box_aspect(1)


def best_figure(out, best, grid, primary, final, n_rows, scope_note=None, line_note=None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.4), dpi=160)
    fig.patch.set_facecolor(SURF)
    for ax, (k, lab) in zip((axes[0, 0], axes[0, 1], axes[1, 0]), ((1, "S21"), (0, "S11"), (2, "S22"))):
        curve_panel(ax, grid, best["y"][k], lab, primary, final, best["margins"])
    px = axes[1, 1]
    px.imshow(best["pattern"].reshape(25, 25), cmap="Greys", origin="upper", interpolation="nearest")
    px.set_xticks([]); px.set_yticks([])
    px.set_title("pattern（25×25，上下緣中央為兩埠饋墊）", loc="left", fontsize=10.5, color=INK)
    px.set_xlabel(f"{best['id']}／{best['lineage']} 家族", fontsize=8.5, color=MUTED)
    fig.suptitle(f"目前最佳設計：階段 spec WM {best['wm']:+.2f} dB、最終 spec WM {best['wm_final']:+.2f} dB",
                 x=0.03, ha="left", fontsize=13, color=INK)
    fig.text(0.03, 0.01, (line_note or "虛線＝階段 spec（S21 阻帶 ≤ −15 dB）、點線＝最終 spec（≤ −20 dB）、淡紅＝階段禁區。")
             + "S11/S22 26.5–29.5 GHz ≤ −10 dB。" + (scope_note or f"單次 HFSS 模擬，共 {n_rows} 筆中的最佳，未重測。"),
             fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, facecolor=SURF)
    return fig


def progress_figure(out, rows):
    batches = sorted({r["batch"] for r in rows})
    run_p = np.maximum.accumulate([max(r["wm"] for r in rows if r["batch"] == b) for b in batches])
    run_f = np.maximum.accumulate([max(r["wm_final"] for r in rows if r["batch"] == b) for b in batches])
    fig, ax = plt.subplots(figsize=(6.4, 6.6), dpi=160)
    fig.patch.set_facecolor(SURF)
    xs = np.arange(len(batches))
    for run, color, name in ((run_p, STAGE, "階段 spec（阻帶 −15＋匹配）"), (run_f, FINAL, "最終 spec（阻帶 −20＋匹配）")):
        ax.step(xs, run, where="post", color=color, lw=2.0)
        ax.plot(xs, run, "o", color=color, ms=4, label=name)
        ax.annotate(f"{run[-1]:+.2f}", xy=(xs[-1], run[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=color)
    ax.axhline(0, color=INK, lw=1.2, ls="--")
    ax.text(0, 0.15, "達標 WM = 0", fontsize=9, color=INK)
    step = max(1, len(batches) // 8)
    ax.set_xticks(xs[::step], [f"b{b}" for b in batches][::step])
    ax.set_ylim(min(run_f.min(), run_p.min()) - 0.5, 0.8)
    ax.set_xlabel("批次", color=MUTED); ax.set_ylabel("累計最佳 WM (dB)", color=MUTED)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("各批累計最佳 WM（同一批資料、兩把尺）", loc="left", fontsize=12, color=INK)
    style(ax)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURF)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--primary", type=Path, default=sm.REPO / "configs" / "dual_r81_v4a.yaml")
    ap.add_argument("--final", type=Path, default=sm.REPO / "configs" / "dual_r81_v4.yaml")
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    primary, final = sm.load_cfg(args.primary), sm.load_cfg(args.final)
    grid = sm.frequency_grid(primary.measurement)
    rows = load_rows(primary, final, grid)
    best = max(rows, key=lambda r: r["wm"])
    args.outdir.mkdir(parents=True, exist_ok=True)
    best_figure(args.outdir / "r81_best_design.png", best, grid, primary, final, len(rows))
    progress_figure(args.outdir / "r81_progress_wm.png", rows)
    print("saved", args.outdir, best["id"], round(best["wm"], 3), round(best["wm_final"], 3), len(rows))


if __name__ == "__main__":
    main()
