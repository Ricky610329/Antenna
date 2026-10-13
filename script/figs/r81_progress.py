"""R81 進度圖（給學長）：最佳設計 S11/S21/S22 對 spec、各批累計最佳 WM、最佳 pattern。

用法：python -m script.figs.r81_progress <out.png> [--primary configs/dual_r81_v4a.yaml] [--final configs/dual_r81_v4.yaml]
primary＝現階段搜尋目標（選最佳設計用）；final＝最終 spec（同圖並列評估）。
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

INK, MUTED, GRID, BAD = "#0b0b0b", "#52514e", "#e4e3df", "#e34948"
LINE = {"S11": "#eb6834", "S21": "#2a78d6", "S22": "#1baf7a"}


def load_rows(primary, final, grid):
    root = sm.WIDE_ROOT / "v2_v001" / "dataset"
    rows = []
    for store in sorted(root.glob("dedust_r81v*b??s?")):
        if not ((store / "results.json").exists() and (store / "measurement.json").exists()):
            continue
        batch = int(re.search(r"b(\d\d)s\d$", store.name).group(1))
        for r in sm.load_store(store, grid):
            a, b = sm.true_wm(r, primary, grid), sm.true_wm(r, final, grid)
            if a and b:
                r.update(batch=batch, wm=a["worst_margin"], wm_final=b["worst_margin"], margins=a["margins"])
                rows.append(r)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--primary", type=Path, default=sm.REPO / "configs" / "dual_r81_v4a.yaml")
    ap.add_argument("--final", type=Path, default=sm.REPO / "configs" / "dual_r81_v4.yaml")
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    primary, final = sm.load_cfg(args.primary), sm.load_cfg(args.final)
    grid = sm.frequency_grid(primary.measurement)
    rows = load_rows(primary, final, grid)
    best = max(rows, key=lambda r: r["wm"])
    batches = sorted({r["batch"] for r in rows})
    run_p = np.maximum.accumulate([max(r["wm"] for r in rows if r["batch"] == b) for b in batches])
    run_f = np.maximum.accumulate([max(r["wm_final"] for r in rows if r["batch"] == b) for b in batches])

    fig = plt.figure(figsize=(14, 5.6), dpi=160)
    fig.patch.set_facecolor("#fcfcfb")
    gs = fig.add_gridspec(1, 3, width_ratios=[2.3, 1.4, 1.0], wspace=0.3)

    ax = fig.add_subplot(gs[0])
    for spec, ls, alpha in ((final, ":", 0.0), (primary, "--", 0.10)):
        for b in spec.score_spec["bands"]:
            lo, hi, thr = b["start_ghz"], b["stop_ghz"], b["threshold"]
            ax.plot([lo, hi], [thr, thr], color=INK if b["label"] == "S21" else MUTED, lw=1.4, ls=ls)
            if alpha and b["label"] == "S21" and b["relation"] == "le":
                ax.fill_between([lo, hi], thr, 2, color=BAD, alpha=alpha, lw=0)
    for k, lab in enumerate(sm.LABELS):
        ax.plot(grid, best["y"][k], color=LINE[lab], lw=2.2 if lab == "S21" else 1.8, label=lab)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)
    ax.set_xlim(22, 40); ax.set_ylim(-45, 2)
    ax.set_xticks([22, 24.5, 26, 28, 30, 32, 34, 36, 38, 40])
    ax.set_xlabel("頻率 (GHz)", color=MUTED); ax.set_ylabel("dB", color=MUTED)
    worst = min(best["margins"], key=best["margins"].get)
    ax.set_title(f"最佳設計三條曲線：階段 WM {best['wm']:+.2f}／最終 WM {best['wm_final']:+.2f}（最差：{worst}）",
                 loc="left", fontsize=11, color=INK)

    bx = fig.add_subplot(gs[1])
    xs = np.arange(len(batches))
    for run, color, name in ((run_p, "#2a78d6", "階段 spec（阻帶 −15＋匹配）"), (run_f, "#6250d6", "最終 spec（阻帶 −20＋匹配）")):
        bx.step(xs, run, where="post", color=color, lw=2.0)
        bx.plot(xs, run, "o", color=color, ms=4.5, label=name)
    bx.axhline(0, color=INK, lw=1.2, ls="--")
    bx.text(xs[-1], 0.15, "達標 WM = 0", ha="right", fontsize=9, color=INK)
    step = max(1, len(batches) // 8)                    # 批數多時每隔幾批標一次，避免擠在一起
    bx.set_xticks(xs[::step], [f"b{b}" for b in batches][::step], fontsize=8.5)
    bx.legend(loc="lower right", frameon=False, fontsize=8.5)
    bx.set_ylabel("累計最佳 WM (dB)", color=MUTED)
    bx.set_title("各批累計最佳（同一批資料兩把尺）", loc="left", fontsize=11, color=INK)

    cx = fig.add_subplot(gs[2])
    cx.imshow(best["pattern"].reshape(25, 25), cmap="Greys", origin="upper", interpolation="nearest")
    cx.set_xticks([]); cx.set_yticks([])
    cx.set_title("最佳設計 pattern", loc="left", fontsize=11, color=INK)
    cx.set_xlabel(f"{best['id']}\n{best['lineage']} 家族", fontsize=8.5, color=MUTED)

    for a in (ax, bx):
        a.set_facecolor("#fcfcfb")
        a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color("#bdbcb6")
        a.tick_params(colors=MUTED)
    fig.suptitle(f"R81 濾波器進度：階段 spec 最佳 WM {best['wm']:+.2f} dB，五項皆差 1 dB 內", x=0.01, ha="left", fontsize=13.5, color=INK)
    fig.text(0.01, -0.03, "規格（虛線＝階段、點線＝最終、紅區＝阻帶禁區）：S21 26–30 GHz ≥ −3 dB；22–24.5、36–40 GHz ≤ −15 dB（階段）／−20 dB（最終）；"
             f"S11、S22 26.5–29.5 GHz ≤ −10 dB。HFSS Discrete 22–40 GHz 共 {len(rows)} 筆，皆單次模擬未重測。",
             fontsize=8.5, color=MUTED)
    fig.savefig(args.out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("saved", args.out, best["id"], round(best["wm"], 3), round(best["wm_final"], 3), len(rows))


if __name__ == "__main__":
    main()
