"""R82 S2 上線試用：同批「原 SM 挑」（climb）與「變化量模型挑」（dclimb）的預測 vs 實測，兩格各一個模型。

用法：python -m script.figs.r82_s2_trial <輸出資料夾> --inputs <r81v6bNN_input 夾> [...]
輸出：r82_s2_trial.png（左＝原 SM 預測、右＝變化量模型預測，橫軸預測、縱軸實測；各格正方形）。
只畫兩個模型都有預測的 climb／dclimb 列（manifest 的 pred_wm_mean 與 delta_wm_mean）。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from script import r81_curve_sm as sm
from script.figs import r81_progress as rp
from script.figs.r81_v6 import V6, rows_v6

ARM = {"climb": ("原 SM 挑", "#eb6834"), "dclimb": ("變化量模型挑", "#2a78d6")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--inputs", nargs="+", required=True)
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    cfg = sm.load_cfg(V6)
    true = {r["id"]: r["wm"] for r in rows_v6(cfg, sm.frequency_grid(cfg.measurement))}
    pts, batches = [], set()
    for d in args.inputs:
        for r in json.loads((Path(d) / "manifest.json").read_text(encoding="utf-8")):
            if r["selection_arm"] in ARM and r.get("delta_wm_mean") is not None and r["id"] in true:
                pts.append((r["selection_arm"], r["pred_wm_mean"], r["delta_wm_mean"], true[r["id"]]))
                batches.add(r["batch"])
    arm = np.array([p[0] for p in pts])
    smp, dlp, act = (np.array([p[i] for p in pts]) for i in (1, 2, 3))
    lo = min(smp.min(), dlp.min(), act.min()) - 0.5
    hi = max(smp.max(), dlp.max(), act.max()) + 0.5
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.6), dpi=160)
    fig.patch.set_facecolor(rp.SURF)
    for ax, pred, name in ((axes[0], smp, "原 SM 預測"), (axes[1], dlp, "變化量模型預測")):
        ax.plot([lo, hi], [lo, hi], color=rp.MUTED, lw=1, ls="--")
        for a, (lab, col) in ARM.items():
            k = arm == a
            ax.scatter(pred[k], act[k], s=46, color=col, label=f"{lab}（最佳 {act[k].max():+.2f}、中位 {np.median(act[k]):+.2f}）",
                       edgecolor="white", lw=0.6)
        rho = spearmanr(pred, act)[0]
        ax.set_title(f"{name}：與實測排序 ρ {rho:+.2f}、平均誤差 {np.abs(pred - act).mean():.2f} dB",
                     loc="left", fontsize=10.5, color=rp.INK)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("預測 WM (dB)", color=rp.MUTED); ax.set_ylabel("HFSS 實測 WM (dB)", color=rp.MUTED)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
        rp.style(ax)
        ax.set_box_aspect(1)
    b = "、".join(f"b{x}" for x in sorted(batches))
    fig.suptitle(f"R82 上線試用（{b}）：同批爬山名額，原 SM 挑 vs 變化量模型挑", x=0.03, ha="left", fontsize=12.5, color=rp.INK)
    fig.text(0.03, 0.01, "每點＝一筆候選；虛線＝預測等於實測。兩個模型對同一批 10 筆都有預測，可直接比誰看得準。",
             fontsize=8.5, color=rp.MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "r82_s2_trial.png"
    fig.savefig(out, facecolor=rp.SURF)
    plt.close(fig)
    print("saved", out, len(pts))


if __name__ == "__main__":
    main()
