"""R85 區塊掃描圖：以 b92_016 為底，每個 4×4 小塊反轉後，低阻帶與高阻帶 margin 的變化（dB），兩格各畫一項。

用法：python -m script.figs.r85_block_scan <輸出資料夾>
輸出：r85_block_scan.png（左＝Δ低阻、右＝Δ高阻；灰框＝拉扯區；底圖淡灰＝底設計金屬；各格正方形）。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from script import r81_curve_sm as sm
from script import r81_block_scan as bs
from script.figs import r81_progress as rp
from script.figs.r81_v6 import V6

BASE = "r81v2_b92_016"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    cfg = sm.load_cfg(V6)
    g = sm.frequency_grid(cfg.measurement)
    R = sm.V2_DATASET
    base = next(r for s in ("s1", "s2", "s3") for r in sm.load_store(R / f"dedust_r81v6b92{s}", g)
                if r["id"].startswith(BASE))
    bm = sm.true_wm(base, cfg, g)["margins"]
    man = {}
    for s in ("s1", "s2", "s3"):
        for r in json.loads((R / f"dedust_r81v6sc01{s}_input" / "manifest.json").read_text(encoding="utf-8")):
            man[r["id"]] = r
    keys = (("s21_low_stop", "Δ 低阻帶 11–19 GHz"), ("s21_high_stop", "Δ 高阻帶 36–40 GHz"))
    maps = {k: np.full((25, 25), np.nan) for k, _ in keys}
    for s in ("s1", "s2", "s3"):
        for r in sm.load_store(R / f"dedust_r81v6sc01{s}", g):
            r0, c0, n = man[r["id"]]["block"]
            mg = sm.true_wm(r, cfg, g)["margins"]
            for k, _ in keys:
                maps[k][r0:r0 + n, c0:c0 + n] = mg[k] - bm[k]
    pat = np.asarray(base["pattern"]).reshape(25, 25) > 0.5
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.4), dpi=160)
    fig.patch.set_facecolor(rp.SURF)
    for ax, (k, title) in zip(axes, keys):
        ax.imshow(pat, cmap="Greys", alpha=0.18, interpolation="nearest")
        lim = 6
        im = ax.imshow(maps[k], cmap="RdBu", vmin=-lim, vmax=lim, alpha=0.85, interpolation="nearest")
        ax.contour(bs.TUG.astype(float), levels=[0.5], colors=[rp.INK], linewidths=1.2)
        ax.set_xticks([]); ax.set_yticks([])
        v = maps[k][np.isfinite(maps[k])]
        ax.set_title(f"{title}：中位 {np.median(v):+.2f} dB", loc="left", fontsize=10.5, color=rp.INK)
        ax.set_box_aspect(1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="dB（藍＝變好、紅＝變差）")
    fig.suptitle("R85 區塊掃描：b92_016 每個 4×4 小塊反轉後的變化（24 筆，單次）", x=0.03, ha="left",
                 fontsize=12.5, color=rp.INK)
    fig.text(0.03, 0.02, "黑框＝拉扯區（縫＋上埠墊右側，不掃）；淡灰＝底設計金屬；超出 ±6 dB 以色階端點表示。",
             fontsize=8.5, color=rp.MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "r85_block_scan.png"
    fig.savefig(out, facecolor=rp.SURF)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
