"""R81 任意切縫示意：v6 最佳斷開設計（直線交界）與三個鋸齒切縫重切變體，上埠主體藍、下埠主體橘、浮島灰。

用法：python -m script.figs.r81_reseam <輸出資料夾> [--id r81v2_b72_000_6b22f97e] [--seed 5]
輸出：r81_reseam_demo.png（四格正方形）。
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from script import r81_curve_sm as sm
from script.figs import r81_progress as rp
from script.figs.r81_v6 import V6, rows_v6

TOP, BOT, ISLAND = (0.16, 0.40, 0.80), (0.86, 0.40, 0.20), (0.62, 0.62, 0.62)


def paint(ax, q, title):
    top, bot = sm.bodies(q)
    img = np.ones((25, 25, 3))
    img[q & ~top & ~bot] = ISLAND
    img[top], img[bot] = TOP, BOT
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=10, color=rp.INK)
    ax.set_box_aspect(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--id", default="r81v2_b72_000_6b22f97e")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    cfg = sm.load_cfg(V6)
    r = next(r for r in rows_v6(cfg, sm.frequency_grid(cfg.measurement)) if r["id"] == args.id)
    p = np.asarray(r["pattern"]).reshape(25, 25) > 0.5
    rng = np.random.default_rng(args.seed)
    fig, axes = plt.subplots(2, 2, figsize=(8, 9), dpi=160)
    fig.patch.set_facecolor(rp.SURF)
    paint(axes[0, 0], p, f"原設計 WM {r['wm']:+.2f}：直線交界")
    for ax, (amp, reach) in zip(axes.ravel()[1:], ((2, 2), (3, 3), (4, 3))):
        q = sm.reseam(p, rng, amp, reach)
        assert not sm.pads_connected(q)
        paint(ax, q, f"重切：切縫擺幅 ±{amp} 列、先補 {reach} 格（改 {int((q != p).sum())} 格）")
    fig.suptitle("任意切縫：兩主體先長進空白，再沿鋸齒切縫切開", x=0.03, ha="left", fontsize=12.5, color=rp.INK)
    fig.text(0.03, 0.01, f"{args.id}；切縫左右貫穿且 4 連通，角接觸也接不上，兩埠必斷開。藍＝上埠主體、橘＝下埠主體、灰＝浮島。",
             fontsize=8.5, color=rp.MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), h_pad=3)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "r81_reseam_demo.png"
    fig.savefig(out, facecolor=rp.SURF)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
