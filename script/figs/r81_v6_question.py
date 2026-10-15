"""R81 問學長用：同一個設計（v4x 最佳 b57_004，11–40 GHz 重量 r81v6_m02_018）在 v4x 與 v6 兩種 spec 下的 S21。

用法：python -m script.figs.r81_v6_question <輸出資料夾>
輸出：r81_v6_question.png（左 v4x spec、右 v6 spec，各格正方形）。
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from script import r81_curve_sm as sm
from script.figs import r81_progress as rp

STORE = sm.V2_DATASET / "dedust_r81v6m02s1"
RID = "r81v6_m02_018_4fb06302"
F = np.arange(11, 40.01, .5)
REPO = Path(__file__).resolve().parents[2]


def bands(name):
    cfg = yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))
    return [b for b in cfg["score_spec"]["bands"] if b["label"] == "S21"]


def margin(y, b):
    m = (F >= b["start_ghz"] - 1e-9) & (F <= b["stop_ghz"] + 1e-9)
    return b["threshold"] - y[m].max() if b["relation"] == "le" else y[m].min() - b["threshold"]


def panel(ax, y, specs, title):
    for bs, ls in specs:
        for b in bs:
            lo, hi, thr = b["start_ghz"], b["stop_ghz"], b["threshold"]
            ax.plot([lo, hi], [thr, thr], color=rp.INK, lw=1.4, ls=ls)
            if ls == "--":
                y0, y1 = (thr, 2) if b["relation"] == "le" else (-45, thr)
                ax.fill_between([lo, hi], y0, y1, color=rp.BAD, alpha=0.08, lw=0)
    ax.plot(F, y, color=rp.CURVE, lw=2.0)
    ax.set_xlim(11, 40)
    ax.set_ylim(-45, 2)
    ax.set_xticks([11, 15, 19, 22, 26, 30, 36, 40])
    ax.set_xlabel("頻率 (GHz)", color=rp.MUTED, fontsize=9.5)
    ax.set_ylabel("S21 (dB)", color=rp.MUTED, fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=10.5, color=rp.INK)
    rp.style(ax)
    ax.set_box_aspect(1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    args = ap.parse_args(argv)
    plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
    e = json.loads((STORE / "results.json").read_text(encoding="utf-8"))[RID]
    _, y = torch.load(STORE / e["sample_file"], weights_only=True, map_location="cpu")
    s21 = np.asarray(y, dtype=float)[1]
    v4x, v6a, v6 = bands("dual_r81_v4x.yaml"), bands("dual_r81_v6a.yaml"), bands("dual_r81_v6.yaml")
    w4 = min(margin(s21, b) for b in v4x)
    w6a, w6 = min(margin(s21, b) for b in v6a), min(margin(s21, b) for b in v6)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), dpi=160)
    fig.patch.set_facecolor(rp.SURF)
    panel(axes[0], s21, [(v4x, "--")], f"v4x spec：阻帶 22–24.5、36–40 ≤ −20\n最差差 {-w4:.2f} dB")
    panel(axes[1], s21, [(v6, ":"), (v6a, "--")],
          f"學長 v6 spec：阻帶 11–19、36–40 ≤ −15（虛）／−20（點）\n最差差 {-w6a:.1f}／{-w6:.1f} dB")
    k = int(np.argmax(np.where(F <= 19, s21, -99)))
    axes[1].annotate(f"{F[k]:g} GHz 漏 {s21[k]:.1f} dB", xy=(F[k], s21[k]), xytext=(19.5, -1),
                     fontsize=9.5, color=rp.BAD, arrowprops=dict(arrowstyle="->", color=rp.BAD, lw=1))
    fig.suptitle("同一個設計（v4x 最佳 b57_004，11–40 GHz 重量；S11/S22 已過），兩種 spec",
                 x=0.03, ha="left", fontsize=12.5, color=rp.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "r81_v6_question.png"
    fig.savefig(out, facecolor=rp.SURF, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print("saved", out, f"v4x {w4:+.2f} v6a {w6a:+.2f} v6 {w6:+.2f}")


if __name__ == "__main__":
    main()
