# -*- coding: utf-8 -*-
"""report_occlusion.py — 報告 §7.3（對外版）：s05 承重溫度圖（單色）。
5×5 區塊遮蔽掃描（dedust_occl,真 HFSS）：每塊輪流遮掉,量 worst-margin 掉多少。
單一色相（白→紅）＝掉幅,格內標數字;nan 塊標 —。
用法: python -m script.figs.report_occlusion --out <path>"""
import argparse
import json
import os
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    GRID, INK, INK2, RED, SURF, AQUA, plt)

from antenna.utils import config as _config, DATASET_PATH  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402

FEED = (24, 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    man = {m["id"]: m for m in json.load(
        open(str(DATASET_PATH.joinpath("dedust_occl_input", "manifest.json")), encoding="utf-8"))}
    occ = json.load(open(str(DATASET_PATH.joinpath("dedust_occl", "results.json")), encoding="utf-8"))
    r9 = json.load(open(str(DATASET_PATH.joinpath("dedust_r9", "results.json")), encoding="utf-8"))
    base = r9["s05_1050"]["wm"][2]
    pat = np.asarray(torch.load(str(DATASET_PATH.joinpath("dedust_r9_input", "s05_1050.pt")),
                                weights_only=True)).reshape(25, 25) > 0.5

    heat = np.full((5, 5), np.nan)
    for i, m in man.items():
        if m["source_id"] == "s05_1050" and i in occ and isinstance(occ[i], dict) and "wm" in occ[i]:
            br, bc = m["block"]
            heat[br, bc] = occ[i]["wm"][2] - base

    drop = -heat                                   #? 掉幅（正值,越大越承重）
    vmax = float(np.nanmax(drop))
    cmap = LinearSegmentedColormap.from_list("heatred", [SURF, "#f5c6a0", RED, "#7a1010"])

    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.imshow(pat.astype(int), cmap=ListedColormap([SURF, "#b9c9de"]), vmin=0, vmax=1,
              origin="upper", interpolation="nearest")
    img = np.kron(np.nan_to_num(drop, nan=0.0), np.ones((5, 5)))
    h = ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, alpha=0.66,
                  origin="upper", interpolation="nearest")
    for k in range(6):
        ax.axhline(k * 5 - 0.5, color=SURF, lw=1.6)
        ax.axvline(k * 5 - 0.5, color=SURF, lw=1.6)
    for br in range(5):
        for bc in range(5):
            v = drop[br, bc]
            if np.isnan(v):
                ax.text(bc * 5 + 2, br * 5 + 2, "—", ha="center", va="center",
                        color=INK2, fontsize=11)
            else:
                ax.text(bc * 5 + 2, br * 5 + 2, f"−{v:.1f}", ha="center", va="center",
                        color=SURF if v > vmax * 0.45 else INK, fontsize=10.5,
                        fontweight="bold")
    ax.scatter([FEED[1]], [FEED[0]], marker="^", s=64, color=AQUA, zorder=5,
               edgecolor=SURF, lw=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    cb = fig.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("遮掉這塊後，worst-margin 掉多少（dB）", color=INK2, fontsize=10)
    cb.ax.tick_params(labelsize=8.5, colors=INK2)
    ax.set_title("承重圖：把每塊 5×5 區域輪流遮掉、HFSS 實測性能掉幅\n越紅＝越承重、動不得；近白＝可自由變異區（—＝空塊）",
                 color=INK, fontsize=12, pad=12)
    fig.subplots_adjust(left=0.03, right=0.92, top=0.88, bottom=0.03)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
