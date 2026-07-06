# -*- coding: utf-8 -*-
"""r8_pattern_gallery.py — 除塵前後 pattern 實例圖（金屬=藍、被拔散點=橘、feed=綠標）。"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = r"C:\Users\Ricky\Documents\GitHub\Antenna"
sys.path.insert(0, REPO)
from antenna.utils import config as _config, DATASET_PATH  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402

ASSETS = os.path.join(REPO, "docs", "log", "assets", "round-08")
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"],
                     "axes.unicode_minus": False})
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
FEED = (24, 12)

r8_in = DATASET_PATH.joinpath("dedust_r8_input")
r7_in = DATASET_PATH.joinpath("dedust_r7_input")
res8 = json.load(open(str(DATASET_PATH.joinpath("dedust_r8", "results.json")), encoding="utf-8"))
res7 = json.load(open(str(DATASET_PATH.joinpath("dedust_r7", "results.json")), encoding="utf-8"))


def load(indir, pid):
    return np.asarray(torch.load(str(indir.joinpath(f"{pid}.pt")), weights_only=True)).reshape(25, 25) > 0.5


# (輸入夾, 家族id, 結果dict, 說明)
EXAMPLES = [
    (r8_in, "a00", res8, "前緣最佳"),
    (r8_in, "a01", res8, "拔最多 (25px)"),
    (r8_in, "a06", res8, "代價最痛"),
    (r8_in, "a05", res8, "近零代價"),
    (r8_in, "a11", res8, "唯一明顯變好"),
    (r7_in, "p03", res7, "R7 整塊型特例"),
]

cmap = ListedColormap([SURF, "#1c5cab", "#eb6834"])   # 0=介質 1=保留金屬 2=被拔散點
fig, axes = plt.subplots(2, 3, figsize=(10.6, 8.3), facecolor=SURF)
for ax, (indir, fam, res, note) in zip(axes.flat, EXAMPLES):
    orig = load(indir, f"{fam}_orig")
    d3 = load(indir, f"{fam}_d3")
    removed = orig & ~d3
    img = orig.astype(int)
    img[removed] = 2
    ax.imshow(img, cmap=cmap, vmin=0, vmax=2, origin="upper", interpolation="nearest")
    ax.scatter([FEED[1]], [FEED[0]], marker="^", s=70, color="#0e7a55", zorder=5,
               edgecolor=SURF, linewidth=0.8)
    wo, wd = res[f"{fam}_orig"]["wm"][2], res[f"{fam}_d3"]["wm"][2]
    ax.set_title(f"{fam} — {note}\n原樣 {wo:+.2f} → 除塵 {wd:+.2f}（Δ {wd - wo:+.2f}，拔 {int(removed.sum())}px）",
                 color=INK, fontsize=10.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
fig.legend(handles=[Patch(color="#1c5cab", label="保留金屬"),
                    Patch(color="#eb6834", label="被拔散點（<4px 碎片，d3）"),
                    plt.Line2D([], [], marker="^", ls="", color="#0e7a55", markersize=9, label="feed 像素")],
           loc="lower center", ncol=3, fontsize=10, frameon=False)
fig.suptitle("除塵前後實例 — 橘色＝被 strip_small(4) 拔掉的散點（worst-margin 單位 dB）",
             color=INK, fontsize=13)
fig.tight_layout(rect=[0, 0.05, 1, 1], h_pad=2.6)
out = os.path.join(ASSETS, "fig_e_patterns.png")
fig.savefig(out, dpi=140, facecolor=SURF)
print("→", out)
