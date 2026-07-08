# -*- coding: utf-8 -*-
"""report_newking.py — 新王 c25 vs 前王 c21（docs/report/assets/newking.png）。
左:兩者 pattern（橘=c25 相對 c21 的差異=加的翼對塊）;右:S11/Gain/rad 四曲線疊圖。
c25=組數階梯 5 塊翼對(R11 ref3),c21=SM 導引 3 塊(R10)。資料:ref3 / ref2v store。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    INK, INK2, GRID, SURF, RED, DBLUE, AQUA, ORANGE, plt, style_ax, save)
from matplotlib.colors import ListedColormap  # noqa: E402

from antenna.utils import config as _config, DATASET_PATH  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402
from antenna.utils.store import SampleStore  # noqa: E402

FEED = (24, 12)


def loadp(folder, pid):
    return np.asarray(torch.load(str(DATASET_PATH.joinpath(folder, f"{pid}.pt")), weights_only=True)).reshape(25, 25) > 0.5


def resp_of(store, pat):
    s = SampleStore(DATASET_PATH.joinpath(store), verbose=False)
    for i in range(len(s)):
        x, y = s[i]
        if ((np.asarray(x).reshape(25, 25) > 0.5) == pat).all():
            return np.asarray(y).reshape(2, -1)
    raise SystemExit(f"{store} 無響應")


def main():
    c25 = loadp("dedust_ref3_input", "c25_a15w10_2_22")
    c21 = loadp("dedust_ref2_input", "c21_sm")
    a15 = loadp("dedust_ref2_input", "a15_k4")           # c25 的母體(加翼對前)
    r25, r21 = resp_of("dedust_ref3", c25), resp_of("dedust_ref2v", c21)
    rad25 = torch.load(str(DATASET_PATH.joinpath("dedust_ref3", "rad", "c25_a15w10_2_22.pt")), weights_only=True)
    rad21 = torch.load(str(DATASET_PATH.joinpath("dedust_ref2v", "rad", "c21_sm.pt")), weights_only=True)
    freq = 26.5 + (np.arange(r25.shape[1]) - 5) * 0.5

    fig = plt.figure(figsize=(13.2, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.35, 1.35], height_ratios=[1, 1],
                          hspace=0.42, wspace=0.34)

    # 左:兩 pattern（c25 標橘=相對 c21 的差異塊）
    axp = fig.add_subplot(gs[:, 0])
    img = c25.astype(int)
    img[c25 != a15] = 2                                   # 橘=相對母體 a15 加的翼對(組數階梯)
    axp.imshow(img, cmap=ListedColormap([SURF, DBLUE, ORANGE]), vmin=0, vmax=2, origin="upper", interpolation="nearest")
    axp.scatter([FEED[1]], [FEED[0]], marker="^", s=52, color=AQUA, zorder=5, edgecolor=SURF, lw=0.9)
    axp.set_title("新王 c25（橘＝相對母體 a15 加的翼對）\n5 組件 · wm +0.22 · rad +0.34", color=INK, fontsize=10)
    axp.set_xticks([]); axp.set_yticks([])
    for s in axp.spines.values():
        s.set_color(GRID)
    axc = fig.add_subplot(gs[:, 1])
    axc.imshow(c21.astype(int), cmap=ListedColormap([SURF, "#9fb4d4"]), vmin=0, vmax=1, origin="upper", interpolation="nearest")
    axc.scatter([FEED[1]], [FEED[0]], marker="^", s=52, color=AQUA, zorder=5, edgecolor=SURF, lw=0.9)
    axc.set_title("前王 c21\n3 組件 · wm +0.20 · rad +0.12", color=INK2, fontsize=10.5)
    axc.set_xticks([]); axc.set_yticks([])
    for s in axc.spines.values():
        s.set_color(GRID)

    # 右:四曲線
    for (gr, idx, spec, nm, low) in ((gs[0, 2], 0, -10, "S11", True),
                                     (gs[0, 3], 1, 4, "Gain", False)):
        ax = fig.add_subplot(gr)
        ax.axvspan(26.5, 29.5, color=GRID, alpha=0.45)
        ax.axhline(spec, color=RED, ls=":", lw=1.3)
        ax.plot(freq, r21[idx], color="#9fb4d4", lw=1.8, label="c21（前王）")
        ax.plot(freq, r25[idx], color=DBLUE, lw=2.3, label="c25（新王）")
        band = r25[idx][5:12]
        m = (spec - band.max()) if low else (band.min() - spec)
        style_ax(ax, "頻率 (GHz)", f"{nm} (dB)", f"{nm}（spec {'≤−10' if low else '≥+4'}, c25 margin {m:+.2f}）", tfs=10)
        if idx == 0:
            ax.legend(fontsize=8.6, loc="lower left", framealpha=0.94).get_frame().set_edgecolor(GRID)
    for (gr, cut) in ((gs[1, 2], "phi0"), (gs[1, 3], "phi90")):
        ax = fig.add_subplot(gr)
        ax.axvspan(-45, 45, color=GRID, alpha=0.45)
        ax.plot(np.asarray(rad21["theta"]), np.asarray(rad21[cut]), color="#9fb4d4", lw=1.8)
        ax.plot(np.asarray(rad25["theta"]), np.asarray(rad25[cut]), color=DBLUE, lw=2.3)
        ax.set_xlim(-90, 90); ax.set_xticks([-90, -45, 0, 45, 90])
        style_ax(ax, "θ (deg)", "Gain (dB)", f"Radiation {cut} — 灰帶＝±45° 窗", tfs=10.5)
    fig.suptitle("目前最佳 pattern：新王 c25（組數階梯 5 塊）vs 前王 c21（SM 導引 3 塊）——加一對翼把 rad +0.12→+0.34",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "newking.png")


if __name__ == "__main__":
    main()
