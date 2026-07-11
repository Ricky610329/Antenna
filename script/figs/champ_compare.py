# -*- coding: utf-8 -*-
"""champ_compare.py — 新舊冠軍對比圖（通用;紀錄易主收檔時渲染進 round 檔 assets）。
用法: python -m script.figs.champ_compare --new <id> --old <id> --out docs/log/assets/round-NN/x.png \
      --title "..." [--metric "wm +0.39 vs +0.35"]
id 自動定位:pattern 掃全部 *_input 夾、響應/rad 從對應 store 撈。左=兩 pattern(橘=差異像素),右=三標曲線疊圖。"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    INK, INK2, GRID, SURF, RED, DBLUE, AQUA, ORANGE, plt, style_ax)
from matplotlib.colors import ListedColormap  # noqa: E402

from antenna.utils import config as _config, DATASET_PATH  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402
from antenna.utils.store import SampleStore  # noqa: E402

FEED = (24, 12)


def _locate(pid):
    """回 (pattern bool25x25, store 名)。掃全部輸入夾找 pid;store=夾名去 _input。"""
    for fol in sorted(os.listdir(str(DATASET_PATH))):
        if not fol.endswith("_input"):
            continue
        f = DATASET_PATH.joinpath(fol, pid + ".pt")
        if f.exists():
            p = np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5
            return p, fol[:-6]
    raise SystemExit(f"找不到 {pid} 的 pattern")


def _resp_rad(pid, pat, store):
    """從 store 撈響應（pattern 比對）與 rad（rad/<pid>.pt）;store 沒有就掃其他 store。"""
    def _try(st):
        d = DATASET_PATH.joinpath(st)
        if not d.joinpath("results.json").exists():
            return None, None
        s = SampleStore(d, verbose=False)
        for i in range(len(s)):
            x, y = s[i]
            if ((np.asarray(x).reshape(25, 25) > 0.5) == pat).all():
                rf = d.joinpath("rad", pid + ".pt")
                rad = torch.load(str(rf), weights_only=True) if rf.exists() else None
                if rad is None and d.joinpath("rad").is_dir():   # 公證批的 rad 檔名=公證 id,退而求其次比對任一
                    for g in os.listdir(str(d.joinpath("rad"))):
                        if pid in g:
                            rad = torch.load(str(d.joinpath("rad", g)), weights_only=True)
                            break
                return np.asarray(y).reshape(2, -1), rad
        return None, None
    resp, rad = _try(store)
    if resp is None:
        for fol in sorted(os.listdir(str(DATASET_PATH))):
            if fol.endswith("_input") or fol.endswith("_src") or not fol.startswith("dedust_"):
                continue
            resp, rad = _try(fol)
            if resp is not None:
                break
    if resp is None:
        raise SystemExit(f"{pid} 無響應")
    return resp, rad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="新舊冠軍對比")
    ap.add_argument("--label-new", default=None, dest="label_new")
    ap.add_argument("--label-old", default=None, dest="label_old")
    args = ap.parse_args()
    pn, stn = _locate(args.new)
    po, sto = _locate(args.old)
    rn, radn = _resp_rad(args.new, pn, stn)
    ro, rado = _resp_rad(args.old, po, sto)
    freq = 26.5 + (np.arange(rn.shape[1]) - 5) * 0.5
    ln = args.label_new or f"{args.new}（新）"
    lo = args.label_old or f"{args.old}（舊）"

    fig = plt.figure(figsize=(13.2, 6.4))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.35, 1.35], hspace=0.42, wspace=0.34)
    for col, (p, base, lab, c) in enumerate(((pn, po, ln, INK), (po, None, lo, INK2))):
        ax = fig.add_subplot(gs[:, col])
        img = p.astype(int)
        if base is not None:
            img[p != base] = 2
        ax.imshow(img, cmap=ListedColormap([SURF, DBLUE, ORANGE]), vmin=0, vmax=2,
                  origin="upper", interpolation="nearest")
        ax.scatter([FEED[1]], [FEED[0]], marker="^", s=44, color=AQUA, zorder=5, edgecolor=SURF, lw=0.8)
        d = int((pn != po).sum())
        ax.set_title(f"{lab}" + (f"\n（橘＝相對舊王差異 {d}px）" if base is not None else ""),
                     color=c, fontsize=9.6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    for (gr, idx, spec, nm, low) in ((gs[0, 2], 0, -10, "S11", True), (gs[0, 3], 1, 4, "Gain", False)):
        ax = fig.add_subplot(gr)
        n = rn.shape[1]
        if n >= 17:
            ax.axvspan(26.5, 29.5, color=GRID, alpha=0.45)
        ax.axhline(spec, color=RED, ls=":", lw=1.3)
        ax.plot(freq, ro[idx], color="#9fb4d4", lw=1.8, label=lo)
        ax.plot(freq, rn[idx], color=DBLUE, lw=2.3, label=ln)
        style_ax(ax, "頻率 (GHz)", f"{nm} (dB)", nm, tfs=10.5)
        if idx == 0:
            ax.legend(fontsize=8.2, loc="lower left", framealpha=0.94).get_frame().set_edgecolor(GRID)
    for (gr, cut) in ((gs[1, 2], "phi0"), (gs[1, 3], "phi90")):
        ax = fig.add_subplot(gr)
        ax.axvspan(-45, 45, color=GRID, alpha=0.45)
        for rad, c, lw in ((rado, "#9fb4d4", 1.8), (radn, DBLUE, 2.3)):
            if rad is not None and rad.get(cut) is not None:
                ax.plot(np.asarray(rad["theta"]), np.asarray(rad[cut]), color=c, lw=lw)
        ax.set_xlim(-90, 90); ax.set_xticks([-90, -45, 0, 45, 90])
        style_ax(ax, "θ (deg)", "Gain (dB)", f"Radiation {cut} — 灰帶＝±45° 窗", tfs=10.5)
    fig.suptitle(args.title, color=INK, fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight", facecolor=SURF)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
