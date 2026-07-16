# -*- coding: utf-8 -*-
"""report_champion.py — 報告用：單一設計的規格卡 [pattern | S11 | Gain | rad 極座標]。
id 定位/響應/rad 沿用 champ_compare（掃 *_input 夾＋store 撈）；餘裕從 store results.json 讀。
用法: python -m script.figs.report_champion --id <完整id> --title "..." [--sub "..."] --out <path>"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    GRID, INK, INK2, RED, SURF, DBLUE, GREEN, plt, style_ax, show_pattern, polar_rad_ax)
from script.figs.champ_compare import _locate, _resp_rad  # noqa: E402
from antenna.utils import DATASET_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--sub", default="", help="圖下方小字（可選）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pat, store = _locate(args.id)
    resp, rad = _resp_rad(args.id, pat, store)
    freq = 26.5 + (np.arange(resp.shape[1]) - 5) * 0.5
    res = json.load(open(str(DATASET_PATH.joinpath(store, "results.json")), encoding="utf-8"))
    e = res.get(args.id, {})
    rad_m = e.get("rad_margin")

    fig = plt.figure(figsize=(13.6, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.25, 1.25, 1.45],
                          left=0.035, right=0.985, top=0.80, bottom=0.16, wspace=0.3)
    ax = fig.add_subplot(gs[0])
    show_pattern(ax, pat, "pattern", tfs=10)
    for c, (idx, spec_y, lab, low) in enumerate(((0, -10, "S11 (dB)", True), (1, 4, "Gain (dB)", False))):
        ax = fig.add_subplot(gs[1 + c])
        ax.axvspan(26.5, 29.5, color=GRID, alpha=0.45)
        ax.plot(freq, resp[idx], color=DBLUE, lw=1.9)
        ax.axhline(spec_y, color=RED, ls=":", lw=1.3)
        band = resp[idx][5:12]
        mg = (spec_y - band.max()) if low else (band.min() - spec_y)
        style_ax(ax, "頻率 (GHz)", lab, f"帶內 {lab.split(' ')[0]} 餘裕 {mg:+.2f} ✓", tfs=9.8)
    axp = fig.add_subplot(gs[3], projection="polar")
    if rad is not None:
        th = np.asarray(rad["theta"])
        bi = int(np.abs(th).argmin())
        g0 = max(float(np.asarray(rad[c])[bi]) for c in ("phi0", "phi90"))
        polar_rad_ax(axp, th, [(rad["phi0"], DBLUE, "φ=0°", 1.7),
                               (rad["phi90"], GREEN, "φ=90°", 1.7)],
                     window=45, floor_db=3, g0_ref=g0)
        rlab = f"radiation 餘裕 {rad_m:+.2f} ✓" if isinstance(rad_m, (int, float)) else "radiation pattern"
        axp.set_title(rlab, color=INK, fontsize=9.8, pad=8)
    if args.title:
        fig.suptitle(args.title, color=INK, fontsize=13, y=0.97)
    if args.sub:
        fig.text(0.5, 0.02, args.sub, ha="center", color=INK2, fontsize=9.5)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
