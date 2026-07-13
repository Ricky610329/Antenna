# -*- coding: utf-8 -*-
"""report_rad_polar.py — 報告用：把指定冠軍的方向圖以「極座標」渲染（達標樣貌一目了然）。
主波束朝上、半徑=gain(dB)、phi0(E,藍)/phi90(H,綠)、金＝±45° 覆蓋窗、紅虛圈＝G0−3dB 門檻。
用法: python -m script.figs.report_rad_polar --ids o23b1_007_k8_042_k7_00,g16_r15 \
      --labels "o23b1_007（rad 王 +1.00）,g16（平衡 rad +0.49）" --out docs/report/assets/rad_polar.png
資料同 champ_compare（掃 *_input 找 pattern、對應 store 撈 rad）。"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    INK, INK2, SURF, DBLUE, GREEN, plt, polar_rad_ax)
from script.figs.champ_compare import _locate, _resp_rad  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="逗號分隔的冠軍完整 id")
    ap.add_argument("--labels", default=None, help="逗號分隔的標籤（對齊 ids）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="通過方向圖規格的設計（極座標）")
    args = ap.parse_args()
    ids = [s.strip() for s in args.ids.split(",")]
    labs = [s.strip() for s in args.labels.split(",")] if args.labels else ids

    rads = []
    for pid in ids:
        pat, store = _locate(pid)
        _resp, rad = _resp_rad(pid, pat, store)
        if rad is None:
            raise SystemExit(f"{pid} 無方向圖資料")
        rads.append(rad)

    n = len(ids)
    fig = plt.figure(figsize=(5.0 * n, 6.0))
    for i, (rad, lab) in enumerate(zip(rads, labs)):
        ax = fig.add_subplot(1, n, i + 1, projection="polar")
        th = np.asarray(rad["theta"])
        bi = int(np.abs(th).argmin())
        g0 = max(float(np.asarray(rad[c])[bi]) for c in ("phi0", "phi90") if rad.get(c) is not None)
        series = []
        if rad.get("phi0") is not None:
            series.append((rad["phi0"], DBLUE, "φ=0° (E)", 2.2))
        if rad.get("phi90") is not None:
            series.append((rad["phi90"], GREEN, "φ=90° (H)", 2.2))
        polar_rad_ax(ax, th, series, window=45, floor_db=3, g0_ref=g0)
        ax.set_title(f"{lab}　G0={g0:.1f} dB", color=INK, fontsize=10.5, pad=8)
        ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.14),
                  ncol=2, framealpha=0.9)
    fig.suptitle(args.title, color=INK, fontsize=12.5, y=0.965)
    fig.text(0.5, 0.035,
             "主波束朝上·半徑＝Gain(dB)·金＝±45° 覆蓋窗·紅虛圈＝G0−3dB 門檻（窗內每個角度都須高於此圈才達標）",
             ha="center", color=INK2, fontsize=8.6)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.82, bottom=0.15, wspace=0.30)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
