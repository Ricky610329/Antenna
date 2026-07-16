# -*- coding: utf-8 -*-
"""report_rad_need.py — 報告 §3.4（對外版）：為什麼需要方向圖判準。
兩個帶內都達標的設計並排：一個方向圖不合格（rad −4.63）、一個合格（rad 王 +1.00）。
pattern 素色呈現（不做差異渲染）、性能差異全由極座標方向圖對比。
用法: python -m script.figs.report_rad_need --out <path>"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    INK, INK2, SURF, DBLUE, GREEN, RED, plt, polar_rad_ax, show_pattern)
from script.figs.champ_compare import _locate, _resp_rad  # noqa: E402

CASES = [
    ("b28b2_010_t07h", "設計 A", "帶內達標（餘裕 +0.35 dB）",
     "radiation pattern：不合格 ✗", "窗內最低點比門檻低 4.6 dB"),
    ("o23b1_007_k8_042_k7_00", "設計 B", "帶內達標（餘裕 +0.09 dB）",
     "radiation pattern：合格 ✓", "窗內全程高於門檻、餘裕 +1.0 dB"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fig = plt.figure(figsize=(14.2, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1.5, 1, 1.5],
                          left=0.03, right=0.985, top=0.76, bottom=0.14, wspace=0.32)
    for i, (pid, name, wm_lab, rad_lab, rad_sub) in enumerate(CASES):
        pat, store = _locate(pid)
        _resp, rad = _resp_rad(pid, pat, store)
        ax = fig.add_subplot(gs[i * 2])
        show_pattern(ax, pat, f"{name}\n{wm_lab}", tfs=10)
        axp = fig.add_subplot(gs[i * 2 + 1], projection="polar")
        th = np.asarray(rad["theta"])
        bi = int(np.abs(th).argmin())
        g0 = max(float(np.asarray(rad[c])[bi]) for c in ("phi0", "phi90"))
        series = [(rad["phi0"], DBLUE, "φ=0°", 2.0), (rad["phi90"], GREEN, "φ=90°", 2.0)]
        polar_rad_ax(axp, th, series, window=45, floor_db=3, g0_ref=g0)
        ok = rad_lab.endswith("✓")
        axp.set_title(f"{rad_lab}\n{rad_sub}", color=(GREEN if ok else RED),
                      fontsize=10.5, pad=10)
        axp.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.17),
                   ncol=2, framealpha=0.9)
    fig.suptitle("兩個帶內都達標的設計——radiation pattern 一個不合格、一個合格；帶內指標看不出這件事",
                 color=INK, fontsize=13, y=0.96)
    fig.text(0.5, 0.025,
             "極座標：主波束朝上·半徑＝Gain(dB)·金＝±45° 覆蓋窗·紅虛圈＝主波束−3dB 門檻（窗內每個角度都須高於此圈）",
             ha="center", color=INK2, fontsize=8.8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
