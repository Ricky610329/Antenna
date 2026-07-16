# -*- coding: utf-8 -*-
"""report_wm_dist.py — 報告 §7（對外版）：兩個資料集的 worst-margin 分布疊圖。
學長池 24,189 筆（達標 18,0.07%）vs 新資料集 ~9.3k 筆（達標 ~22%）——「最後一哩路」framing 的主圖。
資料:tmp/expected_best/harvest_margins.npy（R6 快取）＋現場掃 NAS dedust_*（排公證夾 rNNn*/_rep）。
用法: python -m script.figs.report_wm_dist --out <path>"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    REPO, INK2, RED, SURF, DBLUE, MUTED, plt, style_ax)
from antenna.utils import DATASET_PATH  # noqa: E402


def _scan_ours():
    wms = []
    root = str(DATASET_PATH)
    for d in sorted(os.listdir(root)):
        if not d.startswith("dedust_") or d.endswith(("_input", "_src")):
            continue
        if re.search(r"r\d+n", d):          #? 公證/重測夾＝蓄意重複,排除
            continue
        rj = os.path.join(root, d, "results.json")
        if not os.path.exists(rj):
            continue
        res = json.load(open(rj, encoding="utf-8"))
        for pid, e in res.items():
            if isinstance(e, dict) and isinstance(e.get("wm"), list) and "_rep" not in pid:
                wms.append(e["wm"][2])
    return np.asarray(wms, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    senior = np.load(os.path.join(REPO, "tmp", "expected_best", "harvest_margins.npy"))
    ours = _scan_ours()
    s_pass, o_pass = int((senior >= 0).sum()), int((ours >= 0).sum())

    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    bins = np.arange(-14, 1.25, 0.25)
    ax.hist(np.clip(senior, -14, None), bins=bins, density=True, color=MUTED,
            alpha=0.55, label=f"學長歷史資料集（{len(senior):,} 筆）")
    ax.hist(ours, bins=bins, density=True, color=DBLUE, alpha=0.6,
            label=f"新資料集（{len(ours):,} 筆）")
    ax.axvline(0, color=RED, ls=":", lw=1.8)
    ax.set_yscale("log")
    #? JhengHei 對 log 刻度的 U+2212 出豆腐 → 改平文字刻度
    from matplotlib.ticker import FixedFormatter, FixedLocator
    ax.yaxis.set_major_locator(FixedLocator([1e-3, 1e-2, 1e-1, 1]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.001", "0.01", "0.1", "1"]))

    ax.text(0.15, 1.15, "→ 達標區", color=RED, fontsize=10.5, fontweight="bold")
    ax.annotate(f"學長：達標 {s_pass} 筆（{s_pass / len(senior) * 100:.2f}%）\n"
                "重測存活 8 筆，radiation 合格 0 筆",
                xy=(0.25, 3e-3), xytext=(-5.4, 8e-3), color=INK2, fontsize=10.5,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.4))
    ax.annotate(f"新資料集：達標 {o_pass:,} 筆（{o_pass / len(ours) * 100:.1f}%）\n"
                "其中 1,547 筆連 radiation pattern 一起過",
                xy=(0.35, 0.35), xytext=(-4.6, 0.55), color=DBLUE, fontsize=10.5,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=DBLUE, lw=1.6))

    style_ax(ax, "worst-margin（dB，右＝越好；紅虛線＝達標線）", "密度（log）",
             "兩個資料集的 worst-margin 分布：達標從罕見事件變成日常產出", tfs=13)
    ax.set_xlim(-14, 1.3)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.94)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.11)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF)
    print(f"→ {args.out}  senior={len(senior)} pass={s_pass} | ours={len(ours)} pass={o_pass}")


if __name__ == "__main__":
    main()
