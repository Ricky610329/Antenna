# -*- coding: utf-8 -*-
"""report_diversity.py — 報告 §7.2（對外版）：合格解的款式與變體（規格書式目錄）。
雙過解以 ≤10px 連通成「款式」；取前 N 大款式各一代表（優先挑有 rad 曲線檔者），
每款一列 [pattern | S11 | Gain | rad 極座標]。附兩兩/最近鄰漢明統計（進圖說/正文）。
資料:現場掃 NAS dedust_*（排公證夾）＋ <store>_input/<pid>.pt ＋ store 響應/rad。
用法: python -m script.figs.report_diversity --out <path> [--n 4]"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from script.figs.report_r1r10_style import (  # noqa: E402
    GRID, INK, INK2, RED, SURF, DBLUE, GREEN, plt, style_ax, show_pattern, polar_rad_ax)

from antenna.utils import config as _config, DATASET_PATH  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402
from antenna.utils.store import SampleStore  # noqa: E402


def _collect():
    """全歷史雙過解 (wm>=0 且 rad>=0)：回 (patterns NxM bool, metas)。"""
    root = str(DATASET_PATH)
    pats, metas = [], []
    for d in sorted(os.listdir(root)):
        if not d.startswith("dedust_") or d.endswith(("_input", "_src")) or re.search(r"r\d+n", d):
            continue
        rj = os.path.join(root, d, "results.json")
        if not os.path.exists(rj):
            continue
        res = json.load(open(rj, encoding="utf-8"))
        for pid, e in res.items():
            if not isinstance(e, dict) or not isinstance(e.get("wm"), list) or "_rep" in pid:
                continue
            w, rm = e["wm"][2], e.get("rad_margin")
            if w < 0 or not isinstance(rm, (int, float)) or rm < 0:
                continue
            f = os.path.join(root, d + "_input", pid + ".pt")
            if not os.path.exists(f):
                continue
            p = np.asarray(torch.load(f, weights_only=True)).reshape(-1) > 0.5
            pats.append(p)
            metas.append(dict(store=d, pid=pid, wm=w, rad=rm))
    return np.asarray(pats, bool), metas


def _families(D, t=10):
    """單鏈結連通元件（距離 ≤ t 同款）→ 依規模排序的 index 陣列 list。"""
    n = D.shape[0]
    adj = D <= t
    seen = np.zeros(n, bool)
    fams = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], [i]
        seen[i] = True
        while stack:
            j = stack.pop()
            nb = np.where(adj[j] & ~seen)[0]
            seen[nb] = True
            stack.extend(nb)
            comp.extend(nb)
        fams.append(np.asarray(comp))
    fams.sort(key=len, reverse=True)
    return fams


def _rad_file(meta):
    d = DATASET_PATH.joinpath(meta["store"], "rad", meta["pid"] + ".pt")
    return str(d) if d.exists() else None


def _resp(meta, pat):
    """從 store 撈響應（pattern 逐位比對）。"""
    s = SampleStore(DATASET_PATH.joinpath(meta["store"]), verbose=False)
    for i in range(len(s)):
        x, y = s[i]
        if ((np.asarray(x).reshape(-1) > 0.5) == pat).all():
            return np.asarray(y).reshape(2, -1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    P, metas = _collect()
    n = len(P)
    A = P.astype(np.int32)
    D = A @ (1 - A).T + (1 - A) @ A.T
    iu = np.triu_indices(n, 1)
    pair = D[iu]
    nn = np.where(np.eye(n, dtype=bool), 10**9, D).min(axis=1)
    fams = _families(D, 10)
    print(f"雙過解: {n} 筆 | 款式數(≤10px 同款): {len(fams)} | 前五大 {[len(f) for f in fams[:5]]}")
    print(f"兩兩漢明: 中位 {np.median(pair):.0f} px | 最近鄰中位 {np.median(nn):.0f} px")

    #? 每款代表：族內總距離小者優先,且必須有 rad 曲線檔
    picks = []
    for f in fams[:args.n]:
        order = f[np.argsort(D[np.ix_(f, f)].sum(axis=1))]
        rep = next((i for i in order if _rad_file(metas[i])), order[0])
        picks.append((rep, len(f)))

    rows = len(picks)
    fig = plt.figure(figsize=(13.6, 3.15 * rows))
    gs = fig.add_gridspec(rows, 4, width_ratios=[1, 1.25, 1.25, 1.45],
                          left=0.035, right=0.985, top=0.915, bottom=0.06,
                          hspace=0.52, wspace=0.3)
    for r, (i, size) in enumerate(picks):
        m = metas[i]
        ax = fig.add_subplot(gs[r, 0])
        show_pattern(ax, P[i].reshape(25, 25),
                     f"款式 {r + 1}（{size} 個變體）", tfs=10)
        resp = _resp(m, P[i])
        freq = 26.5 + (np.arange(resp.shape[1]) - 5) * 0.5
        for c, (idx, spec_y, lab, low) in enumerate(((0, -10, "S11 (dB)", True), (1, 4, "Gain (dB)", False))):
            ax = fig.add_subplot(gs[r, 1 + c])
            ax.axvspan(26.5, 29.5, color=GRID, alpha=0.45)
            ax.plot(freq, resp[idx], color=DBLUE, lw=1.9)
            ax.axhline(spec_y, color=RED, ls=":", lw=1.3)
            band = resp[idx][5:12]
            mg = (spec_y - band.max()) if low else (band.min() - spec_y)
            style_ax(ax, "頻率 (GHz)" if r == rows - 1 else "", lab,
                     f"帶內 {lab.split(' ')[0]} 餘裕 {mg:+.2f} ✓", tfs=9.6)
        axp = fig.add_subplot(gs[r, 3], projection="polar")
        rad = torch.load(_rad_file(m), weights_only=True)
        th = np.asarray(rad["theta"])
        bi = int(np.abs(th).argmin())
        g0 = max(float(np.asarray(rad[c])[bi]) for c in ("phi0", "phi90"))
        polar_rad_ax(axp, th, [(rad["phi0"], DBLUE, "φ=0°", 1.7),
                               (rad["phi90"], GREEN, "φ=90°", 1.7)],
                     window=45, floor_db=3, g0_ref=g0)
        axp.set_title(f"radiation 餘裕 +{m['rad']:.2f} ✓", color=INK, fontsize=9.6, pad=8)
    fig.suptitle(f"{n:,} 筆合格解 ＝ {len(fams)} 種基本設計（款式）——規模前 {rows} 大款式各一代表，電性全部合格",
                 color=INK, fontsize=13.5, y=0.975)
    fig.text(0.5, 0.012,
             f"同款判定＝彼此差 10 個像素以內；全部合格解兩兩中位相差 {np.median(pair):.0f} 個像素（625 個中）。"
             "灰帶＝通帶 26.5–29.5 GHz；極座標＝主波束朝上、金＝±45° 窗、紅虛圈＝峰值−3dB 門檻",
             ha="center", color=INK2, fontsize=8.8)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, facecolor=SURF)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
