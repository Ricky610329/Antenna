# -*- coding: utf-8 -*-
"""pattern 空間分布圖（Ricky 2026-07-16「越相似聚一塊,越不相似遠離」;每輪重跑=大陸漂移追蹤）：
全歷史 pattern → PCA 2D;左 panel=家族著色,右 panel=低側 lo 著色（低側戰役地圖）。
用法: python -m script.figs.pattern_map [--out tmp/pattern_map.png]
margin 王自動讀 docs/records.json;HITS（同框命中筆標注）換代時手動更新下方清單。"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from antenna.utils import DATASET_PATH
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = str(DATASET_PATH)
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default=os.path.join(REPO, "tmp", "pattern_map.png"))
_args = _ap.parse_args()
_rec = json.load(open(os.path.join(REPO, "docs", "records.json"), encoding="utf-8"))
KING_ID, KING_WM = _rec["wm"]["id"], _rec["wm"]["value"]
HITS = ("l31b2_005_lb_n09", "l31b3_019_lb_f3t07", "l32b1_023_lb_f3t07")   # 同框命中筆（換代手動更新）
DYN = ("c18", "vg0338", "vg0396", "g1_038", "g1_039", "r2_016", "r3_001")
BRIDGE = ("n09", "t03", "t07", "t09", "p00", "lb_", "brc_", "brdg")

pats, fams, ids = [], [], []
lo_map = {}
for d in os.listdir(BASE):
    rp = os.path.join(BASE, d, "results.json")
    if d.startswith("dedust_") and not d.endswith("_input") and os.path.exists(rp):
        for k, v in json.load(open(rp, encoding="utf-8")).items():
            if v.get("oob_gain_max_lo") is not None:
                lo_map[k] = v["oob_gain_max_lo"]
for fol in os.listdir(BASE):
    mp = os.path.join(BASE, fol, "manifest.json")
    if not fol.endswith("_input") or not os.path.exists(mp):
        continue
    for m in json.load(open(mp, encoding="utf-8")):
        f = os.path.join(BASE, fol, m["id"] + ".pt")
        if not os.path.exists(f):
            continue
        p = np.asarray(torch.load(f, weights_only=True)).reshape(-1) > 0.5
        pats.append(p)
        ids.append(m["id"])
        mid = m["id"] + str(m.get("source_id", ""))
        if any(t in mid for t in DYN):
            fams.append("王朝系")
        elif m.get("kind") == "denovo":
            fams.append("denovo")
        elif m.get("kind") == "selfgen":
            fams.append("selfgen")
        elif any(t in mid for t in BRIDGE):
            fams.append("中繼/碎片系")
        elif m.get("kind") == "grad":
            fams.append("G臂")
        else:
            fams.append("其他")
X = np.stack(pats).astype(np.float32)
print(f"樣本 {len(X)}")
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
Z = U[:, :2] * S[:2]
var = (S[:2] ** 2).sum() / (S ** 2).sum()
print(f"PCA 2D 解釋變異 {var:.1%}")

fams = np.array(fams)
los = np.array([lo_map.get(i, np.nan) for i in ids])
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
COL = {"王朝系": "crimson", "中繼/碎片系": "#2e8b57", "denovo": "#7b3fbf",
       "selfgen": "#c77f2e", "G臂": "#1f5fa8", "其他": "#b8b8b8"}
ax = axes[0]
for fam in ("其他", "selfgen", "G臂", "denovo", "中繼/碎片系", "王朝系"):
    m_ = fams == fam
    ax.scatter(Z[m_, 0], Z[m_, 1], s=4, alpha=0.45, c=COL[fam], label=f"{fam}（{m_.sum()}）")
king = [i for i, x in enumerate(ids) if x == KING_ID]
hit = [i for i, x in enumerate(ids) if x in HITS]
if king:
    ax.scatter(Z[king, 0], Z[king, 1], s=150, marker="*", c="gold", edgecolors="k", zorder=5, label=f"margin 王 {KING_WM:+.2f}")
if hit:
    ax.scatter(Z[hit, 0], Z[hit, 1], s=110, marker="^", c="lime", edgecolors="k", zorder=5, label="同框命中筆")
ax.legend(fontsize=8, markerscale=2)
ax.set_title(f"① pattern 空間（PCA 2D,n={len(X)},解釋變異 {var:.0%}）——家族著色", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

ax = axes[1]
has = ~np.isnan(los)
ax.scatter(Z[~has, 0], Z[~has, 1], s=3, alpha=0.15, c="#cccccc")
sc = ax.scatter(Z[has, 0], Z[has, 1], s=5, alpha=0.6, c=np.clip(los[has], -9, 5), cmap="RdYlGn_r")
plt.colorbar(sc, ax=ax, label="低側 Gain 峰 (dBi;綠=壓得深=好)")
if king:
    ax.scatter(Z[king, 0], Z[king, 1], s=150, marker="*", c="gold", edgecolors="k", zorder=5)
if hit:
    ax.scatter(Z[hit, 0], Z[hit, 1], s=110, marker="^", c="lime", edgecolors="k", zorder=5)
ax.set_title("② 同座標——低側 lo 著色（低側戰役地圖:深綠區=gap 據點）", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("pattern 空間分布（相似聚集;PCA over 25×25 像素;★=現任王 ▲=同框命中筆）", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = _args.out
fig.savefig(out, dpi=130)
print(out)
