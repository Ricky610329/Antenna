# -*- coding: utf-8 -*-
"""wm 地形圖（Ricky 2026-07-17「基於分布圖整理一個 wm 分布的地形圖」）：
PCA 2D（同 pattern_map 基底）× wm 當高度 → 等高地形。
海平面隱喻:wm=0（三標線）=海平面,浮出水面=可用解陸地,王=最高峰。
插值誠實條款:距最近資料點太遠的網格 mask 留白（不做偽地形）——海峽空虛直接可見。"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from antenna.utils import DATASET_PATH
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = str(DATASET_PATH)
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default=os.path.join(REPO, "tmp", "wm_terrain.png"))
_args = _ap.parse_args()
_rec = json.load(open(os.path.join(REPO, "docs", "records.json"), encoding="utf-8"))
KING_ID, KING_WM = _rec["wm"]["id"], _rec["wm"]["value"]
HITS = ("l32b3_018_lb_f3t07", "l31b2_005_lb_n09", "l31b3_019_lb_f3t07")   # 同框命中筆（l32b3_018=wm 首過 0;換代手動更新）
XO_PREFIX = "x32b"                                                        # 海峽臂標注（換輪手動更新）
DYN = ("c18", "vg0338", "vg0396", "g1_038", "g1_039", "r2_016", "r3_001")

wm_map = {}
for d in os.listdir(BASE):
    rp = os.path.join(BASE, d, "results.json")
    if d.startswith("dedust_") and not d.endswith("_input") and os.path.exists(rp):
        for k, v in json.load(open(rp, encoding="utf-8")).items():
            if isinstance(v.get("wm"), list):
                wm_map[k] = v["wm"][2]

pats, ids, fams = [], [], []
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
        fams.append("dyn" if any(t in m["id"] + str(m.get("source_id", "")) for t in DYN) else "o")
X = np.stack(pats).astype(np.float32)
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
Z = U[:, :2] * S[:2]
var = (S[:2] ** 2).sum() / (S ** 2).sum()
wms = np.array([wm_map.get(i, np.nan) for i in ids])
has = ~np.isnan(wms)
print(f"樣本 {len(X)}（有 wm {int(has.sum())}）;PCA 解釋變異 {var:.1%}")

Zh, Wh = Z[has], np.clip(wms[has], -12, 1)
gx = np.linspace(Z[:, 0].min(), Z[:, 0].max(), 260)
gy = np.linspace(Z[:, 1].min(), Z[:, 1].max(), 260)
GX, GY = np.meshgrid(gx, gy)
#? 地形=網格 KNN 局部最大值的平滑（「這一帶最好能到多少」比平均更符合搜尋語義:
#  地形高度=該地帶的潛力上限;平均會被大量废樣本拖沉高地）
tree = cKDTree(Zh)
dist, idx = tree.query(np.c_[GX.ravel(), GY.ravel()], k=8)
H = np.take(Wh, idx).max(axis=1).reshape(GX.shape)
H = gaussian_filter(H, sigma=2.2)
#? 誠實條款:最近資料點 > 閾值的網格=無資料支撐 → mask 留白
span = max(gx[-1] - gx[0], gy[-1] - gy[0])
mask = dist[:, 0].reshape(GX.shape) > span * 0.035
Hm = np.ma.masked_where(mask, H)

fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
ax = axes[0]
levels = np.arange(-12, 1.25, 0.75)
cf = ax.contourf(GX, GY, Hm, levels=levels, cmap="terrain", alpha=0.92)
ax.contour(GX, GY, Hm, levels=[0.0], colors="k", linewidths=2.0)      # 海平面=三標線
ax.contour(GX, GY, Hm, levels=[-3.0], colors="k", linewidths=0.6, linestyles=":")
plt.colorbar(cf, ax=ax, label="局部 wm 潛力上限 (dB;黑實線=海平面 wm=0)")
ax.scatter(Zh[:, 0], Zh[:, 1], s=1.2, c="k", alpha=0.10)
king = [i for i, x in enumerate(ids) if x == KING_ID]
hit = [i for i, x in enumerate(ids) if x in HITS]
xo = [i for i, x in enumerate(ids) if x.startswith(XO_PREFIX)]
if king:
    ax.scatter(Z[king, 0], Z[king, 1], s=170, marker="*", c="gold", edgecolors="k", zorder=6, label=f"margin 王 {KING_WM:+.2f}（最高峰）")
if hit:
    ax.scatter(Z[hit, 0], Z[hit, 1], s=100, marker="^", c="lime", edgecolors="k", zorder=6, label="同框命中筆（中繼高地）")
if xo:
    ax.scatter(Z[xo, 0], Z[xo, 1], s=42, marker="D", c="magenta", edgecolors="k", zorder=5, label="X 海峽臂 b1（24 筆填海）")
ax.legend(fontsize=9, loc="lower right")
ax.set_title(f"① wm 地形（n={int(has.sum())};高度=局部潛力上限 KNN8-max;無資料區留白）", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

ax = axes[1]
sel = Wh >= -3
cf2 = ax.scatter(Zh[sel, 0], Zh[sel, 1], s=14, c=Wh[sel], cmap="RdYlGn", vmin=-3, vmax=0.6, alpha=0.85)
ax.scatter(Zh[~sel, 0], Zh[~sel, 1], s=1, c="#dddddd", alpha=0.25)
plt.colorbar(cf2, ax=ax, label="realized wm (dB)")
if king:
    ax.scatter(Z[king, 0], Z[king, 1], s=170, marker="*", c="gold", edgecolors="k", zorder=6)
if hit:
    ax.scatter(Z[hit, 0], Z[hit, 1], s=100, marker="^", c="lime", edgecolors="k", zorder=6)
if xo:
    ax.scatter(Z[xo, 0], Z[xo, 1], s=42, marker="D", c="none", edgecolors="magenta", zorder=5)
ax.set_title("② 作戰區散點（wm>=-3 上色;灰=其餘;◇=X 臂落點）", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("wm 地形圖——海平面=三標線 wm=0;★=王(最高峰) ▲=同框筆 ◇=X 海峽臂", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = _args.out
fig.savefig(out, dpi=130)
print(out)
