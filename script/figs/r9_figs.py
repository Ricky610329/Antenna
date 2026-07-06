# -*- coding: utf-8 -*-
"""r9_figs.py — R9 歸檔四圖（assets/round-09/）：校正散點 / 探索全景 / SM 體檢 / 冠軍 pattern。"""
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

ASSETS = os.path.join(REPO, "docs", "log", "assets", "round-09")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, DBLUE, LBLUE = "#2a78d6", "#1c5cab", "#86b6ef"
AQUA, ORANGE, VIOLET, RED = "#1baf7a", "#eb6834", "#4a3aa7", "#d03b3b"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"],
                     "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})

man = {m["id"]: m for m in json.load(open(str(DATASET_PATH.joinpath("dedust_r9_input", "manifest.json")), encoding="utf-8"))}
res = json.load(open(str(DATASET_PATH.joinpath("dedust_r9", "results.json")), encoding="utf-8"))
r8m = {m["id"]: m for m in json.load(open(str(DATASET_PATH.joinpath("dedust_r8_input", "manifest.json")), encoding="utf-8"))}
r8 = json.load(open(str(DATASET_PATH.joinpath("dedust_r8", "results.json")), encoding="utf-8"))
ok = {i for i in res if "wm" in res[i]}
W = lambda i: res[i]["wm"][2]


def style(ax, xl="", yl="", ti=""):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.7, alpha=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel(xl, color=INK2)
    ax.set_ylabel(yl, color=INK2)
    if ti:
        ax.set_title(ti, color=INK, fontsize=11.5)


# ==== 圖 1：校正散點 ====
fig, ax = plt.subplots(figsize=(7.6, 7), facecolor=SURF)
lim = (-3.6, 1.2)
ax.plot(lim, lim, ls="--", color=INK2, lw=1.3)
ax.axhline(0, color=RED, ls=":", lw=1.1)
ax.axvline(0, color=RED, ls=":", lw=1.1)
groups = {"t": (DBLUE, "o", "T 帳面達標"), "n": (BLUE, "s", "N 近標 [-1,0)"), "m": (LBLUE, "^", "M 深帶 [-3,-1)")}
pv, cv = [], []
for pre, (c, mk, lab) in groups.items():
    ids = [i for i in ok if i.startswith(pre) and i[1].isdigit()]
    x = [man[i]["pool_wm"][2] for i in ids]
    y = [W(i) for i in ids]
    pv += x; cv += y
    ax.scatter(x, y, s=42, color=c, marker=mk, label=lab, zorder=4, edgecolor=SURF, lw=0.5)
# R8 A 臂疊圖（另一家族的漂移對照）
ax8 = [(r8m[f"a{k:02d}_orig"]["pool_wm"][2], r8[f"a{k:02d}_orig"]["wm"][2]) for k in range(15)]
ax.scatter([a for a, _ in ax8], [b for _, b in ax8], s=36, color=MUTED, marker="x",
           label="R8 A 臂（乾淨版型家族,漂移偏下）", zorder=3)
(b0, b1), *_ = np.linalg.lstsq(np.vstack([np.ones(len(pv)), pv]).T, cv, rcond=None)
xs = np.linspace(*lim, 10)
ax.plot(xs, b0 + b1 * xs, color=DBLUE, lw=1.6,
        label=f"R9 fit: 現行 = {b0:+.2f} + {b1:.2f}·池值 (殘差σ 0.77)")
ax.annotate("帳面達標且現行達標\n8/18 (+n09)", (0.42, 0.6), color=AQUA, fontsize=9.5, ha="right")
ax.set_xlim(lim)
ax.set_ylim(-4.6, 1.2)
style(ax, "池記錄值 worst-margin (dB)", "現行 HFSS worst-margin (dB)",
      "校正曲線：漂移非全池統一（頂帶 ±0.4 可信、深帶偏 −0.4、乾淨版型家族偏 −0.5~−1）")
ax.legend(fontsize=8.6, loc="lower right", framealpha=0.92).get_frame().set_edgecolor(GRID)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "calibration.png"), dpi=140, facecolor=SURF)
plt.close(fig)

# ==== 圖 2：探索全景 strip ====
fams = ["F0", "F1", "F2", "F3", "F4", "F5"]
xpos = {f: i for i, f in enumerate(fams)}
fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=SURF)
rng = np.random.default_rng(0)
for i in ok:
    if i.startswith("e"):
        a = man[i]["anchor"]
        x = xpos[a] + (0 if man[i]["flip_k"] == 0 else rng.uniform(-0.22, 0.22))
        if man[i]["flip_k"] == 0:
            ax.scatter(x, W(i), s=110, color=DBLUE, marker="D", zorder=5, edgecolor=SURF, lw=0.8)
        else:
            ax.scatter(x, W(i), s=22, color=BLUE, alpha=0.55, zorder=3)
    elif i.startswith("g"):
        a = man[i]["anchor"]
        ax.scatter(xpos[a] + rng.uniform(0.26, 0.40), W(i), s=26, color=ORANGE, alpha=0.8, zorder=4)
    elif i.startswith("s"):
        a = man[i]["anchor"]
        ax.scatter(xpos[a] - 0.34, W(i), s=46, color=VIOLET, marker="s", zorder=4, edgecolor=SURF, lw=0.5)
for y, c, lab in ((0, RED, "spec 達標線"), (-1.80, INK2, "R8 乾淨前緣 −1.80"),
                  (-2.68, MUTED, "舊可製造紀錄 p03_d3 −2.68")):
    ax.axhline(y, color=c, ls=":", lw=1.2)
    ax.text(5.62, y + 0.25, lab, color=c, fontsize=8.6, ha="right")
for i, dy in (("s05_1050", 1.0), ("g15_sm", -0.4), ("e73_x16", -1.2)):
    if i in ok:
        a = man[i]["anchor"]
        xx = xpos[a] + (-0.34 if i.startswith("s") else 0.33 if i.startswith("g") else 0.1)
        ax.annotate(f"{i} {W(i):+.2f}", (xx, W(i)), xytext=(xx + 0.28, W(i) + dy),
                    color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_xticks(range(6))
ax.set_xticklabels([f"{f}\n(池 {next(man[i]['anchor_pool_wm'] for i in man if i.startswith('e') and man[i].get('anchor') == f):+.2f})"
                    for f in fams], fontsize=9.5)
style(ax, "探索錨點（池 top-300 跨家族代表）", "現行 HFSS worst-margin (dB)",
      "探索全景：◆ k=0 乾淨投影 · ● E 鄰域 · ● G SM導引(右) · ■ S 對稱(左)")
leg = [Patch(color=DBLUE, label="k=0 乾淨投影 ◆"), Patch(color=BLUE, label="E 鄰域"),
       Patch(color=ORANGE, label="G SM 導引"), Patch(color=VIOLET, label="S 對稱")]
ax.legend(handles=leg, fontsize=8.8, loc="lower left", framealpha=0.92).get_frame().set_edgecolor(GRID)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "explore_overview.png"), dpi=140, facecolor=SURF)
plt.close(fig)

# ==== 圖 3：SM 體檢（乾淨投影區 = 分布外） ====
fig, ax = plt.subplots(figsize=(7, 6.2), facecolor=SURF)
lim = (-27, 2)
ax.plot(lim, lim, ls="--", color=INK2, lw=1.3)
for pre, c, lab in (("e", BLUE, "E 鄰域"), ("g", ORANGE, "G SM 導引"), ("s", VIOLET, "S 對稱")):
    ids = [i for i in ok if i.startswith(pre) and man[i].get("sm_wm")]
    ax.scatter([W(i) for i in ids], [man[i]["sm_wm"][2] for i in ids], s=26, color=c,
               alpha=0.75, label=lab, edgecolor=SURF, lw=0.4)
ax.set_xlim(lim)
ax.set_ylim(lim)
style(ax, "HFSS 真值 (dB)", "SM 預測 (dB)",
      "SM 在乾淨投影區＝分布外：一致樂觀（G 臂中位 +4.3）但排序有訊號（G 均值贏 E 2.4dB）")
ax.legend(fontsize=9, loc="upper left", framealpha=0.92).get_frame().set_edgecolor(GRID)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "sm_check.png"), dpi=140, facecolor=SURF)
plt.close(fig)

# ==== 圖 4：冠軍 pattern gallery ====
pool = np.load(os.path.join(REPO, "tmp", "pattern_anatomy", "pool.npz"))
okp = ~np.isnan(pool["wm"][:, 2])
pats_pool = np.unpackbits(pool["packed"][okp], axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
aidx = {man[i]["anchor"]: man[i]["anchor_pool_idx"] for i in man if i.startswith("e") and man[i]["flip_k"] == 0}
r9in = DATASET_PATH.joinpath("dedust_r9_input")

def loadp(pid):
    return np.asarray(torch.load(str(r9in.joinpath(f"{pid}.pt")), weights_only=True)).reshape(25, 25) > 0.5

cells = [
    (pats_pool[aidx["F2"]], "F2 錨點（含粉塵,池 −0.01）", None),
    (loadp("s05_1050"), "s05：F2 → 10-5-10 對稱化", "wm −0.29 (S11 +0.06✓)　rad −0.91"),
    (loadp("g15_sm"), "g15：F3 鄰域·SM 導引 k=16", "wm −1.49　rad −0.54"),
    (pats_pool[aidx["F3"]], "F3 錨點（含粉塵,池 −0.14）", None),
    (loadp("e39_x0"), "e39：F3 乾淨投影 (k=0)", "wm −2.68　rad +0.24"),
    (loadp("g24_sm"), "g24：F3 鄰域·SM 導引 k=48", "wm −1.85　rad +0.44 ✓"),
]
cmap = ListedColormap([SURF, "#1c5cab"])
fig, axes = plt.subplots(2, 3, figsize=(10.6, 8.0), facecolor=SURF)
for axx, (p, t1, t2) in zip(axes.flat, cells):
    axx.imshow(p.astype(int), cmap=cmap, vmin=0, vmax=1, origin="upper", interpolation="nearest")
    axx.scatter([12], [24], marker="^", s=60, color="#0e7a55", zorder=5, edgecolor=SURF, lw=0.8)
    axx.set_title(t1 + ("\n" + t2 if t2 else ""), color=INK, fontsize=10)
    axx.set_xticks([])
    axx.set_yticks([])
    for s in axx.spines.values():
        s.set_color(GRID)
fig.suptitle("可製造紀錄一夜 −2.68 → −0.29：新冠軍與它們的錨點（全件 ≥4px、零粉塵）",
             color=INK, fontsize=12.5)
fig.tight_layout(h_pad=2.2)
fig.savefig(os.path.join(ASSETS, "champions.png"), dpi=140, facecolor=SURF)
print("4 figs →", ASSETS)
