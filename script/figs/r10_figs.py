# -*- coding: utf-8 -*-
"""r10_figs.py — R10 報告三圖：w17 血統譜系 / w17 三標曲線 / 遮蔽承重熱圖。"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, TwoSlopeNorm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = r"C:\Users\Ricky\Documents\GitHub\Antenna"
sys.path.insert(0, REPO)
from antenna.utils import config as _config, DATASET_PATH  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402
from antenna.utils.store import SampleStore  # noqa: E402

ASSETS = os.path.join(REPO, "docs", "log", "assets", "round-10")
os.makedirs(ASSETS, exist_ok=True)
INK, INK2, MUTED, GRID, SURF, RED = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#d03b3b"
DBLUE, AQUA, ORANGE = "#1c5cab", "#0e7a55", "#eb6834"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"],
                     "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})

FEED = (24, 12)


def loadp(folder, pid):
    return (np.asarray(torch.load(str(DATASET_PATH.joinpath(folder, f"{pid}.pt")), weights_only=True))
            .reshape(25, 25) > 0.5)


def find_resp(store_name, pat):
    store = SampleStore(DATASET_PATH.joinpath(store_name), verbose=False)
    for i in range(len(store)):
        x, y = store[i]
        if ((np.asarray(x).reshape(25, 25) > 0.5) == pat).all():
            return np.asarray(y).reshape(2, -1)
    raise SystemExit(f"{store_name} 找不到響應")


# ==== 圖 1：血統譜系 F2 → s05 → w17 ====
pool = np.load(os.path.join(REPO, "tmp", "pattern_anatomy", "pool.npz"))
okp = ~np.isnan(pool["wm"][:, 2])
pats_pool = np.unpackbits(pool["packed"][okp], axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
r9man = {m["id"]: m for m in json.load(open(str(DATASET_PATH.joinpath("dedust_r9_input", "manifest.json")), encoding="utf-8"))}
f2_idx = next(m["anchor_pool_idx"] for m in r9man.values() if m.get("anchor") == "F2" and m.get("flip_k") == 0)
f2 = pats_pool[f2_idx]
s05 = loadp("dedust_r9_input", "s05_1050")
w17 = loadp("dedust_ref1_input", "w17_k8")

cells = [(f2, None, "① F2 錨點（含粉塵,池 −0.01）\n跨家族普查代表"),
         (s05, None, "② s05 ＝ ① 的 10-5-10 對稱化\nwm −0.29 · rad −0.91（R9 破紀錄）"),
         (w17, s05, "③ w17 ＝ ② 翻 8px 再對稱化（seed 5017）\n★ wm +0.48 · rad +0.26 三標全過")]
cmap3 = ListedColormap([SURF, "#1c5cab", "#eb6834"])
fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.5), facecolor=SURF)
for ax, (p, ref, title) in zip(axes, cells):
    img = p.astype(int)
    if ref is not None:
        img[(p != ref)] = 2                       # 相對前一代的變化像素標橘
    ax.imshow(img, cmap=cmap3, vmin=0, vmax=2, origin="upper", interpolation="nearest")
    ax.scatter([FEED[1]], [FEED[0]], marker="^", s=60, color=AQUA, zorder=5, edgecolor=SURF, lw=0.8)
    ax.set_title(title, color=INK, fontsize=9.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
fig.suptitle("w17 血統：兩步「構造式」編輯,從碎片雲錨點到三標全過（橘=相對上一代的變化像素）",
             color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "w17_lineage.png"), dpi=140, facecolor=SURF)
plt.close(fig)

# ==== 圖 2：w17 vs s05 三標曲線 ====
resp_w = find_resp("dedust_ref1", w17)
resp_s = find_resp("dedust_r9", s05)
n_pts = resp_w.shape[1]
freq = 26.5 + (np.arange(n_pts) - 5) * 0.5
rad_w = torch.load(str(DATASET_PATH.joinpath("dedust_ref1", "rad", "w17_k8.pt")), weights_only=True)
rad_s = torch.load(str(DATASET_PATH.joinpath("dedust_r9", "rad", "s05_1050.pt")), weights_only=True)

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), facecolor=SURF)


def style(ax, xl, yl, ti):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.7, alpha=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel(xl, color=INK2)
    ax.set_ylabel(yl, color=INK2)
    ax.set_title(ti, color=INK, fontsize=11.5)


ax = axes[0][0]
ax.axvspan(26.5, 29.5, color=GRID, alpha=0.4)
ax.axhline(-10, color=RED, ls=":", lw=1.3)
ax.plot(freq, resp_s[0], color=MUTED, lw=1.8, label="s05（前代,wm −0.29）")
ax.plot(freq, resp_w[0], color=DBLUE, lw=2.4, label="★ w17（wm +0.48）")
style(ax, "頻率 (GHz)", "S11 (dB)", "S11 — 帶內全程壓過 −10（margin +0.83）")
ax.legend(fontsize=8.8, framealpha=0.92).get_frame().set_edgecolor(GRID)

ax = axes[0][1]
ax.axvspan(26.5, 29.5, color=GRID, alpha=0.4)
ax.axhline(4, color=RED, ls=":", lw=1.3)
ax.plot(freq, resp_s[1], color=MUTED, lw=1.8)
ax.plot(freq, resp_w[1], color=DBLUE, lw=2.4)
style(ax, "頻率 (GHz)", "Gain (dB)", "Gain — s05 的帶緣低點被抬起（margin −0.29 → +0.48）")

for col, cut in ((0, "phi0"), (1, "phi90")):
    ax = axes[1][col]
    ax.axvspan(-45, 45, color=GRID, alpha=0.4)
    for rad, c, lw in ((rad_s, MUTED, 1.8), (rad_w, DBLUE, 2.4)):
        th = np.asarray(rad["theta"])
        g = np.asarray(rad[cut])
        ax.plot(th, g, color=c, lw=lw)
        g0 = g[np.argmin(np.abs(th))]
        ax.hlines(g0 - 3, -45, 45, color=c, ls=":", lw=1.0, alpha=0.8)
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -45, 0, 45, 90])
    style(ax, "θ (deg)", "Gain (dB)",
          f"Radiation {cut} — " + ("s05 的凹陷曾在此" if cut == "phi90" else "±45° 全程覆蓋"))
fig.suptitle("w17 vs s05 三標曲線：S11 ✓ · Gain ✓ · rad 兩切面 ✓（margin +0.48 / +0.26）",
             color=INK, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "w17_curves.png"), dpi=140, facecolor=SURF)
plt.close(fig)

# ==== 圖 3：遮蔽承重熱圖（Δwm / Δrad × s05 / g24）====
occ_man = {m["id"]: m for m in json.load(open(str(DATASET_PATH.joinpath("dedust_occl_input", "manifest.json")), encoding="utf-8"))}
occ = json.load(open(str(DATASET_PATH.joinpath("dedust_occl", "results.json")), encoding="utf-8"))
BASE = {"s05_1050": (-0.29, -0.91), "g24_sm": (-1.85, 0.44)}
div = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])

fig, axes = plt.subplots(2, 2, figsize=(10.6, 10.2), facecolor=SURF)
for row, sid in enumerate(("s05_1050", "g24_sm")):
    for col, (met, lab) in enumerate((("wm", "Δ worst-margin"), ("rad", "Δ rad 餘裕"))):
        grid = np.full((5, 5), np.nan)
        for i, m in occ_man.items():
            if m["source_id"] != sid or i not in occ or "wm" not in occ[i]:
                continue
            br, bc = m["block"]
            base = BASE[sid][0] if met == "wm" else BASE[sid][1]
            val = (occ[i]["wm"][2] if met == "wm" else occ[i]["rad_margin"]) - base
            grid[br, bc] = val
        ax = axes[row][col]
        vmax = np.nanmax(np.abs(grid))
        im = ax.imshow(grid, cmap=div, norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
                       origin="upper", interpolation="nearest")
        for br in range(5):
            for bc in range(5):
                v = grid[br, bc]
                ax.text(bc, br, "—" if np.isnan(v) else f"{v:+.1f}", ha="center", va="center",
                        color=INK, fontsize=9.5)
        ax.set_title(f"{sid.split('_')[0]} · {lab}（拔掉該 5×5 塊後 − 基準）", color=INK, fontsize=10.5)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
        fig.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(colors=MUTED, labelsize=8)
fig.suptitle("物理遮蔽掃描（真·空間重要度）：紅=拔了變差（承重）· 藍=拔了變好 · —=空塊/待補",
             color=INK, fontsize=12.5)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "occlusion_maps.png"), dpi=140, facecolor=SURF)
print("3 figs →", ASSETS)
