# -*- coding: utf-8 -*-
"""r9_champ_curves.py — 冠軍三傑 (s05/g15/g24) 的 S11 / Gain / radiation 曲線圖。"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = r"C:\Users\Ricky\Documents\GitHub\Antenna"
sys.path.insert(0, REPO)
from antenna.utils import config as _config, DATASET_PATH  # noqa: E402
_config.device = "cpu"
import torch  # noqa: E402
from antenna.utils.store import SampleStore  # noqa: E402

ASSETS = os.path.join(REPO, "docs", "log", "assets", "round-09")
INK, INK2, MUTED, GRID, SURF, RED = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#d03b3b"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"],
                     "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})

CHAMPS = [("s05_1050", "#1c5cab", "s05：F2×10-5-10 對稱（wm −0.29,rad −0.91）"),
          ("g15_sm", "#eb6834", "g15：F3·SM 導引 k=16（wm −1.49,rad −0.54）"),
          ("g24_sm", "#0e7a55", "g24：F3·SM 導引 k=48（wm −1.85,rad +0.44）")]

r9in = DATASET_PATH.joinpath("dedust_r9_input")
store = SampleStore(DATASET_PATH.joinpath("dedust_r9"), verbose=False)
res = json.load(open(str(DATASET_PATH.joinpath("dedust_r9", "results.json")), encoding="utf-8"))

# id → 響應：用 pattern 逐筆比對 store
targets = {}
for cid, _c, _l in CHAMPS:
    targets[cid] = (np.asarray(torch.load(str(r9in.joinpath(f"{cid}.pt")), weights_only=True))
                    .reshape(25, 25) > 0.5)
resp = {}
for i in range(len(store)):
    x, y = store[i]
    p = np.asarray(x).reshape(25, 25) > 0.5
    for cid, tp in targets.items():
        if cid not in resp and (p == tp).all():
            resp[cid] = np.asarray(y).reshape(2, -1)
assert set(resp) == set(targets), f"缺響應: {set(targets)-set(resp)}"
n_pts = resp[CHAMPS[0][0]].shape[1]
# 中央平台 idx 5..11 = 26.5–29.5 GHz (width [5,0,7,0,5], 0.5 GHz/點)
freq = 26.5 + (np.arange(n_pts) - 5) * 0.5

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), facecolor=SURF)


def style(ax, xl, yl, ti):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.7, alpha=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel(xl, color=INK2)
    ax.set_ylabel(yl, color=INK2)
    ax.set_title(ti, color=INK, fontsize=11.5)


# --- S11 ---
ax = axes[0][0]
ax.axvspan(26.5, 29.5, color=GRID, alpha=0.4)
ax.axhline(-10, color=RED, ls=":", lw=1.3)
ax.text(freq[0] + 0.1, -9.6, "spec：帶內 ≤ −10 dB", color=RED, fontsize=8.8)
for cid, c, lab in CHAMPS:
    ax.plot(freq, resp[cid][0], color=c, lw=2)
style(ax, "頻率 (GHz)", "S11 (dB)", "S11 — 灰帶=規格頻帶 26.5–29.5 GHz")

# --- Gain ---
ax = axes[0][1]
ax.axvspan(26.5, 29.5, color=GRID, alpha=0.4)
ax.axhline(4, color=RED, ls=":", lw=1.3)
ax.text(freq[0] + 0.1, 4.15, "spec：帶內 ≥ +4 dB", color=RED, fontsize=8.8)
for cid, c, lab in CHAMPS:
    ax.plot(freq, resp[cid][1], color=c, lw=2)
style(ax, "頻率 (GHz)", "Gain (dB)", "Gain — s05 只差帶內最低點 0.29 dB")

# --- rad phi0 / phi90 ---
for col, cut in ((0, "phi0"), (1, "phi90")):
    ax = axes[1][col]
    ax.axvspan(-45, 45, color=GRID, alpha=0.4)
    for cid, c, lab in CHAMPS:
        rad = torch.load(str(DATASET_PATH.joinpath("dedust_r9", "rad", f"{cid}.pt")), weights_only=True)
        th = np.asarray(rad["theta"])
        g = np.asarray(rad[cut])
        ax.plot(th, g, color=c, lw=2, label=lab if col == 0 else None)
        g0 = g[np.argmin(np.abs(th))]
        ax.hlines(g0 - 3, -45, 45, color=c, ls=":", lw=1.1, alpha=0.8)
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -45, 0, 45, 90])
    style(ax, "θ (deg)", "Gain (dB)",
          f"Radiation {cut} 切面 — 灰帶=±45° 窗,虛線=各自 G(0)−3dB 門檻")
axes[1][0].legend(fontsize=8.8, loc="lower center", framealpha=0.92).get_frame().set_edgecolor(GRID)

fig.suptitle("冠軍三傑的三標曲線：S11 ✓(s05) · Gain 差 0.29(s05) · rad ✓(g24)", color=INK, fontsize=13)
fig.tight_layout()
out = os.path.join(ASSETS, "champions_curves.png")
fig.savefig(out, dpi=140, facecolor=SURF)
print("→", out)
