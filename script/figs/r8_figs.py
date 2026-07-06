# -*- coding: utf-8 -*-
"""r8_figs.py — R8 報告四張圖（落 docs/log/assets/round-08/）。palette=dataviz reference。"""
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

ASSETS = os.path.join(REPO, "docs", "log", "assets", "round-08")
os.makedirs(ASSETS, exist_ok=True)

BLUE, DBLUE, AQUA = "#2a78d6", "#1c5cab", "#1baf7a"
ORANGE, RED, VIOLET = "#eb6834", "#d03b3b", "#4a3aa7"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

man = {m["id"]: m for m in json.load(open(str(DATASET_PATH.joinpath("dedust_r8_input", "manifest.json")), encoding="utf-8"))}
res = json.load(open(str(DATASET_PATH.joinpath("dedust_r8", "results.json")), encoding="utf-8"))
W = lambda i: res[i]["wm"][2]
G = lambda i: res[i]["wm"][1]
R = lambda i: res[i]["rad_margin"]

plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"],
                     "axes.unicode_minus": False, "mathtext.fontset": "dejavusans"})


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.7, alpha=0.8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5)


# ---- 圖 A：漂移 dumbbell + 除塵 Δ bar ----
fams = [f"a{f:02d}" for f in range(15)]
pool_v = [man[f + "_orig"]["pool_wm"][2] for f in fams]
hfss_v = [W(f + "_orig") for f in fams]
dd = [W(f + "_d3") - W(f + "_orig") for f in fams]
y = np.arange(15)[::-1]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), facecolor=SURF)
ax = axes[0]
for yy, pv, hv in zip(y, pool_v, hfss_v):
    ax.plot([pv, hv], [yy, yy], color=GRID, lw=1.6, zorder=1)
ax.scatter(pool_v, y, s=34, color=MUTED, zorder=3, label="池記錄值 (學長當年設定)")
ax.scatter(hfss_v, y, s=38, color=DBLUE, zorder=3, label="HFSS 重跑 (現行設定)")
ax.set_yticks(y)
ax.set_yticklabels(fams, fontsize=8.5)
ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9).get_frame().set_edgecolor(GRID)
style(ax, "worst-margin (dB)", "", "池值 → 現行 HFSS 重跑：14/15 向下（中位 −0.52）")

ax = axes[1]
colors = [AQUA if v > 0 else ORANGE for v in dd]
ax.barh(y, dd, height=0.62, color=colors, zorder=2)
ax.axvspan(-0.5, 0.5, color=GRID, alpha=0.45, zorder=1)
ax.axvline(0, color=INK2, lw=1)
ax.text(-0.48, 14.5, "判準帶 |Δ|<0.5", color=INK2, fontsize=8.5, va="top")
ax.set_yticks(y)
ax.set_yticklabels(fams, fontsize=8.5)
style(ax, "Δ worst-margin：除塵後 − 原樣 (dB)", "", "整塊型除塵代價：|Δ| 中位 1.17，變好僅 3/15")
fig.suptitle("A 臂 — 乾淨前緣 15 名 × (原樣 / 除塵 d3)", color=INK, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "fig_a_dedust.png"), dpi=140, facecolor=SURF)
plt.close(fig)

# ---- 圖 B：補洞 4 對 × 3 指標 ----
pairs = ["b00_holes", "b01_holes", "b02_holes", "b03_holes"]
labels = [f"{p[:3]}\n(補{-man[p]['removed_px']}px)" for p in pairs]
dG = [G(p) - G(man[p]["base_id"]) for p in pairs]
dR = [R(p) - R(man[p]["base_id"]) for p in pairs]
dW = [W(p) - W(man[p]["base_id"]) for p in pairs]
x = np.arange(4)
fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURF)
for off, vals, c, lab in ((-0.27, dG, DBLUE, "Δ Gain"), (0.0, dR, VIOLET, "Δ rad 餘裕"), (0.27, dW, MUTED, "Δ worst")):
    ax.bar(x + off, vals, width=0.24, color=c, label=lab, zorder=2)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.legend(fontsize=9, framealpha=0.9).get_frame().set_edgecolor(GRID)
style(ax, "", "Δ (dB)", "判準=Gain 與 rad 同號變好 → 實測 Gain 混合、rad 四筆全負（噪聲地板 0.00）")
fig.suptitle("B 臂 — 補洞因果檢驗（4 個編輯對）", color=INK, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "fig_b_holes.png"), dpi=140, facecolor=SURF)
plt.close(fig)

# ---- 圖 C：SM 預測 vs HFSS 真值散點 ----
groups = {"orig": (DBLUE, "o", "orig (池內)"), "d3": (BLUE, "s", "d3 (池內編輯)"),
          "probe": ("#5598e7", "^", "probe (池內鄰域)"), "holes": ("#86b6ef", "D", "holes"),
          "blob": (ORANGE, "o", "blob (池外)"), "rand": (RED, "s", "random (池外)")}
fig, ax = plt.subplots(figsize=(8, 6.4), facecolor=SURF)
lim = (-26, 3)
ax.plot(lim, lim, color=INK2, ls="--", lw=1.4, zorder=1)
ax.text(-9.4, -8.2, "SM = HFSS（無誤差線）", color=INK2, fontsize=8.5, rotation=38)
for kind, (c, mk, lab) in groups.items():
    ids = [i for i in res if man[i]["kind"] == kind and man[i].get("sm_wm")]
    ax.scatter([W(i) for i in ids], [man[i]["sm_wm"][2] for i in ids],
               s=34, color=c, marker=mk, alpha=0.85, zorder=3,
               label=f"{lab}  n={len(ids)}", edgecolor=SURF, linewidth=0.5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect("equal")
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92).get_frame().set_edgecolor(GRID)
ax.annotate("池內：貼線、一致偏樂觀 (+1.4~+2.4)", (-4.2, 1.6), color=DBLUE, fontsize=9.5)
ax.annotate("池外：散開、一致偏悲觀 (−3.6~−5.5)", (-25.3, -21.5), color=ORANGE, fontsize=9.5)
style(ax, "HFSS 真值 worst-margin (dB)", "SM 預測 worst-margin (dB)",
      "SM 乾淨區校準（重錨前）：盲區不在池內、在池外")
fig.suptitle("C 臂 — SM 預測 vs HFSS 真值（97 筆）", color=INK, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "fig_c_sm.png"), dpi=140, facecolor=SURF)
plt.close(fig)

# ---- 圖 D：uniform random vs 各參照線 ----
dw = sorted(W(i) for i in res if i.startswith("d"))
fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURF)
ax.scatter(dw, np.zeros(len(dw)), s=54, color=RED, zorder=3, label="真 uniform random 10 筆")
refs = [(-3.47, DBLUE, ":", 1.02, "left", "池抽樣 E[best-of-10] = −3.47"),
        (-4.9, AQUA, ":", 0.72, "right", "我們線上搜尋 best 帶 −4.0~−4.9 "),
        (-7.85, MUTED, ":", 0.42, "right", "池全體中位 = −7.85 "),
        (dw[-1], RED, "--", 1.02, "right", f"uniform best-of-10 = {dw[-1]:.2f} ")]
ax.axvspan(-4.9, -4.0, color=AQUA, alpha=0.12, zorder=1)
for v, c, ls, ty, ha, lab in refs:
    ax.axvline(v, color=c, ls=ls, lw=1.6, zorder=2)
    ax.text(v + (0.12 if ha == "left" else -0.12), ty, lab, color=c, fontsize=8.8,
            ha=ha, va="top")
ax.set_ylim(-0.6, 1.15)
ax.set_yticks([])
ax.set_xlim(-19.5, -2.2)
style(ax, "worst-margin (dB) — 越右越好", "", "uniform random 與池抽樣差 ~5dB（N=10）→「輸 random」輸的是學長分布,不是隨機")
fig.suptitle("D 臂 — 真隨機基線", color=INK, fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "fig_d_random.png"), dpi=140, facecolor=SURF)
plt.close(fig)

print("4 figs →", ASSETS)
