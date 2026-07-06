# -*- coding: utf-8 -*-
"""pool_families.py — 池頂端結構家族普查：top-300 greedy 家族聚類 + 代表 gallery。"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = r"C:\Users\Ricky\Documents\GitHub\Antenna"
sys.path.insert(0, REPO)
from antenna.utils import config as _config  # noqa: E402
_config.device = "cpu"

POOL = os.path.join(REPO, "tmp", "pattern_anatomy", "pool.npz")
ASSETS = os.path.join(REPO, "docs", "log", "assets", "round-09")
os.makedirs(ASSETS, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"],
                     "axes.unicode_minus": False})

d = np.load(POOL)
ok = ~np.isnan(d["wm"][:, 2])
wm = d["wm"][ok]
feats = d["feats"][ok]
pats = np.unpackbits(d["packed"][ok], axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
worst = wm[:, 2]
FEATURES = ("n_comp", "main_frac", "r_feed", "metal_frac", "sym_lr", "perim_ratio", "n_holes", "feed_touch")
F = {k: feats[:, i] for i, k in enumerate(FEATURES)}

TOPN = 300
order = np.argsort(worst)[::-1][:TOPN]

# greedy leader clustering: Hamming > DIST 才開新家族
DIST = 100
leaders, members = [], {}
for i in order:
    hit = None
    for L in leaders:
        if np.count_nonzero(pats[i] != pats[L]) <= DIST:
            hit = L
            break
    if hit is None:
        leaders.append(i)
        members[i] = [i]
    else:
        members[hit].append(i)

print(f"top-{TOPN} (wm {worst[order[-1]]:.2f}~{worst[order[0]]:+.2f}) → {len(leaders)} 個家族 (Hamming>{DIST})")
print("\n| 家族 | 成員數 | 最佳wm | metal_frac | n_comp | 上半金屬佔比 | 下半金屬佔比 |")
print("|---|---|---|---|---|---|---|")
rows = []
for L in leaders:
    ms = members[L]
    best = ms[int(np.argmax(worst[ms]))]
    p = pats[best]
    up, dn = p[:12].mean(), p[13:].mean()
    rows.append((best, len(ms), worst[best]))
    print(f"| F{leaders.index(L)} | {len(ms)} | {worst[best]:+.2f} | {F['metal_frac'][best]:.2f} "
          f"| {int(F['n_comp'][best])} | {up:.2f} | {dn:.2f} |")

# 也掃「乾淨前緣」(main_frac>=0.9) top-15 (=R8 A 臂) 分屬哪些家族
clean = np.where(F["main_frac"] >= 0.9)[0]
clean = clean[np.argsort(worst[clean])[::-1]]
apicked = []
for i in clean:
    if len(apicked) >= 15:
        break
    if all(np.count_nonzero(pats[i] != pats[j]) > 60 for j in apicked):
        apicked.append(int(i))
fam_of = {}
for L in leaders:
    for m in members[L]:
        fam_of[m] = leaders.index(L)
print("\nR8 A 臂 15 錨點所屬家族:", [fam_of.get(i, "top300外") for i in apicked])

# gallery: 前 12 家族代表
cmap = ListedColormap([SURF, "#1c5cab"])
n_show = min(12, len(leaders))
fig, axes = plt.subplots(3, 4, figsize=(11, 9), facecolor=SURF)
for ax, (best, sz, w) in zip(axes.flat, rows[:n_show]):
    ax.imshow(pats[best].astype(int), cmap=cmap, vmin=0, vmax=1, origin="upper",
              interpolation="nearest")
    k = rows.index((best, sz, w))
    ax.set_title(f"F{k} — 池wm {w:+.2f}（成員 {sz}）\nmetal {F['metal_frac'][best]:.2f} · "
                 f"組數 {int(F['n_comp'][best])} · 主件 {F['main_frac'][best]:.2f}",
                 color=INK, fontsize=9.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
for ax in axes.flat[n_show:]:
    ax.axis("off")
fig.suptitle(f"harvest 池 top-{TOPN} 的結構家族代表（greedy 聚類,Hamming>{DIST}；⚠ 池值未折價）",
             color=INK, fontsize=13)
fig.tight_layout(h_pad=2.0)
out = os.path.join(ASSETS, "pool_families.png")
fig.savefig(out, dpi=140, facecolor=SURF)
print("\n→", out)
