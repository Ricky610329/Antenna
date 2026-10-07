# -*- coding: utf-8 -*-
"""Render analysis-18 symmetry census figures from the reproducible CSV."""
import argparse
import csv
import gzip
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_ASSETS = REPO / "docs" / "log" / "assets" / "analysis-18"
DEFAULT_DATA = REPO / "application" / "pattern_browser" / "data"
SURF, INK, MUTED, GRID = "#fcfcfb", "#171716", "#66645f", "#dedcd4"
BLUE, ORANGE, GREEN, RED = "#1c5cab", "#e36a32", "#16886b", "#c94343"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"],
                     "axes.unicode_minus": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})


def _rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, mode="rt", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _floats(rows, name):
    out = []
    for row in rows:
        try:
            out.append(float(row[name]))
        except (KeyError, TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out)


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, color=INK, fontsize=11)
    ax.grid(color=GRID, lw=0.6, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _density_and_medians(ax, x, y, bins, color=ORANGE):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size >= 20:
        hb = ax.hexbin(x, y, gridsize=34, mincnt=1, bins="log", cmap="Blues",
                       linewidths=0.15, alpha=0.82)
    else:
        hb = ax.scatter(x, y, s=14, color=BLUE, alpha=0.65)
    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins + 1) if x.size else np.array([])
    mx, my, mn = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        take = (x >= lo) & (x < hi if hi < edges[-1] else x <= hi)
        if np.count_nonzero(take) >= 10:
            mx.append((lo + hi) / 2)
            my.append(float(np.median(y[take])))
            mn.append(int(np.count_nonzero(take)))
    if mx:
        ax.scatter(mx, my, s=34, marker="D", color=color, edgecolor=SURF, lw=0.7,
                   zorder=5, label=f"分箱中位數（n≥10；{len(mx)} 箱）")
        ax.legend(frameon=False, fontsize=8, loc="best")
    return hb


def geometry_field(rows, out):
    x = _floats(rows, "geometry_mismatch_fraction")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for ax, cut, label in zip(axes, ("phi0", "phi90"), ("φ=0°", "φ=90°")):
        y = _floats(rows, f"{cut}_mirror_power_90")
        _density_and_medians(ax, x, y, 20)
        _style(ax, "幾何左右鏡射 mismatch A（pair XOR mean）",
               "場型鏡射殘差（線性功率，±90°）", label)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    fig.suptitle("歷史幾何不對稱與場型不對稱（cache-level；hex 顏色=log density）",
                 fontsize=13, color=INK)
    fig.text(0.5, -0.01, "重建式統計圖；不是復原舊圖 generator。稀疏分箱不插值。",
             ha="center", color=MUTED, fontsize=8.5)
    fig.savefig(out / "geometry_vs_field_symmetry.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def exact_frontback(rows, out):
    rows = [r for r in rows if r.get("symmetry_class") == "exact"]
    x = _floats(rows, "metal_fraction")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for ax, cut, label in zip(axes, ("phi0", "phi90"), ("φ=0°", "φ=90°")):
        y = _floats(rows, f"{cut}_front_back_imbalance")
        _density_and_medians(ax, x, y, 16, color=GREEN)
        ax.axhline(0, color=RED, ls="--", lw=0.9)
        _style(ax, "exact-metal-symmetric 樣本的金屬比例",
               "切面前/後角區平均線性功率不均（signed）", label)
        ax.set_ylim(-1, 1)
    fig.suptitle("完全金屬對稱樣本：金屬比例與切面前後角區不均（hex 顏色=log density）",
                 fontsize=13, color=INK)
    fig.text(0.5, -0.01, "離散角度樣本算術平均，非 3D 半球積分/效率；不套性能門檻。",
             ha="center", color=MUTED, fontsize=8.5)
    fig.savefig(out / "exact_symmetry_metal_frontback.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def counterexamples(rows, data_dir, out):
    ids = np.load(data_dir / "patterns.npz", allow_pickle=False)
    pats = np.unpackbits(ids["packed"], axis=1)[:, :625].reshape(-1, 25, 25)
    lookup = {str(pid): i for i, pid in enumerate(ids["ids"])}
    valid = []
    for row in rows:
        try:
            g = float(row["geometry_mismatch_fraction"])
            f = float(row["field_mirror_power_90"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(g) and np.isfinite(f):
            valid.append((row, g, f))
    exact = sorted((x for x in valid if x[1] == 0),
                   key=lambda x: abs(float(x[0]["phi0_mirror_power_90"]) -
                                     float(x[0]["phi90_mirror_power_90"])), reverse=True)
    asym = sorted((x for x in valid if x[1] > 0.1),
                  key=lambda x: float(x[0]["phi90_mirror_power_90"]))
    chosen = exact[:2] + asym[:2]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2), constrained_layout=True)
    cmap = ListedColormap([SURF, BLUE])
    for ax, (row, g, f) in zip(axes, chosen):
        ax.imshow(pats[lookup[row["id"]]], cmap=cmap, vmin=0, vmax=1,
                  interpolation="nearest", origin="upper")
        ax.scatter([12], [24], marker="^", s=38, color=GREEN, edgecolor=SURF, lw=0.6)
        role = "幾何 exact／切面差異大" if g == 0 else "幾何不對稱／φ90 殘差低"
        p0 = float(row["phi0_mirror_power_90"])
        p90 = float(row["phi90_mirror_power_90"])
        ax.set_title(f"{role}\n{row['id']}\nA={g:.3f} · φ0={p0:.3f} · φ90={p90:.3f}", fontsize=8.3)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.suptitle("兩切面差異代表（由 census 規則決定性挑選）", fontsize=12.5, color=INK)
    fig.text(0.5, -0.02, "重建式 pattern gallery；饋線邊在下。曲線 store provenance 仍未逐列解決。",
             ha="center", color=MUTED, fontsize=8.2)
    fig.savefig(out / "symmetry_counterexamples.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="繪製 analysis-18 對稱盤點圖")
    parser.add_argument("--csv", type=Path, default=DEFAULT_ASSETS / "symmetry_census.csv.gz")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ASSETS)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(args.csv)
    geometry_field(rows, args.out_dir)
    exact_frontback(rows, args.out_dir)
    counterexamples(rows, args.data_dir, args.out_dir)
    print(f"figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
