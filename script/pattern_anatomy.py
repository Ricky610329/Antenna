# -*- coding: utf-8 -*-
"""
script/pattern_anatomy.py — 離線分析 analysis-01：pattern 解剖（地形隨機性 × S11/Gain 結構歸因）。

回答兩個問題（零 HFSS、開發機可跑、NAS 唯讀）：
  A. 學長「運氣論點」——pattern 距離 (Hamming) vs margin 差有沒有局部結構？（variogram 式曲線；
     全平＝搜尋≈抽獎、短距明顯低於長距漸近線＝引導式搜尋有依據）
  B/C. 結構特徵（連通組數、feed 連通、金屬比例、對稱度…）分別驅動 S11 還是 Gain？
     同 Gain 之下 S11 好壞的 pattern 差在哪？（Ricky 的「連成一塊算一組」假設）

子命令：
    python -m script.pattern_anatomy collect-pool    # harvest_single 全池: 8 特徵 + wm_S11/Gain/worst (~12-15 分)
    python -m script.pattern_anatomy collect-trajs   # 短距配對: 我們 R4/R5 csv(flips) + 學長軌跡相鄰 Hamming
    python -m script.pattern_anatomy report          # 圖 (docs/log/assets/analysis-01/) + markdown 表 (stdout)

快取落 `tmp/pattern_anatomy/`（git 忽略、可重建）。margin 全用 `antenna.losses.worst_margin` + 現行
targets 重算（與 round-06 / benchmark 同一把尺）；連通元件用 4-連通十字（對齊 `FeedReachability`）。
"""
import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from antenna.utils import config as _config, ROOTDIR, DATASET_PATH
_config.device = "cpu"
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from antenna.training import load_config, PORT_SPECS
from antenna.losses import worst_margin

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(REPO, "tmp", "pattern_anatomy")
ASSETS = os.path.join(REPO, "docs", "log", "assets", "analysis-01")
SENIOR_RESULT = r"T:\碩二_吳維文's\Patch Antenna\Experiment\result"   # 唯讀!
FEED = (24, 12)                                  # single feed 像素 (對齊 FeedReachability.single_feed)
N_BITS = 625                                     # 25×25

_cfg = load_config(os.path.join(REPO, "configs", "single_r5_explore.yaml"))
LABELS = PORT_SPECS[_cfg.port]["labels"]         # ["S11", "Gain"]

C_OURS, C_SEN, C_INK, C_INK2 = "#1c5cab", "#1baf7a", "#0b0b0b", "#52514e"
C_MUTED, C_GRID, C_SURF, C_RED = "#898781", "#e1e0d9", "#fcfcfb", "#d03b3b"

#? 4-連通十字 (對齊 FeedReachability 的 structure；「連成一塊算一組」的定義)
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
FEATURES = ("n_comp", "main_frac", "r_feed", "metal_frac",
            "sym_lr", "perim_ratio", "n_holes", "feed_touch")


# ---------------------------------------------------------------- 特徵 (純函式,可單測)
def pattern_features(p) -> dict:
    """p: (25,25) 0/1 或 bool → 8 個結構特徵。

    n_comp=金屬連通組數 (4-連通,Ricky 的「連成一塊算一組」)；main_frac=最大組/金屬數；
    r_feed=含 feed 像素那組/金屬數 (同 FeedReachability 語意,feed 非金屬→0)；metal_frac=金屬佔比；
    sym_lr=左右鏡射一致率；perim_ratio=內部金屬↔介質邊界長/金屬數 (細碎度)；
    n_holes=被金屬包住 (不觸邊) 的介質組數；feed_touch=feed 像素是否金屬。
    """
    from scipy.ndimage import label
    p = np.asarray(p).reshape(25, 25) > 0.5
    metal = int(p.sum())
    sym = float((p == p[:, ::-1]).mean())
    if metal == 0:
        return dict(n_comp=0, main_frac=0.0, r_feed=0.0, metal_frac=0.0,
                    sym_lr=sym, perim_ratio=0.0, n_holes=0, feed_touch=0.0)
    lab, n = label(p, structure=_CROSS)
    sizes = np.bincount(lab.ravel())[1:]
    feed_id = int(lab[FEED])
    h = int(np.abs(np.diff(p.astype(np.int8), axis=0)).sum())
    v = int(np.abs(np.diff(p.astype(np.int8), axis=1)).sum())
    inv_lab, m = label(~p, structure=_CROSS)
    border = set(np.unique(np.concatenate(
        [inv_lab[0, :], inv_lab[-1, :], inv_lab[:, 0], inv_lab[:, -1]])).tolist())
    holes = len(set(range(1, m + 1)) - border)
    return dict(
        n_comp=int(n),
        main_frac=float(sizes.max() / metal),
        r_feed=float(sizes[feed_id - 1] / metal) if feed_id > 0 else 0.0,
        metal_frac=metal / p.size,
        sym_lr=sym,
        perim_ratio=(h + v) / metal,
        n_holes=holes,
        feed_touch=float(feed_id > 0),
    )


def binned_median(d, v, edges):
    """(距離 d, 值 v) 配對 → 依 edges 分箱的 (中位數, 樣本數)。空箱回 nan/0。純函式,可單測。"""
    d = np.asarray(d, float)
    v = np.asarray(v, float)
    med = np.full(len(edges) - 1, np.nan)
    cnt = np.zeros(len(edges) - 1, int)
    for j in range(len(edges) - 1):
        m = (d >= edges[j]) & (d < edges[j + 1]) & np.isfinite(v)
        cnt[j] = int(m.sum())
        if cnt[j]:
            med[j] = float(np.median(v[m]))
    return med, cnt


def _margins(resp):
    """response → (wm_S11, wm_Gain, worst)。"""
    w, per = worst_margin(resp, LABELS, _cfg.targets)
    return float(per["S11"]), float(per["Gain"]), float(w)


# ---------------------------------------------------------------- collect
def collect_pool():
    """harvest_single 全池：pattern → 8 特徵；response → per-label margin。快取 pool.npz。"""
    from antenna.utils.store import SampleStore
    store = SampleStore(DATASET_PATH.joinpath("harvest_single"), verbose=False)
    n = len(store)
    feats = np.full((n, len(FEATURES)), np.nan, np.float32)
    wm = np.full((n, 3), np.nan, np.float32)                 # [S11, Gain, worst]
    packed = np.zeros((n, (N_BITS + 7) // 8), np.uint8)      # pattern 打包位元 (Hamming/示例圖用)
    t0 = time.time()
    for i in range(n):
        try:
            x, y = store[i]
            p = np.asarray(torch.as_tensor(x).float().reshape(25, 25)) > 0.5
            f = pattern_features(p)
            feats[i] = [f[k] for k in FEATURES]
            wm[i] = _margins(y)
            packed[i] = np.packbits(p.ravel())
        except Exception as e:
            print(f"  [err] i={i}: {e}")
        if i % 2000 == 0:
            print(f"  {i}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, "pool.npz"), feats=feats, wm=wm, packed=packed)
    ok = ~np.isnan(wm[:, 2])
    print(f"→ pool.npz  n={int(ok.sum())}  oracle={np.nanmax(wm[:, 2]):.2f}  "
          f"metal_frac 中位={np.nanmedian(feats[:, FEATURES.index('metal_frac')]):.2f}  "
          f"n_comp 中位={np.nanmedian(feats[:, FEATURES.index('n_comp')]):.0f}")


def collect_trajs():
    """短距配對：我們 R4/R5 csv 相鄰 (flips, |Δwm|)＋學長軌跡相鄰 (Hamming, |Δmargin|)。快取 trajs.npz。"""
    # -- 我們：metrics.csv 已有 flips(相鄰翻轉數)+worst_margin,免載 pattern --
    d_ours, dm_ours = [], []
    rd = ROOTDIR.joinpath("result")
    for dname in sorted(os.listdir(str(rd))):
        csvp = rd.joinpath(dname, "metrics.csv")
        if not csvp.exists():
            continue
        with open(str(csvp), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows or "flips" not in rows[0]:
            continue
        prev = None
        n0 = len(d_ours)
        for r in rows:
            try:
                w = float(r["worst_margin"])
            except (ValueError, KeyError, TypeError):
                prev = None
                continue
            fl = r.get("flips", "")
            if prev is not None and fl not in ("", "nan", None):
                d_ours.append(float(fl))
                dm_ours.append(abs(w - prev))
            prev = w
        if len(d_ours) > n0:
            print(f"  ours {dname.split(']')[-1].strip():<44} +{len(d_ours) - n0} 對")
    # -- 學長：online.dataset 相鄰兩筆 Hamming + |Δmargin| (唯讀) --
    d_sen, dm_sen = [], []
    for root, _dirs, files in os.walk(SENIOR_RESULT):
        if "online.dataset" not in files:
            continue
        name = os.path.basename(root)
        try:
            with open(os.path.join(root, "online.dataset"), "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"  [err] {name}: {e}")
            continue
        prev_p, prev_m, n0 = None, None, len(d_sen)
        for item in data:
            try:
                y = item[1]
                y = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
                if y.dim() < 2 or y.shape[0] != 2:
                    break                                    # dual/格式不符 → 整 run 跳過
                p = np.asarray(torch.as_tensor(item[0]).float().reshape(25, 25)) > 0.5
                m = _margins(y)[2]
            except Exception:
                continue
            if prev_p is not None:
                d_sen.append(int((p != prev_p).sum()))
                dm_sen.append(abs(m - prev_m))
            prev_p, prev_m = p, m
        if len(d_sen) > n0:
            print(f"  senior {name:<42} +{len(d_sen) - n0} 對", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, "trajs.npz"),
             d_ours=np.asarray(d_ours, np.float32), dm_ours=np.asarray(dm_ours, np.float32),
             d_sen=np.asarray(d_sen, np.float32), dm_sen=np.asarray(dm_sen, np.float32))
    print(f"→ trajs.npz  ours {len(d_ours)} 對 / senior {len(d_sen)} 對")


# ---------------------------------------------------------------- report
def _style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(C_SURF)
    ax.grid(alpha=0.6, color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_MUTED)
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=C_INK, fontsize=10)
    ax.set_ylabel(ylabel, color=C_INK, fontsize=10)
    if title:
        ax.set_title(title, color=C_INK, fontsize=11)


def _spearman(a, b):
    from scipy.stats import spearmanr
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    return float(spearmanr(a[m], b[m]).statistic)


def report():
    plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"],
                         "axes.unicode_minus": False,
                         "mathtext.fontset": "dejavusans"})
    pool = np.load(os.path.join(CACHE, "pool.npz"))
    feats, wm, packed = pool["feats"], pool["wm"], pool["packed"]
    ok = np.isfinite(wm[:, 2]) & np.isfinite(feats).all(axis=1)
    feats, wm, packed = feats[ok], wm[ok], packed[ok]
    tr = np.load(os.path.join(CACHE, "trajs.npz"))
    os.makedirs(ASSETS, exist_ok=True)
    F = {k: feats[:, i] for i, k in enumerate(FEATURES)}
    s11, gain, worst = wm[:, 0], wm[:, 1], wm[:, 2]
    n = len(worst)

    # ============ A. 地形隨機性 (variogram) ============
    rng = np.random.default_rng(0)
    M = 200_000
    ia, ib = rng.integers(0, n, M), rng.integers(0, n, M)
    keep = ia != ib
    ia, ib = ia[keep], ib[keep]
    dpool = np.zeros(len(ia), np.int32)
    POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)  # popcount 表
    for s in range(0, len(ia), 50_000):                       # 分塊算 Hamming (packed XOR + popcount)
        e = min(s + 50_000, len(ia))
        dpool[s:e] = POP[packed[ia[s:e]] ^ packed[ib[s:e]]].sum(axis=1)
    vpool = np.abs(worst[ia] - worst[ib])
    edges = np.array([1, 3, 6, 11, 21, 41, 81, 161, 321, 626])
    curves = {
        "pool 隨機配對":        binned_median(dpool, vpool, edges),
        "學長軌跡相鄰":         binned_median(tr["d_sen"], tr["dm_sen"], edges),
        "我們軌跡相鄰 (flips)": binned_median(tr["d_ours"], tr["dm_ours"], edges),
    }
    asym = float(np.median(vpool[dpool >= 161]))              # 長距漸近線 = 「不相關」水位
    mids = np.sqrt(edges[:-1] * np.minimum(edges[1:] - 1, 625))

    fig, ax = plt.subplots(figsize=(8, 5))
    for (lbl, (med, cnt)), c in zip(curves.items(), (C_INK2, C_SEN, C_OURS)):
        m = cnt >= 30
        ax.plot(mids[m], med[m], "o-", color=c, lw=2, ms=5, label=f"{lbl} (n={cnt.sum():,})")
    ax.axhline(asym, color=C_RED, ls=":", lw=1.4, label=f"長距漸近線 {asym:.2f} dB (=不相關水位)")
    ax.set_xscale("log")
    _style(ax, "pattern Hamming 距離 (翻轉像素數, log)", "|Δ worst-margin| 中位 (dB)",
           "地形隨機性檢驗：距離越近、margin 越像嗎？")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "variogram.png"), dpi=130)
    plt.close(fig)

    print("### A. 地形隨機性 (variogram)\n")
    print("| Hamming 距離 | pool 隨機配對 | 學長軌跡相鄰 | 我們軌跡相鄰 |")
    print("|---|---|---|---|")
    for j in range(len(edges) - 1):
        row = [f"{edges[j]}-{edges[j + 1] - 1}"]
        for med, cnt in curves.values():
            row.append(f"{med[j]:.2f} (n={cnt[j]:,})" if cnt[j] >= 30 else "—")
        print("| " + " | ".join(row) + " |")
    print(f"\n- 長距漸近線 (d≥161)＝{asym:.2f} dB；短距比值＝訊號強度 (1.0=全隨機/抽獎)。")

    # ============ B. 特徵歸因 (Spearman + n_comp 分箱) ============
    print("\n### B. 結構特徵 × margin (Spearman ρ, 池 n={:,})\n".format(n))
    print("| 特徵 | ρ(wm_S11) | ρ(wm_Gain) | ρ(worst) | 分層ρ(S11)* |")
    print("|---|---|---|---|---|")
    mf = F["metal_frac"]
    qs = np.quantile(mf, [0, .2, .4, .6, .8, 1.0])
    for k in FEATURES:
        strat = []
        for a, b in zip(qs[:-1], qs[1:]):                     # metal_frac 五分層,控制混淆
            m = (mf >= a) & (mf <= b)
            if m.sum() >= 200:
                strat.append(_spearman(F[k][m], s11[m]))
        srho = float(np.nanmedian(strat)) if strat else float("nan")
        print(f"| {k} | {_spearman(F[k], s11):+.2f} | {_spearman(F[k], gain):+.2f} "
              f"| {_spearman(F[k], worst):+.2f} | {srho:+.2f} |")
    print("\n*分層ρ = metal_frac 五分位層內的 ρ(特徵, wm_S11) 中位——控制「特徵天然隨金屬比例動」的混淆。")

    # n_comp 分箱雙箱型圖 (核心圖:「連成一塊算一組」假設)
    bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 8), (9, 99)]
    lbls = ["1", "2", "3", "4", "5-8", "9+"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)
    for ax, y, name, c in ((axes[0], s11, "wm_S11", C_OURS), (axes[1], gain, "wm_Gain", C_SEN)):
        data = [y[(F["n_comp"] >= a) & (F["n_comp"] <= b)] for a, b in bins]
        bp = ax.boxplot(data, labels=[f"{l}\n(n={len(d):,})" for l, d in zip(lbls, data)],
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color=C_INK, lw=1.6))
        for box in bp["boxes"]:
            box.set(facecolor=c, alpha=0.35, edgecolor=c)
        _style(ax, "金屬連通組數 n_comp", f"{name} (dB)", f"{name} vs 連通組數")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "ncomp_box.png"), dpi=130)
    plt.close(fig)
    print("\n| n_comp | wm_S11 中位 | wm_Gain 中位 | n |")
    print("|---|---|---|---|")
    for (a, b), l in zip(bins, lbls):
        m = (F["n_comp"] >= a) & (F["n_comp"] <= b)
        if m.sum():
            print(f"| {l} | {np.median(s11[m]):.2f} | {np.median(gain[m]):.2f} | {int(m.sum()):,} |")

    # top-1% (worst) 尾巴 vs 其餘：標準化差
    thr = np.quantile(worst, 0.99)
    top = worst >= thr
    fig, ax = plt.subplots(figsize=(7, 4.2))
    dz = []
    for k in FEATURES:
        sd = np.std(F[k])
        dz.append((np.mean(F[k][top]) - np.mean(F[k][~top])) / sd if sd > 0 else 0.0)
    order = np.argsort(dz)
    ax.barh([FEATURES[i] for i in order], [dz[i] for i in order],
            color=[C_OURS if d > 0 else C_RED for d in np.array(dz)[order]], alpha=0.8)
    ax.axvline(0, color=C_MUTED, lw=1)
    _style(ax, "標準化差 (top-1% 平均 − 其餘平均)/σ", "",
           f"頂端 1% (worst ≥ {thr:.2f} dB, n={int(top.sum())}) 的結構長相")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "tail_contrast.png"), dpi=130)
    plt.close(fig)

    # r_feed 分箱曲線
    redges = np.array([0, .2, .4, .6, .8, 1.0001])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for y, name, c in ((s11, "wm_S11", C_OURS), (gain, "wm_Gain", C_SEN)):
        med, cnt = binned_median(F["r_feed"], y, redges)
        m = cnt >= 30
        ax.plot(((redges[:-1] + redges[1:]) / 2)[m], med[m], "o-", color=c, lw=2, label=name)
    _style(ax, "r_feed (feed 連通塊佔金屬比例)", "margin 中位 (dB)", "feed 連通 vs margin")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "rfeed_bins.png"), dpi=130)
    plt.close(fig)

    # ============ C. 配對對比 (同 Gain 比 S11 / 同 S11 比 Gain) ============
    def _held_contrast(held, contrast, hname, cname):
        """held 同箱 (0.5dB) 內,contrast 前/後十分位的特徵差 (跨箱中位)。"""
        deltas = {k: [] for k in FEATURES}
        pair_pool = []
        for lo in np.arange(np.floor(held.min()), np.ceil(held.max()), 0.5):
            m = (held >= lo) & (held < lo + 0.5)
            if m.sum() < 50:
                continue
            idx = np.where(m)[0]
            v = contrast[idx]
            hi_i = idx[v >= np.quantile(v, 0.9)]
            lo_i = idx[v <= np.quantile(v, 0.1)]
            if np.median(contrast[hi_i]) - np.median(contrast[lo_i]) < 3.0:
                continue                                       # 該箱 contrast 拉不開 3dB → 略過
            for k in FEATURES:
                deltas[k].append(float(np.median(F[k][hi_i]) - np.median(F[k][lo_i])))
            pair_pool.append((lo, hi_i, lo_i))
        print(f"\n**{hname} 同箱 (0.5dB)、{cname} 前/後十分位差 ≥3dB 的箱共 {len(pair_pool)} 個**")
        print(f"\n| 特徵 | Δ ({cname} 好 − 差) 跨箱中位 |")
        print("|---|---|")
        for k in FEATURES:
            if deltas[k]:
                print(f"| {k} | {np.median(deltas[k]):+.3f} |")
        return pair_pool

    print("\n### C. 配對對比")
    pairs_g = _held_contrast(gain, s11, "Gain", "S11")
    _held_contrast(s11, gain, "S11", "Gain")

    # 示例圖:挑 Gain 最高的 3 個有效箱,各出一對 (S11 最好 vs 最差)
    if pairs_g:
        show = sorted(pairs_g, key=lambda t: -t[0])[:3]
        fig, axes = plt.subplots(2, len(show), figsize=(3.4 * len(show), 7))
        axes = np.atleast_2d(axes)
        for j, (lo, hi_i, lo_i) in enumerate(show):
            for row, i in ((0, hi_i[np.argmax(s11[hi_i])]), (1, lo_i[np.argmin(s11[lo_i])])):
                p = np.unpackbits(packed[i], count=N_BITS).reshape(25, 25)
                ax = axes[row, j]
                ax.imshow(p, cmap="gray_r", vmin=0, vmax=1)
                ax.set_title(f"Gain {gain[i]:.1f} / S11 {s11[i]:.1f} dB\n"
                             f"n_comp={int(F['n_comp'][i])}  r_feed={F['r_feed'][i]:.2f}",
                             fontsize=8.5, color=C_INK)
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].set_ylabel("S11 好", fontsize=10, color=C_OURS)
        axes[1, 0].set_ylabel("S11 差", fontsize=10, color=C_RED)
        fig.suptitle("同 Gain 箱內：S11 最好 vs 最差的 pattern", color=C_INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(ASSETS, "pairs_gain_held.png"), dpi=130)
        plt.close(fig)

    print(f"\n圖: docs/log/assets/analysis-01/ (variogram / ncomp_box / tail_contrast / rfeed_bins / pairs_gain_held)")


def main():
    ap = argparse.ArgumentParser(description="pattern 解剖:地形隨機性 × S11/Gain 結構歸因 (離線)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect-pool")
    sub.add_parser("collect-trajs")
    sub.add_parser("report")
    args = ap.parse_args()
    {"collect-pool": collect_pool, "collect-trajs": collect_trajs, "report": report}[args.cmd]()


if __name__ == "__main__":
    main()
