# -*- coding: utf-8 -*-
"""
script/expected_best.py — 離線歷史基準：「每跑一輪 HFSS，期望拿到的 best worst_margin 是多少?」(Round 06)

把三塊歷史資料放到同一把尺上（margin 全用 `antenna.losses.worst_margin` + 現行 targets 重算 → 跨來源可比），
回答「達成效率」的期望/機率視角（單一軌跡隨機性大 → 看分布，不看單條）。**零 HFSS 成本、開發機可跑。**

子命令：
    python -m script.expected_best collect-ours     # 自家 greedy 家族 metrics.csv → best-so-far 曲線 (~秒)
    python -m script.expected_best collect-pool     # harvest_single 全池逐筆 margin (~12 分鐘, NAS I/O)
    python -m script.expected_best collect-senior   # 學長原始 result 樹(**唯讀**)各 run online.dataset → 軌跡 (~分鐘)
    python -m script.expected_best report           # 彙整快取 → 4 張圖 (docs/log/assets/round-06/) + markdown 表 (stdout)

快取落 `tmp/expected_best/`（git 忽略）。random best-of-N 用池經驗分布**閉式解**（iid 有放回）：
    P(max_N ≤ v_(i)) = (i/n)^N  →  期望/分位數；P(達標 T by N) = 1 − F(T)^N。
⚠ 池 = 學長各 run 搜尋軌跡的**聯集**（偏向好區），所以 random 線是「分布上界參照」，不是 uniform random。
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
CACHE = os.path.join(REPO, "tmp", "expected_best")
ASSETS = os.path.join(REPO, "docs", "log", "assets", "round-06")
SENIOR_RESULT = r"T:\碩二_吳維文's\Patch Antenna\Experiment\result"   # 唯讀!
FAMS = ("single_guided", "single_r2", "single_r3", "single_r4", "single_r5")
T_LIST = (-5.0, -3.0, -2.0, -1.0, 0.0)      # 達成門檻 (dB)
K_GRID = np.asarray([10, 25, 50, 100, 150, 200, 250, 300, 400, 500, 750, 1000])

_cfg = load_config(os.path.join(REPO, "configs", "single_r5_explore.yaml"))   # targets 與 R2-R5 一致(已驗)
LABELS = PORT_SPECS[_cfg.port]["labels"]

# 圖表色票 (dataviz reference palette, light)
C_OURS, C_SEN, C_INK, C_INK2 = "#1c5cab", "#1baf7a", "#0b0b0b", "#52514e"
C_MUTED, C_GRID, C_SURF, C_RED = "#898781", "#e1e0d9", "#fcfcfb", "#d03b3b"


def _margin(resp) -> float:
    return float(worst_margin(resp, LABELS, _cfg.targets)[0])


def _best_so_far(xs, vals):
    """(x, val) 序列 → 去重 x 的 best-so-far（wm 越高越好 → 累計 max）。"""
    ex, eb, cur = [], [], -1e9
    for x, v in zip(xs, vals):
        cur = max(cur, v)
        if ex and x == ex[-1]:
            eb[-1] = cur
        else:
            ex.append(x)
            eb.append(cur)
    return np.asarray(ex, float), np.asarray(eb, float)


# ---------------------------------------------------------------- collect
def collect_ours():
    """自家 greedy 家族：metrics.csv 的 worst_margin（訓練時同函式落盤）→ best-so-far vs hfss_calls。"""
    rd = ROOTDIR.joinpath("result")
    out = {}
    for d in sorted(os.listdir(str(rd))):
        short = d.split("]")[-1].strip().replace("pixel_", "")
        if not any(f in short for f in FAMS):
            continue
        try:
            machine = d.split("Patch-single-")[1].split("-")[0]
        except Exception:
            machine = "?"
        csvp = rd.joinpath(d, "metrics.csv")
        if not csvp.exists():
            continue
        with open(str(csvp), newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        xs, wms = [], []
        for i, r in enumerate(rows):
            v = r.get("worst_margin", "")
            if v in ("", "nan", None):
                continue
            try:
                w = float(v)
            except ValueError:
                continue
            hc = r.get("hfss_calls", "")
            try:
                x = int(float(hc)) if hc not in ("", "nan", None) else i + 1
            except ValueError:
                x = i + 1
            xs.append(x)
            wms.append(w)
        if len(wms) < 8:
            continue
        ex, eb = _best_so_far(xs, wms)
        out[f"{machine}:{short}"] = np.stack([ex, eb])
        print(f"  {machine}:{short:<42} n={len(ex):>4}  best={eb[-1]:.2f}")
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, "ours_curves.npz"), **out)
    print(f"→ {CACHE}\\ours_curves.npz ({len(out)} runs)")


def collect_pool():
    """harvest_single 全池 (學長歷史 HFSS 樣本) 逐筆 margin。"""
    from antenna.utils.store import SampleStore
    store = SampleStore(DATASET_PATH.joinpath("harvest_single"), verbose=False)
    n = len(store)
    ms = np.full(n, np.nan, np.float32)
    t0 = time.time()
    for i in range(n):
        try:
            _x, y = store[i]
            ms[i] = _margin(y)
        except Exception as e:
            print(f"  [err] i={i}: {e}")
        if i % 2000 == 0:
            print(f"  {i}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    np.save(os.path.join(CACHE, "harvest_margins.npy"), ms)
    ok = ms[~np.isnan(ms)]
    print(f"→ harvest_margins.npy  n={len(ok)}  oracle={ok.max():.2f}  median={np.median(ok):.2f}")


def collect_senior():
    """學長原始 result 樹（**唯讀**、裸 pickle、不經 DataManager）→ 各 single run 的有序軌跡。"""
    out = {}
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
        ms = []
        for item in data:
            try:
                y = item[1]
                y = y if isinstance(y, torch.Tensor) else torch.as_tensor(y)
                if y.dim() < 2 or y.shape[0] != 2:
                    break                     # dual(3ch)/格式不符 → 整 run 跳過 (同 harvest_legacy 判別)
                ms.append(_margin(y))
            except Exception:
                continue
        if len(ms) < 30:
            continue
        ex, eb = _best_so_far(range(1, len(ms) + 1), ms)
        out[name] = np.stack([ex, eb])
        print(f"  {name:<50} n={len(ms):>5}  best={eb[-1]:.2f}", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, "senior_curves.npz"), **out)
    print(f"→ senior_curves.npz ({len(out)} runs)")


# ---------------------------------------------------------------- 統計工具
def _stats_at(curves, ks, qs=(0.25, 0.75)):
    """每個 k 取「有跑到 k 的 run」的 best@k → (中位, q25, q75, min, max, n)。有右截斷(censoring)：
    只用存活 run，長預算端樣本會變少 → 呈現時一律帶 n。"""
    out = np.full((len(ks), 6), np.nan)
    for j, k in enumerate(ks):
        vals = [b[np.searchsorted(x, k, side="right") - 1] for x, b in curves if x[-1] >= k]
        if vals:
            out[j] = (np.median(vals), np.quantile(vals, qs[0]), np.quantile(vals, qs[1]),
                      min(vals), max(vals), len(vals))
    return out


def _best_of_n(sorted_vals, N, qs=(0.1, 0.9)):
    """池經驗分布下 best-of-N 的 (期望, 分位數…)。"""
    n = len(sorted_vals)
    i = np.arange(1, n + 1)
    w = (i / n) ** N - ((i - 1) / n) ** N
    mean = float((sorted_vals * w).sum())
    qv = [float(sorted_vals[min(n - 1, int(np.ceil(n * q ** (1.0 / N))) - 1)]) for q in qs]
    return (mean, *qv)


def _km_attain(curves, T, min_risk=5):
    """Kaplan–Meier 估計 P(首次達 T 的輪數 ≤ k)：run 沒達標就停跑＝右截斷(censoring)。
    「存活 run 中的比例」在截斷下非單調、尾端會因組成暴跳 → 必須用 KM。
    回傳 (ks, probs) step 曲線；風險集 < min_risk 即截止（避免小樣本尾端亂跳）。"""
    events = []                                # (k, 達標?)：達標=event、跑完沒達=censored
    for x, b in curves:
        idx = int(np.argmax(b >= T)) if (b >= T).any() else -1
        events.append((float(x[idx]), True) if idx >= 0 else (float(x[-1]), False))
    events.sort()
    n_risk, surv = len(events), 1.0
    ks, ps = [0.0], [0.0]
    k_stop = events[-1][0]
    for k, hit in events:
        if n_risk < min_risk:
            k_stop = k
            break
        if hit:
            surv *= 1.0 - 1.0 / n_risk
            ks.append(k)
            ps.append(1.0 - surv)
        n_risk -= 1
    ks.append(k_stop)
    ps.append(ps[-1])
    return np.asarray(ks), np.asarray(ps)


def _reach_k(curves, T):
    """各 run 首次達 T 的 HFSS 輪數（沒達 → None）。"""
    ks = []
    for x, b in curves:
        idx = np.argmax(b >= T) if (b >= T).any() else -1
        ks.append(int(x[idx]) if idx >= 0 else None)
    return ks


# ---------------------------------------------------------------- report
def _plainlog(axis):
    """log 軸刻度改純文字（預設 \\mathdefault 會拿 CJK 主字型渲染 10^−1 的 −,變豆腐字）。"""
    axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))


def _style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(C_SURF)
    ax.grid(color=C_GRID, lw=0.7, alpha=0.8)
    for s in ax.spines.values():
        s.set_color(C_GRID)
    ax.tick_params(colors=C_MUTED)
    ax.set_xlabel(xlabel, color=C_INK2)
    ax.set_ylabel(ylabel, color=C_INK2)
    if title:
        ax.set_title(title, color=C_INK, fontsize=12)


def report():
    plt.rcParams.update({"font.family": ["Microsoft JhengHei", "sans-serif"],
                         "axes.unicode_minus": False,
                         "mathtext.fontset": "dejavusans"})   # log 軸負指數的 − 需要 mathtext 字型
    ours = {k: (v[0], v[1]) for k, v in np.load(os.path.join(CACHE, "ours_curves.npz")).items()}
    senior = [(v[0], v[1]) for v in np.load(os.path.join(CACHE, "senior_curves.npz")).values()]
    pool = np.load(os.path.join(CACHE, "harvest_margins.npy"))
    pool = np.sort(pool[~np.isnan(pool)])
    os.makedirs(ASSETS, exist_ok=True)

    fam = [(x, b) for k, (x, b) in ours.items()
           if any(f in k for f in FAMS[1:])]                    # R2-R5 = 現行機制;R1 只畫細線
    oracle = float(pool[-1])
    ks = K_GRID
    fam_s, sen_s = _stats_at(fam, ks), _stats_at(senior, ks)
    rnd = np.asarray([_best_of_n(pool, int(N)) for N in ks])    # (mean, p10, p90)

    # log fit（家族中位、樣本 n≥3 的 k）
    m = (fam_s[:, 5] >= 3) & ~np.isnan(fam_s[:, 0])
    (a, b), *_ = np.linalg.lstsq(np.vstack([np.ones(m.sum()), np.log(ks[m])]).T,
                                 fam_s[m, 0], rcond=None)
    k_spec = float(np.exp(-a / b))                              # 期望路徑到 margin 0 所需輪數

    # 我們 / 學長的「史上最佳」錨點
    our_best = max(((float(bst[-1]), int(x[np.argmax(bst)])) for x, bst in fam), key=lambda t: t[0])
    sen_best = max(((float(bst[-1]), int(x[np.argmax(bst)])) for x, bst in senior), key=lambda t: t[0])

    # 躍遷主導度：k≥10 後,最大單筆 record 跳升佔總改善的比例（中位）
    def _jump_frac(curves):
        fr = []
        for x, bst in curves:
            i0 = np.searchsorted(x, 10)
            if i0 >= len(bst) - 1 or bst[-1] <= bst[i0]:
                continue
            jumps = np.diff(bst[i0:])
            fr.append(float(jumps.max() / (bst[-1] - bst[i0])))
        return float(np.median(fr)) if fr else float("nan")

    # ---- 表 1：E[best@k] ----
    print("### 表 1 — best worst-margin @ k（中位;帶 IQR 與存活 n）\n")
    print("| k | 我們 greedy 中位 (IQR, n) | 學長中位 (IQR, n) | random 池抽樣期望 (p10…p90) | fit |")
    print("|---|---|---|---|---|")
    for j, k in enumerate(ks):
        f = (f"{fam_s[j, 0]:.2f} ({fam_s[j, 1]:.2f}…{fam_s[j, 2]:.2f}, n={fam_s[j, 5]:.0f})"
             if fam_s[j, 5] > 0 else "—")
        s = (f"{sen_s[j, 0]:.2f} ({sen_s[j, 1]:.2f}…{sen_s[j, 2]:.2f}, n={sen_s[j, 5]:.0f})"
             if sen_s[j, 5] > 0 else "—")
        print(f"| {k} | {f} | {s} | {rnd[j, 0]:.2f} ({rnd[j, 1]:.2f}…{rnd[j, 2]:.2f}) "
              f"| {a + b * np.log(k):.2f} |")
    print(f"\n- 家族期望曲線 fit：**best(k) ≈ {a:.2f} + {b:.3f}·ln k**（n≥3 的 k 點）")
    print(f"- 期望路徑走到 spec(margin 0) 需 **k ≈ {k_spec:.1e} 輪** → 期望爬升到不了,靠躍遷。")
    print(f"- 躍遷主導度（最大單跳/總改善,k≥10 後,中位）：我們 {_jump_frac(fam):.0%}、學長 {_jump_frac(senior):.0%}")
    print(f"- oracle（池內最佳）= **{oracle:+.2f} dB**；我們史上最佳 {our_best[0]:.2f}@{our_best[1]}、"
          f"學長 {sen_best[0]:+.2f}@{sen_best[1]}")

    # ---- 表 2：到達門檻的效率 ----
    print("\n### 表 2 — 到達門檻 T 的效率（隨機性視角;KM = Kaplan–Meier,右截斷校正）\n")
    print("| T (dB) | 我們:達成/中位輪數 | 我們 KM P(≤500輪) | 學長:達成/中位輪數 | 學長 KM P(≤500輪) | random:P50 / P90 抽樣數 |")
    print("|---|---|---|---|---|---|")

    def _km_at(curves, T, k):
        kk, pp = _km_attain(curves, T)
        i = np.searchsorted(kk, k, side="right") - 1
        return pp[max(i, 0)]

    for T in T_LIST:
        ro, rs = _reach_k(fam, T), _reach_k(senior, T)
        F = float(np.searchsorted(pool, T, side="left")) / len(pool)   # P(單抽 < T)
        if F < 1.0:
            k50 = int(np.ceil(np.log(0.5) / np.log(F))) if F > 0 else 1
            k90 = int(np.ceil(np.log(0.1) / np.log(F))) if F > 0 else 1
            r = f"{k50} / {k90}"
        else:
            r = "池內無此點"
        fo = [k for k in ro if k]
        fs = [k for k in rs if k]
        print(f"| {T:+.0f} | {len(fo)}/{len(ro)}{' / ' + str(int(np.median(fo))) + '輪' if fo else ''} "
              f"| {_km_at(fam, T, 500):.0%} "
              f"| {len(fs)}/{len(rs)}{' / ' + str(int(np.median(fs))) + '輪' if fs else ''} "
              f"| {_km_at(senior, T, 500):.0%} | {r} |")

    # ==== 圖 1：主圖 best vs k ====
    fig, ax = plt.subplots(figsize=(9.5, 6), facecolor=C_SURF)
    for key, (x, bst) in ours.items():
        ax.plot(x, bst, color=C_OURS, lw=0.9, alpha=0.35, zorder=2)
    for x, bst in senior:
        ax.plot(x, bst, color=C_SEN, lw=0.8, alpha=0.28, zorder=1)
    mm = fam_s[:, 5] >= 2
    ax.fill_between(ks[mm], fam_s[mm, 1], fam_s[mm, 2], color=C_OURS, alpha=0.15, lw=0)
    ax.plot(ks[mm], fam_s[mm, 0], color=C_OURS, lw=2.5, zorder=5,
            label=f"我們 greedy 家族中位±IQR (R2–R5, {len(fam)} runs)")
    msn = sen_s[:, 5] >= 10                                     # 長預算端存活太少 → 截掉組成假象
    ax.plot(ks[msn], sen_s[msn, 0], color=C_SEN, lw=2.5, zorder=4,
            label=f"學長方法中位 ({len(senior)} runs, n≥10 段)")
    ax.plot(ks, rnd[:, 0], color=C_INK2, ls="--", lw=2, zorder=4, label="random best-of-N (池抽樣期望)")
    ax.fill_between(ks, rnd[:, 1], rnd[:, 2], color=C_INK2, alpha=0.12, lw=0, label="random p10–p90")
    kk = np.geomspace(10, 1500, 60)
    ax.plot(kk, a + b * np.log(kk), color=C_OURS, ls=":", lw=1.4, zorder=3,
            label=f"fit: {a:.1f}+{b:.2f}·ln k")
    ax.axhline(oracle, color=C_INK, ls=":", lw=1.4)
    ax.text(5.5, oracle + 0.3, f"oracle = 池內最佳 {oracle:+.2f} dB", color=C_INK, fontsize=9)
    ax.axhline(0, color=C_RED, ls=":", lw=1.2)
    ax.text(5.5, -1.05, "spec 達標線 (margin = 0)", color=C_RED, fontsize=9)
    ax.scatter([our_best[1]], [our_best[0]], color=C_OURS, s=42, zorder=6, edgecolor=C_SURF)
    ax.annotate(f"我們史上最佳 {our_best[0]:.2f} @{our_best[1]}", (our_best[1], our_best[0]),
                xytext=(22, -2.5), color=C_OURS, fontsize=9)
    ax.scatter([sen_best[1]], [sen_best[0]], color=C_SEN, s=42, zorder=6, edgecolor=C_SURF)
    ax.annotate(f"學長達標 {sen_best[0]:+.2f} @{sen_best[1]}", (sen_best[1], sen_best[0]),
                xytext=(70, 1.0), color="#0e7a55", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(5, 2200)
    ax.set_ylim(-24, 2.6)
    ax.set_xticks([5, 10, 25, 50, 100, 250, 500, 1000, 2000])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    _style(ax, "HFSS calls (log)", "best worst-margin so far (dB) — 越高越好",
           "每跑一輪 HFSS 期望拿到的 best worst-margin")
    leg = ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    leg.get_frame().set_edgecolor(C_GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "best_vs_k.png"), dpi=140, facecolor=C_SURF)
    plt.close(fig)

    # ==== 圖 2：達成機率 P(best@k ≥ T) ====
    kg = np.unique(np.geomspace(5, 2000, 48).astype(int))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True, facecolor=C_SURF)
    for ax, T in zip(axes, (-3.0, -1.0, 0.0)):
        F = float(np.searchsorted(pool, T, side="left")) / len(pool)
        ax.plot(kg, 1 - F ** kg.astype(float), color=C_INK2, ls="--", lw=2, label="random 池抽樣")
        for curves, c, lab in ((fam, C_OURS, "我們 greedy (R2–R5, KM)"),
                               (senior, C_SEN, "學長方法 (KM)")):
            kk, pp = _km_attain(curves, T)
            ax.plot(kk, pp, color=c, lw=2.2, drawstyle="steps-post", label=lab)
        ax.set_xscale("log")
        _plainlog(ax.xaxis)
        ax.set_xlim(5, 2200)
        ax.set_ylim(-0.03, 1.03)
        _style(ax, "HFSS calls (log)", "P(達成)" if T == -3.0 else "",
               f"門檻 T = {T:+.0f} dB" + ("（spec 達標）" if T == 0 else ""))
    axes[0].legend(fontsize=9, loc="upper left", framealpha=0.9).get_frame().set_edgecolor(C_GRID)
    fig.suptitle("達成機率曲線：P(k 輪內 best ≥ T)（Kaplan–Meier,右截斷校正）", color=C_INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "attain_prob.png"), dpi=140, facecolor=C_SURF)
    plt.close(fig)

    # ==== 圖 3：池分布（生存函數,尾巴＝上界所在） ====
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=C_SURF)
    surv = 1.0 - np.arange(1, len(pool) + 1) / len(pool)
    ax.plot(pool, np.maximum(surv, 1.0 / len(pool)), color=C_INK2, lw=2)
    ax.set_yscale("log")
    _plainlog(ax.yaxis)
    for T in (-3.0, -1.0, 0.0):
        p = 1 - float(np.searchsorted(pool, T, side="left")) / len(pool)
        if p > 0:
            ax.scatter([T], [p], color=C_INK, s=24, zorder=5)
            ax.annotate(f"P(≥{T:+.0f}) = {p:.2e}", (T, p), xytext=(T - 7.5, p * 0.55),
                        color=C_INK, fontsize=9)
    ax.axvline(0, color=C_RED, ls=":", lw=1.2)
    ax.axvline(oracle, color=C_INK, ls=":", lw=1.2)
    ax.text(oracle + 0.3, 4e-3, f"oracle {oracle:+.2f}", color=C_INK, fontsize=9, rotation=90)
    ax.axvline(our_best[0], color=C_OURS, ls=":", lw=1.2)
    ax.text(our_best[0] - 0.75, 6e-4, f"我們史上最佳 {our_best[0]:.2f}", color=C_OURS,
            fontsize=9, rotation=90)
    _style(ax, "worst-margin (dB)", "P(單抽 ≥ x)  (log)",
           f"harvest 池 margin 生存函數（n={len(pool):,}；學長歷史分布的尾巴＝上界所在）")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "pool_dist.png"), dpi=140, facecolor=C_SURF)
    plt.close(fig)

    # ==== 圖 4：每輪期望邊際增益 ====
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=C_SURF)
    kk = np.unique(np.geomspace(10, 1000, 40).astype(int))
    ax.plot(kk, b / kk, color=C_OURS, lw=2.2, label=f"我們 fit：dE/dk = {b:.2f}/k")
    dr = np.asarray([_best_of_n(pool, int(N) + 1)[0] - _best_of_n(pool, int(N))[0] for N in kk])
    ax.plot(kk, np.maximum(dr, 1e-6), color=C_INK2, ls="--", lw=2, label="random 池抽樣 ΔE[best]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plainlog(ax.xaxis)
    _plainlog(ax.yaxis)
    _style(ax, "HFSS calls k (log)", "多跑一輪的期望增益 (dB, log)",
           "邊際報酬：第 k 輪再多跑一輪,期望多賺幾 dB")
    ax.legend(fontsize=9, framealpha=0.9).get_frame().set_edgecolor(C_GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "marginal_gain.png"), dpi=140, facecolor=C_SURF)
    plt.close(fig)

    print(f"\n圖 ×4 → {ASSETS}\\ (best_vs_k / attain_prob / pool_dist / marginal_gain)")


def main():
    ap = argparse.ArgumentParser(description="離線歷史基準：每輪 HFSS 的期望 best 與達成機率 (Round 06)")
    ap.add_argument("cmd", choices=["collect-ours", "collect-pool", "collect-senior", "report"])
    args = ap.parse_args()
    {"collect-ours": collect_ours, "collect-pool": collect_pool,
     "collect-senior": collect_senior, "report": report}[args.cmd]()


if __name__ == "__main__":
    main()
