# -*- coding: utf-8 -*-
"""script/diffsim/eval.py — 驗收：同一把尺 + rank ρ。

尺完全沿用批次線（`docs/diffsim.md` §2「下游零改動」）：
  - `worst_margin`：中央平台 = 索引 5:12（26.5–29.5GHz）；S11 method=low center −10、
    Gain method=high center 4（`configs/single_r5_explore.yaml` targets）。
  - `oob_bad`：遠帶外兩側各 4 點的 `gain_max − s11_min`（`script.dedust.oob_metrics`）。
本檔的向量化版本有 `test_diffsim.py::test_ruler_matches_worst_margin` 對 `antenna.losses.worst_margin`
逐筆比對把關——**不是另一把尺**。

主 KPI = `rank_rho(diffsim_wm, hfss_wm)`（Spearman，與 SM 前瞻 ρ 同口徑：`script.analyze` 用
`scipy.stats.spearmanr`）。
"""
import numpy as np

BAND = slice(5, 12)          # 中央平台（26.5–29.5GHz）
S11_CENTER = -10.0
GAIN_CENTER = 4.0
OOB_SIDE = 4                 # 遠帶外每側點數


def margins(y):
    """(N,34) 或 (34,) → (wm, s11_margin, gain_margin)，皆 (N,)。正 = 達標。"""
    y = np.asarray(y, dtype=np.float64).reshape(-1, 34)
    s11, gain = y[:, :17], y[:, 17:]
    m_s11 = S11_CENTER - s11[:, BAND].max(axis=1)
    m_gain = gain[:, BAND].min(axis=1) - GAIN_CENTER
    return np.minimum(m_s11, m_gain), m_s11, m_gain


def oob_bad(y):
    """(N,34) → 帶外總帳 oob_bad（越低越好）。"""
    y = np.asarray(y, dtype=np.float64).reshape(-1, 34)
    s11, gain = y[:, :17], y[:, 17:]
    far = list(range(OOB_SIDE)) + list(range(17 - OOB_SIDE, 17))
    return gain[:, far].max(axis=1) - s11[:, far].min(axis=1)


def rank_rho(a, b):
    """Spearman ρ（與 analyze 同口徑）。回 (rho, p)。"""
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan"), float("nan")
    r = spearmanr(a[m], b[m])
    return float(r.statistic), float(r.pvalue)


def boot_ci(a, b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """ρ 的 bootstrap 信賴區間。驗證集只有 120 筆，SE(ρ) ≈ 0.09——
    只報點估計會把精度講得比實際好，判準在門檻附近時尤其誤導。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        ix = rng.integers(0, n, n)
        vals[i] = rank_rho(a[ix], b[ix])[0]
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def report_rho(pred_wm, true_wm, strat, tag="") -> dict:
    """分層 rank ρ 表（主 KPI）。回 {stratum: rho}，同時印表（含 95% bootstrap CI）。"""
    out = {}
    print(f"\n== rank ρ(diffsim wm, HFSS wm){(' — ' + tag) if tag else ''} ==")
    print("| 分層 | n | ρ | 95% CI | p |")
    print("|---|---|---|---|---|")
    for s in ["ALL"] + sorted(set(np.asarray(strat).tolist())):
        m = np.ones(len(true_wm), bool) if s == "ALL" else (np.asarray(strat) == s)
        if m.sum() < 3:
            continue
        pw, tw = np.asarray(pred_wm)[m], np.asarray(true_wm)[m]
        rho, p = rank_rho(pw, tw)
        lo, hi = boot_ci(pw, tw)
        out[s] = rho
        print(f"| {s} | {int(m.sum())} | {rho:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.2g} |")
    return out
