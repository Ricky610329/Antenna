# -*- coding: utf-8 -*-
"""R81 v2 濾波器：全曲線代理模型（SM）＋下一批選樣，產出 dedust worker 可吃的 input 夾。

只用 dual（濾波器）資料，絕不混入 single 天線資料：
- 寬頻真值：wide root 底下任何 ``measurement.json`` 描述為 dual／p01／Discrete、
  solver 與 v2 config 相同的 store（results.json 中 status=="ok"），依頻率對位到
  22–40 GHz／0.5（37 點）網格並帶 mask；16–40 舊 survey 與日後 22–40 新 store 都吃。
- 歷史 dual：24–32 GHz 17 點（本機 prior 快照），只監督該段，其餘 mask。

模型：MLP 625→512→256→3×37 預測 S11/S21/S22 dB，masked MSE（S21 權重 1、S11/S22 0.2），
5 顆 seed ensemble。先歷史預訓，再「全部歷史＋寬頻真值（樣本權重 ×20）」微調。
WM 由預測 S21 依 v2 score_spec 計（向量化版與 ``antenna.measurement.score_response``
同規則，測試對帳）；寬頻真值的 WM 一律直接呼叫 score_response。

用法（repo 根）::

    python -m script.r81_curve_sm build --batch 1 --out <dir> [--n 60] [--wide-root ...]
    python -m script.r81_curve_sm score --stores <store dir> [<store dir> ...]

選樣臂（預設 n=60）：exploit 30（預測 WM 均值前段＋maxmin Hamming、每 parent ≤4）、
explore 18（UCB=均值+1.0·std，均值須 ≥ 候選中位數）、random 12（候選均勻抽，SM 對照臂）。
holdout 品質不過（ρ≤0.3 或 WM MAE 不勝常數基準）時，exploit 降為 n/6，釋出名額平分給
explore/random。不跑 HFSS、不寫 NAS（只讀）。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from antenna.measurement import (frequency_grid, measurement_id, score_response,
                                 score_spec_id)
from script.profiled_batch import load_profile_config, pattern_sha256, validate_input

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "dual_r81_v4a.yaml"    # 10/13 先過 v4a（阻帶 −15 + S11/S22 匹配）再收緊到 v4（−20）；量測同 v2/v3
HIST_DIR = REPO / "tmp" / "r81_filter_prior_full_v2_20261007"
WIDE_ROOT = Path(r"T:\碩二_鄒穎麒's\antenna\experiments\r81_wide_filter_20261007")
LABELS = ["S11", "S21", "S22"]
LABEL_WEIGHT = np.array([1.0, 1.0, 1.0], dtype=np.float32)   # v4 起 S11/S22 也計分，三條曲線同權
SKIP_DIRS = {"rad", "jobs_state", "snapshots", "raw_snapshot"}


def load_cfg(path=CONFIG):
    cfg = load_profile_config(path)
    if cfg is None or cfg.port != "dual":
        raise ValueError(f"{path} 不是 dual 具名量測 config")
    return cfg


# ---------------------------------------------------------------- 資料

def align(freqs, response, grid):
    """把 (3,F) 響應依頻率對位到 grid，回傳 (3,G) 值與 mask；不在 grid 上的頻點丟棄。"""
    freqs = np.asarray(freqs, dtype=float)
    y = np.asarray(response, dtype=np.float32)
    if y.shape != (len(LABELS), freqs.size):
        raise ValueError(f"response 形狀 {y.shape} 與頻率數 {freqs.size} 不符")
    out = np.zeros((len(LABELS), grid.size), dtype=np.float32)
    mask = np.zeros_like(out, dtype=bool)
    for j, f in enumerate(grid):
        hit = np.flatnonzero(np.isclose(freqs, f, rtol=0.0, atol=1e-6))
        if hit.size == 1:
            out[:, j] = y[:, hit[0]]
            mask[:, j] = np.isfinite(y[:, hit[0]])
    out[~mask] = 0.0
    return out, mask


def _same_instrument(m, cfg):
    ref = cfg.measurement
    return (m.get("port") == "dual" and m.get("labels") == LABELS
            and m.get("geometry") == ref["geometry"] and m.get("solver") == ref["solver"]
            and m.get("sweep", {}).get("type") == "Discrete")


def wide_store_dirs(root, cfg):
    """wide root 下所有同儀器（dual/p01/Discrete/同 solver）且有 results.json 的 store。"""
    found = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS)
        if "measurement.json" in fn and "results.json" in fn:
            m = json.loads(Path(dp, "measurement.json").read_text(encoding="utf-8"))
            if _same_instrument(m, cfg):
                found.append(Path(dp))
    return found


def input_hashes(root):
    """wide root 下所有 *_input 夾 manifest 的 pattern_sha256（已排入＝不再選）。"""
    out = set()
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in SKIP_DIRS]
        if Path(dp).name.endswith("_input") and "manifest.json" in fn:
            for row in json.loads(Path(dp, "manifest.json").read_text(encoding="utf-8")):
                if isinstance(row, dict) and row.get("pattern_sha256"):
                    out.add(row["pattern_sha256"])
    return out


def load_store(store, grid):
    """讀一個 store 的 ok 條目 → list of dict(pattern, y, mask, sha, lineage, id)。"""
    store = Path(store)
    m = json.loads((store / "measurement.json").read_text(encoding="utf-8"))
    results = json.loads((store / "results.json").read_text(encoding="utf-8"))
    rows = []
    for rid, e in sorted(results.items()):
        if not isinstance(e, dict) or e.get("status") != "ok":
            continue
        p, y = torch.load(store / e["sample_file"], weights_only=True, map_location="cpu")
        p = np.asarray(p) > 0.5
        freqs = e.get("freqs") or frequency_grid(m).tolist()
        yy, mk = align(freqs, np.asarray(y), grid)
        rows.append({"id": rid, "store": store.name, "pattern": p, "y": yy, "mask": mk,
                     "sha": pattern_sha256(p), "lineage": str(e.get("lineage_id", rid))})
    return rows


def load_wide(root, cfg, grid):
    rows, seen = [], set()
    for store in wide_store_dirs(root, cfg):
        for r in load_store(store, grid):
            if r["sha"] not in seen:          # 同圖重量測只留第一筆
                seen.add(r["sha"])
                rows.append(r)
    return rows


def load_history(hist_dir, grid):
    """本機 24–32 GHz dual prior（manifest.json + raw_snapshot/）→ 對位到 grid。"""
    hist_dir = Path(hist_dir)
    man = json.loads((hist_dir / "manifest.json").read_text(encoding="utf-8"))
    P, Y, M, sha, lin = [], [], [], [], []
    for r in man:
        if r.get("labels") != LABELS:
            continue
        p, y = torch.load(hist_dir / "raw_snapshot" / r["sample_file"], weights_only=True,
                          map_location="cpu")
        p = np.asarray(p) > 0.5
        yy, mk = align(r["freqs_ghz"], np.asarray(y), grid)
        P.append(p.reshape(-1))
        Y.append(yy)
        M.append(mk)
        sha.append(r.get("pattern_sha256") or pattern_sha256(p))
        lin.append(str(r.get("lineage_id", r["id"])))
    return {"P": np.asarray(P, dtype=np.float32), "Y": np.asarray(Y), "M": np.asarray(M),
            "sha": sha, "lineage": lin}


# ---------------------------------------------------------------- 評分

def band_slices(score_spec, grid):
    out = []
    for b in score_spec["bands"]:
        lo = int(np.flatnonzero(np.isclose(grid, b["start_ghz"]))[0])
        hi = int(np.flatnonzero(np.isclose(grid, b["stop_ghz"]))[0])
        out.append((b["name"], LABELS.index(b["label"]), lo, hi + 1, b["relation"], b["threshold"]))
    return out


def margins_vec(Y, slices):
    """Y (N,3,G) → (N,B) 各段 margin（與 score_response 同規則：最差點、端點含）。"""
    cols = []
    for _, li, lo, hi, rel, thr in slices:
        v = Y[:, li, lo:hi]
        cols.append((v - thr).min(axis=1) if rel == "ge" else (thr - v).min(axis=1))
    return np.stack(cols, axis=1)


def true_wm(row, cfg, grid):
    """寬頻真值的 v2 WM（官方 score_response）；需要的 S21 頻點沒全覆蓋時回 None。"""
    s21 = LABELS.index("S21")
    for _, li, lo, hi, *_ in band_slices(cfg.score_spec, grid):
        if not row["mask"][li, lo:hi].all():
            return None
    lo = min(s[2] for s in band_slices(cfg.score_spec, grid))
    hi = max(s[3] for s in band_slices(cfg.score_spec, grid))
    if not row["mask"][s21, lo:hi].all():
        return None
    return score_response(row["y"][:, lo:hi], LABELS, grid[lo:hi], cfg.score_spec)


# ---------------------------------------------------------------- 模型

class CurveMLP(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(625, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                 nn.Linear(256, n_out))

    def forward(self, x):
        return self.net(x)


def masked_mse(pred, target, mask, label_w, sample_w=None):
    """sum(w·mask·(pred−y)²)/sum(mask)；有 sample_w 時分子分母都乘樣本權重。"""
    m = mask.float()
    if sample_w is not None:
        m = m * sample_w.view(-1, *([1] * (mask.dim() - 1)))
    err = (pred - target) ** 2 * label_w * m
    return err.sum() / m.sum().clamp_min(1e-8)


def _norm_stats(P, Y, M):
    xm, xs = P.mean(0), P.std(0)
    xs[xs < 1e-3] = 1.0
    cnt = M.sum(0).clip(min=1)
    ym = (Y * M).sum(0) / cnt
    ys = np.sqrt((((Y - ym) ** 2) * M).sum(0) / cnt)
    ys[(ys < 1e-3) | (M.sum(0) == 0)] = 1.0
    return xm, xs, ym, ys


def _fit_one(seed, P, Y, M, W, stats, pre_idx, pre_epochs, ft_epochs, batch=256):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xm, xs, ym, ys = stats
    X = torch.tensor((P - xm) / xs, dtype=torch.float32)
    T = torch.tensor((Y - ym) / ys, dtype=torch.float32).reshape(len(Y), -1)
    Mk = torch.tensor(M).reshape(len(M), -1)
    Wt = torch.tensor(W, dtype=torch.float32)
    lw = torch.tensor(np.repeat(LABEL_WEIGHT, Y.shape[2]), dtype=torch.float32)
    model = CurveMLP(T.shape[1])
    for idx, epochs, lr in ((pre_idx, pre_epochs, 1e-3), (np.arange(len(P)), ft_epochs, 3e-4)):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        for _ in range(epochs):
            order = rng.permutation(idx)
            for k in range(0, len(order), batch):
                b = torch.as_tensor(order[k:k + batch])
                loss = masked_mse(model(X[b]), T[b], Mk[b], lw, Wt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
    model.eval()
    return model


class Ensemble:
    def __init__(self, models, stats, shape):
        self.models, self.stats, self.shape = models, stats, shape

    @torch.no_grad()
    def predict(self, P, chunk=8192):
        """P (N,625) → (K,N,3,G) dB。"""
        xm, xs, ym, ys = self.stats
        out = []
        for m in self.models:
            parts = []
            for k in range(0, len(P), chunk):
                x = torch.tensor((np.asarray(P[k:k + chunk], dtype=np.float32) - xm) / xs)
                parts.append(m(x).numpy().reshape(-1, *self.shape))
            out.append(np.concatenate(parts) * ys + ym)
        return np.stack(out)


def train_ensemble(hist, wide_rows, *, seeds=(0, 1, 2, 3, 4), wide_weight=20.0,
                   pre_epochs=30, ft_epochs=20):
    wP = np.asarray([r["pattern"].reshape(-1) for r in wide_rows], dtype=np.float32).reshape(-1, 625)
    P = np.concatenate([hist["P"], wP])
    Y = np.concatenate([hist["Y"], np.asarray([r["y"] for r in wide_rows]).reshape(-1, *hist["Y"].shape[1:])])
    M = np.concatenate([hist["M"], np.asarray([r["mask"] for r in wide_rows]).reshape(-1, *hist["M"].shape[1:])])
    W = np.concatenate([np.ones(len(hist["P"])), np.full(len(wide_rows), wide_weight)])
    stats = _norm_stats(P, Y, M)
    pre_idx = np.arange(len(hist["P"]))
    models = [_fit_one(s, P, Y, M, W, stats, pre_idx, pre_epochs, ft_epochs) for s in seeds]
    return Ensemble(models, stats, Y.shape[1:])


def _spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def holdout_eval(hist, wide_rows, cfg, grid, *, frac=0.2, seed=81, **train_kw):
    """寬頻真值按 lineage 家族切 holdout（家族不跨），回報 S21 各段 MAE、WM ρ 與常數基準。"""
    scored = [(r, true_wm(r, cfg, grid)) for r in wide_rows]
    scored = [(r, s) for r, s in scored if s is not None]
    fams = sorted({r["lineage"] for r, _ in scored})
    rng = np.random.default_rng(seed)
    hold_f, n_hold = set(), 0
    for f in rng.permutation(fams):
        if n_hold >= frac * len(scored):
            break
        hold_f.add(str(f))
        n_hold += sum(r["lineage"] == f for r, _ in scored)
    train = [r for r, _ in scored if r["lineage"] not in hold_f]
    hold = [(r, s) for r, s in scored if r["lineage"] in hold_f]
    hold_sha = {r["sha"] for r, _ in hold}
    # 同圖、同家族的歷史窄頻都拿掉：歷史裡同家族近親上百筆，留著會讓 holdout 通帶看起來偏準
    keep = np.array([s not in hold_sha and f not in hold_f
                     for s, f in zip(hist["sha"], hist["lineage"])])
    h = {k: (np.asarray(v)[keep] if k in ("P", "Y", "M") else [x for x, k2 in zip(v, keep) if k2])
         for k, v in hist.items()}
    uncovered = [r for r in wide_rows if true_wm(r, cfg, grid) is None and r["lineage"] not in hold_f]
    ens = train_ensemble(h, train + uncovered, **train_kw)
    rep_excluded = int((~keep).sum())
    pred = ens.predict(np.asarray([r["pattern"].reshape(-1) for r, _ in hold], dtype=np.float32))
    slices = band_slices(cfg.score_spec, grid)
    pwm = margins_vec(pred.reshape(-1, *pred.shape[2:]), slices).min(1).reshape(pred.shape[:2]).mean(0)
    twm = np.array([s["worst_margin"] for _, s in hold])
    train_wm = np.array([true_wm(r, cfg, grid)["worst_margin"] for r in train])
    mean_pred = pred.mean(0)
    s21 = LABELS.index("S21")
    band_mae = {name: float(np.mean([np.abs(mean_pred[i, s21, lo:hi] - r["y"][s21, lo:hi]).mean()
                                     for i, (r, _) in enumerate(hold)]))
                for name, _, lo, hi, *_ in slices}
    const = float(train_wm.mean())
    med = float(np.median(train_wm))
    rep = {"holdout_families": sorted(hold_f), "n_holdout": len(hold), "n_train_wide": len(train),
           "n_history_excluded": rep_excluded,
           "s21_band_mae_db": band_mae,
           "wm_mae": float(np.abs(pwm - twm).mean()),
           "wm_mae_const": float(np.abs(const - twm).mean()), "const_wm": const,
           # 參考用：train 中位數常數（不受隨機爛設計拉低，比平均嚴格）；不參與 quality_ok
           "wm_mae_const_median": float(np.abs(med - twm).mean()), "const_wm_median": med,
           "true_wm_range": [float(twm.min()), float(twm.max())] if len(twm) else None,
           "pred_wm_range": [float(pwm.min()), float(pwm.max())] if len(pwm) else None,
           "spearman_wm": _spearman(pwm, twm),
           "holdout": [{"id": r["id"], "lineage": r["lineage"], "true_wm": float(t), "pred_wm": float(p)}
                       for (r, _), t, p in zip(hold, twm, pwm)]}
    rho = rep["spearman_wm"]
    rep["quality_ok"] = bool(np.isfinite(rho) and rho > 0.3 and rep["wm_mae"] < rep["wm_mae_const"])
    return rep


# ---------------------------------------------------------------- 候選與選樣

def pick_parents(wide_rows, hist, cfg, grid, n_wide=12, n_hist=12):
    parents = []
    scored = sorted(((true_wm(r, cfg, grid), r) for r in wide_rows), key=lambda t: -(t[0] or {}).get("worst_margin", -1e9))
    for s, r in scored:
        if s is not None and len(parents) < n_wide:
            parents.append({"id": f"{r['store']}:{r['id']}", "pattern": r["pattern"], "family": r["lineage"],
                            "source": "wide", "wm": s["worst_margin"], "sha": r["sha"]})
    # 歷史：24–24.5 已 ≤−15 dB，按 26–30 通帶 margin 排
    s21 = LABELS.index("S21")
    i24, i245 = (int(np.flatnonzero(np.isclose(grid, f))[0]) for f in (24.0, 24.5))
    lo, hi = (int(np.flatnonzero(np.isclose(grid, f))[0]) for f in (26.0, 30.0))
    Y, M = hist["Y"], hist["M"]
    ok = M[:, s21, [i24, i245]].all(1) & (Y[:, s21, [i24, i245]].max(1) <= -15) & M[:, s21, lo:hi + 1].all(1)
    passm = (Y[:, s21, lo:hi + 1] + 3).min(1)
    used = {p["sha"] for p in parents}
    for i in np.argsort(-np.where(ok, passm, -np.inf)):
        if not ok[i] or len(parents) >= n_wide + n_hist:
            break
        if hist["sha"][i] in used:
            continue
        used.add(hist["sha"][i])
        parents.append({"id": f"hist:{hist['sha'][i][:12]}", "pattern": hist["P"][i].reshape(25, 25) > 0.5,
                        "family": hist["lineage"][i], "source": "history", "pass_margin": float(passm[i]),
                        "sha": hist["sha"][i]})
    return parents


def enforce_dual_geometry(pattern):
    """強制 p01 dual 模擬器需要的上下兩個 5×5 埠墊為金屬。"""
    p = np.asarray(pattern, dtype=np.float32)
    if p.size != 625:
        raise ValueError(f"dual pattern must contain 625 pixels, got {p.size}")
    p = p.reshape(25, 25) > 0.5
    p[:5, 10:15] = True
    p[20:25, 10:15] = True
    return p


def mutate_pattern(parent, rng, *, min_flips=2, max_flips=40, allowed=None):
    """隨機翻 min_flips..max_flips 個不重複像素，再補回埠墊。"""
    p = (np.asarray(parent).reshape(25, 25) > 0.5).copy()
    n = int(rng.integers(min_flips, max_flips + 1))
    pool = np.arange(625) if allowed is None else np.flatnonzero(allowed)
    idx = rng.choice(pool, size=min(n, len(pool)), replace=False)
    p.reshape(-1)[idx] = ~p.reshape(-1)[idx]
    return enforce_dual_geometry(p)


def focus_mask(spec):
    """"5-9,15-19" → 625 長度 bool，只允許這些列；空字串＝全板。"""
    if not spec:
        return None
    m = np.zeros((25, 25), bool)
    for part in spec.split(","):
        a, _, b = part.partition("-")
        m[int(a):int(b or a) + 1] = True
    return m.reshape(-1)


def crossover(a, b, rng):
    """同家族交換：一段連續行或一個矩形區塊取自 b。"""
    c = a.copy()
    if rng.random() < 0.5:
        r0 = int(rng.integers(0, 24))
        r1 = int(rng.integers(r0 + 1, 26))
        c[r0:r1] = b[r0:r1]
    else:
        r0, c0 = (int(x) for x in rng.integers(0, 20, size=2))
        h, w = (int(x) for x in rng.integers(3, 11, size=2))
        c[r0:r0 + h, c0:c0 + w] = b[r0:r0 + h, c0:c0 + w]
    return enforce_dual_geometry(c)


def generate(parents, exclude, n, rng, *, min_flips=2, max_flips=40, p_cross=0.1, allowed=None):
    """變異＋同家族交叉產 n 個去重候選；歷史 parent 本身（新窗沒量過）也列為候選。"""
    fam = {}
    for i, p in enumerate(parents):
        fam.setdefault(p["family"], []).append(i)
    seen = set(exclude)
    pats, meta = [], []

    def add(pat, pi, op):
        pat = enforce_dual_geometry(pat)
        key = pattern_sha256(pat)
        if key in seen:
            return
        seen.add(key)
        pats.append(pat)
        meta.append({"parent": parents[pi]["id"], "family": parents[pi]["family"], "op": op, "sha": key})

    for i, p in enumerate(parents):
        if p["source"] == "history":
            add(p["pattern"], i, "copy")
    tries = 0
    while len(pats) < n and tries < 20 * n:
        tries += 1
        i = int(rng.integers(len(parents)))
        mates = [j for j in fam[parents[i]["family"]] if j != i]
        if mates and rng.random() < p_cross:
            j = int(rng.choice(mates))
            add(crossover(parents[i]["pattern"], parents[j]["pattern"], rng), i, "cross")
        else:
            add(mutate_pattern(parents[i]["pattern"], rng, min_flips=min_flips,
                               max_flips=max_flips, allowed=allowed), i, "mutate")
    return np.asarray(pats, dtype=bool).reshape(-1, 625), meta


def climb_candidates(tops, exclude, rng, *, n_double=400, allowed=None):
    """爬山鄰居：每個 top 設計的全部單格翻轉，加 n_double 個 2–3 格隨機翻轉；去重。"""
    pads = enforce_dual_geometry(np.zeros((25, 25), bool)).reshape(-1)
    seen, P, meta = set(exclude), [], []
    for t in tops:
        base = np.asarray(t["pattern"]).reshape(-1) > 0.5
        moves = [[int(i)] for i in np.flatnonzero(~pads if allowed is None else (~pads & allowed))]
        free = np.flatnonzero(~pads if allowed is None else (~pads & allowed))
        moves += [list(rng.choice(free, size=int(rng.integers(2, 4)), replace=False)) for _ in range(n_double)]
        for mv in moves:
            c = base.copy()
            c[mv] = ~c[mv]
            c = enforce_dual_geometry(c)
            key = pattern_sha256(c)
            if key in seen:
                continue
            seen.add(key)
            P.append(c.reshape(-1))
            meta.append({"parent": t["id"], "family": t["family"], "op": f"climb{len(mv)}", "sha": key})
    return np.asarray(P, dtype=np.float32).reshape(-1, 625), meta


def pick_climb(mean, meta, tops, n):
    """最好的 top 拿一半名額，其餘平分；各自按 SM 預測均值取前段。"""
    if not tops or n <= 0:
        return []
    share = [n - (n // 2) if k == 0 else 0 for k in range(len(tops))]
    rest = n - share[0]
    for k in range(1, len(tops)):
        share[k] = rest // (len(tops) - 1) + (1 if k - 1 < rest % (len(tops) - 1) else 0)
    picks = []
    for t, m in zip(tops, share):
        idx = [i for i in np.argsort(-mean, kind="stable") if meta[i]["parent"] == t["id"]]
        picks += [(int(i), "climb") for i in idx[:m]]
    return picks


def quotas(n, quality_ok):
    if quality_ok:
        e = round(n * 0.5)
        x = round(n * 0.3)
    else:
        e = n // 6
        x = (n - e) // 2
    return {"exploit": e, "explore": x, "random": n - e - x}


def select(P, mean, std, meta, q, rng, *, per_parent=4, pool_mult=3, exploit_lcb=0.0):
    """回傳 [(index, arm)]。exploit：前段池內 maxmin Hamming；explore：UCB；random：均勻。
    每臂每 parent ≤ per_parent；前兩臂湊不滿的缺額由 random 補。"""
    picked, count = [], {}          # count 以 (arm, parent) 計：每臂每 parent ≤ per_parent

    def ok(i, arm):
        return i not in taken and count.get((arm, meta[i]["parent"]), 0) < per_parent

    def take(i, arm):
        picked.append((int(i), arm))
        taken.add(int(i))
        count[(arm, meta[i]["parent"])] = count.get((arm, meta[i]["parent"]), 0) + 1

    taken = set()
    # exploit 用保守分數（均值 − k·std）：b07 實測 SM 對預測最高者高估 1–8 dB（贏家詛咒）
    lcb = mean - exploit_lcb * std
    order = np.argsort(-lcb, kind="stable")
    pool, in_pool = [], {}       # 前段池：每 parent 至多 2·per_parent 筆，避免池被少數 parent 佔滿
    for i in order:
        if len(pool) >= pool_mult * q["exploit"]:
            break
        par = meta[i]["parent"]
        if in_pool.get(par, 0) < 2 * per_parent:
            in_pool[par] = in_pool.get(par, 0) + 1
            pool.append(int(i))
    if q["exploit"] and pool:
        take(pool[0], "exploit")
        Pb = P.astype(bool)
        dmin = (Pb[pool] != Pb[pool[0]]).sum(1).astype(float)
        while sum(a == "exploit" for _, a in picked) < q["exploit"]:
            cand = [k for k, i in enumerate(pool) if ok(i, "exploit")]
            if not cand:
                break
            k = max(cand, key=lambda k: (dmin[k], lcb[pool[k]]))
            take(pool[k], "exploit")
            dmin = np.minimum(dmin, (Pb[pool] != Pb[pool[k]]).sum(1))
    floor = float(np.median(mean))
    for i in np.argsort(-(mean + std), kind="stable"):
        if sum(a == "explore" for _, a in picked) >= q["explore"]:
            break
        if mean[i] >= floor and ok(int(i), "explore"):
            take(int(i), "explore")
    rest = np.array([i for i in range(len(P)) if i not in taken])
    n_rand = sum(q.values()) - len(picked)     # 前兩臂若因上限湊不滿，缺額併入 random
    for i in rng.choice(rest, size=min(n_rand, len(rest)), replace=False):
        take(int(i), "random")
    return picked


# ---------------------------------------------------------------- 輸出

def write_input(out, cfg, config_text, P, picks, meta, pred, batch):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"拒絕覆寫非空輸出夾：{out}")
    out.mkdir(parents=True, exist_ok=True)
    mid, sid = measurement_id(cfg.measurement), score_spec_id(cfg.score_spec)
    rows, sel = [], []
    for seq, (i, arm) in enumerate(picks):
        pat = P[i].reshape(25, 25).astype(np.float32)
        sha = pattern_sha256(pat)
        rid = f"r81v2_b{batch:02d}_{seq:03d}_{sha[:8]}"
        torch.save(torch.tensor(pat), out / f"{rid}.pt")
        rows.append({"id": rid, "batch": batch, "port": "dual", "kind": "r81v2", "measurement_id": mid,
                     "score_spec_id": sid, "pattern_file": f"{rid}.pt", "pattern_sha256": sha,
                     "lineage_id": meta[i]["family"], "parent_id": meta[i]["parent"],
                     "selection_arm": arm, "op": meta[i]["op"],
                     "pred_wm_mean": pred["wm_mean"][i], "pred_wm_std": pred["wm_std"][i]})
        sel.append({"id": rid, "arm": arm, "parent": meta[i]["parent"], "family": meta[i]["family"],
                    "op": meta[i]["op"], "pred_wm_mean": pred["wm_mean"][i], "pred_wm_std": pred["wm_std"][i],
                    "pred_margins": pred["margins"][i]})
    (out / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "measurement.json").write_text(json.dumps(cfg.measurement, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "score_spec.json").write_text(json.dumps(cfg.score_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(config_text, encoding="utf-8")
    (out / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    validate_input(out, cfg)
    return rows, sel


def build(args):
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    cfg = load_cfg(args.config)
    grid = frequency_grid(cfg.measurement)
    hist = load_history(args.hist_dir, grid)
    wide = load_wide(args.wide_root, cfg, grid)
    print(f"歷史 {len(hist['P'])} 筆、寬頻真值 {len(wide)} 筆（{time.time() - t0:.0f}s）")
    kw = {"wide_weight": args.wide_weight, "pre_epochs": args.pre_epochs, "ft_epochs": args.ft_epochs,
          "seeds": tuple(range(args.ensemble))}
    t1 = time.time()
    rep = holdout_eval(hist, wide, cfg, grid, seed=args.seed, **kw)
    rep["holdout_train_s"] = time.time() - t1
    print(f"holdout：n={rep['n_holdout']} ρ={rep['spearman_wm']:.3f} WM MAE {rep['wm_mae']:.2f} "
          f"vs 常數(平均) {rep['wm_mae_const']:.2f}／常數(中位) {rep['wm_mae_const_median']:.2f} "
          f"→ quality_ok={rep['quality_ok']}")
    t2 = time.time()
    ens = train_ensemble(hist, wide, **kw)
    rep["final_train_s"] = time.time() - t2

    rng = np.random.default_rng(args.seed + args.batch)
    allowed = focus_mask(args.focus_rows)
    parents = pick_parents(wide, hist, cfg, grid)
    exclude = {r["sha"] for r in wide} | input_hashes(args.wide_root)
    P, meta = generate(parents, exclude, args.n_candidates, rng,
                       min_flips=args.min_flips, max_flips=args.max_flips, allowed=allowed)
    tops = [p for p in parents if p["source"] == "wide"][:args.climb_top]
    Pc, metac = climb_candidates(tops, exclude | {m["sha"] for m in meta}, rng, allowed=allowed)
    n_gen = len(P)
    P, meta = np.concatenate([P, Pc]), meta + metac
    pred = ens.predict(P.astype(np.float32))                       # (K,N,3,G)
    slices = band_slices(cfg.score_spec, grid)
    marg = np.stack([margins_vec(p, slices) for p in pred])        # (K,N,B)
    wm = marg.min(2)
    mean, std = wm.mean(0), wm.std(0)
    names = [s[0] for s in slices]
    pm = marg.mean(0)
    predd = {"wm_mean": [float(x) for x in mean], "wm_std": [float(x) for x in std],
             "margins": [{n: float(v) for n, v in zip(names, row)} for row in pm]}
    climb_n = min(args.climb_n, args.n) if tops else 0
    cmean = np.full(len(P), -np.inf)
    cmean[n_gen:] = mean[n_gen:] - args.exploit_lcb * std[n_gen:]   # 爬山同樣用保守分數
    picks = pick_climb(cmean, meta, tops, climb_n)
    q = quotas(args.n - len(picks), rep["quality_ok"])
    gen_idx = np.arange(n_gen)
    picks += [(int(gen_idx[i]), a) for i, a in select(P[:n_gen], mean[:n_gen], std[:n_gen], meta[:n_gen], q, rng,
                                                         exploit_lcb=args.exploit_lcb)]
    q = {"climb": len([1 for _, a in picks if a == "climb"]), **q}
    rows, sel = write_input(args.out, cfg, Path(args.config).read_text(encoding="utf-8"),
                            P, picks, meta, predd, args.batch)
    arms = {}
    for s in sel:
        arms.setdefault(s["arm"], []).append(s)
    rep.update({"quotas": q, "n_candidates": int(len(P)), "n_excluded_known": len(exclude),
                "parents": [{k: v for k, v in p.items() if k != "pattern"} for p in parents],
                "candidate_pred_wm": {"max": float(mean.max()), "p90": float(np.quantile(mean, .9)),
                                      "median": float(np.median(mean))},
                "arms": {a: {"n": len(v), "pred_wm_min": min(x["pred_wm_mean"] for x in v),
                             "pred_wm_max": max(x["pred_wm_mean"] for x in v),
                             "parents": sorted({x["parent"] for x in v})} for a, v in arms.items()},
                "config": str(args.config), "measurement_id": measurement_id(cfg.measurement),
                "score_spec_id": score_spec_id(cfg.score_spec), "train_kw": {k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()},
                "n_history": int(len(hist["P"])), "n_wide": len(wide),
                "wide_stores": sorted({r["store"] for r in wide}), "total_s": time.time() - t0})
    Path(args.out, "sm_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    for a, v in rep["arms"].items():
        print(f"{a:8s} n={v['n']:2d} pred WM {v['pred_wm_min']:+.2f}..{v['pred_wm_max']:+.2f}  parents={len(v['parents'])}")
    print(f"→ {args.out}（validate_input 通過；發車前照常 check-dup）")

V2_DATASET = WIDE_ROOT / "v2_v001" / "dataset"


def enqueue(args):
    """選好的 input 夾 → 切 k 片 → 上 NAS → check-dup → jobs-add（同一 prio、不綁機器）。"""
    src, root = Path(args.input), Path(args.dataset_root)
    cfg = load_profile_config(args.config)
    shutil.copy2(args.config, src / "config.yaml")          # jobs-add 要求與 config bytes 一致
    validate_input(src, cfg)
    rows = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    base = f"dedust_{args.name}"
    if (root / f"{base}_input").exists():
        raise SystemExit(f"{base}_input 已存在，換名字")
    shutil.copytree(src, root / f"{base}_input")
    dedust = [sys.executable, "-m", "script.dedust", "--dataset-root", str(root)]
    if subprocess.call(dedust + ["check-dup", "--input", f"{base}_input"]) != 0:
        raise SystemExit("check-dup 有重複，不發車")
    for k in range(args.shards):
        d = root / f"{base}s{k + 1}_input"
        d.mkdir()
        for f in ("config.yaml", "measurement.json", "score_spec.json"):
            shutil.copy2(src / f, d / f)
        sub = rows[k::args.shards]
        for r in sub:
            shutil.copy2(src / r["pattern_file"], d / r["pattern_file"])
        (d / "manifest.json").write_text(json.dumps(sub, ensure_ascii=False, indent=1), encoding="utf-8")
        validate_input(d, cfg)
        subprocess.check_call(dedust + ["jobs-add", "--scope", cfg.scope, "--config", str(args.config),
                                        "--input", d.name, "--store", f"{base}s{k + 1}",
                                        "--prio", str(args.prio)])


def score(args):
    cfg = load_cfg(args.config)
    grid = frequency_grid(cfg.measurement)
    for store in args.stores:
        rows = load_store(store, grid)
        print(f"== {Path(store).name}: {len(rows)} ok")
        out = []
        for r in rows:
            s = true_wm(r, cfg, grid)
            out.append((s["worst_margin"] if s else -np.inf, r, s))
        for wm, r, s in sorted(out, key=lambda t: -t[0]):
            if s is None:
                print(f"  {r['id']:32s} {r['lineage']:12s}  （未覆蓋 v2 頻段）")
            else:
                m = s["margins"]
                print(f"  {r['id']:32s} {r['lineage']:12s} WM {wm:+6.2f}  " +
                      " ".join(f"{k}={v:+.2f}" for k, v in m.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--batch", type=int, required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=60)
    b.add_argument("--wide-root", type=Path, default=WIDE_ROOT)
    b.add_argument("--hist-dir", type=Path, default=HIST_DIR)
    b.add_argument("--config", type=Path, default=CONFIG)
    b.add_argument("--n-candidates", type=int, default=20000)
    b.add_argument("--ensemble", type=int, default=5)
    b.add_argument("--wide-weight", type=float, default=20.0)
    b.add_argument("--pre-epochs", type=int, default=30)
    b.add_argument("--ft-epochs", type=int, default=20)
    b.add_argument("--seed", type=int, default=81)
    # b01 用 2–40 格變異，實測多數把通帶打壞；b02 起預設改小步 1–8 格（2026-10-13）
    # 爬山臂：實測前 climb_top 名的 1–3 格鄰居，SM 排序取 climb_n 筆（2026-10-13 b02 出 −1.33 後加）
    b.add_argument("--climb-top", type=int, default=3)
    # 只在這些列翻格，如 "5-9,15-19"＝兩個饋墊附近（匹配 S11/S22 主要由饋入點阻抗決定）
    b.add_argument("--focus-rows", default="")
    b.add_argument("--exploit-lcb", type=float, default=1.0)
    b.add_argument("--climb-n", type=int, default=24)
    b.add_argument("--min-flips", type=int, default=1)
    b.add_argument("--max-flips", type=int, default=8)
    q = sub.add_parser("enqueue", help="切片上 NAS、查重、入佇列（主批 prio 5／填空池 prio 9）")
    q.add_argument("--input", required=True)
    q.add_argument("--name", required=True, help="如 r81v2b03 → store dedust_r81v2b03s1..")
    q.add_argument("--prio", type=int, default=5)
    q.add_argument("--shards", type=int, default=6)
    q.add_argument("--dataset-root", type=Path, default=V2_DATASET)
    q.add_argument("--config", type=Path, default=CONFIG)
    s = sub.add_parser("score")
    s.add_argument("--stores", nargs="+", required=True)
    s.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args(argv)
    {"build": build, "score": score, "enqueue": enqueue}[args.cmd](args)


if __name__ == "__main__":
    main()
