# -*- coding: utf-8 -*-
"""script/diffsim/run.py — diffsim 驅動：預測、擬合純量、仿射校準、gate 報數。

紀律（`docs/log/analysis-08-diffsim.md` §1）：
  - 調參/診斷 **只看 `dev`**；`val` 只在 gate 報數時看一次。
  - 仿射校準的係數只能在 `fit` 上擬（與 val/dev 不相交）；凍結尺永不進擬合。

用法：
    python -m script.diffsim.run predict --split dev --n 200            # 跑 L1 存快取
    python -m script.diffsim.run scan --split dev --n 200               # (er, Q) 網格掃 ρ
    python -m script.diffsim.run gate1                                  # L1 gate：val 報數
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from . import data as D                        # noqa: E402
from .eval import margins, rank_rho, report_rho  # noqa: E402

CACHE = os.path.join(D.CACHE_DIR, "pred")


def pick(idx, split_arr, which: str, n_per: int = None, strata=None, seed: int = 0):
    """挑樣本：回 index 陣列（每 stratum 取 n_per 筆，決定性）。"""
    strata = strata or ["clean", "neg", "senior", "frozen"]
    out = []
    for s in strata:
        m = np.where((split_arr == which) & (idx["stratum"] == s))[0]
        if n_per is not None and len(m) > n_per:
            #! 種子不可用 Python 的 hash()——它每個 process 都不同（PYTHONHASHSEED 隨機化），
            #  同一條指令跑兩次會抽到不同樣本。改用內容穩定的雜湊。
            tag = int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)
            m = np.sort(np.random.default_rng(seed + tag).choice(m, n_per, replace=False))
        out.append(m)
    return np.concatenate(out) if out else np.array([], dtype=int)


def run_l1(idx, sel, *, er, q, n_modes=None, device="cpu", batch=16, dtype=None, **kw):
    from .l1 import CavityL1
    import torch
    m = CavityL1(n_modes=n_modes, er_eff=er, q=q, device=device,
                 dtype=dtype or torch.float64, **kw)
    t = time.time()
    pred = m.predict(idx["x"][sel].astype(np.float64), batch=batch)
    return pred, time.time() - t


def affine_fit(pred_fit, y_fit):
    """每頻點仿射 a·x+b（最小平方）。回 (a (34,), b (34,))。"""
    a = np.ones(34)
    b = np.zeros(34)
    for j in range(34):
        x, y = pred_fit[:, j], y_fit[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5 or np.std(x[ok]) < 1e-9:
            b[j] = np.mean(y[ok]) - np.mean(x[ok]) if ok.any() else 0.0
            continue
        A = np.stack([x[ok], np.ones(ok.sum())], 1)
        sol, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        a[j], b[j] = sol
    return a, b


def affine_apply(pred, a, b):
    return pred * a[None, :] + b[None, :]


def _cache_path(tag):
    os.makedirs(CACHE, exist_ok=True)
    return os.path.join(CACHE, f"{tag}.npz")


def cmd_predict(args):
    idx = D.load()
    split, _ = D.assign_split(idx)
    sel = pick(idx, split, args.split, args.n)
    print(f"{args.split}: {len(sel)} 筆 | er={args.er} Q={args.q} modes={args.modes} dev={args.device}")
    pred, dt = run_l1(idx, sel, er=args.er, q=args.q, n_modes=args.modes,
                      device=args.device, batch=args.batch)
    print(f"  {dt:.1f}s ({dt / max(len(sel), 1) * 1000:.0f} ms/筆)")
    np.savez_compressed(_cache_path(args.tag or f"l1_{args.split}"), pred=pred, sel=sel,
                        er=args.er, q=args.q, modes=args.modes)
    wm_p, _, _ = margins(pred)
    wm_t, _, _ = margins(idx["y"][sel])
    report_rho(wm_p, wm_t, idx["stratum"][sel], tag=f"L1 裸 ({args.split})")


def cmd_scan(args):
    """(er, Q) 網格：**只在 dev 上掃**，找可用起點。"""
    idx = D.load()
    split, _ = D.assign_split(idx)
    sel = pick(idx, split, args.split, args.n)
    y = idx["y"][sel]
    wm_t, _, _ = margins(y)
    ers = [float(v) for v in args.ers.split(",")]
    qs = [float(v) for v in args.qs.split(",")]
    print(f"掃描 {len(sel)} 筆（{args.split}）：er {ers} × Q {qs}")
    print("| er | Q | ρ(ALL) | ρ(clean) | ρ(neg) | ρ(senior) |")
    print("|---|---|---|---|---|---|")
    best = None
    for er in ers:
        for q in qs:
            pred, dt = run_l1(idx, sel, er=er, q=q, n_modes=args.modes,
                              device=args.device, batch=args.batch)
            wm_p, _, _ = margins(pred)
            row = [er, q]
            for s in ["ALL", "clean", "neg", "senior"]:
                m = np.ones(len(sel), bool) if s == "ALL" else (idx["stratum"][sel] == s)
                row.append(rank_rho(wm_p[m], wm_t[m])[0] if m.sum() > 3 else float("nan"))
            print("| " + " | ".join(f"{v:+.3f}" if i > 1 else f"{v:g}" for i, v in enumerate(row)) + " |",
                  flush=True)
            if best is None or row[2] > best[2]:
                best = row
    print(f"\n最佳（dev）: er={best[0]} Q={best[1]} ρ={best[2]:+.3f}")


L1_GRID = [dict(er=er, q=q, gap=g, diag=d)
           for er in (3.0, 3.3, 3.55, 3.9)
           for q in (8.0, 15.0, 30.0)
           for g, d in ((0, 0), (1, 1), (2, 2), (3, 3))]


def cmd_fitscan(args):
    """L1 純量擬合：**只在 fit 分割上選**（與 dev/val 不相交），選 pooled ρ(wm) 最大者。"""
    idx = D.load()
    split, _ = D.assign_split(idx)
    sel = pick(idx, split, "fit", args.n)
    y = idx["y"][sel]
    wm_t, _, _ = margins(y)
    st = idx["stratum"][sel]
    print(f"fit 選參：{len(sel)} 筆（{args.n}/stratum）× {len(L1_GRID)} 組")
    print("| er | Q | gap | diag | ρ(pooled) | ρ(clean) | ρ(neg) | ρ(senior) |")
    print("|---|---|---|---|---|---|---|---|")
    rows = []
    for cfg in L1_GRID:
        pred, _ = run_l1(idx, sel, batch=args.batch, device=args.device, **cfg)
        wm_p, _, _ = margins(pred)
        pooled = rank_rho(wm_p, wm_t)[0]
        per = [rank_rho(wm_p[st == s], wm_t[st == s])[0] for s in ("clean", "neg", "senior")]
        rows.append((pooled, cfg, per))
        print(f"| {cfg['er']} | {cfg['q']:g} | {cfg['gap']} | {cfg['diag']} | {pooled:+.3f} | "
              + " | ".join(f"{v:+.3f}" for v in per) + " |", flush=True)
    rows.sort(key=lambda t: -t[0])
    print(f"\n**fit 最佳**：{rows[0][1]} → ρ(pooled)={rows[0][0]:+.3f}")
    with open(os.path.join(D.CACHE_DIR, "l1_params.json"), "w", encoding="utf-8") as fh:
        import json
        json.dump(dict(best=rows[0][1], rho_fit=rows[0][0], n=len(sel)), fh, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="diffsim 驅動")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fs = sub.add_parser("fitscan")
    fs.add_argument("--n", type=int, default=200, help="每 stratum 取幾筆（fit 分割）")
    fs.add_argument("--batch", type=int, default=24)
    fs.add_argument("--device", default="cpu")
    for name in ("predict", "scan"):
        p = sub.add_parser(name)
        p.add_argument("--split", default="dev")
        p.add_argument("--n", type=int, default=None, help="每 stratum 取幾筆")
        p.add_argument("--modes", type=int, default=30)
        p.add_argument("--device", default="cpu")
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--tag", default=None)
        if name == "predict":
            p.add_argument("--er", type=float, default=3.55)
            p.add_argument("--q", type=float, default=20.0)
        else:
            p.add_argument("--ers", default="2.8,3.1,3.55")
            p.add_argument("--qs", default="8,20,50")
    a = ap.parse_args()
    {"predict": cmd_predict, "scan": cmd_scan, "fitscan": cmd_fitscan}[a.cmd](a)


if __name__ == "__main__":
    main()
