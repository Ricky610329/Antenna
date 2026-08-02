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


#? 網格刻意小（27 組 × 600 筆）——參數多、樣本少就變成對 fit 分割過擬合。
#  rad_eff（輻射效率）是**先驗的物理選擇**不是超參數：D₀ 只管方向性，抓不到
#  「會共振但不輻射」；dev 上量到 pooled ρ +0.363→+0.413，主要修的正是 Gain 那一路。
#  精度：一律 CPU float64——float32 下 ρ 掉 0.07（Cholesky+eigh 對 B 的條件數敏感），
#  GPU float64 的 cusolverDnXsyevd 在本機直接報 INTERNAL_ERROR。
L1_GRID = [dict(er=er, q=q, gap=g, diag=g, rad_eff=True)
           for er in (3.0, 3.3, 3.55)
           for q in (8.0, 15.0, 30.0)
           for g in (1, 2, 3)]


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


def cmd_gate1(args):
    """L1 gate：**val 只跑這一次**。判準寫死在 `docs/log/analysis-08-diffsim.md` §1。

    ① 主 KPI = 裸 diffsim-wm 對 HFSS-wm 的 pooled Spearman ρ（判準 ≥0.40）
    ② 仿射校準（係數只在 fit 上擬）後的 S11 每頻點 MAE
    ③ 負片域 ρ 單獨報；凍結尺 30 單獨報
    """
    import json
    idx = D.load()
    split, _ = D.assign_split(idx)
    pf = os.path.join(D.CACHE_DIR, "l1_params.json")
    cfg = json.load(open(pf, encoding="utf-8"))["best"] if os.path.exists(pf) else \
        dict(er=3.55, q=15.0, gap=2, diag=2)
    print(f"L1 參數（fit 分割選出，未看過 val）：{cfg}")

    sel = pick(idx, split, "val")
    pred, dt = run_l1(idx, sel, batch=args.batch, device=args.device, **cfg)
    st, y = idx["stratum"][sel], idx["y"][sel]
    wm_p, _, _ = margins(pred)
    wm_t, _, _ = margins(y)
    print(f"val {len(sel)} 筆，{dt:.0f}s")
    rhos = report_rho(wm_p, wm_t, st, tag="L1 裸（val，主 KPI ①）")

    selc = pick(idx, split, "fit", args.calib)
    pc, _ = run_l1(idx, selc, batch=args.batch, device=args.device, **cfg)
    a, b = affine_fit(pc, idx["y"][selc])
    pa = affine_apply(pred, a, b)
    mae = np.abs(pa[:, :17] - y[:, :17]).mean(0)
    print(f"\n② 仿射校準（fit {len(selc)} 筆擬）後 S11 每頻點 MAE(dB)：中位 {np.median(mae):.2f}，"
          f"帶內 5:12 {mae[5:12].mean():.2f}，全帶 {mae.mean():.2f}")
    wm_a, _, _ = margins(pa)
    print(f"   （校準後 wm 的 pooled ρ = {rank_rho(wm_a, wm_t)[0]:+.3f}，僅供對照，非判準）")

    print(f"\n③ 負片域 ρ = {rhos.get('neg', float('nan')):+.3f}；"
          f"凍結尺 ρ = {rhos.get('frozen', float('nan')):+.3f}")
    ok = rhos.get("ALL", 0) >= 0.40
    print(f"\n===== GATE 1：pooled ρ = {rhos.get('ALL', float('nan')):+.3f} vs 判準 0.40 → "
          f"{'通過' if ok else '**不通過**'} =====")
    return ok


def main():
    ap = argparse.ArgumentParser(description="diffsim 驅動")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g1 = sub.add_parser("gate1")
    g1.add_argument("--batch", type=int, default=24)
    g1.add_argument("--device", default="cpu")
    g1.add_argument("--calib", type=int, default=300, help="仿射校準用的 fit 筆數/stratum")
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
    {"predict": cmd_predict, "scan": cmd_scan, "fitscan": cmd_fitscan, "gate1": cmd_gate1}[a.cmd](a)


if __name__ == "__main__":
    main()
