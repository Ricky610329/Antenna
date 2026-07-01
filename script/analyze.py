# -*- coding: utf-8 -*-
"""
script/analyze.py — 可重現的診斷分析工具（把散在對話裡的一次性分析收成子命令）。純讀 NAS。

子命令：
  volatility  各 run 每 epoch 的像素翻轉數 + sim_loss/r_feed 波動（探索量;trust/ensemble/refit 的效果）
  rad-repr    方向圖用 K 個 cosine mode 的最佳擬合殘差 vs K（表達力上限;全域 vs ±45°窗內）
  rad-error   已訓 rad head 的窗內 pred-vs-real 誤差（需載 SM;判「凍 trunk」是否是瓶頸）

用法：
  python -m script.analyze volatility --runs single_r3_explore single_r3_dip --labels E D
  python -m script.analyze rad-repr   --run single_r2_enstrust_harvest
  python -m script.analyze rad-error  --run single_r2_enstrust_harvest
"""
import argparse
import csv
import os
import statistics as st

import numpy as np
import torch

from antenna.utils import config, ROOTDIR
config.device = "cpu"


def _resolve(sub, must=None):
    rd = ROOTDIR.joinpath("result")
    for d in os.listdir(str(rd)):
        if sub in d and (must is None or must in d):
            return rd.joinpath(d)
    raise SystemExit(f"找不到 run 含 '{sub}'" + (f" & '{must}'" if must else ""))


def _rows(run):
    with open(str(run / "metrics.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _col(rows, key):
    return [float(r[key]) for r in rows if r.get(key, "") not in ("", "nan")]


def _mad(vals):
    d = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
    return st.median(d) if d else float("nan")


def _cos_basis(theta, K):
    theta = np.asarray(theta).reshape(-1).astype(float)
    span = (theta.max() - theta.min()) or 1.0
    phi = np.pi * (theta - theta.min()) / span
    return np.cos(np.arange(K).reshape(-1, 1) * phi.reshape(1, -1))    # (K, n_theta)


# ── volatility ────────────────────────────────────────────────────────────
def cmd_volatility(args):
    print(f"{'run':26}{'N':>5}{'翻轉/ep':>9}{'|Δsim|':>8}{'|Δr_feed|':>10}")
    labels = args.labels or args.runs
    for name, lab in zip(args.runs, labels):
        run = _resolve(name)
        rows = [r for r in _rows(run) if (r.get("pattern_hash") or "").strip() not in ("", "nan")]
        flips, prev = [], None
        for r in rows:
            pt = run / "patterns" / f"{r['pattern_hash']}.pt"
            if not pt.exists():
                continue
            pat = torch.load(str(pt), weights_only=True)[0].reshape(-1)
            if prev is not None:
                flips.append(int((pat != prev).sum()))
            prev = pat
        sl, rf = _col(rows, "sim_loss"), _col(rows, "r_feed")
        print(f"{lab:26}{len(flips):>5}{(st.median(flips) if flips else 0):>9}"
              f"{_mad(sl):>8.2f}{_mad(rf):>10.3f}")


# ── rad-repr（表達力上限：最佳擬合殘差 vs K）───────────────────────────────
def cmd_rad_repr(args):
    run = _resolve(args.run)
    raddir = str(run / "radiation")
    files = [x for x in os.listdir(raddir) if x.endswith(".pt")][: args.n]
    if not files:
        raise SystemExit(f"{run} 無 radiation/ 資料")
    _, y0 = torch.load(os.path.join(raddir, files[0]), weights_only=True)
    theta = y0[0].numpy()
    mask = np.abs(theta) <= args.window
    curves = []
    for fn in files:
        _, y = torch.load(os.path.join(raddir, fn), weights_only=True)
        curves += [y[1].numpy(), y[2].numpy()]          # phi0, phi90
    print(f"曲線={len(curves)} 窗內點={int(mask.sum())}/{len(theta)}  (最佳擬合=該K表達力上限)")
    print(f"{'K':>4}{'全域RMSE':>10}{f'窗內±{args.window:.0f}°':>10}")
    for K in args.k:
        B = _cos_basis(theta, K)
        g, w = [], []
        for r in curves:
            c, *_ = np.linalg.lstsq(B.T, r, rcond=None)
            res = B.T @ c - r
            g.append(float(np.sqrt((res ** 2).mean())))
            w.append(float(np.sqrt((res[mask] ** 2).mean())))
        print(f"{K:>4}{st.median(g):>10.2f}{st.median(w):>10.2f}")


# ── rad-error（已訓 head 的窗內 pred-vs-real）──────────────────────────────
def cmd_rad_error(args):
    from antenna.pattern import AntennaPattern
    from antenna.training import load_config, build_surrogate, setup_responses
    AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
    run = _resolve(args.run)
    cfg = load_config(str(run / "config.yaml"))
    config.checkpoint_save_path = str(run / "checkpoint")
    sm = build_surrogate(cfg, str(run / "checkpoint"), setup_responses(cfg))
    sm.pre_load_model(str(run / "checkpoint" / "sm.pth"), strict=True)
    sm.model.eval()
    raddir = str(run / "radiation")
    files = [x for x in os.listdir(raddir) if x.endswith(".pt")]
    files.sort(key=lambda fn: os.path.getmtime(os.path.join(raddir, fn)))
    files = files[-args.n:]                              # 最近 N（forgetting 最小）
    _, y0 = torch.load(os.path.join(raddir, files[0]), weights_only=True)
    theta = y0[0].float()
    sm.set_rad_theta(theta)
    mask = theta.abs() <= args.window
    gl, win = [], []
    with torch.no_grad():
        for fn in files:
            pat, y = torch.load(os.path.join(raddir, fn), weights_only=True)
            err = sm.rad_predict(pat.reshape(-1).float()) - y[1:3].float()
            gl.append(float((err ** 2).mean().sqrt()))
            win.append(float((err[:, mask] ** 2).mean().sqrt()))
    print(f"{run.name.split(']')[-1].strip()}  最近 {len(files)} 筆")
    print(f"  全域 RMSE 中位={st.median(gl):.2f} dB")
    print(f"  窗內 ±{args.window:.0f}° RMSE 中位={st.median(win):.2f} dB  (對比 floor;越大=凍 trunk 越擬不準)")


def main():
    ap = argparse.ArgumentParser(description="可重現診斷分析（純讀 NAS）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("volatility"); v.add_argument("--runs", nargs="+", required=True); v.add_argument("--labels", nargs="*")
    v.set_defaults(func=cmd_volatility)
    rr = sub.add_parser("rad-repr"); rr.add_argument("--run", required=True); rr.add_argument("--n", type=int, default=40)
    rr.add_argument("--k", nargs="+", type=int, default=[4, 6, 8, 10, 12, 16, 24]); rr.add_argument("--window", type=float, default=45)
    rr.set_defaults(func=cmd_rad_repr)
    re = sub.add_parser("rad-error"); re.add_argument("--run", required=True); re.add_argument("--n", type=int, default=30)
    re.add_argument("--window", type=float, default=45); re.set_defaults(func=cmd_rad_error)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
