# -*- coding: utf-8 -*-
"""
script/analyze.py — 可重現的診斷分析工具（把散在對話裡的一次性分析收成子命令）。純讀 NAS。

子命令：
  volatility  各 run 每 epoch 的像素翻轉數 + sim_loss/r_feed 波動（探索量;trust/ensemble/refit 的效果）
  rad-repr    方向圖用 K 個 cosine mode 的最佳擬合殘差 vs K（表達力上限;全域 vs ±45°窗內）
  rad-error   已訓 rad head 的窗內 pred-vs-real 誤差（需載 SM;判「凍 trunk」是否是瓶頸）
  gain        性能期望儀表（三層帳:階梯命中率/學習曲線斜率/近王→紀錄轉換）——
              本意=用實測數字擋「過早悲觀/過早樂觀」,不做尾巴外推（偽精確禁區）

用法：
  python -m script.analyze volatility --runs single_r3_explore single_r3_dip --labels E D
  python -m script.analyze rad-repr   --run single_r2_enstrust_harvest
  python -m script.analyze rad-error  --run single_r2_enstrust_harvest
  python -m script.analyze gain --line r21 [--record 0.39]
"""
import argparse
import csv
import os
import statistics as st
import sys

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # cp950 console 印全形/符號

from antenna.utils import config, ROOTDIR, DATASET_PATH
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


def cmd_components(args):
    """組件尺寸分布 vs 三標（Ricky 觀察「N 組塊大小很不平均」的量化;2026-07-09,analysis-02）。
    掃全部 dedust store 去重真值 → 每 pattern 的組件尺寸統計（n/main/second/minc/cv/metal）
    對 wm/rad/oob 做 Spearman（全樣本+作戰區）＋三標過 profile。零 HFSS,可重跑。"""
    import json
    from scipy.ndimage import label as _label
    from scipy.stats import spearmanr
    from script.dedust import _CROSS, oob_metrics
    from antenna.utils.store import SampleStore
    stores = [d for d in os.listdir(str(DATASET_PATH))
              if d.startswith("dedust_") and not d.endswith("_input") and not d.endswith("_src")]
    data = {}
    for stname in stores:
        inp = stname + "_input"
        if not DATASET_PATH.joinpath(stname, "results.json").exists() or not DATASET_PATH.joinpath(inp).is_dir():
            continue
        res = json.load(open(str(DATASET_PATH.joinpath(stname, "results.json")), encoding="utf-8"))
        smap = {}
        sto = SampleStore(DATASET_PATH.joinpath(stname), verbose=False)
        for k in range(len(sto)):
            x, y = sto[k]
            smap[(np.asarray(x).reshape(-1) > 0.5).tobytes()] = np.asarray(y).reshape(2, -1)
        for i, r in res.items():
            if "wm" not in r:
                continue
            f = DATASET_PATH.joinpath(inp, i + ".pt")
            if not f.exists():
                continue
            p = np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5
            key = p.reshape(-1).tobytes()
            if key in data:
                continue
            resp = smap.get(key)
            ob = oob_metrics(resp)["oob_bad"] if resp is not None else None
            data[key] = (p, r["wm"][2], r.get("rad_margin"), ob)
    rows = []
    for p, wm, rad, ob in data.values():
        lab, n = _label(p, structure=_CROSS)
        if n == 0:
            continue
        sizes = np.sort(np.bincount(lab.ravel())[1:])[::-1].astype(float)
        rows.append(dict(n=n, main=sizes[0], main_frac=sizes[0] / sizes.sum(),
                         second=sizes[1] if n > 1 else 0, minc=sizes[-1],
                         cv=float(sizes.std() / sizes.mean()) if n > 1 else 0.0,
                         metal=sizes.sum(), wm=wm, rad=rad, ob=ob))
    print("樣本: " + str(len(rows)) + " 互異 pattern（現行 HFSS 真值,全 store 去重）")

    def corr(feat, target, subset=None):
        xs = [r[feat] for r in rows if (subset is None or subset(r)) and r[target] is not None]
        ys = [r[target] for r in rows if (subset is None or subset(r)) and r[target] is not None]
        if len(xs) < 20:
            return "  —"
        rho, pv = spearmanr(xs, ys)
        star = "**" if pv < 0.01 else ("*" if pv < 0.05 else "")
        return format(rho, "+.2f") + star

    feats = ("n", "main", "main_frac", "second", "minc", "cv", "metal")
    for title, sub2 in (("全樣本", None), ("作戰區(wm≥−3)", lambda r: r["wm"] >= -3)):
        print("\n== " + title + " Spearman（** p<.01 / * p<.05）==")
        print("| 特徵 | wm | rad | oob(低=好) |")
        print("|---|---|---|---|")
        for f in feats:
            print("| " + f + " | " + corr(f, "wm", sub2) + " | " + corr(f, "rad", sub2) + " | " + corr(f, "ob", sub2) + " |")
    tp = [r for r in rows if r["wm"] >= 0 and (r["rad"] if r["rad"] is not None else -9) >= 0]
    print("\n三標過 " + str(len(tp)) + " 筆 profile（含缺陷變體,非全可製造）:")
    for f in ("n", "main", "second", "minc", "metal"):
        v = [r[f] for r in tp]
        print("  " + f + ": 中位 " + format(np.median(v), ".0f") + " 範圍 [" + format(min(v), ".0f") + ", " + format(max(v), ".0f") + "]")


def cmd_gain(args):
    """性能期望三層帳（2026-07-12,Ricky「像 AdamW 一樣對提升有預期」＋防過早悲觀）:
    L1 階梯命中率——每批×臂在近門檻的實測命中率（不外推;下滑=礦脈枯竭早警）
    L2 學習曲線——best-so-far vs 累積 N 擬 R(N)=a+b·lnN,邊際增益 b/N（每百筆期望 dB）
    L3 轉換率——歷史「近王級(≥--near)→紀錄推進」比率;新臂零歷史=走預註冊存活測試,不給 dB 期望。"""
    import json
    LADDER = (0.0, 0.20, 0.30, 0.35)
    stores = []                                       # (mtime, 批標籤, manifest, results)
    for fol in os.listdir(str(DATASET_PATH)):
        if not (fol.startswith(f"dedust_{args.line}") and fol.endswith("_input")):
            continue
        st_name = fol[:-6]
        rp = DATASET_PATH.joinpath(st_name, "results.json")
        mp = DATASET_PATH.joinpath(fol, "manifest.json")
        if not (rp.exists() and mp.exists()):
            continue
        man = json.load(open(str(mp), encoding="utf-8"))
        res = json.load(open(str(rp), encoding="utf-8"))
        batch = st_name[len("dedust_"):].rstrip("abcdefgh")   # r21b1a → r21b1
        stores.append((os.path.getmtime(str(rp)), batch, man, res))
    if not stores:
        raise SystemExit(f"找不到 dedust_{args.line}*_input 批次")
    stores.sort()
    samp = []                                         # 時序展平（store 粒度排序）
    for _, batch, man, res in stores:
        for m in man:
            r = res.get(m["id"])
            if r is None or "error" in r:
                continue
            samp.append(dict(batch=batch, kind=m.get("kind", "?"), wm=r["wm"][2],
                             rad=r.get("rad_margin"), oob=r.get("oob_bad")))

    def tri(s):
        return s["wm"] >= 0 and (s["rad"] if s["rad"] is not None else -9) >= 0

    print(f"—— L1 階梯命中率（{args.line};三標樣本計;n=非error 筆數;oob=旗艦軸 2026-07-12）——")
    OOBL = (10.0, 9.5, 9.0)
    hdr = ("| 批 | 臂 | n | 三標 | " + " | ".join(f"wm≥{t:+.2f}" for t in LADDER)
           + f" | wm>{args.record:+.2f} | " + " | ".join(f"oob<{t}" for t in OOBL)
           + f" | oob<{args.oob_record} |")
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    batches = sorted({s["batch"] for s in samp})
    for b in batches:
        for k in sorted({s["kind"] for s in samp if s["batch"] == b}):
            g = [s for s in samp if s["batch"] == b and s["kind"] == k]
            t3 = [s for s in g if tri(s)]
            cells = [f"{100*len([s for s in t3 if s['wm'] >= t])/len(g):.0f}%" for t in LADDER]
            rec = len([s for s in t3 if s["wm"] > args.record])
            ocells = [f"{100*len([s for s in t3 if (s['oob'] or 99) < t])/len(g):.0f}%" for t in OOBL]
            orec = len([s for s in t3 if (s["oob"] or 99) < args.oob_record])
            print(f"| {b} | {k} | {len(g)} | {100*len(t3)/len(g):.0f}% | " + " | ".join(cells)
                  + f" | {rec} | " + " | ".join(ocells) + f" | {orec} |")

    print("\n—— L2 學習曲線（best-so-far wm,三標;R(N)=a+b·lnN）——")
    best, curve = -9.0, []
    for i, s in enumerate([s for s in samp if tri(s)], 1):
        if s["wm"] > best:
            best = s["wm"]
        curve.append((i, best))
    if len(curve) >= 10:
        N = np.array([c[0] for c in curve], float)
        R = np.array([c[1] for c in curve], float)
        b_, _a = np.polyfit(np.log(N), R, 1)
        marg = b_ / N[-1] * 100
        need = int(0.01 * N[-1] / b_) if b_ > 1e-6 else -1
        print(f"  累積三標 {int(N[-1])} 筆,best {best:+.2f};擬合 b={b_:.3f} → 邊際增益 ≈ {marg:+.4f} dB/百筆（遞減中）")
        print(f"  以當前斜率估再 +0.01 ≈ {need if need >= 0 else '∞'} 筆三標"
              f"（全程擬合=樂觀上界;換 HFSS 預算再 ×(1/三標率)）")
    else:
        print("  三標樣本 <10,曲線不擬")

    print("\n—— L2b 旗艦軸曲線（可用帶外 best-so-far,wm≥0.15∧rad≥0;越低越好）——")
    ubest, ucurve = 99.0, []
    for i, s in enumerate([s for s in samp
                           if s["wm"] >= 0.15 and (s["rad"] if s["rad"] is not None else -9) >= 0
                           and s["oob"] is not None], 1):
        if s["oob"] < ubest:
            ubest = s["oob"]
        ucurve.append((i, ubest))
    if len(ucurve) >= 8:
        N = np.array([c[0] for c in ucurve], float)
        R = np.array([c[1] for c in ucurve], float)
        b_, _a = np.polyfit(np.log(N), R, 1)
        print(f"  合格解 {int(N[-1])} 筆,best {ubest};擬合 b={b_:.3f} → 邊際 ≈ {b_/N[-1]*100:+.4f} dB/百筆合格解")
    else:
        print(f"  合格解 {len(ucurve)} 筆(<8 不擬),best {ubest if ucurve else '—'}")

    near = [s for s in samp if tri(s) and s["wm"] >= args.near]
    ups = sum(1 for i in range(1, len(curve)) if curve[i][1] > curve[i - 1][1])
    print("\n—— L3 近王→紀錄轉換（本線內）——")
    print(f"  近王級(wm≥{args.near:+.2f}三標) {len(near)} 筆;best-so-far 推進 {ups} 次 → 轉換 ≈ 每 {len(near)/max(ups,1):.1f} 筆近王出 1 次推進")
    fresh = {k for k in {s['kind'] for s in samp} if len([s for s in samp if s['kind'] == k]) < 30}
    if fresh:
        print(f"  ⚠ 新臂 {sorted(fresh)}: 歷史不足,走 round 檔預註冊存活測試（資訊帳）,不給 dB 期望")


def main():
    ap = argparse.ArgumentParser(description="可重現診斷分析（純讀 NAS）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("volatility"); v.add_argument("--runs", nargs="+", required=True); v.add_argument("--labels", nargs="*")
    v.set_defaults(func=cmd_volatility)
    rr = sub.add_parser("rad-repr"); rr.add_argument("--run", required=True); rr.add_argument("--n", type=int, default=40)
    rr.add_argument("--k", nargs="+", type=int, default=[4, 6, 8, 10, 12, 16, 24]); rr.add_argument("--window", type=float, default=45)
    rr.set_defaults(func=cmd_rad_repr)
    re = sub.add_parser("rad-error"); re.add_argument("--run", required=True); re.add_argument("--n", type=int, default=30)
    cp = sub.add_parser("components", help="組件尺寸分布 vs 三標 (全 store 真值,零 HFSS;analysis-02)")
    cp.set_defaults(func=cmd_components)
    gn = sub.add_parser("gain", help="性能期望三層帳（階梯/曲線/轉換;防過早悲觀）")
    gn.add_argument("--line", default="r22", help="批次線前綴（掃 dedust_<line>*_input）")
    gn.add_argument("--record", type=float, default=0.39, help="現任 wm 紀錄")
    gn.add_argument("--oob-record", type=float, default=8.61, dest="oob_record", help="現任帶外紀錄（旗艦軸）")
    gn.add_argument("--near", type=float, default=0.30, help="近王級門檻")
    gn.set_defaults(func=cmd_gain)
    re.add_argument("--window", type=float, default=45); re.set_defaults(func=cmd_rad_error)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
