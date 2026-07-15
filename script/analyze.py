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


def _load_truths():
    """全 store 去重真值 loader（components/oobnav/terrain 共用）：掃 DATASET_PATH 全部
    dedust_*（排除 _input/_src）→ list[(pattern bool25×25, wm, rad, resp|None)]。
    resp=店內 (2,17) 響應,查無配對時 None。純讀 NAS、零 HFSS。"""
    import json
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
            data[key] = (p, r["wm"][2], r.get("rad_margin"), smap.get(key))
    return list(data.values())


def cmd_components(args):
    """組件尺寸分布 vs 三標（Ricky 觀察「N 組塊大小很不平均」的量化;2026-07-09,analysis-02）。
    掃全部 dedust store 去重真值 → 每 pattern 的組件尺寸統計（n/main/second/minc/cv/metal）
    對 wm/rad/oob 做 Spearman（全樣本+作戰區）＋三標過 profile。零 HFSS,可重跑。"""
    from scipy.ndimage import label as _label
    from scipy.stats import spearmanr
    from script.dedust import _CROSS, oob_metrics
    rows = []
    for p, wm, rad, resp in _load_truths():
        ob = oob_metrics(resp)["oob_bad"] if resp is not None else None
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


def cmd_oobnav(args):
    """帶外拆側導航統計（analysis-03 口徑常態化;2026-07-13）。
    ⚠ 原 2026-07-09 一次性腳本已佚,結構特徵改用固定列/欄帶重規格（feed 在 (24,12)=底列）——
    與 analysis-03 原文數字**不可直接比,方向可比**：
      toprows=頂 5 列（row 0-4）金屬 / midband=中帶列 10-14 金屬 / edgecols=左右各 3 欄金屬 /
      cloudpx=上半（row 0-9）金屬 / cloud_w=上半金屬欄跨度。
    輸出：①低/高側 Gain 峰分布 ②作戰區結構載體 Spearman vs 低側峰 ③rad top/bot 10% 特徵對比。"""
    from scipy.ndimage import label as _label
    from scipy.stats import spearmanr
    from script.dedust import _CROSS, oob_metrics
    rows = []
    for p, wm, rad, resp in _load_truths():
        if resp is None:
            continue
        m = oob_metrics(resp)
        up = p[:10]
        cols = np.where(up.any(axis=0))[0]
        rows.append(dict(metal=int(p.sum()), n_comp=int(_label(p, structure=_CROSS)[1]),
                         toprows=int(p[:5].sum()), midband=int(p[10:15].sum()),
                         edgecols=int(p[:, :3].sum() + p[:, 22:].sum()),
                         cloudpx=int(up.sum()), cloud_w=int(cols[-1] - cols[0] + 1) if len(cols) else 0,
                         wm=wm, rad=rad, lo=m["oob_gain_max_lo"], hi=m["oob_gain_max_hi"],
                         ob=m["oob_bad"]))
    print("樣本: " + str(len(rows)) + " 互異 pattern（有響應者;現行 HFSS 真值,全 store 去重）")
    for side, key in (("低側", "lo"), ("高側", "hi")):
        v = sorted(r[key] for r in rows)
        print(side + " Gain 峰 min/中位/max = " + format(v[0], "+.2f") + " / "
              + format(v[len(v) // 2], "+.2f") + " / " + format(v[-1], "+.2f"))

    feats = ("edgecols", "metal", "toprows", "cloudpx", "cloud_w", "midband", "n_comp")
    war = [r for r in rows if r["wm"] >= -1]
    print("\n== 作戰區(wm≥−1,n=" + str(len(war)) + ") Spearman vs 低側 Gain 峰（低=好;** p<.01）==")
    print("| 特徵 | ρ(lo峰) | ρ(oob_bad) |")
    print("|---|---|---|")
    for f in feats:
        line = "| " + f
        xs = [r[f] for r in war]
        for tgt in ("lo", "ob"):
            if len(set(xs)) < 2:
                line += " | —(常數)"
                continue
            rho, pv = spearmanr(xs, [r[tgt] for r in war])
            line += " | " + format(rho, "+.2f") + ("**" if pv < 0.01 else ("*" if pv < 0.05 else ""))
        print(line + " |")

    rr = sorted((r for r in rows if r["wm"] >= 0 and r["rad"] is not None), key=lambda r: r["rad"])
    k = max(1, len(rr) // 10)
    print("\n== rad 對比（wm≥0 且有 rad,n=" + str(len(rr)) + ";bot10% vs top10% 特徵中位）==")
    for f in ("midband", "metal", "cloudpx", "toprows", "n_comp"):
        bot = np.median([r[f] for r in rr[:k]])
        top = np.median([r[f] for r in rr[-k:]])
        print("  " + f + ": rad差 " + format(bot, ".0f") + " vs rad好 " + format(top, ".0f"))


def cmd_terrain(args):
    """地形 variogram（analysis-01 A 部口徑,改用自家 HFSS 真值;2026-07-13）：
    全 store 去重真值兩兩配對 → |Δwm| 中位 vs Hamming 距離分箱。
    ⚠ 侷限同 analysis-01：短距對多為刻意變體（搜尋選出的移動）,短距平滑度可能略被高估。"""
    truths = _load_truths()
    P = np.packbits(np.array([p.reshape(-1) for p, _, _, _ in truths], dtype=np.uint8), axis=1)
    wm = np.array([w for _, w, _, _ in truths], dtype=np.float32)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
    bins = ((1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 80), (81, 160), (161, 320), (321, 625))
    acc = [[] for _ in bins]
    n = len(truths)
    for i in range(n - 1):
        d = lut[np.bitwise_xor(P[i], P[i + 1:])].sum(axis=1)
        dw = np.abs(wm[i] - wm[i + 1:])
        for bi, (a, b) in enumerate(bins):
            m = (d >= a) & (d <= b)
            if m.any():
                acc[bi].append(dw[m])
    print("樣本: " + str(n) + " 互異 pattern（現行 HFSS 真值,全 store 去重;全兩兩配對）")
    print("| Hamming 距離 | n_pairs | |Δwm| 中位 |")
    print("|---|---|---|")
    meds = {}
    for (a, b), lst in zip(bins, acc):
        v = np.concatenate(lst) if lst else np.array([])
        meds[(a, b)] = float(np.median(v)) if len(v) else float("nan")
        print("| " + str(a) + "-" + str(b) + " | " + str(len(v)) + " | "
              + (format(meds[(a, b)], ".2f") if len(v) else "—") + " |")
    far = np.concatenate([x for (a, _), lst in zip(bins, acc) if a >= 161 for x in lst] or [np.array([0.0])])
    asym = float(np.median(far))
    near = meds.get((1, 2), float("nan"))
    print("\n長距漸近線(161-625) " + format(asym, ".2f") + " dB;d1-2 中位 " + format(near, ".2f")
          + " = " + format(100 * near / asym, ".0f") + "% —— analysis-01(學長池口徑): 0.46-0.61 = 12-16%")


def cmd_gain(args):
    """性能期望三層帳（2026-07-12,Ricky「像 AdamW 一樣對提升有預期」＋防過早悲觀）:
    L1 階梯命中率——每批×臂在近門檻的實測命中率（不外推;下滑=礦脈枯竭早警）
    L2 學習曲線——best-so-far vs 累積 N 擬 R(N)=a+b·lnN,邊際增益 b/N（每百筆期望 dB）
    L3 轉換率——歷史「近王級(≥--near)→紀錄推進」比率;新臂零歷史=走預註冊存活測試,不給 dB 期望。
    門檻預設自動讀 docs/records.json（旗標可覆蓋;弱模型化 2026-07-12）。"""
    if args.record is None or args.oob_record is None or args.near is None:
        _rec = _records()
        args.record = args.record if args.record is not None else _rec["wm"]["value"]
        args.oob_record = args.oob_record if args.oob_record is not None else _rec["oob"]["value"]
        args.near = args.near if args.near is not None else round(_rec["wm"]["value"] - 0.09, 2)
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

    #? L0 常升目標（Ricky 2026-07-12「持續增加探索的範圍與資料量;確保有一個目標是提升的」）
    tot, rads = 0, 0
    for fol in os.listdir(str(DATASET_PATH)):
        rp2 = DATASET_PATH.joinpath(fol, "results.json")
        if fol.endswith("_input") or not fol.startswith("dedust_") or not rp2.exists():
            continue
        try:
            rr = json.load(open(str(rp2), encoding="utf-8"))
        except Exception:
            continue
        tot += sum(1 for v in rr.values() if "error" not in v)
        rd = DATASET_PATH.joinpath(fol, "rad")
        if rd.is_dir():
            rads += len(os.listdir(str(rd)))
    expl = sum(1 for s in samp if s["kind"] in ("denovo", "wild", "selfgen", "coldmine", "infogain", "fragfix"))
    print("—— L0 資料與覆蓋（常升目標:探索範圍×資料量）——")
    print(f"  全史 HFSS 真值 {tot} 筆/方向圖 {rads};本線 {len(samp)} 筆,探索類(C/D/F/I/W/自產) {expl}"
          f"（{100*expl/max(len(samp),1):.0f}%）")

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


def cmd_data(args):
    """資料總帳＋健檢一鍵化（2026-07-13,取代一次性普查腳本）:
    ①總量帳（量測次數/唯一樣本〔hash 檔名聯集=內容定址零重複計算〕/方向圖/error 殘留）
    ②按 round 系列分組 ③完整性（manifest↔results 缺口/孤兒 input/rad 缺漏）
    ④重複健康度（跨店 hash 碰撞;蓄意公證店之外的碰撞=查重洩漏警報）。
    每輪 /close-round 跑一次;接手盤點/使用者問資料量時先跑這個。"""
    import json
    import re
    meas, errs, rads, files_total = 0, 0, 0, 0
    uniq = {}                                             # hash 檔名 → [store,...]
    groups, gaps, orphans = {}, [], []
    stores_seen = set()
    for fol in sorted(os.listdir(str(DATASET_PATH))):
        d = DATASET_PATH.joinpath(fol)
        if not fol.startswith("dedust_"):
            continue
        if fol.endswith("_input"):
            if not DATASET_PATH.joinpath(fol[:-6], "results.json").exists():
                orphans.append(fol)
            continue
        rp = d.joinpath("results.json")
        if not rp.exists():
            continue
        stores_seen.add(fol)
        try:
            res = json.load(open(str(rp), encoding="utf-8"))
        except Exception:
            gaps.append(f"{fol}: results.json 壞損")
            continue
        ok = sum(1 for v in res.values() if "error" not in v)
        errs += len(res) - ok
        meas += ok
        mp = DATASET_PATH.joinpath(fol + "_input", "manifest.json")
        if mp.exists():
            man_n = len(json.load(open(str(mp), encoding="utf-8")))
            if ok < man_n:
                gaps.append(f"{fol}: {ok}/{man_n}")
        rd = d.joinpath("rad")
        rn = len(os.listdir(str(rd))) if rd.is_dir() else 0
        rads += rn
        name = fol[len("dedust_"):]
        mo = re.match(r"(r\d+)", name)
        if mo and int(mo.group(1)[1:]) >= 19:
            g = mo.group(1)
        elif name.startswith(("vgen", "auto")):
            g = name[:4]
        else:
            g = "R7-R18 前期"
        groups[g] = groups.get(g, 0) + ok
        for f in os.listdir(str(d)):
            if f != "results.json" and d.joinpath(f).is_file():
                files_total += 1
                uniq.setdefault(f, []).append(fol)
    print("== 總量帳 ==")
    print(f"  量測次數 {meas}（error 殘留 {errs}）/ store 樣本檔 {files_total} / **唯一樣本 {len(uniq)}**"
          f" / 方向圖 {rads}")
    print(f"  重複量測開銷 {files_total - len(uniq)} 檔（公證/重驗=品質成本）")
    print("\n== 按系列 ==")
    for g, n in sorted(groups.items()):
        print(f"  {g}: {n}")
    print("\n== 完整性 ==")
    print(("  ⚠ 未收全/壞損: " + "; ".join(gaps[:10])) if gaps else "  manifest↔results 全對齊 ✓")
    print(("  ⚠ 孤兒 input（有輸入無 store,可能待跑）: " + ", ".join(orphans[:10])) if orphans
          else "  無孤兒 input ✓")
    dup = {f: sl for f, sl in uniq.items() if len(sl) > 1}
    #? 蓄意重測店（公證/重驗/穩健/消融基準）＋前期歷史（check-dup 2026-07-10 上線前）不算洩漏;
    #  警報只對 r19+ 時代的非蓄意碰撞拉——那才是防線破口。
    INTEN = r"n\d|repeat|verify|w17rep|champ|crown|bakeoff|ablate|tol|occl|ref2v|probes"

    def _new_era(s):
        mo = re.match(r"dedust_r(\d+)", s)
        return bool(mo and int(mo.group(1)) >= 19)

    inten = {f for f, sl in dup.items() if any(re.search(INTEN, s) for s in sl)}
    hist_dup = {f for f, sl in dup.items() if f not in inten and not any(_new_era(s) for s in sl)}
    leak = {f: sl for f, sl in dup.items() if f not in inten and f not in hist_dup}
    print("\n== 重複健康度 ==")
    print(f"  跨店碰撞 {len(dup)} 檔＝蓄意重測 {len(inten)}＋前期歷史 {len(hist_dup)}＋其餘 {len(leak)}")
    if leak:
        print(f"  ⚠ r19+ 非蓄意碰撞 {len(leak)} 檔——先查 manifest kind:批內搭載的 notarize/repeat"
              "（如 r19a 搭 cc 公證）屬蓄意;kind 非豁免類才是真洩漏:")
        for f, sl in list(leak.items())[:5]:
            print(f"    {f[:16]}… ← {', '.join(sl)}")
    else:
        print("  r19+ 非蓄意碰撞 0 ＝ check-dup 防線完好 ✓")

    #? 自產收穫（2026-07-13,Ricky「tier2 會不會拿來分析」）:selfgen 是隨機翻全史,可能撞到好 pattern——
    #  掃 auto 夾的三標/紀錄候選,別讓它們隱形（自產已自動餵 SM,見 sm_reanchor._load_clean_stores）。
    rec = _records()
    print("\n== 自產收穫（selfgen dedust_auto*;三標/紀錄候選）==")
    hits, autos = [], 0
    for fol in os.listdir(str(DATASET_PATH)):
        if not fol.startswith("dedust_auto"):
            continue
        rp = DATASET_PATH.joinpath(fol, "results.json")
        if not rp.exists():
            continue
        for i, r in json.load(open(str(rp), encoding="utf-8")).items():
            if "wm" not in r:
                continue
            autos += 1
            tri = r["wm"][2] >= 0 and (r.get("rad_margin") if r.get("rad_margin") is not None else -9) >= 0
            if not tri:
                continue
            tag = []
            if r["wm"][2] > rec["wm"]["value"]:
                tag.append(f"wm{r['wm'][2]:+.2f}>王")
            if (r.get("oob_bad") or 99) < rec["usable_oob"]["value"] and r["wm"][2] >= rec["buffer"]:
                tag.append(f"可用oob{r['oob_bad']}")
            if (r.get("rad_margin") or -9) > rec["rad"]["value"]:
                tag.append(f"rad{r['rad_margin']}>王")
            hits.append((fol, i, r["wm"][2], r.get("oob_bad"), tag))
    tri_n = len(hits)
    print(f"  自產 {autos} 筆,三標 {tri_n}"
          + ("——★ 含紀錄候選,下方列(照 /notarize 公證)" if any(h[4] for h in hits) else "（無紀錄候選;已自動餵 SM）"))
    for fol, i, wm, oob, tag in sorted(hits, key=lambda h: h[3] or 99)[:8]:
        star = "★ " + ",".join(tag) if tag else ""
        print(f"    {i[:24]} [{fol[7:]}] wm{wm:+.2f} oob{oob} {star}")


def _records():
    """docs/records.json＝紀錄與門檻的機器真相源（換王先改它;analyze gain/batch 預設讀它）。"""
    import json
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "records.json")
    return json.load(open(p, encoding="utf-8"))


def cmd_batch(args):
    """收檔判讀一鍵化（弱模型化 2026-07-12,取代每批手寫 judge script）:
    自動發現 dedust_r<round>b<batch>* 夾,輸出完成度/臂別表/可用帶外推進/前瞻 ρ（含 rad 頭退鍵線）/
    紀錄候選＋現成公證指令/「→ 行動」摘要——弱模型照「→」行執行即可,判斷都在程式裡。"""
    import json
    import re
    from scipy.stats import spearmanr
    rec = _records()
    buf = rec["buffer"]
    stores = []
    for suf in "abcdefgh":
        st = f"dedust_r{args.round}b{args.batch}{suf}"
        if DATASET_PATH.joinpath(st, "results.json").exists() \
                and DATASET_PATH.joinpath(st + "_input", "manifest.json").exists():
            stores.append(st)
    if not stores:
        raise SystemExit(f"找不到 dedust_r{args.round}b{args.batch}* 的 store（還沒收檔?）")
    rows, incomplete = [], []
    for st in stores:
        man = json.load(open(str(DATASET_PATH.joinpath(st + "_input", "manifest.json")), encoding="utf-8"))
        res = json.load(open(str(DATASET_PATH.joinpath(st, "results.json")), encoding="utf-8"))
        okn = sum(1 for m in man if m["id"] in res and "error" not in res[m["id"]])
        if okn < len(man):
            incomplete.append(f"{st}: {okn}/{len(man)}")
        for m in man:
            r = res.get(m["id"])
            if r is None or "error" in r:
                continue
            rows.append(dict(id=m["id"], st=st, kind=m.get("kind", "?"), src=m.get("source_id", ""),
                             wm=r["wm"][2], rad=r.get("rad_margin"), oob=r.get("oob_bad"),
                             pwm=m.get("pred_wm"), poob=m.get("pred_oob"), prad=m.get("pred_rad"),
                             d=m.get("diff_px")))
    print(f"== r{args.round} b{args.batch} 收檔判讀（{len(stores)} 夾 {len(rows)} 筆;門檻源 records.json {rec['updated']}）==")
    if incomplete:
        print("⚠ 未收全（先決定等/補跑,別急著下結論）: " + "; ".join(incomplete))

    def tri(s):
        return s["wm"] >= 0 and (s["rad"] if s["rad"] is not None else -9) >= 0

    def usable(s):
        return s["wm"] >= buf and (s["rad"] if s["rad"] is not None else -9) >= 0

    print("\n-- 臂別（合格=wm≥buffer∧rad≥0）--")
    for kind in sorted({s["kind"] for s in rows}):
        g = [s for s in rows if s["kind"] == kind]
        t3 = [s for s in g if tri(s)]
        us = [s for s in g if usable(s)]
        best = max(g, key=lambda s: s["wm"])
        buo = min(us, key=lambda s: s["oob"] or 99) if us else None
        print(f"  {kind}: n={len(g)} 三標 {len(t3)}({100 * len(t3) / len(g):.0f}%)"
              f" 合格 {len(us)}({100 * len(us) / len(g):.0f}%) | best {best['id']} wm{best['wm']:+.2f}"
              + (f" | 合格最佳oob {buo['id']} {buo['oob']}" if buo else ""))

    print(f"\n-- 可用帶外（紀錄 {rec['usable_oob']['value']}＝{rec['usable_oob']['id']}）--")
    us_all = sorted([s for s in rows if usable(s)], key=lambda s: s["oob"] or 99)
    adv = [s for s in us_all if (s["oob"] or 99) < rec["usable_oob"]["value"]]
    for s in us_all[:5]:
        print(f"  {s['oob']}  {s['id']} [{s['kind']}] wm{s['wm']:+.2f} rad{s['rad']:+.2f}"
              + ("  ★ 推進!" if s in adv else ""))

    print("\n-- 前瞻（M 臂 mlotto,pred × realized）--")
    mm = [s for s in rows if s["kind"] == "mlotto"]
    rad_rho = None
    for pk, ak, nm in (("pwm", "wm", "wm"), ("poob", "oob", "oob"), ("prad", "rad", "rad頭")):
        xs = [(s[pk], s[ak]) for s in mm if s[pk] is not None and s[ak] is not None]
        if len(xs) > 5:
            rho, p = spearmanr([a for a, _ in xs], [b for _, b in xs])
            print(f"  {nm}: rho={rho:+.3f} (p={p:.3f}, n={len(xs)})")
            if pk == "prad":
                rad_rho = rho

    print("\n-- 紀錄候選（單次;鐵則=下批公證）--")
    cands = []
    for s in rows:
        tags = []
        # margin 王候選=必須三標(非三標高 wm 只有破帶內參考點 inband 才算,且那不換王)
        if tri(s) and s["wm"] > rec["wm"]["value"]:
            tags.append(f"wm{s['wm']:+.2f}>{rec['wm']['value']}(margin王,三標)")
        elif (not tri(s)) and s["wm"] > rec["inband"]["value"]:
            tags.append(f"wm{s['wm']:+.2f}>{rec['inband']['value']}(帶內參考,非三標;不換王)")
        if tri(s) and (s["oob"] or 99) < rec["oob"]["value"]:
            tags.append(f"oob{s['oob']}<{rec['oob']['value']}")
        if tri(s) and (s["rad"] or -9) > rec["rad"]["value"]:
            tags.append(f"rad{s['rad']}>{rec['rad']['value']}")
        if usable(s) and (s["oob"] or 99) < rec["usable_oob"]["value"]:
            tags.append(f"可用oob{s['oob']}<{rec['usable_oob']['value']}")
        if tags:
            cands.append((s, tags))
    if cands:
        nums = [int(mo.group(1)) for f in os.listdir(str(DATASET_PATH))
                if (mo := re.match(rf"dedust_r{args.round}n(\d+)", f))]
        nx = (max(nums) + 1) if nums else 1
        for j, (s, tags) in enumerate(cands):
            sub = "abcdefgh"[j % 8]
            print(f"  ★ {s['id']} [{s['kind']}] {','.join(tags)}")
            print(f"    → python -m script.dedust select-repeat --source-input {s['st']}_input"
                  f" --id {s['id']} --n 2 --input dedust_r{args.round}n{nx}{sub}_input")
            print(f"    → python -m script.dedust jobs-add --input dedust_r{args.round}n{nx}{sub}_input"
                  f" --store dedust_r{args.round}n{nx}{sub} --prio 2")
    else:
        print("  （無）")

    #? KPI 面板②覆蓋/③水位（decisions 2026-07-15 戰略換軸——每批必報,跨輪畫曲線）
    DYN = ("c18", "vg0338", "vg0396", "g1_038", "g1_039", "r2_016", "r3_001")
    root_idx, dyn_pats = {}, []
    for fol in os.listdir(str(DATASET_PATH)):
        mp = DATASET_PATH.joinpath(fol, "manifest.json")
        if not fol.endswith("_input") or not mp.exists():
            continue
        for m in json.load(open(str(mp), encoding="utf-8")):
            root_idx.setdefault(m["id"], m.get("source_id"))
            if any(t in m["id"] for t in DYN):
                f = DATASET_PATH.joinpath(fol, m["id"] + ".pt")
                if f.exists():
                    dyn_pats.append(np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5)
    dpk = np.packbits(np.stack(dyn_pats).astype(np.uint8), axis=1)
    POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

    def _rt(name):
        seen, cur = set(), name
        while cur in root_idx and root_idx[cur] and cur not in seen:
            seen.add(cur)
            cur = root_idx[cur]
        return cur
    bkeys, near, blood, fresh, nall = set(), 0, 0, 0, 0
    dds, bpats = [], []
    for st in stores:
        for m in json.load(open(str(DATASET_PATH.joinpath(st + "_input", "manifest.json")), encoding="utf-8")):
            f = DATASET_PATH.joinpath(st + "_input", m["id"] + ".pt")
            if not f.exists():
                continue
            p = np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5
            bkeys.add(p.tobytes())
            bpats.append(p)
            dd = int(POP[np.bitwise_xor(dpk, np.packbits(p.astype(np.uint8)))].sum(axis=1).min())
            dds.append(dd)
            nall += 1
            near += int(dd < 20)
            blood += int(any(t in _rt(m["id"]) for t in DYN))
            fresh += int(m.get("kind") in ("denovo", "selfgen") or m.get("diff_px") == -1
                         or "rand" in str(m.get("source_id", "")))
    #? 批內互異度（Ricky 視覺質疑 2026-07-15:d_dyn 量「離王朝」不量「彼此像不像」——盲點補上）
    bpk2 = np.packbits(np.stack(bpats).astype(np.uint8), axis=1)
    dmat = POP[np.bitwise_xor(bpk2[:, None, :], bpk2[None, :, :])].sum(axis=2)
    np.fill_diagonal(dmat, 9999)
    intra = int(np.median(dmat.min(axis=1)))
    print("\n-- 覆蓋/多樣性（KPI②）--")
    print(f"  近王(d_dyn<20) {near}/{nall}={near / max(nall, 1):.0%} | 王系血統根 {blood / max(nall, 1):.0%}"
          f" | d_dyn 中位 {int(np.median(dds))} | 無親新血 {fresh}/{nall}={fresh / max(nall, 1):.0%}")
    print(f"  批內最近鄰 Hamming 中位 {intra}（隨機基準 ~260;<50=批內高度同質）")

    wms = [s["wm"] for s in rows]
    hist = []
    from script.dedust import oob_metrics as _oobm
    for p, wm, rad, resp in _load_truths():
        if p.reshape(-1).tobytes() in bkeys or rad is None or resp is None:
            continue
        hist.append((wm, rad, _oobm(resp)["oob_bad"]))
    H = np.array(hist) if hist else np.zeros((0, 3))
    newf = []
    for s in rows:
        if s["rad"] is None or s["oob"] is None:
            continue
        dom = ((H[:, 0] >= s["wm"]) & (H[:, 1] >= s["rad"]) & (H[:, 2] <= s["oob"])
               & ((H[:, 0] > s["wm"]) | (H[:, 1] > s["rad"]) | (H[:, 2] < s["oob"])))
        if not dom.any():
            newf.append(s)
    print("-- 整體水位/前緣（KPI③）--")
    print(f"  本批 wm 中位 {np.median(wms):+.2f} / P90 {np.percentile(wms, 90):+.2f}"
          f" / 作戰區(wm≥−1) {sum(w >= -1 for w in wms)}/{len(wms)}")
    print(f"  帕累托前緣增量（wm×rad×oob 對全歷史非支配）: +{len(newf)} 筆"
          + ("  例: " + ",".join(s["id"] for s in newf[:3]) if newf else ""))

    for kind in ("slotchain", "denovo", "infogain", "hslot", "repair"):
        g = sorted([s for s in rows if s["kind"] == kind], key=lambda s: -s["wm"])[:5]
        if g:
            print(f"\n-- {kind} top5 --")
            for s in g:
                print(f"  {s['id']} wm{s['wm']:+.2f} rad {s['rad']} oob {s['oob']}"
                      f" {'★三標' if tri(s) else ''} (src {s['src']}, d{s['d']})")

    print("\n== → 行動（照抄執行;細節見 /batch-cycle skill）==")
    print(f"  ① 公證候選 {len(cands)} 件" + ("——照上方 select-repeat/jobs-add 指令發車(prio 2),收檔走 /notarize" if cands else "——無,跳過"))
    print("  ② 可用帶外: " + (f"★ 推進 {adv[0]['oob']}（{adv[0]['id']},單次→列入公證）" if adv
                            else "零推進（連續零推進批數對照 round 檔 §1 回報線）"))
    if rad_rho is not None:
        print(f"  ③ rad 頭前瞻 {rad_rho:+.3f}: " + ("≥0.3 續鍵（下批保留 --rad-key）" if rad_rho >= 0.3
                                                  else "<0.3——若連兩批<0.3 → 下批移除 --rad-key"))
    print("  ④ 重錨: python -m script.sm_reanchor train --add \"" + ",".join(stores) + "\" --out sm_reanchorNN.pth")
    print("  ⑤ 下批 select → check-dup（exit 1 停）→ jobs-add → 池存量<48 補 → dedust watch 掛偵測")


def cmd_credit(args):
    """血統貢獻分（R24 探索誘因包 D,Ricky 核准 2026-07-12）:紀錄 id 沿 source_id 鏈回溯,
    每個祖先給其出身臂記一分——探索的延遲報酬記帳（例:margin 王經 g1 填空池=池記功）。
    R23 期間純報表校準,R24 起進配額股息計分（當批效率 40%+血統 40%+新穎產出 20%）。"""
    import json
    idx = {}
    for fol in os.listdir(str(DATASET_PATH)):
        if not (fol.startswith("dedust_") and fol.endswith("_input")):
            continue
        mp = DATASET_PATH.joinpath(fol, "manifest.json")
        if not mp.exists():
            continue
        for m in json.load(open(str(mp), encoding="utf-8")):
            idx.setdefault(m["id"], (m.get("kind", "?"), m.get("source_id")))
    credit = {}
    for rid in args.ids.split(","):
        rid = rid.strip()
        seen, chain, cur = set(), [], rid
        while cur and cur in idx and cur not in seen:
            seen.add(cur)
            kind, src = idx[cur]
            chain.append(f"{cur}[{kind}]")
            credit[kind] = credit.get(kind, 0) + 1
            cur = src
        print(f"{rid}:\n  " + " ← ".join(chain) + (f" ← {cur}(池根)" if cur and cur not in idx else ""))
    print("\n臂別血統貢獻分（配額股息計分輸入）:")
    for k, v in sorted(credit.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


def cmd_ikpi(args):
    """I 資訊臂 KPI（「模型更新量」量測;R24 §1 承諾三輪未辦,R28 前置落地）：
    比較「該批發車用的 SM（--pre）」vs「吃完該批後的重錨版（--post）」在本批各臂樣本上的
    |wm 誤差| 改善量（pre−post,正=模型從這批學到東西）——I 臂樣本若真是「資訊量最高的量測點」,
    其改善中位應高於對照臂 M（純隨機,同批同訓練）。判準:I−M 差 >+0.1=資訊增量成立。"""
    import json
    from antenna.training import load_config, setup_responses, PORT_SPECS
    from antenna.zoo import SURROGATES
    from antenna.losses import worst_margin
    from script.dedust import DEFAULT_CFG
    cfg = load_config(DEFAULT_CFG)
    setup_responses(cfg)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "ikpi")
    os.makedirs(cache, exist_ok=True)
    sms = {}
    for tag, pth in (("pre", args.pre), ("post", args.post)):
        sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
        sm.pre_load_model(DATASET_PATH.joinpath(pth), strict=True)
        sm.model.eval()
        sms[tag] = sm
    per_arm = {}
    for suf in "abcdefgh":
        st = f"dedust_r{args.round}b{args.batch}{suf}"
        mp = DATASET_PATH.joinpath(st + "_input", "manifest.json")
        rp = DATASET_PATH.joinpath(st, "results.json")
        if not (mp.exists() and rp.exists()):
            continue
        man = json.load(open(str(mp), encoding="utf-8"))
        res = json.load(open(str(rp), encoding="utf-8"))
        for m in man:
            r = res.get(m["id"])
            if r is None or "wm" not in r:
                continue
            x = torch.tensor(np.asarray(torch.load(
                str(DATASET_PATH.joinpath(st + "_input", m["id"] + ".pt")), weights_only=True)),
                dtype=torch.float32).reshape(-1)
            errs = {}
            with torch.no_grad():
                for tag, sm in sms.items():
                    w, _ = worst_margin(sm.model(x), labels, cfg.targets)
                    errs[tag] = abs(float(w) - r["wm"][2])
            per_arm.setdefault(m.get("kind", "?"), []).append(errs["pre"] - errs["post"])
    print(f"== I 臂 KPI（模型更新量;r{args.round} b{args.batch};{args.pre} → {args.post}）==")
    print("| 臂 | n | |wm err| 改善中位（pre−post） |")
    print("|---|---|---|")
    meds = {}
    for kind, v in sorted(per_arm.items(), key=lambda kv: -float(np.median(kv[1]))):
        meds[kind] = float(np.median(v))
        print(f"| {kind} | {len(v)} | {meds[kind]:+.2f} |")
    if "infogain" in meds and "mlotto" in meds:
        d = meds["infogain"] - meds["mlotto"]
        verdict = ("I 樣本資訊增量成立（>+0.1）" if d > 0.1 else
                   "I 與對照同級,無讀數（±0.1 內）" if d > -0.1 else
                   "I 反而低於對照——分歧選點策略檢討")
        print(f"→ I−M 對照差 {d:+.2f} dB：{verdict}")


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
    on = sub.add_parser("oobnav", help="帶外拆側導航統計（分側分布+結構載體相關+rad對比;analysis-03 口徑）")
    on.set_defaults(func=cmd_oobnav)
    tr = sub.add_parser("terrain", help="地形 variogram——|Δwm| 中位 vs Hamming,自家真值（analysis-01 口徑）")
    tr.set_defaults(func=cmd_terrain)
    gn = sub.add_parser("gain", help="性能期望三層帳（階梯/曲線/轉換;防過早悲觀;門檻自動讀 records.json）")
    gn.add_argument("--line", default="r23", help="批次線前綴（掃 dedust_<line>*_input）")
    gn.add_argument("--record", type=float, default=None, help="現任 wm 紀錄（預設讀 docs/records.json）")
    gn.add_argument("--oob-record", type=float, default=None, dest="oob_record", help="現任帶外紀錄（預設讀 records.json）")
    gn.add_argument("--near", type=float, default=None, help="近王級門檻（預設=wm 紀錄−0.09）")
    gn.set_defaults(func=cmd_gain)
    bt = sub.add_parser("batch", help="收檔判讀一鍵化（臂別/可用帶外/前瞻/紀錄候選+公證指令/→行動;/batch-cycle step①）")
    bt.add_argument("--round", type=int, required=True)
    bt.add_argument("--batch", type=int, required=True)
    bt.set_defaults(func=cmd_batch)
    dt = sub.add_parser("data", help="資料總帳+健檢（唯一樣本/分組/完整性/查重洩漏警報;每輪收檔跑）")
    dt.set_defaults(func=cmd_data)
    cr = sub.add_parser("credit", help="血統貢獻分（探索延遲報酬記帳;R24 配額股息計分輸入）")
    cr.add_argument("--ids", default="k23b1_021_m22g1_025_cc,o23b1_007_k8_042_k7_00,"
                    "o6_001_o4_035_o3_05,m5_054_m3_026_m1_01,h7_010_g16",
                    help="逗號分隔紀錄 id（預設=現任五頭銜/紀錄）")
    cr.set_defaults(func=cmd_credit)
    ik = sub.add_parser("ikpi", help="I 資訊臂 KPI:pre/post SM 在該批各臂的 |wm err| 改善（I−M >+0.1=資訊增量成立）")
    ik.add_argument("--round", type=int, required=True)
    ik.add_argument("--batch", type=int, required=True)
    ik.add_argument("--pre", required=True, help="該批發車用的 SM（如 sm_reanchor28.pth）")
    ik.add_argument("--post", required=True, help="吃完該批後的重錨版（如 sm_reanchor29.pth）")
    ik.set_defaults(func=cmd_ikpi)
    re.add_argument("--window", type=float, default=45); re.set_defaults(func=cmd_rad_error)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
