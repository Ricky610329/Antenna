"""R82 變化量代理模型：上下文（量過的鄰居 pattern＋實測曲線）＋查詢 pattern → 查詢曲線。

S1＝上下文 1 筆 c，學 Δ = y(查詢) − y(c)，預測＝y(c) + Δ̂。
- 鄰居以查詢為中心找（Ricky 2026-10-19）；v6 回測可選「最近實測」或「血緣父代」。
- 訓練配對：每筆設計抽 d ~ 均勻(1..50)，挑距離最接近 d 的設計（Ricky：50 格內均勻）。
- 架構：mlp（攤平輸入）或 cnn（上下文圖＋翻轉遮罩當兩張 25×25 圖）。
- 混合式（hyb-*）：以錨定預測 y(c) + SM(q) − SM(c) 為底，只學殘差；SM 為現行絕對曲線 ensemble，
  訓練配對的 SM(q) 用交叉擬合（out-of-fold），模擬「查詢是新設計、上下文已在訓練集」的部署情境。
資料：24–32 GHz 歷史＋寬頻真值，全部對位到 v6 的 11–40 GHz 格點，沒量到的頻點以遮罩略過。

  python -m script.r82_delta_sm histval [--variants mlp,cnn,hyb-mlp,hyb-cnn]     # 本機歷史驗證
  python -m script.r82_delta_sm backtest --ref <anchor_backtest.json> [--anchor parent]  # v6 時間切分回測
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from script import r81_curve_sm as sm

V6 = sm.REPO / "configs" / "dual_r81_v6.yaml"
DMAX = 50
_POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def batch_of(store):
    m = re.search(r"r81v6b(\d\d)s\d$", store or "")
    return int(m.group(1)) if m else -1


def load_all(cfg, grid):
    """歷史＋寬頻 → dict(P bool N×625, Y N×3×G, M bool, wide bool, batch int, key, src)。同圖只留一筆。"""
    hist = sm.load_history(sm.HIST_DIR, grid)
    wide = sm.load_wide(sm.WIDE_ROOT, cfg, grid)
    P, Y, M, W, B, K, src, seen = [], [], [], [], [], [], [], set()
    for wi, r in enumerate(wide):
        seen.add(r["sha"])
        P.append(np.asarray(r["pattern"]).reshape(-1) > 0.5)
        Y.append(r["y"]); M.append(r["mask"]); W.append(True); B.append(batch_of(r["store"]))
        K.append(f"{r['store']}:{r['id']}"); src.append(("wide", wi))
    for i, s in enumerate(hist["sha"]):
        if s in seen:
            continue
        seen.add(s)
        P.append(hist["P"][i] > 0.5); Y.append(hist["Y"][i]); M.append(hist["M"][i])
        W.append(False); B.append(-1); K.append(f"hist:{s[:12]}"); src.append(("hist", i))
    return {"P": np.asarray(P), "Y": np.asarray(Y, np.float32), "M": np.asarray(M), "wide": np.asarray(W),
            "batch": np.asarray(B), "key": K, "src": src, "hist": hist, "wide_rows": wide}


def hamming(Pk_q, Pk_all, chunk=256):
    """packbits 後的漢明距離 (q, N) uint16。"""
    out = np.empty((len(Pk_q), len(Pk_all)), np.uint16)
    for s in range(0, len(Pk_q), chunk):
        out[s:s + chunk] = _POP[Pk_q[s:s + chunk, None, :] ^ Pk_all[None, :, :]].sum(2)
    return out


def make_pairs(Pk, rng, draws=8, dmax=DMAX, chunk=32, allow=None):
    """每筆查詢抽 draws 次 d~U(1..dmax)，挑距離最接近 d 的設計（同距離隨機）；只在 allow 內配對。
    距離逐塊算，不存整張矩陣。回傳 (q, c)。"""
    idx = np.flatnonzero(np.ones(len(Pk), bool) if allow is None else allow)
    Pa = Pk[idx]
    qs, cs = [], []
    for s in range(0, len(idx), chunk):
        D = hamming(Pa[s:s + chunk], Pa).astype(np.int32)
        for k, row in enumerate(D):
            ok = (row >= 1) & (row <= dmax)
            if not ok.any():
                continue
            cand, dist = np.flatnonzero(ok), row[ok]
            for d in rng.integers(1, dmax + 1, size=draws):
                gap = np.abs(dist - d)
                hit = cand[gap == gap.min()]
                qs.append(idx[s + k]); cs.append(idx[int(rng.choice(hit))])
    return np.asarray(qs), np.asarray(cs)


class DeltaNet(nn.Module):
    """輸入：上下文圖、翻轉遮罩、上下文曲線＋遮罩、距離（hyb 另加 SM 差）→ Δ 曲線（dB/10）。"""

    def __init__(self, g, arch="mlp", hyb=False):
        super().__init__()
        self.g, self.arch = g, arch
        n_curve = 3 * g * (3 if hyb else 2) + 1
        if arch == "cnn":
            self.enc = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),     # 13×13
                                     nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),     # 7×7
                                     nn.Flatten(), nn.Linear(64 * 49, 512), nn.ReLU())
            n_img = 512
        else:
            self.enc = nn.Identity()
            n_img = 625 * 2
        self.head = nn.Sequential(nn.Linear(n_img + n_curve, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(),
                                  nn.Linear(512, 3 * g))

    def forward(self, img, curve):
        z = self.enc(img.view(-1, 2, 25, 25)) if self.arch == "cnn" else img
        return self.head(torch.cat([z, curve], 1)).view(-1, 3, self.g)


def features(Pc, Pq, Yc, Mc, dS=None):
    """torch 張量 → (img (n,1250), curve)。dS＝SM(q)−SM(c)（hyb 用）。"""
    flip = (Pc != Pq).float()
    img = torch.cat([Pc.float(), flip], 1)
    parts = [(Yc * Mc / 10).flatten(1), Mc.float().flatten(1)]
    if dS is not None:
        parts.append((dS / 10).flatten(1))
    parts.append(flip.sum(1, keepdim=True) / DMAX)
    return img, torch.cat(parts, 1)


def train(data, q, c, *, arch="mlp", S_q=None, S_c=None, seeds=(0, 1), epochs=30, wide_weight=5.0,
          dev="cpu", log=print):
    """masked MSE。hyb：S_q/S_c 為 (len(q),3,G) 的 SM 預測，目標改為殘差 (yq−yc) − (Sq−Sc)。"""
    P = torch.tensor(data["P"], dtype=torch.uint8, device=dev)
    Y = torch.tensor(data["Y"], device=dev)
    M = torch.tensor(data["M"], device=dev)
    W = torch.tensor(data["wide"], device=dev)
    q, c = torch.tensor(q, device=dev), torch.tensor(c, device=dev)
    hyb = S_q is not None
    dS = torch.tensor(S_q - S_c, device=dev) if hyb else None
    keep = (M[q] & M[c]).flatten(1).any(1)
    q, c = q[keep], c[keep]
    if hyb:
        dS = dS[keep]
    models = []
    for s in seeds:
        torch.manual_seed(s)
        net = DeltaNet(Y.shape[2], arch, hyb).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(q) + 1023) // 1024))
        n = len(q)
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            tot = 0.0
            for i in range(0, n, 1024):
                b = perm[i:i + 1024]
                qb, cb = q[b], c[b]
                db = dS[b] if hyb else None
                img, cur = features(P[cb], P[qb], Y[cb], M[cb], db)
                t = (Y[qb] - Y[cb] - (db if hyb else 0)) / 10
                mb = (M[qb] & M[cb]).float()
                wb = torch.where(W[qb] & W[cb], wide_weight, 1.0)
                err = ((net(img, cur) - t) ** 2 * mb).flatten(1).sum(1) / mb.flatten(1).sum(1).clamp(min=1)
                loss = (err * wb).sum() / wb.sum()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += float(loss) * len(b)
        log(f"  {arch}{'-hyb' if hyb else ''} seed {s}: {n} 配對，末 epoch loss {tot / n:.4f}")
        models.append(net.eval())
    return models


@torch.no_grad()
def predict(models, Pc, Pq, Yc, Mc, dS=None, dev="cpu", chunk=4096):
    """numpy 輸入 → (K, n, 3, G) 預測曲線 dB：y(c) + [SM 差] + 模型輸出。"""
    outs = []
    for s in range(0, len(Pc), chunk):
        sl = slice(s, s + chunk)
        img, cur = features(torch.tensor(Pc[sl], dtype=torch.uint8, device=dev),
                            torch.tensor(Pq[sl], dtype=torch.uint8, device=dev),
                            torch.tensor(Yc[sl], device=dev), torch.tensor(Mc[sl], device=dev),
                            None if dS is None else torch.tensor(dS[sl], device=dev))
        base = Yc[sl] + (0 if dS is None else dS[sl])
        outs.append(np.stack([base + m(img, cur).cpu().numpy() * 10 for m in models]))
    return np.concatenate(outs, 1)


# ---------------------------------------------------------------- 絕對曲線 SM（對照組與混合式的底）

def split_sources(data, idx):
    hs = [i for k in idx for (src, i) in [data["src"][k]] if src == "hist"]
    ws = [i for k in idx for (src, i) in [data["src"][k]] if src == "wide"]
    h = data["hist"]
    hsub = {"P": h["P"][hs], "Y": h["Y"][hs], "M": h["M"][hs], "sha": [h["sha"][i] for i in hs],
            "lineage": [h["lineage"][i] for i in hs]}
    return hsub, [data["wide_rows"][i] for i in ws]


def fit_sm(data, idx, seeds=(0, 1, 2)):
    return sm.train_ensemble(*split_sources(data, idx), seeds=seeds)


def sm_predict(ens, P):
    return ens.predict(np.asarray(P, np.float32)).mean(0)


def oof_prior(data, idx, rng, folds=4, seeds=(0,)):
    """idx 內各設計的 out-of-fold SM 預測 (len(idx),3,G)。"""
    out = np.zeros((len(idx),) + data["Y"].shape[1:], np.float32)
    fold = rng.integers(folds, size=len(idx))
    for f in range(folds):
        ens = fit_sm(data, idx[fold != f], seeds)
        out[fold == f] = sm_predict(ens, data["P"][idx[fold == f]])
    return out


# ---------------------------------------------------------------- 評估

def inband_score(Y, M, slices, names=("s21_pass", "s11_match", "s22_match")):
    """歷史資料只有 24–32 GHz：用帶內三項（通帶、S11、S22）的最差 margin 當分數；頻點不全回 nan。"""
    out = np.full(len(Y), np.inf)
    for name, li, lo, hi, rel, thr in slices:
        if name not in names:
            continue
        v = Y[:, li, lo:hi + 1]
        mg = (thr - v.max(1)) if rel == "le" else (v.min(1) - thr)
        out = np.minimum(out, np.where(M[:, li, lo:hi + 1].all(1), mg, np.nan))
    return out


def parse_range(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def _device(args):
    torch.set_num_threads(4)
    return "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"


def histval(args):
    """本機歷史驗證：按設計留出 10%；查詢＝留出設計、上下文＝訓練設計。
    ①距離分組（1..dmax_eval 均勻）的曲線誤差與帶內分數 ρ；②爬山情境：同一上下文周圍 ≥5 個留出查詢的組內排序。"""
    t0 = time.time()
    dev = _device(args)
    cfg = sm.load_cfg(V6)
    grid = sm.frequency_grid(cfg.measurement)
    slices = sm.band_slices(cfg.score_spec, grid)
    data = load_all(cfg, grid)
    P, Y, M = data["P"], data["Y"], data["M"]
    N = len(P)
    rng = np.random.default_rng(args.seed)
    test = np.zeros(N, bool)
    test[rng.choice(N, size=N // 10, replace=False)] = True
    trn = np.flatnonzero(~test)
    Pk = np.packbits(P, 1)
    q, c = make_pairs(Pk, rng, draws=args.draws, dmax=args.dmax, allow=~test)
    print(f"資料 {N}（留出 {test.sum()}），訓練配對 {len(q)}，{time.time() - t0:.0f}s；裝置 {dev}", flush=True)
    variants = args.variants.split(",")
    ens = fit_sm(data, trn)
    S_in = np.zeros_like(Y)
    S_in[trn] = sm_predict(ens, P[trn])                           # 上下文：已在訓練集（in-sample）
    S_in[test] = sm_predict(ens, P[test])                         # 查詢：新設計
    S_q = S_in.copy()
    if any(v.startswith("hyb") for v in variants):
        S_q[trn] = oof_prior(data, trn, rng)                      # 訓練配對的查詢端用 out-of-fold
    print(f"SM 完成，{time.time() - t0:.0f}s", flush=True)
    models = {}
    for v in variants:
        hyb = v.startswith("hyb")
        models[v] = train(data, q, c, arch=v.split("-")[-1], S_q=S_q[q] if hyb else None,
                          S_c=S_in[c] if hyb else None, seeds=tuple(range(args.ensemble)), epochs=args.epochs,
                          wide_weight=args.wide_weight, dev=dev)
        print(f"{v} 完成，{time.time() - t0:.0f}s", flush=True)

    def preds(eq, ec):
        out = {"照抄": Y[ec], "絕對SM": S_in[eq], "錨定": Y[ec] + S_in[eq] - S_in[ec]}
        for v, ms in models.items():
            dS = S_in[eq] - S_in[ec] if v.startswith("hyb") else None
            out[v] = predict(ms, P[ec], P[eq], Y[ec], M[ec], dS, dev).mean(0)
        return out

    # ① 距離分組
    tq = np.flatnonzero(test)
    eq, ec = [], []
    for s0 in range(0, len(tq), 32):
        D = hamming(Pk[tq[s0:s0 + 32]], Pk[trn]).astype(np.int32)
        for k, row in enumerate(D):
            for d in rng.integers(1, args.dmax_eval + 1, size=4):
                hit = np.flatnonzero(row == d)
                if len(hit):
                    eq.append(tq[s0 + k]); ec.append(trn[int(rng.choice(hit))])
    eq, ec = np.asarray(eq), np.asarray(ec)
    dd = _POP[Pk[eq] ^ Pk[ec]].sum(1)
    mk = M[eq] & M[ec]
    pr = preds(eq, ec)
    truth = inband_score(Y[eq], mk, slices)
    print(f"\n① 距離分組（評估配對 {len(eq)}）：曲線 MAE dB／帶內分數 ρ")
    edges = [1, 4, 13, 31, 51, 81, 121]
    for lo, hi in zip(edges, edges[1:]):
        k = (dd >= lo) & (dd < hi)
        if k.sum() < 20:
            continue
        ok = k & np.isfinite(truth)
        cells = []
        for n, v in pr.items():
            mae = ((np.abs(v[k] - Y[eq][k]) * mk[k]).sum((1, 2)) / mk[k].sum((1, 2)).clip(1)).mean()
            rho = spearmanr(inband_score(v[ok], mk[ok], slices), truth[ok])[0]
            cells.append(f"{n} {mae:.2f}/{rho:+.2f}")
        print(f"  {lo:3d}–{hi - 1:3d} n={k.sum():5d}  " + "  ".join(cells))
    # ② 爬山情境：上下文周圍 ≤gmax 格的留出查詢，組內排序
    for gmax in (3, 12, 30):
        gq, gc, gid = [], [], []
        ctx = rng.choice(trn, size=min(4000, len(trn)), replace=False)
        D = hamming(Pk[ctx], Pk[tq]).astype(np.int32)
        for k, row in enumerate(D):
            hit = tq[(row >= 1) & (row <= gmax)]
            if len(hit) >= 5:
                hit = rng.choice(hit, size=min(15, len(hit)), replace=False)
                gq += list(hit); gc += [ctx[k]] * len(hit); gid += [k] * len(hit)
            if len(set(gid)) >= 300:
                break
        if not gq:
            continue
        gq, gc, gid = np.asarray(gq), np.asarray(gc), np.asarray(gid)
        gm = M[gq] & M[gc]
        pr = preds(gq, gc)
        tg = inband_score(Y[gq], gm, slices)
        line = []
        for n, v in pr.items():
            sv = inband_score(v, gm, slices)
            rs = []
            for g in np.unique(gid):
                k = (gid == g) & np.isfinite(tg)
                if k.sum() >= 5 and np.ptp(tg[k]) > 0 and np.ptp(sv[k]) > 0:
                    rs.append(spearmanr(sv[k], tg[k])[0])
            line.append(f"{n} {np.mean(rs):+.2f}" if rs else f"{n} nan")
        print(f"② 同上下文組內排序（≤{gmax} 格，{len(np.unique(gid))} 組）平均 ρ：" + "  ".join(line), flush=True)
    print(f"總耗時 {time.time() - t0:.0f}s")


def backtest(args):
    """v6 時間切分回測：b72..b78 每批只用該批之前的資料訓練，預測該批實測候選（集合同 anchor_backtest.json）。"""
    t0 = time.time()
    dev = _device(args)
    cfg = sm.load_cfg(V6)
    grid = sm.frequency_grid(cfg.measurement)
    slices = sm.band_slices(cfg.score_spec, grid)
    need = np.zeros((3, len(grid)), bool)
    for _, li, lo, hi, *_ in slices:
        need[li, lo:hi + 1] = True
    data = load_all(cfg, grid)
    P, Y, M = data["P"], data["Y"], data["M"]
    full = (M | ~need).all((1, 2))                               # 可當上下文（能算完整 v6）的設計
    Pk = np.packbits(P, 1)
    rng = np.random.default_rng(args.seed)
    print(f"資料 {len(P)} 筆（寬頻 {data['wide'].sum()}，可算 v6 {full.sum()}），{time.time() - t0:.0f}s；裝置 {dev}")
    ref = {o["id"]: o for o in json.loads(Path(args.ref).read_text(encoding="utf-8"))}
    parent_of = {}
    for f in Path(args.ref).parent.glob("r81v6b*_input/manifest.json"):
        for r in json.loads(f.read_text(encoding="utf-8")):
            parent_of[r["id"]] = r.get("parent_id", "")
    key_idx = {k.split(":", 1)[1]: i for i, k in enumerate(data["key"]) if not k.startswith("hist:")}
    full_key = {k: i for i, k in enumerate(data["key"]) if full[i]}
    arch = args.variant.split("-")[-1]
    hyb = args.variant.startswith("hyb")
    rows = []
    for b in parse_range(args.batches):
        ok = data["batch"] < b                                    # 時間切分（歷史 batch=−1 一律可用）
        q, c = make_pairs(Pk, rng, draws=args.draws, dmax=args.dmax, allow=ok)
        S_in = S_q = None
        if hyb:
            idx = np.flatnonzero(ok)
            ens = fit_sm(data, idx)
            S_in = np.zeros_like(Y)
            S_in[idx] = sm_predict(ens, P[idx])
            S_q = S_in.copy()
            S_q[idx] = oof_prior(data, idx, rng)
        models = train(data, q, c, arch=arch, S_q=S_q[q] if hyb else None, S_c=S_in[c] if hyb else None,
                       seeds=tuple(range(args.ensemble)), epochs=args.epochs, wide_weight=args.wide_weight, dev=dev)
        targets = [key_idx[i] for i, o in ref.items() if o["batch"] == b and i in key_idx]
        pool = np.flatnonzero(ok & full)
        Dt = hamming(Pk[targets], Pk[pool]) if targets else np.zeros((0, len(pool)))
        if hyb and targets:
            S_in[targets] = sm_predict(ens, P[targets])
        for t, drow in zip(targets, Dt):
            j = pool[np.argmin(drow)]                             # 離查詢最近的實測（以查詢為中心）
            if args.anchor == "parent":                           # 改用血緣父代（有實測且可算 v6 才用）
                pj = full_key.get(parent_of.get(data["key"][t].split(":", 1)[1], ""))
                if pj is not None and ok[pj]:
                    j = pj
            dS = (S_in[t:t + 1] - S_in[j:j + 1]) if hyb else None
            pred = predict(models, P[j:j + 1], P[t:t + 1], Y[j:j + 1], M[j:j + 1], dS, dev).mean(0)
            o = ref[data["key"][t].split(":", 1)[1]]
            rows.append(dict(o, delta=float(sm.margins_vec(pred, slices).min(1)[0]),
                             ctx_d=int(_POP[Pk[t] ^ Pk[j]].sum()),
                             ctx_true=float(sm.margins_vec(Y[j:j + 1], slices).min(1)[0])))
        print(f"b{b}: 訓練配對 {len(q)}，查詢 {len(targets)}，{time.time() - t0:.0f}s", flush=True)
    report(rows)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


def report(rows):
    t = np.array([r["true"] for r in rows])
    cols = {"原始SM": np.array([r["raw"] for r in rows]), "錨定": np.array([r["anc"] for r in rows]),
            "變化量": np.array([r["delta"] for r in rows]), "上下文照抄": np.array([r["ctx_true"] for r in rows])}
    d = np.array([r["ctx_d"] for r in rows])
    bt = np.array([r["batch"] for r in rows])
    raw = cols["原始SM"]
    subsets = [("全部", np.ones(len(t), bool)), ("上下文 ≤3 格", d <= 3), ("4–12 格", (d > 3) & (d <= 12)),
               ("13 格以上", d > 12), ("實際 WM>-8", t > -8), ("原始SM預測前半", raw >= np.median(raw))]
    for name, k in subsets:
        if k.sum() < 5:
            continue
        print(f"{name:12s} n={k.sum():3d} " + " | ".join(
            f"{c} ρ={spearmanr(v[k], t[k])[0]:+.2f} MAE={np.abs(v[k] - t[k]).mean():.2f}" for c, v in cols.items()))
    for c, v in cols.items():
        k = v >= np.median(v)
        top5 = np.mean([t[bt == b][np.argsort(-v[bt == b])[:5]].mean() for b in np.unique(bt)])
        print(f"{c:6s}：自己的預測前半 ρ={spearmanr(v[k], t[k])[0]:+.2f}；每批取前 5 實測平均 {top5:+.2f}；偏差 {np.mean(t - v):+.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("histval")
    h.add_argument("--variants", default="mlp,cnn,hyb-mlp,hyb-cnn")
    h.add_argument("--dmax-eval", type=int, default=120, help="評估配對距離上限（看 50 夠不夠）")
    b = sub.add_parser("backtest")
    b.add_argument("--ref", required=True, help="anchor_backtest.json（原始 SM 與錨定預測、實測 WM）")
    b.add_argument("--batches", default="72-78")
    b.add_argument("--variant", default="hyb-cnn")
    b.add_argument("--anchor", choices=("nearest", "parent"), default="nearest",
                   help="上下文＝離查詢最近的實測，或血緣父代（Ricky 10/19：兩種都試）")
    b.add_argument("--out", default="")
    for x in (h, b):
        x.add_argument("--dmax", type=int, default=DMAX, help="訓練配對距離上限（均勻取樣）")
        x.add_argument("--draws", type=int, default=16)
        x.add_argument("--epochs", type=int, default=30)
        x.add_argument("--ensemble", type=int, default=2)
        x.add_argument("--wide-weight", type=float, default=5.0)
        x.add_argument("--seed", type=int, default=82)
        x.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    {"histval": histval, "backtest": backtest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
