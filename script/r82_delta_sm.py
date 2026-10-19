"""R82 變化量代理模型：上下文（量過的鄰居 pattern＋實測曲線）＋查詢 pattern → 查詢曲線。

上下文 1 筆 c，學 Δ = y(查詢) − y(c)，預測＝y(c) + Δ̂。
- 鄰居以查詢為中心找（Ricky 2026-10-19）；v6 回測同時算「最近實測」與「血緣父代」兩種上下文。
- 配對距離 d ~ 均勻(1..50)（Ricky：50 格內均勻）；每個 epoch 從鄰居庫重抽，30 個 epoch 看到上千萬組不同配對。
- 架構：cnn（上下文圖＋翻轉遮罩當兩張 25×25 圖，歷史驗證勝 mlp 與錨定殘差混合式）或 mlp。
- 兩段式：先全部資料（遮罩訓練），再只用「重點子集」（v6＝寬頻真值）配對、小學習率微調。
資料：24–32 GHz 歷史＋寬頻真值，全部對位到 v6 的 11–40 GHz 格點，沒量到的頻點以遮罩略過。

  python -m script.r82_delta_sm scarcesim     # 用歷史資料模擬「某頻段只有少數設計量過」，選方案（Ricky 10/19）
  python -m script.r82_delta_sm histval       # 本機歷史驗證（按設計留出）
  python -m script.r82_delta_sm backtest --ref <anchor_backtest.json>   # v6 時間切分回測
"""
import argparse
import copy
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


def load_all(cfg, grid, hist=None, wide=None):
    """歷史＋寬頻 → dict(P bool N×625, Y N×3×G, M bool, wide bool, batch int, key, src)。同圖只留一筆。"""
    hist = sm.load_history(sm.HIST_DIR, grid) if hist is None else hist
    wide = sm.load_wide(sm.WIDE_ROOT, cfg, grid) if wide is None else wide
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


# ---------------------------------------------------------------- 配對：鄰居庫＋每 epoch 重抽

def neighbor_bank(Pk, rng, allow=None, dmax=DMAX, per=12, chunk=64):
    """每筆設計、每個距離 d=1..dmax 各存至多 per 個鄰居（只在 allow 內）。
    回傳 dict(nb int32 N×dmax×per, cnt N×dmax, alt N×dmax)：alt[i,d]＝最接近 d 且非空的距離格（無鄰居 −1）。"""
    N = len(Pk)
    allow = np.ones(N, bool) if allow is None else allow
    idx = np.flatnonzero(allow)
    nb = np.full((N, dmax, per), -1, np.int32)
    cnt = np.zeros((N, dmax), np.int16)
    for s in range(0, len(idx), chunk):
        D = hamming(Pk[idx[s:s + chunk]], Pk[idx]).astype(np.int16)
        for k, row in enumerate(D):
            i = idx[s + k]
            sel = np.flatnonzero((row >= 1) & (row <= dmax))
            if not len(sel):
                continue
            sel = sel[rng.permutation(len(sel))]
            dd = row[sel] - 1
            order = np.argsort(dd, kind="stable")
            sel, dd = sel[order], dd[order]
            starts = np.searchsorted(dd, np.arange(dmax))
            ends = np.searchsorted(dd, np.arange(dmax), side="right")
            for d in np.flatnonzero(ends > starts):
                take = sel[starts[d]:min(ends[d], starts[d] + per)]
                nb[i, d, :len(take)] = idx[take]
                cnt[i, d] = len(take)
    return {"nb": nb, "cnt": cnt, "alt": _nearest_bin(cnt)}


def _nearest_bin(cnt):
    """alt[i,d]＝最接近 d 且有鄰居的距離格（整列無鄰居 −1）。"""
    has = cnt > 0
    dmax = has.shape[1]
    alt = np.full(has.shape, -1, np.int16)
    ar = np.arange(dmax)
    for i in np.flatnonzero(has.any(1)):
        ne = np.flatnonzero(has[i])
        pos = np.searchsorted(ne, ar)
        lo, hi = ne[np.clip(pos - 1, 0, len(ne) - 1)], ne[np.clip(pos, 0, len(ne) - 1)]
        alt[i] = np.where(np.abs(hi - ar) <= np.abs(ar - lo), hi, lo)
    return alt


def sample_pairs(bank, rng, draws):
    """draws（每筆設計的抽樣次數）→ (q, c)：d~U(1..dmax)，取最接近 d 的非空距離格，格內隨機一個鄰居。"""
    alt, cnt, nb = bank["alt"], bank["cnt"], bank["nb"]
    draws = np.broadcast_to(np.asarray(draws), (len(alt),))
    q = np.repeat(np.arange(len(alt)), draws)
    d = rng.integers(0, alt.shape[1], size=len(q))
    a = alt[q, d]
    ok = a >= 0
    q, a = q[ok], a[ok].astype(np.int64)
    j = (rng.random(len(q)) * cnt[q, a]).astype(np.int64)
    return q, nb[q, a, j].astype(np.int64)


# ---------------------------------------------------------------- 模型

class DeltaNet(nn.Module):
    """輸入：上下文圖、翻轉遮罩、上下文曲線＋遮罩、距離 → Δ 曲線（dB/10）。"""

    def __init__(self, g, arch="cnn"):
        super().__init__()
        self.g, self.arch = g, arch
        n_curve = 3 * g * 2 + 1
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


def features(Pc, Pq, Yc, Mc):
    """torch 張量 → (img (n,1250), curve)。"""
    flip = (Pc != Pq).float()
    img = torch.cat([Pc.float(), flip], 1)
    curve = torch.cat([(Yc * Mc / 10).flatten(1), Mc.float().flatten(1), flip.sum(1, keepdim=True) / DMAX], 1)
    return img, curve


def train(P, Y, M, focus, bank, draws, *, arch="cnn", seeds=(0, 1), epochs=30, focus_weight=5.0,
          ft_bank=None, ft_draws=0, ft_epochs=0, init=None, rng=None, dev="cpu", log=print):
    """masked MSE；兩端都在 focus（重點子集）的配對加權。每個 epoch 從 bank 重抽配對。
    ft_bank：第二段只用重點子集配對、學習率 3e-4 微調 ft_epochs。init：沿用已訓練模型（複製後只做微調）。"""
    rng = rng or np.random.default_rng(0)
    Pt = torch.tensor(P, dtype=torch.uint8, device=dev)
    Yt = torch.tensor(Y, device=dev)
    Mt = torch.tensor(M, device=dev)
    Ft = torch.tensor(focus, device=dev)

    def run(net, opt, sched, bk, dr, n_ep, tag):
        for ep in range(n_ep):
            q, c = sample_pairs(bk, rng, dr)
            q, c = torch.tensor(q, device=dev), torch.tensor(c, device=dev)
            keep = (Mt[q] & Mt[c]).flatten(1).any(1)
            q, c = q[keep], c[keep]
            perm = torch.randperm(len(q), device=dev)
            tot = 0.0
            for i in range(0, len(q), 1024):
                b = perm[i:i + 1024]
                qb, cb = q[b], c[b]
                img, cur = features(Pt[cb], Pt[qb], Yt[cb], Mt[cb])
                t = (Yt[qb] - Yt[cb]) / 10
                mb = (Mt[qb] & Mt[cb]).float()
                wb = torch.where(Ft[qb] & Ft[cb], focus_weight, 1.0)
                err = ((net(img, cur) - t) ** 2 * mb).flatten(1).sum(1) / mb.flatten(1).sum(1).clamp(min=1)
                loss = (err * wb).sum() / wb.sum()
                opt.zero_grad(); loss.backward(); opt.step()
                if sched:
                    sched.step()
                tot += float(loss) * len(b)
        log(f"    {tag}: 末 epoch {len(q)} 配對，loss {tot / max(len(q), 1):.4f}")

    n_per_ep = int(np.broadcast_to(np.asarray(draws), (len(P),)).sum())
    models = []
    for k, s in enumerate(seeds):
        torch.manual_seed(s)
        if init is not None:
            net = copy.deepcopy(init[k]).train()
        else:
            net = DeltaNet(Y.shape[2], arch).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (n_per_ep // 1024 + 2))
            run(net, opt, sched, bank, draws, epochs, f"{arch} seed {s}")
        if ft_bank is not None and ft_epochs:
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
            run(net, opt, None, ft_bank, ft_draws, ft_epochs, f"{arch} seed {s} 微調")
        models.append(net.eval())
    return models


@torch.no_grad()
def predict(models, Pc, Pq, Yc, Mc, dev="cpu", chunk=4096):
    """numpy 輸入 → (K, n, 3, G) 預測曲線 dB：y(c) + 模型輸出。"""
    outs = []
    for s in range(0, len(Pc), chunk):
        sl = slice(s, s + chunk)
        img, cur = features(torch.tensor(Pc[sl], dtype=torch.uint8, device=dev),
                            torch.tensor(Pq[sl], dtype=torch.uint8, device=dev),
                            torch.tensor(Yc[sl], device=dev), torch.tensor(Mc[sl], device=dev))
        outs.append(np.stack([Yc[sl] + m(img, cur).cpu().numpy() * 10 for m in models]))
    return np.concatenate(outs, 1)


def score_candidates(Pq, slices, data, *, seeds=(0, 1), epochs=30, draws=16, focus_draws=64, focus_weight=10.0,
                     seed=82, log=print):
    """選樣用：以全部資料訓練變化量模型，對每個候選取「最近的實測設計」（須量齊 slices 用到的頻點）當上下文，
    回傳 dict(wm_mean, wm_std, ctx_d)；上下文超過 DMAX 格（訓練範圍外）時 wm_mean＝nan。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P, Y, M = data["P"], data["Y"], data["M"]
    need = np.zeros(M.shape[1:], bool)
    for _, li, lo, hi, *_ in slices:
        need[li, lo:hi + 1] = True
    full = np.flatnonzero((M | ~need).all((1, 2)))
    rng = np.random.default_rng(seed)
    Pk = np.packbits(P, 1)
    bank = neighbor_bank(Pk, rng)
    models = train(P, Y, M, data["wide"], bank, np.where(data["wide"], focus_draws, draws), seeds=seeds,
                   epochs=epochs, focus_weight=focus_weight, rng=rng, dev=dev, log=log)
    Pq = np.asarray(Pq).reshape(len(Pq), -1) > 0.5
    D = hamming(np.packbits(Pq, 1), Pk[full])
    j, d = full[D.argmin(1)], D.min(1).astype(int)
    pred = predict(models, P[j], Pq, Y[j], M[j], dev)            # (K,n,3,G)
    wm = np.stack([sm.margins_vec(k, slices).min(1) for k in pred])
    mean = np.where(d <= DMAX, wm.mean(0), np.nan)
    return {"wm_mean": mean, "wm_std": wm.std(0), "ctx_d": d}


# ---------------------------------------------------------------- 絕對曲線 SM（對照組）

def fit_sm(data, idx, M=None, seeds=(0, 1, 2)):
    """以 idx 內設計訓練現行絕對曲線 SM；M 可覆寫遮罩（頻段稀缺模擬用）。"""
    M = data["M"] if M is None else M
    h = data["hist"]
    hs = [(k, i) for k in idx for (src, i) in [data["src"][k]] if src == "hist"]
    hsub = {"P": h["P"][[i for _, i in hs]], "Y": h["Y"][[i for _, i in hs]], "M": M[[k for k, _ in hs]],
            "sha": [h["sha"][i] for _, i in hs], "lineage": [h["lineage"][i] for _, i in hs]}
    wsub = [dict(data["wide_rows"][i], mask=M[k]) for k in idx for (src, i) in [data["src"][k]] if src == "wide"]
    return sm.train_ensemble(hsub, wsub, seeds=seeds)


def sm_predict(ens, P):
    return ens.predict(np.asarray(P, np.float32)).mean(0)


# ---------------------------------------------------------------- 評估工具

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


def group_rho(score, truth, gid, min_n=4):
    """同上下文組內 Spearman 平均。"""
    rs = []
    for g in np.unique(gid):
        k = gid == g
        if k.sum() >= min_n and np.ptp(truth[k]) > 0 and np.ptp(score[k]) > 0:
            rs.append(spearmanr(score[k], truth[k])[0])
    return (float(np.mean(rs)) if rs else float("nan")), len(rs)


def parse_range(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def _device(args):
    torch.set_num_threads(4)
    return "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"


def _setup(args):
    cfg = sm.load_cfg(V6)
    grid = sm.frequency_grid(cfg.measurement)
    return cfg, grid, sm.band_slices(cfg.score_spec, grid), load_all(cfg, grid)


# ---------------------------------------------------------------- 頻段稀缺模擬（選方案）

def scarcesim(args):
    """用歷史資料模擬 v6 的處境：S21 某段（預設 24–25.5 GHz）只有一群相近設計（n_scarce 筆）量過，其餘全遮掉。
    群取自密集區（像 v6 冠軍家族）；留出群內部分設計當查詢（上下文＝最近的已量設計），比較：照抄、絕對 SM、
    變化量（一段式）、變化量（＋微調）。看該段曲線誤差、該段分數（S21 最大值）的整體排序，以及「同一已量設計周圍
    ≤12 格查詢」的組內排序。重複 n_clusters 群取平均。"""
    t0 = time.time()
    dev = _device(args)
    cfg, grid, slices, data = _setup(args)
    P, Y, M0 = data["P"], data["Y"], data["M"]
    N = len(P)
    lo, hi = (float(x) for x in args.band.split("-"))
    band = np.zeros_like(M0[0])
    band[1, (grid >= lo - 1e-9) & (grid <= hi + 1e-9)] = True
    has = (M0 & band).sum((1, 2)) == band.sum()
    Pk = np.packbits(P, 1)
    rng = np.random.default_rng(args.seed)
    cand = np.flatnonzero(has & ~data["wide"])
    need = int(args.n_scarce * 1.4)
    probe = rng.choice(cand, size=min(3000, len(cand)), replace=False)
    dense = probe[(hamming(Pk[probe], Pk[cand]) <= args.region_d).sum(1) >= need]   # 像 v6：一群彼此相近的設計
    print(f"資料 {N}，該段量齊的歷史設計 {len(cand)}，可當密集群中心 {len(dense)}／{len(probe)}；"
          f"模擬頻段 S21 {lo}–{hi} GHz，{time.time() - t0:.0f}s；裝置 {dev}", flush=True)
    res = {}
    used = np.zeros(N, bool)
    for ci in range(args.n_clusters):
        seed = int(rng.choice([d for d in dense if not used[d]]))
        d0 = hamming(Pk[seed:seed + 1], Pk[cand])[0]
        region = cand[np.argsort(d0, kind="stable")[:need]]
        used[region] = True
        test = np.zeros(N, bool)
        test[rng.choice(region, size=need - args.n_scarce, replace=False)] = True
        scarce = np.zeros(N, bool)
        scarce[region] = True
        scarce &= ~test
        M = M0.copy()
        M[~scarce] &= ~band                                       # 群外、留出都看不到該段
        tq, sc = np.flatnonzero(test), np.flatnonzero(scarce)
        D = hamming(Pk[tq], Pk[sc]).astype(np.int32)
        ctx, dctx = sc[D.argmin(1)], D.min(1)                     # 上下文＝最近的已量設計（以查詢為中心）
        gi, gj = np.nonzero((D >= 1) & (D <= 12))                 # 爬山情境：每個已量設計周圍 ≤12 格的查詢成一組
        gq, gc = tq[gi], sc[gj]
        bank = neighbor_bank(Pk, rng, allow=~test, dmax=args.dmax)
        ft_bank = neighbor_bank(Pk, rng, allow=scarce, dmax=args.dmax)
        print(f"群 {ci}：已量 {scarce.sum()}、查詢 {len(tq)}（上下文距離中位 {int(np.median(dctx))}，"
              f"最大 {int(dctx.max())}；組內配對 {len(gq)}），{time.time() - t0:.0f}s", flush=True)
        draws = np.where(scarce, args.focus_draws, args.draws)
        ens = fit_sm(data, np.flatnonzero(~test), M=M, seeds=(0, 1))
        ms1 = train(P, Y, M, scarce, bank, draws, arch=args.arch, seeds=tuple(range(args.ensemble)),
                    epochs=args.epochs, focus_weight=args.focus_weight, rng=rng, dev=dev)
        ms2 = train(P, Y, M, scarce, bank, draws, arch=args.arch, seeds=tuple(range(args.ensemble)),
                    ft_bank=ft_bank, ft_draws=args.focus_draws, ft_epochs=args.ft_epochs, init=ms1, rng=rng, dev=dev)
        sm_q, sm_g = sm_predict(ens, P[tq]), sm_predict(ens, P[gq])
        preds = {"照抄": (Y[ctx], Y[gc]), "絕對SM": (sm_q, sm_g)}
        for n, ms in (("變化量", ms1), ("變化量+微調", ms2)):
            preds[n] = (predict(ms, P[ctx], P[tq], Y[ctx], M[ctx], dev).mean(0),
                        predict(ms, P[gc], P[gq], Y[gc], M[gc], dev).mean(0))
        truth, gtruth = Y[tq][:, 1][:, band[1]].max(1), Y[gq][:, 1][:, band[1]].max(1)
        for n, (v, vg) in preds.items():
            pv = v[:, 1][:, band[1]]
            mae = float(np.abs(pv - Y[tq][:, 1][:, band[1]]).mean())
            rho = float(spearmanr(pv.max(1), truth)[0])
            g, ng = group_rho(vg[:, 1][:, band[1]].max(1), gtruth, gc)
            res.setdefault(n, []).append((mae, rho, g))
            print(f"  {n:8s} 該段 MAE {mae:.2f} dB｜整體 ρ {rho:+.2f}｜組內 ρ {g:+.2f}（{ng} 組）", flush=True)
    print(f"\n平均（{args.n_clusters} 群）：")
    for n, v in res.items():
        v = np.array(v)
        print(f"  {n:8s} MAE {v[:, 0].mean():.2f}｜整體 ρ {np.nanmean(v[:, 1]):+.2f}｜組內 ρ {np.nanmean(v[:, 2]):+.2f}")
    print(f"總耗時 {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- 本機歷史驗證

def histval(args):
    """按設計留出 10%；查詢＝留出設計、上下文＝訓練設計。距離分組誤差＋同上下文組內排序。"""
    t0 = time.time()
    dev = _device(args)
    cfg, grid, slices, data = _setup(args)
    P, Y, M = data["P"], data["Y"], data["M"]
    N = len(P)
    rng = np.random.default_rng(args.seed)
    test = np.zeros(N, bool)
    test[rng.choice(N, size=N // 10, replace=False)] = True
    trn = np.flatnonzero(~test)
    Pk = np.packbits(P, 1)
    bank = neighbor_bank(Pk, rng, allow=~test, dmax=args.dmax)
    print(f"資料 {N}（留出 {test.sum()}），{time.time() - t0:.0f}s；裝置 {dev}", flush=True)
    ens = fit_sm(data, trn)
    models = train(P, Y, M, data["wide"], bank, np.where(data["wide"], args.focus_draws, args.draws), arch=args.arch,
                   seeds=tuple(range(args.ensemble)), epochs=args.epochs, focus_weight=args.focus_weight,
                   rng=rng, dev=dev)
    tq = np.flatnonzero(test)
    for gmax in (3, 12, 30):
        gq, gc = [], []
        ctx = rng.choice(trn, size=min(4000, len(trn)), replace=False)
        D = hamming(Pk[ctx], Pk[tq]).astype(np.int32)
        for k, row in enumerate(D):
            hit = tq[(row >= 1) & (row <= gmax)]
            if len(hit) >= 5:
                hit = rng.choice(hit, size=min(15, len(hit)), replace=False)
                gq += list(hit); gc += [ctx[k]] * len(hit)
            if len(set(gc)) >= 300:
                break
        gq, gc = np.asarray(gq), np.asarray(gc)
        gm = M[gq] & M[gc]
        tg = inband_score(Y[gq], gm, slices)
        pr = {"絕對SM": sm_predict(ens, P[gq]), "變化量": predict(models, P[gc], P[gq], Y[gc], M[gc], dev).mean(0)}
        ok = np.isfinite(tg)
        line = []
        for n, v in pr.items():
            mae = ((np.abs(v - Y[gq]) * gm).sum((1, 2)) / gm.sum((1, 2)).clip(1)).mean()
            g, _ = group_rho(inband_score(v, gm, slices)[ok], tg[ok], gc[ok], 5)
            line.append(f"{n} MAE {mae:.2f}／組內 ρ {g:+.2f}")
        print(f"≤{gmax} 格（{len(set(gc))} 組）：" + "  ".join(line), flush=True)
    print(f"總耗時 {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- v6 時間切分回測

def backtest(args):
    """b72..b78 每批只用該批之前的資料訓練，預測該批實測候選（集合同 anchor_backtest.json）。"""
    t0 = time.time()
    dev = _device(args)
    cfg, grid, slices, data = _setup(args)
    need = np.zeros((3, len(grid)), bool)
    for _, li, lo, hi, *_ in slices:
        need[li, lo:hi + 1] = True
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
    bank_all = neighbor_bank(Pk, rng, dmax=args.dmax)
    print(f"鄰居庫完成，{time.time() - t0:.0f}s", flush=True)
    rows = []
    for b in parse_range(args.batches):
        ok = data["batch"] < b                                    # 時間切分（歷史 batch=−1 一律可用）
        bank = {k: v.copy() for k, v in bank_all.items()}
        bank["nb"][~ok] = -1                                      # 未來的設計不當查詢
        _drop_future(bank, ok)                                    # 也不當上下文
        focus = data["wide"] & ok
        ft_bank = None
        if args.ft_epochs:
            ft_bank = {k: v.copy() for k, v in bank.items()}
            _drop_future(ft_bank, focus)
        models = train(P, Y, M, focus, bank, np.where(focus, args.focus_draws, args.draws) * ok, arch=args.arch,
                       seeds=tuple(range(args.ensemble)), epochs=args.epochs, focus_weight=args.focus_weight,
                       ft_bank=ft_bank, ft_draws=np.where(focus, args.focus_draws, 0), ft_epochs=args.ft_epochs,
                       rng=rng, dev=dev)
        targets = [key_idx[i] for i, o in ref.items() if o["batch"] == b and i in key_idx]
        pool = np.flatnonzero(ok & full)
        Dt = hamming(Pk[targets], Pk[pool]) if targets else np.zeros((0, len(pool)))
        for t, drow in zip(targets, Dt):
            o = dict(ref[data["key"][t].split(":", 1)[1]])
            jn = pool[np.argmin(drow)]                            # 離查詢最近的實測（以查詢為中心）
            jp = full_key.get(parent_of.get(data["key"][t].split(":", 1)[1], ""))
            jp = jp if jp is not None and ok[jp] else jn          # 血緣父代（有實測且可算 v6 才用）
            for tag, j in (("", jn), ("_p", jp)):
                pred = predict(models, P[j:j + 1], P[t:t + 1], Y[j:j + 1], M[j:j + 1], dev).mean(0)
                o["delta" + tag] = float(sm.margins_vec(pred, slices).min(1)[0])
                o["ctx_d" + tag] = int(_POP[Pk[t] ^ Pk[j]].sum())
                o["ctx_true" + tag] = float(sm.margins_vec(Y[j:j + 1], slices).min(1)[0])
            rows.append(o)
        print(f"b{b}: 查詢 {len(targets)}，{time.time() - t0:.0f}s", flush=True)
    report(rows)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


def _drop_future(bank, ok):
    """鄰居庫中只保留 ok 內的鄰居（ok 外的設計也不當查詢），並重算 cnt／alt。"""
    nb = bank["nb"]
    valid = (nb >= 0) & ok[np.clip(nb, 0, None)]
    nb[~valid] = -1
    order = np.argsort(~valid, axis=2, kind="stable")             # 有效的鄰居擠到前面
    bank["nb"] = np.take_along_axis(nb, order, 2)
    bank["cnt"] = valid.sum(2).astype(np.int16)
    bank["alt"] = _nearest_bin(bank["cnt"])


def report(rows):
    t = np.array([r["true"] for r in rows])
    cols = {"原始SM": np.array([r["raw"] for r in rows]), "錨定(父)": np.array([r["anc"] for r in rows]),
            "變化量(最近)": np.array([r["delta"] for r in rows]), "變化量(父)": np.array([r["delta_p"] for r in rows]),
            "照抄最近": np.array([r["ctx_true"] for r in rows])}
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
    s = sub.add_parser("scarcesim")
    s.add_argument("--band", default="24-25.5", help="模擬稀缺的 S21 頻段（GHz）")
    s.add_argument("--n-scarce", type=int, default=436, help="該段有量的設計數（v6 11–40 完整資料量）")
    s.add_argument("--n-clusters", type=int, default=4)
    s.add_argument("--region-d", type=int, default=30, help="群中心 30 格內至少要有 1.4×n_scarce 個設計（密集群）")
    h = sub.add_parser("histval")
    b = sub.add_parser("backtest")
    b.add_argument("--ref", required=True, help="anchor_backtest.json（原始 SM 與錨定預測、實測 WM）")
    b.add_argument("--batches", default="72-78")
    b.add_argument("--out", default="")
    for x in (s, h, b):
        x.add_argument("--arch", choices=("cnn", "mlp"), default="cnn")
        x.add_argument("--dmax", type=int, default=DMAX, help="配對距離上限（均勻取樣）")
        x.add_argument("--draws", type=int, default=16, help="每個 epoch 每筆設計抽幾組配對")
        x.add_argument("--focus-draws", type=int, default=64, help="重點子集（寬頻／稀缺群）每 epoch 抽樣次數")
        x.add_argument("--focus-weight", type=float, default=10.0)
        x.add_argument("--epochs", type=int, default=30)
        x.add_argument("--ft-epochs", type=int, default=10, help="第二段只用重點子集配對微調的 epoch 數（0＝不微調）")
        x.add_argument("--ensemble", type=int, default=2)
        x.add_argument("--seed", type=int, default=82)
        x.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    {"scarcesim": scarcesim, "histval": histval, "backtest": backtest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
