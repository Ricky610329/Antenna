"""R82 第 3 步：上下文接龍（Transformer）——K 個量過的鄰居（以查詢為中心）一起當上下文，預測查詢的曲線。

每個鄰居 k 帶（pattern、與查詢不同的格子、實測曲線＋遮罩），先由 CNN 編碼、各自提出「y_k + Δ_k」，
Transformer 讓 K 個鄰居互相參考，逐頻點輸出信任權重（鄰居沒量到的頻點權重歸零），加權合成最後曲線。
K=1 時退化成第 2 步的變化量模型，所以多出來的效果都可歸給「多看幾個鄰居」。
訓練上下文：一半取最近 K 個，一半在 50 格內按距離均勻抽（Ricky 2026-10-19：上下文 3–4 筆、挑法多樣）。

  python -m script.r82_ctx_sm histval   [--k 4]   # 歷史資料留出驗證：絕對 SM vs 變化量（K=1）vs 上下文（K）
  python -m script.r82_ctx_sm scarcesim [--k 4]   # 頻段稀缺模擬（同 r82_delta_sm scarcesim 的密集群設計）
  python -m script.r82_ctx_sm backtest --ref <anchor_backtest.json> [--k 4]   # v6 時間切分回測
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn

from script import r81_curve_sm as sm
from script import r82_delta_sm as dm


def near_list(bank, K):
    """鄰居庫中每筆設計最近的 K 個鄰居（依距離格由近到遠）；不足補 −1。"""
    nb = bank["nb"].reshape(len(bank["nb"]), -1)
    valid = nb >= 0
    order = np.argsort(~valid, axis=1, kind="stable")[:, :K]
    out = np.take_along_axis(nb, order, 1)
    out[~np.take_along_axis(valid, order, 1)] = -1
    return out


def sample_episodes(bank, near, rng, draws, K, p_near=0.5):
    """每筆查詢 draws 個情境：K 個上下文。一半取最近 K 個，其餘每格各抽 d~U(1..dmax) 的鄰居。回傳 (q, C)。"""
    alt, cnt, nb = bank["alt"], bank["cnt"], bank["nb"]
    q = np.repeat(np.arange(len(alt)), np.broadcast_to(np.asarray(draws), (len(alt),)))
    q = q[alt[q, 0] >= 0]
    C = np.empty((len(q), K), np.int64)
    for k in range(K):
        a = alt[q, rng.integers(0, alt.shape[1], size=len(q))].astype(np.int64)
        j = (rng.random(len(q)) * cnt[q, a]).astype(np.int64)
        C[:, k] = nb[q, a, j]
    use = (rng.random(len(q)) < p_near) & (near[q] >= 0).all(1)
    C[use] = near[q[use]]
    return q, C


class CtxNet(nn.Module):
    def __init__(self, g, d=256):
        super().__init__()
        self.g = g
        self.enc = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(64 * 49, 512), nn.ReLU())
        self.cur = nn.Sequential(nn.Linear(3 * g * 2 + 1, 256), nn.ReLU())
        self.tok = nn.Linear(512 + 256, d)
        layer = nn.TransformerEncoderLayer(d, 4, 512, dropout=0.0, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.delta = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 3 * g))
        self.logit = nn.Linear(d, 3 * g)

    def forward(self, img, cur, Yc, Mc):
        """img (B,K,1250)、cur (B,K,n)、Yc (B,K,3,G) dB/10、Mc (B,K,3,G) bool → (B,3,G) dB/10。"""
        B, K = img.shape[:2]
        z = self.enc(img.reshape(B * K, 2, 25, 25))
        h = self.tok(torch.cat([z, self.cur(cur.reshape(B * K, -1))], 1)).view(B, K, -1)
        h = self.tf(h)
        dl = self.delta(h).view(B, K, 3, self.g)
        lg = self.logit(h).view(B, K, 3, self.g).masked_fill(~Mc, -1e4)
        w = torch.softmax(lg, 1)
        return (w * (Yc + dl)).sum(1)


def _batch(Pt, Yt, Mt, q, C):
    B, K = C.shape
    cf = C.reshape(-1)
    qf = q.repeat_interleave(K)
    img, cur = dm.features(Pt[cf], Pt[qf], Yt[cf], Mt[cf])
    return (img.view(B, K, -1), cur.view(B, K, -1), (Yt[cf] / 10).view(B, K, *Yt.shape[1:]),
            Mt[cf].view(B, K, *Mt.shape[1:]))


def train_ctx(P, Y, M, focus, bank, draws, K, *, seeds=(0, 1), epochs=30, focus_weight=10.0, rng=None,
              dev="cpu", bs=512, log=print):
    rng = rng or np.random.default_rng(0)
    near = near_list(bank, K)
    Pt = torch.tensor(P, dtype=torch.uint8, device=dev)
    Yt = torch.tensor(Y, device=dev)
    Mt = torch.tensor(M, device=dev)
    Ft = torch.tensor(focus, device=dev)
    n_ep = int(np.broadcast_to(np.asarray(draws), (len(P),)).sum())
    models = []
    for s in seeds:
        torch.manual_seed(s)
        net = CtxNet(Y.shape[2]).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * (n_ep // bs + 2))
        for ep in range(epochs):
            q, C = sample_episodes(bank, near, rng, draws, K)
            q, C = torch.tensor(q, device=dev), torch.tensor(C, device=dev)
            perm = torch.randperm(len(q), device=dev)
            tot = 0.0
            for i in range(0, len(q), bs):
                b = perm[i:i + bs]
                qb, Cb = q[b], C[b]
                img, cur, Yc, Mc = _batch(Pt, Yt, Mt, qb, Cb)
                mb = (Mt[qb] & Mc.any(1)).float()
                wb = torch.where(Ft[qb] & Ft[Cb].all(1), focus_weight, 1.0)
                err = ((net(img, cur, Yc, Mc) - Yt[qb] / 10) ** 2 * mb).flatten(1).sum(1) / mb.flatten(1).sum(1).clamp(min=1)
                loss = (err * wb).sum() / wb.sum()
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += float(loss) * len(b)
        log(f"    ctx K={K} seed {s}: 末 epoch {len(q)} 情境，loss {tot / max(len(q), 1):.4f}")
        models.append(net.eval())
    return models


@torch.no_grad()
def predict_ctx(models, P, Y, M, C, Pq, dev="cpu", chunk=1024):
    """C (n,K) 上下文索引、Pq (n,625) 查詢 pattern → (n,3,G) dB（各 seed 平均）。"""
    Pt = torch.tensor(np.concatenate([P, Pq]), dtype=torch.uint8, device=dev)
    Yt = torch.tensor(np.concatenate([Y, np.zeros((len(Pq),) + Y.shape[1:], np.float32)]), device=dev)
    Mt = torch.tensor(np.concatenate([M, np.zeros((len(Pq),) + M.shape[1:], bool)]), device=dev)
    qi = torch.arange(len(P), len(P) + len(Pq), device=dev)
    Ct = torch.tensor(C, device=dev)
    out = []
    for s in range(0, len(Pq), chunk):
        img, cur, Yc, Mc = _batch(Pt, Yt, Mt, qi[s:s + chunk], Ct[s:s + chunk])
        out.append(torch.stack([m(img, cur, Yc, Mc) for m in models]).mean(0).cpu().numpy() * 10)
    return np.concatenate(out)


def nearest_k(Pk, q_idx_or_P, pool, K):
    """以查詢為中心：pool 中最近的 K 個（距離 0＝自己，排除）。回傳 (n,K) 索引與 (n,K) 距離。"""
    Qk = q_idx_or_P
    D = dm.hamming(Qk, Pk[pool]).astype(np.int32)
    D[D == 0] = 10 ** 6
    o = np.argsort(D, 1, kind="stable")[:, :K]
    return pool[o], np.take_along_axis(D, o, 1)


# ---------------------------------------------------------------- 評估

def histval(args):
    """按設計留出 10%；同一訓練設計周圍 ≤gmax 格的留出查詢成組，比較組內排序與誤差（上下文以查詢為中心取最近）。"""
    t0 = time.time()
    dev = dm._device(args)
    cfg, grid, slices, data = dm._setup(args)
    P, Y, M = data["P"], data["Y"], data["M"]
    N = len(P)
    rng = np.random.default_rng(args.seed)
    test = np.zeros(N, bool)
    test[rng.choice(N, size=N // 10, replace=False)] = True
    trn = np.flatnonzero(~test)
    Pk = np.packbits(P, 1)
    bank = dm.neighbor_bank(Pk, rng, allow=~test)
    draws = np.where(data["wide"], args.focus_draws, args.draws)
    print(f"資料 {N}（留出 {test.sum()}），{time.time() - t0:.0f}s；裝置 {dev}", flush=True)
    ens = dm.fit_sm(data, trn)
    m1 = dm.train(P, Y, M, data["wide"], bank, draws, seeds=tuple(range(args.ensemble)), epochs=args.epochs,
                  focus_weight=args.focus_weight, rng=rng, dev=dev)
    mk = train_ctx(P, Y, M, data["wide"], bank, np.maximum(draws // 2, 1), args.k, seeds=tuple(range(args.ensemble)),
                   epochs=args.epochs, focus_weight=args.focus_weight, rng=rng, dev=dev)
    print(f"模型完成，{time.time() - t0:.0f}s", flush=True)
    tq = np.flatnonzero(test)
    for gmax in (3, 12, 30):
        gq, gid = [], []
        ctx = rng.choice(trn, size=min(4000, len(trn)), replace=False)
        D = dm.hamming(Pk[ctx], Pk[tq]).astype(np.int32)
        for k, row in enumerate(D):
            hit = tq[(row >= 1) & (row <= gmax)]
            if len(hit) >= 5:
                hit = rng.choice(hit, size=min(15, len(hit)), replace=False)
                gq += list(hit); gid += [k] * len(hit)
            if len(set(gid)) >= 300:
                break
        gq, gid = np.asarray(gq), np.asarray(gid)
        C, _ = nearest_k(Pk, Pk[gq], trn, args.k)
        c1 = C[:, 0]
        pr = {"絕對SM": dm.sm_predict(ens, P[gq]),
              "變化量K=1": dm.predict(m1, P[c1], P[gq], Y[c1], M[c1], dev).mean(0),
              f"上下文K={args.k}": predict_ctx(mk, P, Y, M, C, P[gq], dev)}
        gm = M[gq]
        tg = dm.inband_score(Y[gq], gm, slices)
        ok = np.isfinite(tg)
        line = []
        for n, v in pr.items():
            mae = ((np.abs(v - Y[gq]) * gm).sum((1, 2)) / gm.sum((1, 2)).clip(1)).mean()
            g, _ = dm.group_rho(dm.inband_score(v, gm, slices)[ok], tg[ok], gid[ok], 5)
            line.append(f"{n} MAE {mae:.2f}／組內 ρ {g:+.2f}")
        print(f"≤{gmax} 格（{len(set(gid))} 組）：" + "  ".join(line), flush=True)
    print(f"總耗時 {time.time() - t0:.0f}s")


def scarcesim(args):
    """頻段稀缺模擬（同 r82_delta_sm.scarcesim 的密集群設計），比較絕對 SM、變化量 K=1、上下文 K。"""
    t0 = time.time()
    dev = dm._device(args)
    cfg, grid, slices, data = dm._setup(args)
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
    dense = probe[(dm.hamming(Pk[probe], Pk[cand]) <= args.region_d).sum(1) >= need]
    res, used = {}, np.zeros(N, bool)
    for ci in range(args.n_clusters):
        seed = int(rng.choice([d for d in dense if not used[d]]))
        region = cand[np.argsort(dm.hamming(Pk[seed:seed + 1], Pk[cand])[0], kind="stable")[:need]]
        used[region] = True
        test = np.zeros(N, bool)
        test[rng.choice(region, size=need - args.n_scarce, replace=False)] = True
        scarce = np.zeros(N, bool)
        scarce[region] = True
        scarce &= ~test
        M = M0.copy()
        M[~scarce] &= ~band
        tq, sc = np.flatnonzero(test), np.flatnonzero(scarce)
        D = dm.hamming(Pk[tq], Pk[sc]).astype(np.int32)
        gi, gj = np.nonzero((D >= 1) & (D <= 12))
        gq, gs = tq[gi], sc[gj]                                   # 組＝每個已量設計周圍 ≤12 格的查詢
        bank = dm.neighbor_bank(Pk, rng, allow=~test)
        draws = np.where(scarce, args.focus_draws, args.draws)
        ens = dm.fit_sm(data, np.flatnonzero(~test), M=M, seeds=(0, 1))
        m1 = dm.train(P, Y, M, scarce, bank, draws, seeds=tuple(range(args.ensemble)), epochs=args.epochs,
                      focus_weight=args.focus_weight, rng=rng, dev=dev)
        mk = train_ctx(P, Y, M, scarce, bank, np.maximum(draws // 2, 1), args.k, seeds=tuple(range(args.ensemble)),
                       epochs=args.epochs, focus_weight=args.focus_weight, rng=rng, dev=dev)
        Cq, _ = nearest_k(Pk, Pk[tq], sc, args.k)                # 上下文＝最近的 K 個已量設計
        c1 = Cq[:, 0]
        pq = {"絕對SM": dm.sm_predict(ens, P[tq]),
              "變化量K=1": dm.predict(m1, P[c1], P[tq], Y[c1], M[c1], dev).mean(0),
              f"上下文K={args.k}": predict_ctx(mk, P, Y, M, Cq, P[tq], dev)}
        truth = Y[tq][:, 1][:, band[1]].max(1)
        pos = {q: i for i, q in enumerate(tq)}
        gpos = np.array([pos[q] for q in gq])
        print(f"群 {ci}：查詢 {len(tq)}、組內配對 {len(gq)}，{time.time() - t0:.0f}s", flush=True)
        for n, v in pq.items():
            pv = v[:, 1][:, band[1]]
            mae = float(np.abs(pv - Y[tq][:, 1][:, band[1]]).mean())
            rho = float(spearmanr(pv.max(1), truth)[0])
            g, ng = dm.group_rho(pv.max(1)[gpos], truth[gpos], gs)
            res.setdefault(n, []).append((mae, rho, g))
            print(f"  {n:10s} 該段 MAE {mae:.2f} dB｜整體 ρ {rho:+.2f}｜組內 ρ {g:+.2f}（{ng} 組）", flush=True)
    print(f"\n平均（{args.n_clusters} 群）：")
    for n, v in res.items():
        v = np.array(v)
        print(f"  {n:10s} MAE {v[:, 0].mean():.2f}｜整體 ρ {np.nanmean(v[:, 1]):+.2f}｜組內 ρ {np.nanmean(v[:, 2]):+.2f}")
    print(f"總耗時 {time.time() - t0:.0f}s")


def backtest(args):
    """v6 時間切分回測：上下文＝最近的 K 個量齊 v6 頻點的實測（該批之前）。輸出與 r82_delta_sm.report 相容。"""
    t0 = time.time()
    dev = dm._device(args)
    cfg, grid, slices, data = dm._setup(args)
    need = np.zeros((3, len(grid)), bool)
    for _, li, lo, hi, *_ in slices:
        need[li, lo:hi + 1] = True
    P, Y, M = data["P"], data["Y"], data["M"]
    full = (M | ~need).all((1, 2))
    Pk = np.packbits(P, 1)
    rng = np.random.default_rng(args.seed)
    ref = {o["id"]: o for o in json.loads(Path(args.ref).read_text(encoding="utf-8"))}
    key_idx = {k.split(":", 1)[1]: i for i, k in enumerate(data["key"]) if not k.startswith("hist:")}
    bank_all = dm.neighbor_bank(Pk, rng)
    print(f"資料 {len(P)}，鄰居庫完成，{time.time() - t0:.0f}s；裝置 {dev}", flush=True)
    rows = []
    for b in dm.parse_range(args.batches):
        ok = data["batch"] < b
        bank = {k: v.copy() for k, v in bank_all.items()}
        bank["nb"][~ok] = -1
        dm._drop_future(bank, ok)
        focus = data["wide"] & ok
        draws = np.where(focus, args.focus_draws, args.draws) * ok
        mk = train_ctx(P, Y, M, focus, bank, np.maximum(draws // 2, ok.astype(int)), args.k,
                       seeds=tuple(range(args.ensemble)), epochs=args.epochs, focus_weight=args.focus_weight,
                       rng=rng, dev=dev)
        targets = [key_idx[i] for i, o in ref.items() if o["batch"] == b and i in key_idx]
        C, Dk = nearest_k(Pk, Pk[targets], np.flatnonzero(ok & full), args.k)
        pred = predict_ctx(mk, P, Y, M, C, P[targets], dev)
        wm = sm.margins_vec(pred, slices).min(1)
        for t, w, c, d in zip(targets, wm, C, Dk):
            o = dict(ref[data["key"][t].split(":", 1)[1]])
            o.update(delta=float(w), delta_p=float(w), ctx_d=int(d[0]),
                     ctx_true=float(sm.margins_vec(Y[c[0]:c[0] + 1], slices).min(1)[0]))
            rows.append(o)
        print(f"b{b}: 查詢 {len(targets)}，{time.time() - t0:.0f}s", flush=True)
    dm.report(rows)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("histval")
    s = sub.add_parser("scarcesim")
    s.add_argument("--band", default="24-25.5")
    s.add_argument("--n-scarce", type=int, default=436)
    s.add_argument("--n-clusters", type=int, default=4)
    s.add_argument("--region-d", type=int, default=30)
    b = sub.add_parser("backtest")
    b.add_argument("--ref", required=True)
    b.add_argument("--batches", default="72-78")
    b.add_argument("--out", default="")
    for x in (h, s, b):
        x.add_argument("--k", type=int, default=4, help="上下文鄰居數（Ricky：3–4 筆）")
        x.add_argument("--draws", type=int, default=16)
        x.add_argument("--focus-draws", type=int, default=64)
        x.add_argument("--focus-weight", type=float, default=10.0)
        x.add_argument("--epochs", type=int, default=30)
        x.add_argument("--ensemble", type=int, default=2)
        x.add_argument("--seed", type=int, default=82)
        x.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    {"histval": histval, "scarcesim": scarcesim, "backtest": backtest}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
