# -*- coding: utf-8 -*-
"""script/gnn_bakeoff.py — 方向③ 金屬像素圖 GNN bakeoff（規格 v3,Ricky 2026-07-30 定調）。

表示（金屬像素圖）:節點=金屬像素（座標/距feed,幾乎零手工特徵）;邊四型=面鄰接/對角鄰接/
跨縫 gap1(d=2,LOS)/gap2(d=3,LOS);虛擬全域節點（大件內部傳播捷徑+readout）;淺層+residual。
「訊息路徑=電流路徑」——治「組拓撲≠電流路徑」;d=1 翻轉=增刪節點（離散事件）。

判準（發車前寫死,scratch 2026-07-30 GNN v3）:
  紙筆考=142 鏈包 d=1 考卷——包內 pred×real Spearman ρ 中位 ≥0.30 ∧ pred_sd/real_sd ≥0.5
  （拓撲類/幾何類分層判;現任基線 mlp 0.068/cnn2 0.004）;過線→影子制度,不裸換。

子命令:
  build-cache                    全乾淨資料轉圖快取（tmp/gnn_cache/）
  train --op gine --layers 4     單組合訓練+三尺（held-out/凍結/harvest）
  grid                           {gine,gatv2}×{2,4,6}×{64,128} 12 組合格掃
  exam --ckpt <pth>              d=1 考卷（拓撲/幾何分層 ρ）
用法一律 repo 根 `python -m script.gnn_bakeoff <cmd>`;GPU 自動（有卡才）。
"""
import argparse
import os
import sys
import json
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from antenna.utils import DATASET_PATH  # noqa: E402

FEED = (24, 12)
CACHE_DIR = os.path.join(REPO, "tmp", "gnn_cache")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- 轉換器
#? 邊 offsets（半空間去重:只取 (dr>0) or (dr==0 and dc>0),雙向邊由對稱補）
def _offsets():
    out = []
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            d = max(abs(dr), abs(dc))
            if d == 0 or d > 3:
                continue
            if not (dr > 0 or (dr == 0 and dc > 0)):
                continue
            if d == 1:
                t = 0 if (dr == 0 or dc == 0) else 1          # 面/對角
            else:
                t = d                                          # 2=gap1, 3=gap2
            out.append((dr, dc, t))
    return out


_OFFS = _offsets()


def _between_cells(dr, dc):
    """嚴格中間格（LOS 檢查:cheb 距兩端皆 < d 的格）——跨縫邊要求全空。"""
    d = max(abs(dr), abs(dc))
    cells = []
    for r in range(min(0, dr) + 1 - 1, max(0, dr) + 1):
        for c in range(min(0, dc), max(0, dc) + 1):
            if (r, c) in ((0, 0), (dr, dc)):
                continue
            if max(abs(r), abs(c)) < d and max(abs(r - dr), abs(c - dc)) < d:
                cells.append((r, c))
    return cells


_BETWEEN = {(dr, dc): _between_cells(dr, dc) for dr, dc, t in _OFFS if t >= 2}


def pattern_to_graph(p):
    """25×25 bool → (node_feat[N,5], edge_index[2,E], edge_type[E])。
    節點=金屬像素+末位虛擬全域節點;邊含雙向+虛擬雙向(type 4)。"""
    p = np.asarray(p).reshape(25, 25).astype(bool)
    rr, cc = np.where(p)
    n = len(rr)
    idx = -np.ones((25, 25), dtype=np.int32)
    idx[rr, cc] = np.arange(n)
    fr, fc = FEED
    feat = np.zeros((n + 1, 5), dtype=np.float32)
    feat[:n, 0] = rr / 24.0
    feat[:n, 1] = cc / 24.0
    feat[:n, 2] = np.sqrt((rr - fr) ** 2 + (cc - fc) ** 2) / 34.0
    feat[:n, 3] = (idx[fr, fc] == np.arange(n)) if n else 0.0
    feat[n, 4] = 1.0                                           # 虛擬節點 flag
    src, dst, ety = [], [], []
    for dr, dc, t in _OFFS:
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < 25) & (c2 >= 0) & (c2 < 25)
        if not ok.any():
            continue
        j = idx[r2[ok], c2[ok]]
        hit = j >= 0
        if t >= 2 and hit.any():                               # LOS:中間格全空
            a_r, a_c = rr[ok][hit], cc[ok][hit]
            los = np.ones(len(a_r), dtype=bool)
            for br, bc in _BETWEEN[(dr, dc)]:
                mr, mc = a_r + br, a_c + bc
                inb = (mr >= 0) & (mr < 25) & (mc >= 0) & (mc < 25)
                occ = np.zeros(len(a_r), dtype=bool)
                occ[inb] = p[mr[inb], mc[inb]]
                los &= ~occ
            i_ = idx[a_r, a_c][los]
            j_ = j[hit][los]
        else:
            i_ = idx[rr[ok], cc[ok]][hit]
            j_ = j[hit]
        src += [i_, j_]
        dst += [j_, i_]
        ety += [np.full(len(i_), t, np.int64), np.full(len(i_), t, np.int64)]
    #? 虛擬節點雙向
    ar = np.arange(n)
    src += [ar, np.full(n, n)]
    dst += [np.full(n, n), ar]
    ety += [np.full(n, 4, np.int64), np.full(n, 4, np.int64)]
    ei = np.stack([np.concatenate(src), np.concatenate(dst)]) if src else np.zeros((2, 0), np.int64)
    return (torch.tensor(feat), torch.tensor(ei, dtype=torch.long),
            torch.tensor(np.concatenate(ety) if ety else np.zeros(0, np.int64)))


# ---------------------------------------------------------------- 模型
class GNN(nn.Module):
    def __init__(self, op="gine", layers=4, hidden=128, n_out=34):
        super().__init__()
        self.op, self.layers = op, layers
        self.emb = nn.Linear(5, hidden)
        self.temb = nn.Embedding(5, hidden)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        if op == "gine":
            self.mlps = nn.ModuleList(
                nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
                for _ in range(layers))
            self.eps = nn.Parameter(torch.zeros(layers))
        else:                                                  # gatv2（單頭,型別 embedding 進 attention）
            self.W = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
            self.att = nn.ModuleList(nn.Linear(hidden * 3, 1) for _ in range(layers))
            self.mlps = nn.ModuleList(
                nn.Sequential(nn.Linear(hidden, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, hidden))
                for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden * 2), nn.ReLU(),
                                  nn.Linear(hidden * 2, n_out + 2))   # 曲線 34 + wm/lo 標量

    def forward(self, feat, ei, ety, gid, n_graph, vmask):
        h = self.emb(feat)
        te_all = self.temb(ety)
        s, d = ei[0], ei[1]
        for L in range(self.layers):
            if self.op == "gine":
                msg = torch.relu(h[s] + te_all)
                agg = torch.zeros_like(h).index_add_(0, d, msg)
                h = h + self.norms[L](self.mlps[L]((1 + self.eps[L]) * h + agg))
            else:
                z = self.W[L](h)
                e = self.att[L](torch.cat([z[s], z[d], te_all], -1)).squeeze(-1)
                e = nn.functional.leaky_relu(e, 0.2)
                emax = torch.full((h.size(0),), -1e30, device=h.device).index_reduce_(
                    0, d, e, "amax", include_self=True)
                ex = torch.exp(e - emax[d])
                den = torch.zeros(h.size(0), device=h.device).index_add_(0, d, ex) + 1e-9
                w = (ex / den[d]).unsqueeze(-1)
                agg = torch.zeros_like(h).index_add_(0, d, w * z[s])
                h = h + self.norms[L](self.mlps[L](agg))
        #? readout:虛擬節點 + 真節點 sum/max（sum 保總量語意）
        hv = h[vmask]
        real = ~vmask
        gsum = torch.zeros(n_graph, h.size(1), device=h.device).index_add_(0, gid[real], h[real])
        gmax = torch.full((n_graph, h.size(1)), -1e30, device=h.device).index_reduce_(
            0, gid[real], h[real], "amax", include_self=True)
        return self.head(torch.cat([hv, gsum, gmax], -1))


def _collate(graphs, ys, device):
    """block-diagonal 批次。graphs=[(feat,ei,ety)],ys=[(34,), wm, lo]"""
    feats, eis, etys, gids, vmask = [], [], [], [], []
    off = 0
    for g, (feat, ei, ety) in enumerate(graphs):
        n = feat.size(0)
        feats.append(feat)
        eis.append(ei + off)
        etys.append(ety)
        gids.append(torch.full((n,), g, dtype=torch.long))
        vm = torch.zeros(n, dtype=torch.bool)
        vm[-1] = True
        vmask.append(vm)
        off += n
    y = torch.stack(ys)
    return (torch.cat(feats).to(device), torch.cat(eis, 1).to(device), torch.cat(etys).to(device),
            torch.cat(gids).to(device), len(graphs), torch.cat(vmask).to(device), y.to(device))


# ---------------------------------------------------------------- 資料
def _load_pool():
    """乾淨真值同鍋（reuse sm_reanchor:_load_clean/_load_harvest/_build_ds 的 reps 口徑）。
    回 (keys, graphs, targets, tr_idx_with_reps, ho_idx, fz_idx, hv_items)。"""
    from script.sm_reanchor import _load_clean, _load_harvest, _cfg, LABELS
    from antenna.losses import worst_margin
    from script.dedust import oob_metrics
    import hashlib
    tr, ho = _load_clean()
    replay, hval = _load_harvest(2000, 500)
    fzp = os.path.join(REPO, "configs", "heldout_frozen.json")
    fzk = set(json.load(open(fzp, encoding="utf-8"))["keys"]) if os.path.exists(fzp) else set()

    def tgt(y):
        w, _ = worst_margin(torch.as_tensor(y), LABELS, _cfg.targets)
        lo = oob_metrics(np.asarray(y).reshape(2, -1))["oob_gain_max_lo"]
        return np.concatenate([np.asarray(y).reshape(-1), [float(w)], [lo]]).astype(np.float32)

    items = [("tr", x, y) for x, y in tr] + [("ho", x, y) for x, y in ho] + \
            [("rp", x, y) for x, y in replay] + [("hv", x, y) for x, y in hval]
    return items, tgt, fzk, hashlib


def cmd_build_cache(args):
    os.makedirs(CACHE_DIR, exist_ok=True)
    items, tgt, fzk, hashlib = _load_pool()
    graphs, targets, kinds, iskeys = [], [], [], []
    t0 = time.time()
    for k, (kind, x, y) in enumerate(items):
        p = np.asarray(x).reshape(25, 25) > 0.5
        graphs.append(pattern_to_graph(p))
        targets.append(torch.tensor(tgt(y)))
        kinds.append(kind)
        iskeys.append(hashlib.md5(p.tobytes()).hexdigest())
        if k % 2000 == 0:
            print(f"{k}/{len(items)} {time.time()-t0:.0f}s", flush=True)
    torch.save(dict(graphs=graphs, targets=targets, kinds=kinds, keys=iskeys, frozen=list(fzk)),
               os.path.join(CACHE_DIR, "cache.pt"))
    n_nodes = [g[0].size(0) for g in graphs]
    n_edges = [g[1].size(1) for g in graphs]
    print(f"快取 {len(graphs)} 圖 → {CACHE_DIR}  節點中位 {int(np.median(n_nodes))} 邊中位 {int(np.median(n_edges))}")


def _train_one(op, layers, hidden, epochs=30, batch=48, seed=0, verbose=True):
    torch.manual_seed(seed)
    C = torch.load(os.path.join(CACHE_DIR, "cache.pt"), weights_only=False)
    graphs, targets, kinds, keys = C["graphs"], C["targets"], C["kinds"], C["keys"]
    fzk = set(C["frozen"])
    tr_i = [i for i, k in enumerate(kinds) if k == "tr"]
    ho_i = [i for i, k in enumerate(kinds) if k == "ho"]
    rp_i = [i for i, k in enumerate(kinds) if k == "rp"]
    hv_i = [i for i, k in enumerate(kinds) if k == "hv"]
    fz_i = [i for i in ho_i if keys[i] in fzk]
    #? 密度反權重（response 口徑,同鍋公平）:借 sm_reanchor._build_ds 的 reps
    from script.sm_reanchor import _load_clean, _load_harvest, _build_ds
    tr, _ho = _load_clean()
    replay, _hv = _load_harvest(2000, 500)
    _ds, reps = _build_ds(tr, replay, 8, mode="response")
    train_pool = [i for i, r in zip(tr_i, reps) for _ in range(int(max(1, min(int(r), 4))))] + rp_i
    #? reps cap 4（複製式在圖批次太貴;分布形狀保留,總量 ~2×——bakeoff 探針口徑,非投產口徑）
    model = GNN(op, layers, hidden).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-5)
    rng = np.random.default_rng(seed)

    def evaluate(idx):
        model.eval()
        errs = []
        with torch.no_grad():
            for b0 in range(0, len(idx), 256):
                bi = idx[b0:b0 + 256]
                fe, ei, et, gi, ng, vm, y = _collate([graphs[i] for i in bi],
                                                     [targets[i] for i in bi], DEV)
                out = model(fe, ei, et, gi, ng, vm)
                errs.append((out[:, 34] - y[:, 34]).abs().cpu())
        return float(torch.cat(errs).median())

    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_pool))
        tot, nb = 0.0, 0
        for b0 in range(0, len(order), batch):
            bi = [train_pool[j] for j in order[b0:b0 + batch]]
            fe, ei, et, gi, ng, vm, y = _collate([graphs[i] for i in bi],
                                                 [targets[i] for i in bi], DEV)
            out = model(fe, ei, et, gi, ng, vm)
            loss = nn.functional.mse_loss(out[:, :34], y[:, :34]) + \
                0.3 * nn.functional.mse_loss(out[:, 34], y[:, 34]) + \
                0.3 * nn.functional.mse_loss(out[:, 35], y[:, 35])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sch.step(tot / max(nb, 1))
        if verbose and (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}/{epochs} loss {tot/max(nb,1):.3f} ho {evaluate(ho_i):.3f}", flush=True)
    res = dict(op=op, layers=layers, hidden=hidden,
               ho=evaluate(ho_i), frozen=evaluate(fz_i), harvest=evaluate(hv_i))
    return model, res


def cmd_train(args):
    t0 = time.time()
    model, res = _train_one(args.op, args.layers, args.hidden, args.epochs)
    res["min"] = round((time.time() - t0) / 60, 1)
    print(json.dumps(res))
    out = os.path.join(CACHE_DIR, f"gnn_{args.op}_{args.layers}_{args.hidden}.pth")
    torch.save(model.state_dict(), out)
    print("→", out)


def cmd_grid(args):
    rows = []
    for op in ("gine", "gatv2"):
        for layers in (2, 4, 6):
            for hidden in (64, 128):
                t0 = time.time()
                model, res = _train_one(op, layers, hidden, args.epochs, verbose=False)
                res["min"] = round((time.time() - t0) / 60, 1)
                rows.append(res)
                torch.save(model.state_dict(),
                           os.path.join(CACHE_DIR, f"gnn_{op}_{layers}_{hidden}.pth"))
                print(json.dumps(res), flush=True)
    rows.sort(key=lambda r: r["frozen"])
    print("\n| op | L | H | ho | 凍結 | harvest | 分 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['op']} | {r['layers']} | {r['hidden']} | {r['ho']:.3f} | {r['frozen']:.3f} "
              f"| {r['harvest']:.3f} | {r['min']} |")


def cmd_exam(args):
    """d=1 考卷:142 鏈包,包內 pred×real Spearman;拓撲類（4-conn 組數或對角接點數改變）/幾何類分層。"""
    from scipy.stats import spearmanr
    from scipy.ndimage import label as _lab
    S8 = np.ones((3, 3), bool)
    m = GNN(args.op, args.layers, args.hidden).to(DEV)
    m.load_state_dict(torch.load(args.ckpt, weights_only=True))
    m.eval()

    def pred_wm(pats):
        outs = []
        with torch.no_grad():
            for b0 in range(0, len(pats), 256):
                gs = [pattern_to_graph(p) for p in pats[b0:b0 + 256]]
                fe, ei, et, gi, ng, vm, _y = _collate(gs, [torch.zeros(36)] * len(gs), DEV)
                o = m(fe, ei, et, gi, ng, vm)
                outs.append(o[:, 34].cpu())
        return torch.cat(outs).numpy()

    def topo_sig(p):
        _l4, n4 = _lab(p)
        _l8, n8 = _lab(p, structure=S8)
        return n4, n8

    packs = sorted(d for d in os.listdir(str(DATASET_PATH))
                   if d.startswith("dedust_c") and not d.endswith("_input")
                   and "_p" in d and DATASET_PATH.joinpath(d, "results.json").exists())
    rows_all, rows_topo, rows_geo, sd_ratio = [], [], [], []
    for st in packs:
        ind = DATASET_PATH.joinpath(st + "_input")
        res = json.load(open(str(DATASET_PATH.joinpath(st, "results.json")), encoding="utf-8"))
        ids = [k for k, v in res.items() if "wm" in v and "error" not in v]
        if len(ids) < 8:
            continue
        pats, real = [], []
        for k in ids:
            f = ind.joinpath(k + ".pt")
            if not f.exists():
                continue
            pats.append(np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5)
            real.append(res[k]["wm"][2])
        if len(pats) < 8:
            continue
        pw = pred_wm(pats)
        rho = spearmanr(pw, real)[0]
        rows_all.append(rho)
        sd_ratio.append(np.std(pw) / (np.std(real) + 1e-9))
        sigs = [topo_sig(p) for p in pats]
        base = max(set(sigs), key=sigs.count)
        ti = [i for i, s in enumerate(sigs) if s != base]
        gi_ = [i for i, s in enumerate(sigs) if s == base]
        if len(ti) >= 6:
            rows_topo.append(spearmanr(pw[ti], np.array(real)[ti])[0])
        if len(gi_) >= 6:
            rows_geo.append(spearmanr(pw[gi_], np.array(real)[gi_])[0])
    print(f"包數 {len(rows_all)} | 包內 ρ 中位 {np.nanmedian(rows_all):+.3f}"
          f" | pred_sd/real_sd 中位 {np.nanmedian(sd_ratio):.3f}")
    print(f"拓撲類（n={len(rows_topo)} 包）ρ 中位 {np.nanmedian(rows_topo):+.3f}"
          f" | 幾何類（n={len(rows_geo)} 包）ρ 中位 {np.nanmedian(rows_geo):+.3f}")
    print("判準:全包 ρ≥0.30 ∧ sd 比 ≥0.5 → 過線進影子;分層帳照記（拓撲過幾何死=部分勝利）")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(required=True)
    s = sub.add_parser("build-cache")
    s.set_defaults(fn=cmd_build_cache)
    s = sub.add_parser("train")
    s.add_argument("--op", default="gine", choices=["gine", "gatv2"])
    s.add_argument("--layers", type=int, default=4)
    s.add_argument("--hidden", type=int, default=128)
    s.add_argument("--epochs", type=int, default=30)
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("grid")
    s.add_argument("--epochs", type=int, default=25)
    s.set_defaults(fn=cmd_grid)
    s = sub.add_parser("exam")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--op", default="gine", choices=["gine", "gatv2"])
    s.add_argument("--layers", type=int, default=4)
    s.add_argument("--hidden", type=int, default=128)
    s.set_defaults(fn=cmd_exam)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
