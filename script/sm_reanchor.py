# -*- coding: utf-8 -*-
"""
script/sm_reanchor.py — SM 乾淨區重錨（R10 Stage A ／ R8 C 臂判準的後半）。

背景：SM（sm_harvest.pth,學長池上訓的）在乾淨投影區＝分布外——一致樂觀 +1.4~+4.3（R8/R9 實測），
排序有訊號但絕對值不可信。把 r7+r8+r9 的乾淨區 HFSS 真值（~270 筆,去重）餵回去重錨,
配 harvest 重放（防災難性遺忘）——「週期 harvest 重錨」候選的第一次落地。
（起點 sm_harvest.pth 由 `script/train_sm_offline.py` 初訓;同族譜:初訓→重錨。）

用法（開發機,零 HFSS）：
    python -m script.sm_reanchor train   # sm_harvest.pth 起點 → 訓練 → DATASET_PATH/sm_reanchor.pth
    python -m script.sm_reanchor eval    # 前(sm_harvest) vs 後(sm_reanchor)：held-out 乾淨 / harvest 驗證

判準（R8 C 臂）：held-out 乾淨區 |wm 誤差| 中位進 **~2 dB 帶** → 精修 round 導航儀合格；
harvest 驗證誤差不得明顯惡化（遺忘檢查）。切分決定性（hash 排序取每第 5 筆當 held-out）。
"""
import argparse
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from antenna.utils import config as _config, DATASET_PATH
_config.device = "cpu"
import torch
from torch.utils.data import TensorDataset, ConcatDataset

from antenna.training import load_config, setup_responses, PORT_SPECS
from antenna.losses import worst_margin
from antenna.utils.store import SampleStore
from antenna.zoo import SURROGATES

import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CFG = os.path.join(REPO, "configs", "single_r5_explore.yaml")
#? 去重「先見先贏」→ certified 店排最前:同 pattern 若在 ref2(37+舊萃取碼,Gain 有已知污染個案,
#  如 w17 分身 +0.48)也出現,以 verify/公證店的正確響應為準。ref2 其餘未知風險=誠實記錄、靠量取勝。
CLEAN_STORES = ("dedust_ref2v", "dedust_champ_disc",                       # 修復版重驗 (擋 ref2 毒樣本,如 b20)
                "dedust_verify_interp", "dedust_verify_disc2", "dedust_w17rep", "dedust_repeat",
                "dedust_repeat_218", "dedust_r7", "dedust_r8", "dedust_r9",
                "dedust_ref1", "dedust_occl", "dedust_ref2",
                # v4 追加（R11-R14 全批,2026-07-09;certified 優先序維持=前面的店先見先贏）
                "dedust_ref3", "dedust_probes", "dedust_wide", "dedust_crown", "dedust_family2",
                "dedust_bakeoff", "dedust_blocks", "dedust_ablate", "dedust_resize",
                "dedust_occl2", "dedust_tol",
                # v5 追加（R15-R19a,2026-07-10）:手術/低側族/王鄰域變異=新區域覆蓋。
                # ⚠ dedust_r19b 刻意不進——與 r19a 交錯分夾同分布,整夾保留當 R19 門檻 held-out
                #   （round-19 §1:vargen held-out wm 排序 ρ≥0.5 且 oob 顯著 → R20 GA 發車）。
                "dedust_r15ga", "dedust_r15inf", "dedust_r15v", "dedust_addmap",
                "dedust_r16b", "dedust_r17", "dedust_r18", "dedust_r19a",
                # v6 追加（2026-07-11）:r19b 考卷任務已卸（R20 起改前瞻性驗證）+ R20 gen1 真值
                "dedust_r19b", "dedust_r20g1a", "dedust_r20g1b", "dedust_r20g1c",
                # v7 追加（2026-07-11）:gen2 三夾 + vgen2 資料批
                "dedust_r20g2a", "dedust_r20g2b", "dedust_r20g2c", "dedust_vgen2a", "dedust_vgen2b",
                # v8 追加（2026-07-11）:gen3+g20n 公證+vgen3+R21 batch1
                "dedust_r20g3a", "dedust_r20g3b", "dedust_r20g3c", "dedust_r20n",
                "dedust_vgen3a", "dedust_vgen3b", "dedust_r21b1a", "dedust_r21b1b", "dedust_r21b1c",
                # v9 追加（2026-07-11）:R21 batch2
                "dedust_r21b2a", "dedust_r21b2b", "dedust_r21b2c",
                # v10 追加（2026-07-11）:R21 batch3
                "dedust_r21b3a", "dedust_r21b3b", "dedust_r21b3c",
                # v11 追加（2026-07-11）:R21 batch4（150/150 零 error 首例）
                "dedust_r21b4a", "dedust_r21b4b", "dedust_r21b4c",
                # v12 追加（2026-07-12）:R21 batch5（六夾切片）＋g1 填空批
                "dedust_r21b5a", "dedust_r21b5b", "dedust_r21b5c",
                "dedust_r21b5d", "dedust_r21b5e", "dedust_r21b5f",
                "dedust_r21g1a", "dedust_r21g1b",
                # v13 追加（2026-07-12）:R22 b1 六夾＋n1 公證＋g2a 填空（b/c 未完待後補）
                "dedust_r22b1a", "dedust_r22b1b", "dedust_r22b1c",
                "dedust_r22b1d", "dedust_r22b1e", "dedust_r22b1f",
                "dedust_r22n1", "dedust_r21g2a",
                # v14 追加（2026-07-12）:R22 b2 六夾＋n2 三公證＋g2b/c 填空
                "dedust_r22b2a", "dedust_r22b2b", "dedust_r22b2c",
                "dedust_r22b2d", "dedust_r22b2e", "dedust_r22b2f",
                "dedust_r22n2o", "dedust_r22n2h", "dedust_r22n2w",
                "dedust_r21g2b", "dedust_r21g2c")
#? ref2 殘餘風險: 已實錘假象觸發率 ~9% (11 抽 1),無 certified 對照的 ref2 條目可能還有 ~10 筆髒 Gain——
#  佔訓練集 <0.3%,MSE 回歸可容忍;隨後續重驗逐步被 certified 店覆蓋。store 不存在時自動略過。
OUT_PTH = "sm_reanchor.pth"                                  # DATASET_PATH 下（--out 可換版本名）

_cfg = load_config(DEFAULT_CFG)
LABELS = PORT_SPECS[_cfg.port]["labels"]

from antenna.pattern import AntennaPattern
AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))   # train_by_datas 的 size_converter 需要（同 train.py）
setup_responses(_cfg)                                 # AntennaResponse spec 安裝（size_converter 靠它讀形狀）


def _load_clean():
    """r7+r8+r9 乾淨區真值 → 去重（pattern bytes）→ hash 排序 → 每第 5 筆 held-out。決定性。"""
    seen = {}
    for name in CLEAN_STORES:
        if not DATASET_PATH.joinpath(name).is_dir():
            continue                                     # 尚未跑出來的 store (如 champ_disc) 自動略過
        store = SampleStore(DATASET_PATH.joinpath(name), verbose=False)
        for i in range(len(store)):
            x, y = store[i]
            key = np.asarray(x).tobytes()
            if key not in seen:
                seen[key] = (torch.as_tensor(x, dtype=torch.float32),
                             torch.as_tensor(y, dtype=torch.float32))
    items = [seen[k] for k in sorted(seen)]      # bytes 排序＝決定性切分
    tr = [it for j, it in enumerate(items) if j % 5 != 0]
    ho = [it for j, it in enumerate(items) if j % 5 == 0]
    return tr, ho


def _load_harvest(n_replay: int, n_val: int):
    store = SampleStore(DATASET_PATH.joinpath("harvest_single"), verbose=False)
    idx = np.random.default_rng(0).choice(len(store), size=n_replay + n_val, replace=False)
    grab = lambda ii: [(torch.as_tensor(store[i][0], dtype=torch.float32),
                        torch.as_tensor(store[i][1], dtype=torch.float32)) for i in ii]
    return grab(idx[:n_replay]), grab(idx[n_replay:])


def _tds(items):
    return TensorDataset(torch.stack([x for x, _ in items]), torch.stack([y for _, y in items]))


def _make_sm():
    cache = os.path.join(REPO, "tmp", "sm_reanchor")
    os.makedirs(cache, exist_ok=True)
    n_pts = sum(_cfg.targets[LABELS[0]]["width"])
    return SURROGATES["mlp"](cache, 25 * 25, (len(LABELS), n_pts))


def _wm_errs(sm, items):
    errs = []
    sm.model.eval()
    with torch.no_grad():
        for x, y in items:
            pred = sm.model(x.flatten())
            w_pred, _ = worst_margin(pred, LABELS, _cfg.targets)
            w_true, _ = worst_margin(y, LABELS, _cfg.targets)
            errs.append(abs(float(w_pred) - float(w_true)))
    return np.asarray(errs)


def train(args):
    tr, ho = _load_clean()
    replay, _ = _load_harvest(args.replay, args.val)
    print(f"乾淨真值 {len(tr) + len(ho)} 筆（train {len(tr)} / held-out {len(ho)}）＋ harvest 重放 {len(replay)}")
    sm = _make_sm()
    sm.pre_load_model(DATASET_PATH.joinpath("sm_harvest.pth"), strict=True)
    ds = ConcatDataset([_tds(tr)] * args.over + [_tds(replay)])
    print(f"訓練集 {len(ds)} 筆（乾淨 ×{args.over} 過採樣 + 重放）,epochs={args.epochs}, batch={args.batch}")
    losses = sm.train_by_datas(ds, epochs=args.epochs, batch_size=args.batch, verbose=True)
    out = DATASET_PATH.joinpath(args.out)
    sm.save_as(out)
    print(f"loss: 首 {losses[0]:.3f} → 末 {losses[-1]:.3f}；權重 → {out}")


def evaluate(args):
    tr, ho = _load_clean()
    _, hval = _load_harvest(args.replay, args.val)
    print(f"| 模型 | 乾淨 held-out ({len(ho)}) 中位/p90 | 乾淨 train ({len(tr)}) 中位 | harvest 驗證 ({len(hval)}) 中位 |")
    print("|---|---|---|---|")
    for tag, pth in (("重錨前 sm_harvest", "sm_harvest.pth"), ("v1 sm_reanchor", "sm_reanchor.pth"),
                     (f"本版 {args.out}", args.out)):
        if tag.startswith("本版") and args.out == "sm_reanchor.pth":
            continue                                   # --out 沒換名 → v1 那列已涵蓋
        f = DATASET_PATH.joinpath(pth)
        if not f.exists():
            print(f"| {tag} | （{pth} 不存在,跳過） | | |")
            continue
        sm = _make_sm()
        sm.pre_load_model(f, strict=True)
        e_ho, e_tr, e_hv = _wm_errs(sm, ho), _wm_errs(sm, tr), _wm_errs(sm, hval)
        print(f"| {tag} | **{np.median(e_ho):.2f}** / {np.percentile(e_ho, 90):.2f} "
              f"| {np.median(e_tr):.2f} | {np.median(e_hv):.2f} |")
    print("\n判準：held-out 中位 ≤ ~2 → 導航儀合格；harvest 欄惡化過多＝遺忘（調 --over/--replay 重訓）。")


def tune(args):
    """小網格超參搜尋（零 HFSS,開發機）：資料預載一次 → 每組合從 sm_harvest 重訓 → held-out 中位選最佳。
    ⚠ 誠實條款：以 held-out 選模型=輕度選擇性過擬合(94 點),數字比單次訓練樂觀一點;p90 同列供對照。"""
    import itertools
    torch.manual_seed(0)
    tr, ho = _load_clean()
    max_replay = max(args.grid_replay)
    replay_all, hval = _load_harvest(max_replay, args.val)
    print(f"資料預載完成: 乾淨 train {len(tr)} / held-out {len(ho)} / 重放池 {len(replay_all)}")
    rows = []
    best = (None, 1e9)
    for epochs, over, replay, batch in itertools.product(args.grid_epochs, args.grid_over,
                                                         args.grid_replay, args.grid_batch):
        torch.manual_seed(0)                     # 每組合同 seed → 差異來自超參,非亂數
        sm = _make_sm()
        sm.pre_load_model(DATASET_PATH.joinpath("sm_harvest.pth"), strict=True)
        ds = ConcatDataset([_tds(tr)] * over + [_tds(replay_all[:replay])])
        sm.train_by_datas(ds, epochs=epochs, batch_size=batch, verbose=False)
        e_ho, e_hv = _wm_errs(sm, ho), _wm_errs(sm, hval)
        med, p90, hv = float(np.median(e_ho)), float(np.percentile(e_ho, 90)), float(np.median(e_hv))
        rows.append((med, p90, hv, epochs, over, replay, batch))
        print(f"  ep={epochs:<3} over={over:<2} replay={replay:<4} batch={batch:<3} → "
              f"held-out {med:.2f}/{p90:.2f}  harvest {hv:.2f}", flush=True)
        if med < best[1]:
            torch.manual_seed(0)
            best = ((epochs, over, replay, batch), med)
            sm.save_as(DATASET_PATH.joinpath(args.out))
    rows.sort()
    print("\n| held-out 中位 | p90 | harvest | epochs | over | replay | batch |")
    print("|---|---|---|---|---|---|---|")
    for med, p90, hv, e, o, rp, b in rows:
        print(f"| {med:.2f} | {p90:.2f} | {hv:.2f} | {e} | {o} | {rp} | {b} |")
    print(f"\n最佳 {best[0]} → 已存 {args.out}（⚠ 依判準複核 harvest 欄再採用）")


# ---------------------------------------------------------------- rad 頭（K=16 cosine,2026-07-12）
RAD_K = 16


def _rad_dataset():
    """全史 (pattern, phi0/phi90) 配對:掃有 rad/ 的 store,id 對回 *_input 的 .pt。
    回 X(n,625)、C(n,2,91)=|θ|≤90 子網格曲線、M(n)=真 rad_margin、keys=pattern bytes、θ 子網格。"""
    import json
    from script.dedust import rad_window_margin
    X, C, M, keys = [], [], [], []
    theta_sub = None
    for fol in sorted(os.listdir(str(DATASET_PATH))):
        d = DATASET_PATH.joinpath(fol)
        rd = d.joinpath("rad")
        ind = DATASET_PATH.joinpath(fol + "_input")
        if fol.endswith("_input") or not fol.startswith("dedust_") or not rd.is_dir() \
                or not ind.joinpath("manifest.json").exists():
            continue
        for m in json.load(open(str(ind.joinpath("manifest.json")), encoding="utf-8")):
            rf, pf = rd.joinpath(m["id"] + ".pt"), ind.joinpath(m["id"] + ".pt")
            if not (rf.exists() and pf.exists()):
                continue
            r = torch.load(str(rf), weights_only=True)
            if r.get("phi0") is None or r.get("phi90") is None:
                continue
            th = np.asarray(r["theta"], float).reshape(-1)
            sub = np.abs(th) <= 90
            if theta_sub is None:
                theta_sub = th[sub]
            c0 = np.asarray(r["phi0"], float).reshape(-1)
            c9 = np.asarray(r["phi90"], float).reshape(-1)
            pat = np.asarray(torch.load(str(pf), weights_only=True)).reshape(-1) > 0.5
            X.append(pat.astype(np.float32))
            C.append(np.stack([c0[sub], c9[sub]]).astype(np.float32))
            M.append(min(rad_window_margin(th, c0), rad_window_margin(th, c9)))
            keys.append(pat.tobytes())
    return np.stack(X), np.stack(C), np.asarray(M), keys, theta_sub


def train_rad(args):
    """rad 頭:625 → 2×K cosine 係數（|θ|≤90 半圓）→ 雙切面曲線;±45° 窗 2× 加權。
    K=16 依 rad-repr 表達力分析（Ricky 2026-07-12「感覺可以補 K=16」);資料=R7 起順收的全史方向圖。
    **判準（發車前寫死）**:held-out（pattern-hash 切分,防公證重複洩漏）rad_margin 排序 ρ≥0.4
    → pred_rad 進 pred_sel 罰項;否則只隨 manifest 記 pred_rad 供前瞻,不進選批鍵。"""
    import hashlib
    import torch.nn as nn
    from scipy.stats import spearmanr
    from script.dedust import rad_window_margin as rwm
    X, C, M, keys, th = _rad_dataset()
    print(f"rad 資料 {len(X)} 筆;θ 子網格 {len(th)} 點;真 margin 中位 {np.median(M):+.2f}")
    phi = np.pi * (th - th.min()) / (th.max() - th.min())
    B = torch.tensor(np.cos(np.arange(RAD_K).reshape(-1, 1) * phi.reshape(1, -1)), dtype=torch.float32)
    side = np.array([int(hashlib.md5(k).hexdigest(), 16) % 7 == 0 for k in keys])
    w = torch.tensor(np.where(np.abs(th) <= 45, 2.0, 1.0), dtype=torch.float32)
    Xt, Ct = torch.tensor(X[~side]), torch.tensor(C[~side])
    Xh, Mh = torch.tensor(X[side]), M[side]
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(625, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                        nn.Linear(256, 2 * RAD_K))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(Xt)
    for ep in range(args.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            fit = net(Xt[idx]).reshape(-1, 2, RAD_K) @ B
            loss = (((fit - Ct[idx]) ** 2) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if ep == 0 or ep % 5 == 4:
            print(f"  ep {ep + 1}/{args.epochs} loss {tot / n:.4f}", flush=True)
    with torch.no_grad():
        fit = (net(Xh).reshape(-1, 2, RAD_K) @ B).numpy()
    pm = np.array([min(rwm(th, fit[i][0]), rwm(th, fit[i][1])) for i in range(len(fit))])
    rho, p = spearmanr(pm, Mh)
    mae = float(np.abs(pm - Mh).mean())
    print(f"held-out {int(side.sum())} 筆:rad_margin 排序 ρ={rho:+.3f} (p={p:.1e}) / MAE {mae:.3f} dB")
    torch.save(dict(state=net.state_dict(), K=RAD_K, theta=np.asarray(th)),
               str(DATASET_PATH.joinpath(args.out)))
    print(f"→ {args.out}（判準:ρ≥0.4 才進 pred_sel 選批鍵）")


def main():
    ap = argparse.ArgumentParser(description="SM 乾淨區重錨（R10 Stage A;train 開發機零 HFSS）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", train), ("eval", evaluate), ("tune", tune)):
        s = sub.add_parser(name)
        s.add_argument("--epochs", type=int, default=40)
        s.add_argument("--batch", type=int, default=64)
        s.add_argument("--over", type=int, default=8, help="乾淨 train 過採樣倍數 (預設 8 ≈ 與重放等量)")
        s.add_argument("--replay", type=int, default=2000, help="harvest 重放筆數")
        s.add_argument("--val", type=int, default=500, help="harvest 驗證筆數 (不進訓練)")
        s.add_argument("--out", default="sm_reanchor.pth", help="輸出權重名 (DATASET_PATH 下;v2 建議 sm_reanchor2.pth)")
        s.add_argument("--grid-epochs", type=int, nargs="+", default=[40, 80])
        s.add_argument("--grid-over", type=int, nargs="+", default=[4, 8, 16])
        s.add_argument("--grid-replay", type=int, nargs="+", default=[1000, 2000])
        s.add_argument("--grid-batch", type=int, nargs="+", default=[64])
        s.set_defaults(fn=fn)
    s = sub.add_parser("train-rad", help="rad 頭:pattern→K=16 cosine 雙切面(±45 窗加權);held-out ρ≥0.4 才進 pred_sel")
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--out", default="rad_head1.pth")
    s.set_defaults(fn=train_rad)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
