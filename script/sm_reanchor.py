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
CLEAN_STORES = ("dedust_r7", "dedust_r8", "dedust_r9")      # repeat 店不收（同 pattern 重複）
OUT_PTH = "sm_reanchor.pth"                                  # DATASET_PATH 下（NAS,跨機可用）

_cfg = load_config(DEFAULT_CFG)
LABELS = PORT_SPECS[_cfg.port]["labels"]

from antenna.pattern import AntennaPattern
AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))   # train_by_datas 的 size_converter 需要（同 train.py）
setup_responses(_cfg)                                 # AntennaResponse spec 安裝（size_converter 靠它讀形狀）


def _load_clean():
    """r7+r8+r9 乾淨區真值 → 去重（pattern bytes）→ hash 排序 → 每第 5 筆 held-out。決定性。"""
    seen = {}
    for name in CLEAN_STORES:
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
    out = DATASET_PATH.joinpath(OUT_PTH)
    sm.save_as(out)
    print(f"loss: 首 {losses[0]:.3f} → 末 {losses[-1]:.3f}；權重 → {out}")


def evaluate(args):
    tr, ho = _load_clean()
    _, hval = _load_harvest(args.replay, args.val)
    print(f"| 模型 | 乾淨 held-out ({len(ho)}) 中位/p90 | 乾淨 train ({len(tr)}) 中位 | harvest 驗證 ({len(hval)}) 中位 |")
    print("|---|---|---|---|")
    for tag, pth in (("重錨前 sm_harvest", "sm_harvest.pth"), ("重錨後 sm_reanchor", OUT_PTH)):
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


def main():
    ap = argparse.ArgumentParser(description="SM 乾淨區重錨（R10 Stage A;train 開發機零 HFSS）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", train), ("eval", evaluate)):
        s = sub.add_parser(name)
        s.add_argument("--epochs", type=int, default=40)
        s.add_argument("--batch", type=int, default=64)
        s.add_argument("--over", type=int, default=8, help="乾淨 train 過採樣倍數 (預設 8 ≈ 與重放等量)")
        s.add_argument("--replay", type=int, default=2000, help="harvest 重放筆數")
        s.add_argument("--val", type=int, default=500, help="harvest 驗證筆數 (不進訓練)")
        s.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
