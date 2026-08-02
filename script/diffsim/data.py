# -*- coding: utf-8 -*-
"""script/diffsim/data.py — diffsim 資料層：掃 NAS 建索引 + 決定性分割。

**只讀 NAS**（`DATASET_PATH` 下的 SampleStore `.pt`），索引快取落本機 scratch/tmp。

分割鐵則（`docs/diffsim.md` §5，發車前寫死）：
  - 驗證集 與 擬合核/仿射校準/殘差頭 用的樣本 **必須不相交**。
  - 凍結尺 `dedust_r50b1b_frozen`（30 筆 OOD）**只做最終報數，永不進擬合**。
分割靠「內容 hash → [0,1) 決定性亂數」，同一筆樣本永遠落同一側（重跑/擴充索引都不會漂）。

用法：
    python -m script.diffsim.data build          # 掃 NAS 建索引（~66k 筆，多執行緒；有快取就跳過）
    python -m script.diffsim.data summary        # 看分層/分割統計
"""
import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from antenna.utils import config as _config, DATASET_PATH   # noqa: E402
_config.device = "cpu"
import torch                                                # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(REPO, "tmp", "diffsim")
INDEX_NPZ = os.path.join(CACHE_DIR, "index.npz")

FROZEN_STORE = "dedust_r50b1b_frozen"     # OOD 凍結尺（30 筆）——最終報數專用
SENIOR_STORE = "harvest_single"           # 學長收割（24k）
NEG_LIST = os.path.join(REPO, "configs", "neg_stores.txt")
EXCLUDE = {"harvest_dual",                # 雙埠：幾何不同，不在本鏈範圍
           "harvest_single_random",       # 空夾
           "jobs_state"}                  # 佇列狀態，非樣本


def _neg_stores() -> set:
    out = set()
    with open(NEG_LIST, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                out.add(line)
    return out


def _list_stores() -> list:
    """所有單埠樣本店（NAS 唯讀）。"""
    root = str(DATASET_PATH)
    out = []
    for name in sorted(os.listdir(root)):
        if name in EXCLUDE or name.endswith("_input"):
            continue
        if not os.path.isdir(os.path.join(root, name)):
            continue
        out.append(name)
    return out


def _stratum(store: str, neg: set) -> str:
    """分層標籤：驗收要分層抽樣（`docs/diffsim.md` §5），負片域單獨報。"""
    if store == FROZEN_STORE:
        return "frozen"
    if store == SENIOR_STORE:
        return "senior"
    if store in neg:
        return "neg"
    return "clean"


def _load_store(args) -> tuple:
    store, path = args
    xs, ys, names = [], [], []
    for fn in sorted(os.listdir(path)):
        if not fn.endswith(".pt"):
            continue
        try:
            x, y = torch.load(os.path.join(path, fn), weights_only=True)
        except Exception:
            continue                                   # 半截壞檔只損一筆
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if x.size != 625 or y.size != 34:
            continue
        xs.append((x > 0.5).astype(np.uint8))
        ys.append(y)
        names.append(fn[:-3])
    return store, xs, ys, names


def build(force: bool = False, workers: int = 24) -> dict:
    """掃 NAS 全部單埠店 → 本機索引快取。回傳 dict of arrays。"""
    if os.path.exists(INDEX_NPZ) and not force:
        return load()
    os.makedirs(CACHE_DIR, exist_ok=True)
    neg = _neg_stores()
    stores = _list_stores()
    jobs = [(s, os.path.join(str(DATASET_PATH), s)) for s in stores]
    X, Y, S, N = [], [], [], []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for store, xs, ys, names in ex.map(_load_store, jobs):
            done += 1
            if xs:
                X.extend(xs)
                Y.extend(ys)
                S.extend([store] * len(xs))
                N.extend(names)
            print(f"  [{done}/{len(jobs)}] {store}: {len(xs)}", flush=True)
    x = np.asarray(X, dtype=np.uint8)
    y = np.asarray(Y, dtype=np.float32)
    store_arr = np.asarray(S)
    name_arr = np.asarray(N)
    strat = np.asarray([_stratum(s, neg) for s in store_arr])
    np.savez_compressed(INDEX_NPZ, x=x, y=y, store=store_arr, name=name_arr, stratum=strat)
    print(f"索引落地 {INDEX_NPZ}：{len(x)} 筆")
    return load()


def load() -> dict:
    if not os.path.exists(INDEX_NPZ):
        raise SystemExit(f"找不到 {INDEX_NPZ}——先跑 python -m script.diffsim.data build")
    z = np.load(INDEX_NPZ, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- 決定性分割
def hash_u01(x_row: np.ndarray) -> float:
    """內容 → [0,1) 決定性亂數（分割用；同一筆樣本永遠同值）。"""
    h = hashlib.sha1(np.ascontiguousarray(x_row, dtype=np.uint8).tobytes()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def assign_split(idx: dict, val_per_stratum: int = 30, dev_per_stratum: int = 40,
                 seed_tag: str = "diffsim-v1"):
    """回傳 (split (N,) of 'val'/'dev'/'fit', u01 (N,))。

    三層（Goodhart 護欄——**val 只在 gate 報數時看一次**，迭代一律看 dev）：
      - `val`：驗收集。`frozen` 全進 val 且永不外流；其餘 stratum 取 hash-u01 最小的
        `val_per_stratum` 筆。
      - `dev`：迭代診斷用（下一段 `dev_per_stratum` 筆）——調參對著它，不對著 val。
      - `fit`：其餘全部，供擬合核/仿射校準/殘差頭。
    切分是內容 hash 的函數 → 同一個 pattern 永遠落同一側，重建索引也不漂。
    """
    x = idx["x"]
    strat = idx["stratum"]
    tag = seed_tag.encode()
    u = np.empty(len(x), dtype=np.float64)
    for i in range(len(x)):
        h = hashlib.sha1(tag + np.ascontiguousarray(x[i]).tobytes()).digest()
        u[i] = int.from_bytes(h[:8], "big") / float(1 << 64)
    split = np.full(len(x), "fit", dtype=object)
    for s in np.unique(strat):
        m = np.where(strat == s)[0]
        if s == "frozen":
            split[m] = "val"
            continue
        order = m[np.argsort(u[m])]
        split[order[:val_per_stratum]] = "val"
        split[order[val_per_stratum:val_per_stratum + dev_per_stratum]] = "dev"
    return np.asarray(split), u


def summary():
    idx = load()
    split, _ = assign_split(idx)
    from .eval import margins
    wm, _, _ = margins(idx["y"])
    print(f"總樣本 {len(idx['x'])}")
    print("| stratum | n | val | dev | fit | wm 中位 | wm≥0 |")
    print("|---|---|---|---|---|---|---|")
    for s in ["clean", "neg", "senior", "frozen"]:
        m = idx["stratum"] == s
        if not m.any():
            continue
        print(f"| {s} | {m.sum()} | {(m & (split == 'val')).sum()} | {(m & (split == 'dev')).sum()} "
              f"| {(m & (split == 'fit')).sum()} | {np.median(wm[m]):+.2f} | {(wm[m] >= 0).sum()} |")


def main():
    ap = argparse.ArgumentParser(description="diffsim 資料索引")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--force", action="store_true")
    b.add_argument("--workers", type=int, default=24)
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "build":
        build(force=a.force, workers=a.workers)
        summary()
    else:
        summary()


if __name__ == "__main__":
    main()
