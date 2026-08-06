# -*- coding: utf-8 -*-
"""Pattern Browser 資料索引產線(契約=SPEC.md)。

掃 NAS dataset 一次,產出:
  data/patterns.npz  : packed uint8 [N,79](np.packbits(625 bits)) + ids(同序)
  data/meta.json     : list[N],欄位見 SPEC.md「資料契約」

用法(repo 根,ant env):
  python -m application.pattern_browser.build_index --cache-dir <scratchpad>
    --cache-dir   : 讀既有 res_index.json / pt_index.json 當起點,只補掃比快取新的
                    results.json / 輸入夾(mtime 判定,留 1 天安全邊)。
    --full-rescan : 忽略快取新鮮度,全店重掃 results.json 與輸入夾(仍沿用快取當底)。
    --rebuild     : 忽略既有 patterns.npz,全部 .pt 重讀(預設=增量,只讀新 id)。
    --ds / --out  : 覆寫 NAS 根 / 輸出夾(預設見下)。

重跑=增量刷新:bits 從舊 npz 沿用、只讀新 .pt;指標/特徵每次重算(快)。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from antenna.patch.patch_simulator.single_port import diag_bridge_sites  # noqa: E402

DS_DEFAULT = r"T:\碩二_鄒穎麒's\antenna\dataset"
OUT_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAFETY_S = 86400  # mtime 增量判定的安全邊(1 天)
N8_STRUCT = np.ones((3, 3), dtype=int)


def log(*a):
    print(*a, flush=True)


def list_dirs(ds):
    return sorted(d for d in os.listdir(ds) if os.path.isdir(os.path.join(ds, d)))


def load_cache(cache_dir):
    """回傳 (res_index, pt_index, res_mtime, pt_mtime);無快取=({},{},0,0)。"""
    res_index, pt_index, res_mt, pt_mt = {}, {}, 0.0, 0.0
    if not cache_dir:
        return res_index, pt_index, res_mt, pt_mt
    rp = os.path.join(cache_dir, "res_index.json")
    pp = os.path.join(cache_dir, "pt_index.json")
    if os.path.exists(rp):
        res_index = json.load(open(rp, encoding="utf-8"))
        res_mt = os.path.getmtime(rp)
        log(f"cache res_index: {len(res_index)} ids")
    if os.path.exists(pp):
        pt_index = json.load(open(pp, encoding="utf-8"))
        pt_mt = os.path.getmtime(pp)
        log(f"cache pt_index: {len(pt_index)} ids")
    return res_index, pt_index, res_mt, pt_mt


def scan_inputs(ds, dirs, pt_index, since):
    """掃輸入夾(*_input)補 pt_index(id → 夾名,先到先贏)。since>0 時只掃 mtime 較新的夾。"""
    n_scan = 0
    for d in dirs:
        if not d.endswith("_input"):
            continue
        p = os.path.join(ds, d)
        if since and os.path.getmtime(p) < since - SAFETY_S:
            continue
        n_scan += 1
        try:
            for f in os.listdir(p):
                if f.endswith(".pt"):
                    pt_index.setdefault(f[:-3], d)
        except OSError as e:
            log(f"  ERR list {d}: {e}")
    log(f"scan inputs: {n_scan} dirs -> pt_index {len(pt_index)} ids")


def scan_stores(ds, dirs, res_index, since):
    """掃結果店 results.json 補 res_index(id → [store, metrics],先到先贏)。"""
    n_scan = 0
    for d in dirs:
        if d.endswith("_input"):
            continue
        rj = os.path.join(ds, d, "results.json")
        try:
            if not os.path.exists(rj):
                continue
            if since and os.path.getmtime(rj) < since - SAFETY_S:
                continue
            n_scan += 1
            r = json.load(open(rj, encoding="utf-8"))
        except (OSError, ValueError) as e:
            log(f"  ERR read {d}: {e}")
            continue
        for k, v in r.items():
            res_index.setdefault(k, [d, v])
    log(f"scan stores: {n_scan} results.json -> res_index {len(res_index)} ids")


def load_existing_npz(out_dir):
    """舊 npz → {id: packed_row}(增量刷新的 bits 來源)。"""
    p = os.path.join(out_dir, "patterns.npz")
    if not os.path.exists(p):
        return {}
    try:
        z = np.load(p, allow_pickle=False)
        ids, packed = z["ids"], z["packed"]
        log(f"existing npz: {len(ids)} ids (bits 沿用)")
        return {str(i): packed[k] for k, i in enumerate(ids)}
    except Exception as e:
        log(f"  WARN 舊 npz 讀取失敗,全部重讀: {e}")
        return {}


def save_npz(out_dir, ids, packed_rows):
    """寫 patterns.npz(先寫 tmp 再 os.replace,中斷不留半檔)。"""
    packed = np.stack(packed_rows).astype(np.uint8) if packed_rows else np.zeros((0, 79), np.uint8)
    p = os.path.join(out_dir, "patterns.npz")
    np.savez_compressed(p + ".tmp.npz", packed=packed, ids=np.array(ids))
    os.replace(p + ".tmp.npz", p)
    return packed


def read_pt_bits(ds, folder, pid):
    """讀一筆輸入 .pt → (625,) uint8 bits。失敗丟例外由呼叫端記帳。"""
    import torch
    x = torch.load(os.path.join(ds, folder, pid + ".pt"), weights_only=True)
    mat = np.asarray(x, dtype=np.float32).reshape(25, 25) > 0.5
    return mat.astype(np.uint8).reshape(-1)


def pattern_features(bits):
    """bits(625,) → (total, n4, n8, largest8_frac, ndiag)。契約:n8=ones(3,3);ndiag=diag_bridge_sites(mat,0.10,0.2) 的 len(sites)。"""
    mat = bits.reshape(25, 25).astype(bool)
    total = int(mat.sum())
    if total == 0:
        return 0, 0, 0, 0.0, 0
    _, n4 = ndimage.label(mat)
    lab8, n8 = ndimage.label(mat, structure=N8_STRUCT)
    largest8 = int(np.bincount(lab8.reshape(-1))[1:].max()) if n8 else 0
    sites, _skipped = diag_bridge_sites(mat, 0.10, 0.2)
    return total, int(n4), int(n8), round(largest8 / total, 4), len(sites)


def metric_fields(entry):
    """res_index 一筆 [store, metrics] → (store, wm, rad, lo, sel);缺=None。契約:wm=metrics['wm'][2]。"""
    if not entry:
        return None, None, None, None, None
    store, m = entry
    if not isinstance(m, dict):
        return store, None, None, None, None
    w = m.get("wm")
    wm = float(w[2]) if isinstance(w, (list, tuple)) and len(w) >= 3 else None
    rad = m.get("rad_margin")
    lo = m.get("oob_gain_max_lo")
    sel = m.get("sel")
    return (store, wm,
            float(rad) if rad is not None else None,
            float(lo) if lo is not None else None,
            float(sel) if sel is not None else None)


def variant_wm(res_index, suffix):
    """{parent: wm} — 掃 res_index 中 `~db100`/`~sl100` 變體的 wm 收進親本欄。"""
    out = {}
    for k, v in res_index.items():
        if k.endswith(suffix):
            _, wm, _, _, _ = metric_fields(v)
            if wm is not None:
                out[k[: -len(suffix)]] = wm
    return out


def main():
    ap = argparse.ArgumentParser(description="Pattern Browser 資料索引產線(契約=SPEC.md)")
    ap.add_argument("--ds", default=DS_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--full-rescan", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    dirs = list_dirs(args.ds)
    log(f"NAS dirs: {len(dirs)}")

    res_index, pt_index, res_mt, pt_mt = load_cache(args.cache_dir)
    if args.full_rescan:
        res_mt = pt_mt = 0.0
    scan_inputs(args.ds, dirs, pt_index, pt_mt)
    scan_stores(args.ds, dirs, res_index, res_mt)

    main_ids = sorted(i for i in pt_index if "~" not in i)
    log(f"main ids(不含 ~ 變體): {len(main_ids)}")

    old_bits = {} if args.rebuild else load_existing_npz(args.out)
    db100 = variant_wm(res_index, "~db100")
    sl100 = variant_wm(res_index, "~sl100")
    log(f"變體結果: db100={len(db100)} sl100={len(sl100)}")

    ids, packed_rows, meta = [], [], []
    n_reuse = n_read = n_fail = last_ckpt = 0
    fails = []
    t_read = time.time()
    for i, pid in enumerate(main_ids):
        row = old_bits.get(pid)
        if row is not None:
            bits = np.unpackbits(row)[:625]
            n_reuse += 1
        else:
            try:
                bits = read_pt_bits(args.ds, pt_index[pid], pid)
                row = np.packbits(bits)
                n_read += 1
            except Exception as e:
                n_fail += 1
                if len(fails) < 20:
                    fails.append(f"{pid} ({pt_index[pid]}): {e}")
                continue
        total, n4, n8, l8f, ndiag = pattern_features(bits)
        store, wm, rad, lo, sel = metric_fields(res_index.get(pid))
        ids.append(pid)
        packed_rows.append(row)
        meta.append({
            "id": pid, "folder": pt_index[pid], "store": store,
            "wm": wm, "rad": rad, "lo": lo, "sel": sel,
            "total": total, "n4": n4, "n8": n8, "largest8_frac": l8f, "ndiag": ndiag,
            "has_db100": pid in db100, "has_sl100": pid in sl100,
            "db100_wm": db100.get(pid), "sl100_wm": sl100.get(pid),
            "kind": None,  # v1 不掃 manifest
        })
        if n_read - last_ckpt >= 4000:  # 期中 checkpoint:中斷後重跑可沿用已讀 bits
            save_npz(args.out, ids, packed_rows)
            last_ckpt = n_read
            log(f"  checkpoint npz @ read={n_read}")
        if (i + 1) % 2000 == 0:
            dt = time.time() - t_read
            rate = n_read / dt if dt > 0 and n_read else 0
            eta = (len(main_ids) - i - 1) / rate / 60 if rate else 0
            log(f"  {i+1}/{len(main_ids)}  read={n_read} reuse={n_reuse} fail={n_fail}"
                f"  {dt:.0f}s  eta~{eta:.1f}min")

    packed = save_npz(args.out, ids, packed_rows)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, separators=(",", ":"))

    n_res_only = sum(1 for k in res_index if "~" not in k and k not in pt_index)
    n_qual = sum(1 for r in meta if r["wm"] is not None and r["wm"] > 0
                 and r["rad"] is not None and r["rad"] > 0)
    n_wm = sum(1 for r in meta if r["wm"] is not None)
    log(f"done {time.time()-t0:.0f}s  N={len(ids)}  packed={packed.shape}"
        f"  read={n_read} reuse={n_reuse} fail={n_fail}")
    log(f"  有指標(wm)={n_wm}  合格(wm>0∧rad>0)={n_qual}"
        f"  有結果無輸入.pt(不進列表)={n_res_only}")
    if fails:
        log("  失敗樣本(前 20):")
        for s in fails:
            log("   ", s)


if __name__ == "__main__":
    main()
