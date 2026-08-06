# -*- coding: utf-8 -*-
"""Pattern Browser 資料索引產線(契約=SPEC.md)。

掃 NAS dataset 一次,產出:
  data/patterns.npz     : packed uint8 [N,79](np.packbits(625 bits)) + ids(同序)
  data/meta.json        : list[N],欄位見 SPEC.md「資料契約」(v2 加 has_resp/has_rad)
  data/resp.npz         : resp float16 [N,2,17](S11,Gain;缺=NaN)+ has_resp bool[N] + ids
  data/rad.npz          : theta f16[181] + phi0/phi90 f16[N,181](缺=NaN)+ has_rad bool[N] + ids
  data/variant_resp.json: {變體id: {s11,gain,phi0,phi90}}(~db100/~sl100;缺=null)
  data/curve_state.json : {store: 讀取當下 mtime}=曲線層增量真相源(內部用,非 SPEC 契約)

用法(repo 根,ant env):
  python -m application.pattern_browser.build_index --cache-dir <scratchpad>
    --cache-dir   : 讀既有 res_index.json / pt_index.json 當起點,只補掃比快取新的
                    results.json / 輸入夾(mtime 判定,留 1 天安全邊)。
    --full-rescan : 忽略快取新鮮度,全店重掃 results.json 與輸入夾(仍沿用快取當底)。
    --rebuild     : 忽略既有 patterns.npz 與曲線 npz,全部重讀(預設=增量)。
    --skip-curves : 跳過曲線層 NAS 重讀(舊 resp/rad/variant 對齊新 ids 重寫,不讀店)。
    --ds / --out  : 覆寫 NAS 根 / 輸出夾(預設見下)。

重跑=增量刷新:bits 從舊 npz 沿用、只讀新 .pt;指標/特徵每次重算(快)。
曲線層增量:curve_state 沒讀過、或店的 results.json / rad 夾 mtime 比上次讀時新、
或該店有新 id/新變體才重讀;每 50 店 checkpoint(tmp+replace,中斷可續)。
resp 對映=店內樣本 pattern packbits bytes → id。
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
VARIANT_SUFFIXES = ("~db100", "~sl100")  # 進 variant_resp.json 的變體(SPEC v2)


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


# ---------------------------------------------------------------- v2 曲線層


def load_old_curves(out_dir):
    """舊 resp.npz / rad.npz / variant_resp.json / curve_state.json →
    (resp_map, rad_map, theta, variants, state)。
    resp_map: {id: (has, row(2,17))};rad_map: {id: (has, phi0, phi90)};
    state: {store: 讀取當下的 mtime}=增量真相源(哪些店真的讀過;npz 內容不當證據,
    避免 --skip-curves 對齊重寫出的 NaN 列被誤認「已嘗試」)。"""
    rp = os.path.join(out_dir, "resp.npz")
    gp = os.path.join(out_dir, "rad.npz")
    vp = os.path.join(out_dir, "variant_resp.json")
    sp = os.path.join(out_dir, "curve_state.json")
    resp_map, rad_map, theta, variants, state = {}, {}, None, {}, {}
    try:
        if os.path.exists(rp) and os.path.exists(gp):
            z = np.load(rp, allow_pickle=False)
            for k, i in enumerate(z["ids"]):
                resp_map[str(i)] = (bool(z["has_resp"][k]), z["resp"][k])
            z = np.load(gp, allow_pickle=False)
            theta = z["theta"]
            for k, i in enumerate(z["ids"]):
                rad_map[str(i)] = (bool(z["has_rad"][k]), z["phi0"][k], z["phi90"][k])
            log(f"existing curves: resp {len(resp_map)} / rad {len(rad_map)} ids (沿用)")
    except Exception as e:
        log(f"  WARN 舊曲線檔讀取失敗,全店重讀: {e}")
        resp_map, rad_map, theta = {}, {}, None
    if os.path.exists(vp):
        try:
            variants = json.load(open(vp, encoding="utf-8"))
        except (OSError, ValueError):
            variants = {}
    if resp_map and os.path.exists(sp):
        try:
            state = json.load(open(sp, encoding="utf-8"))
            log(f"curve state: {len(state)} stores 已讀過")
        except (OSError, ValueError):
            state = {}
    return resp_map, rad_map, theta, variants, state


def store_mtime(ds, store):
    """店的增量指紋=max(results.json, rad 夾) mtime;讀不到=None(當作要重讀)。"""
    try:
        mt = os.path.getmtime(os.path.join(ds, store, "results.json"))
        rd = os.path.join(ds, store, "rad")
        if os.path.isdir(rd):
            mt = max(mt, os.path.getmtime(rd))
        return mt
    except OSError:
        return None


def read_store_samples(sp):
    """店根 hash .pt → ({pattern_packbits_bytes: resp(2,17) f32}, n_ok, n_bad)。
    同店重複 pattern(公證/重測)=任取先到者(setdefault)。"""
    import torch
    out, n_ok, n_bad = {}, 0, 0
    for f in sorted(os.listdir(sp)):
        if not f.endswith(".pt"):
            continue
        try:
            pat, resp = torch.load(os.path.join(sp, f), weights_only=True)
            mat = np.asarray(pat, dtype=np.float32).reshape(25, 25) > 0.5
            resp = np.asarray(resp, dtype=np.float32).reshape(2, 17)
            out.setdefault(np.packbits(mat.astype(np.uint8).reshape(-1)).tobytes(), resp)
            n_ok += 1
        except Exception:
            n_bad += 1
    return out, n_ok, n_bad


def read_rad_file(path):
    """store/rad/{id}.pt(dict theta/phi0/phi90 各 181)→ (theta, phi0, phi90) f32;壞/缺=None。"""
    import torch
    try:
        r = torch.load(path, weights_only=True)
        return (np.asarray(r["theta"], dtype=np.float32).reshape(181),
                np.asarray(r["phi0"], dtype=np.float32).reshape(181),
                np.asarray(r["phi90"], dtype=np.float32).reshape(181))
    except Exception:
        return None


def json_curve(arr):
    """float 陣列 → JSON 安全 list(round 4;非有限=None,避免 NaN 進 JSON)。"""
    return [round(float(x), 4) if np.isfinite(x) else None for x in arr]


def save_curves(out_dir, ids, resp, has_resp, theta, phi0, phi90, has_rad, variants, state):
    """寫 resp.npz / rad.npz / variant_resp.json / curve_state.json(tmp+os.replace,
    中斷不留半檔;state 與資料同時落地 → checkpoint 續跑一致)。"""
    ids_arr = np.array(ids)
    p = os.path.join(out_dir, "resp.npz")
    np.savez_compressed(p + ".tmp.npz", resp=resp, has_resp=has_resp, ids=ids_arr)
    os.replace(p + ".tmp.npz", p)
    p = os.path.join(out_dir, "rad.npz")
    th = theta if theta is not None else np.full(181, np.nan, np.float16)
    np.savez_compressed(p + ".tmp.npz", theta=np.asarray(th, np.float16),
                        phi0=phi0, phi90=phi90, has_rad=has_rad, ids=ids_arr)
    os.replace(p + ".tmp.npz", p)
    for name, obj in (("variant_resp.json", variants), ("curve_state.json", state)):
        p = os.path.join(out_dir, name)
        with open(p + ".tmp", "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(p + ".tmp", p)


def build_curves(ds, out_dir, ids, packed_rows, meta, res_index, pt_index, rebuild, skip):
    """v2 曲線層主流程。回傳 (has_resp, has_rad) bool[N](與 ids 同序,供 meta 打旗)。

    增量:先把舊 npz 依 id 對齊回填,再只重讀「curve_state 沒讀過、results.json/rad 夾
    mtime 比上次讀時新、或該店有新 id / 新變體」的店;skip=True 完全不讀店
    (僅對齊重寫,保持與 ids 同序契約;state 原樣保留)。"""
    t0 = time.time()
    n = len(ids)
    resp = np.full((n, 2, 17), np.nan, np.float16)
    has_resp = np.zeros(n, bool)
    phi0 = np.full((n, 181), np.nan, np.float16)
    phi90 = np.full((n, 181), np.nan, np.float16)
    has_rad = np.zeros(n, bool)

    resp_map, rad_map, theta, old_variants, state = load_old_curves(out_dir)
    if rebuild:
        state = {}

    store_pids = {}  # store -> [idx into ids]
    for k, r in enumerate(meta):
        if r["store"]:
            store_pids.setdefault(r["store"], []).append(k)
    var_store = {k: v[0] for k, v in res_index.items() if k.endswith(VARIANT_SUFFIXES)}
    store_vars = {}  # store -> [變體id]
    for v, s in var_store.items():
        store_vars.setdefault(s, []).append(v)

    # 舊資料依 id 對齊回填(skip / 未重讀的店沿用;重讀的店稍後覆寫)
    for k, pid in enumerate(ids):
        e = resp_map.get(pid)
        if e is not None:
            has_resp[k], resp[k] = e
        e = rad_map.get(pid)
        if e is not None:
            has_rad[k], phi0[k], phi90[k] = e

    stores = sorted(set(store_pids) | set(store_vars))
    read_set, mtimes = [], {}
    if not skip:
        for s in stores:
            mt = store_mtime(ds, s)
            mtimes[s] = mt
            if (mt is None or s not in state or mt > state[s] + 1
                    or any(ids[k] not in resp_map for k in store_pids.get(s, ()))
                    or any(v not in old_variants for v in store_vars.get(s, ()))):
                read_set.append(s)
    log(f"curves: {len(stores)} stores → 重讀 {len(read_set)} / 沿用 {len(stores) - len(read_set)}"
        + (" [skip-curves]" if skip else ""))

    read_s = set(read_set)
    variants = {v: old_variants[v] for v in var_store
                if v in old_variants and var_store[v] not in read_s}
    if not skip:  # 重讀店的舊 state 先移除,完成一店補一店(checkpoint 續跑一致)
        state = {s: m for s, m in state.items() if s not in read_s}

    n_samp = n_bad = n_unmatch = 0
    for si, s in enumerate(read_set):
        sp = os.path.join(ds, s)
        try:
            samples, n_ok, nb = read_store_samples(sp)
        except OSError as e:
            log(f"  ERR store {s}: {e}")
            continue
        n_samp += n_ok
        n_bad += nb
        rad_dir = os.path.join(sp, "rad")
        try:
            rad_files = set(os.listdir(rad_dir)) if os.path.isdir(rad_dir) else set()
        except OSError:
            rad_files = set()
        for k in store_pids.get(s, ()):
            r = samples.get(packed_rows[k].tobytes())
            has_resp[k] = r is not None
            resp[k] = r if r is not None else np.nan
            if r is None:
                n_unmatch += 1
            rr = (read_rad_file(os.path.join(rad_dir, ids[k] + ".pt"))
                  if ids[k] + ".pt" in rad_files else None)
            has_rad[k] = rr is not None
            if rr is not None:
                if theta is None:
                    theta = rr[0]
                phi0[k], phi90[k] = rr[1], rr[2]
            else:
                phi0[k] = phi90[k] = np.nan
        for v in store_vars.get(s, ()):
            ent = {"s11": None, "gain": None, "phi0": None, "phi90": None}
            fold = pt_index.get(v)
            if fold:
                try:
                    r = samples.get(np.packbits(read_pt_bits(ds, fold, v)).tobytes())
                    if r is not None:
                        ent["s11"], ent["gain"] = json_curve(r[0]), json_curve(r[1])
                except Exception:
                    pass
            rr = (read_rad_file(os.path.join(rad_dir, v + ".pt"))
                  if v + ".pt" in rad_files else None)
            if rr is not None:
                if theta is None:
                    theta = rr[0]
                ent["phi0"], ent["phi90"] = json_curve(rr[1]), json_curve(rr[2])
            variants[v] = ent
        if mtimes.get(s) is not None:
            state[s] = mtimes[s]  # 這一店完整讀完才記帳(mtime=讀前快照)
        if (si + 1) % 50 == 0:  # 期中 checkpoint:中斷後重跑沿用已讀店
            save_curves(out_dir, ids, resp, has_resp, theta, phi0, phi90, has_rad,
                        variants, state)
            log(f"  checkpoint curves @ store {si + 1}")
        if (si + 1) % 25 == 0:
            dt = time.time() - t0
            eta = dt / (si + 1) * (len(read_set) - si - 1) / 60
            log(f"  curves {si + 1}/{len(read_set)} stores  samples={n_samp}"
                f"  {dt:.0f}s  eta~{eta:.1f}min")

    save_curves(out_dir, ids, resp, has_resp, theta, phi0, phi90, has_rad, variants, state)
    n_store_ids = sum(len(x) for x in store_pids.values())
    n_v_resp = sum(1 for e in variants.values() if e["s11"] is not None)
    n_v_rad = sum(1 for e in variants.values() if e["phi0"] is not None)
    log(f"curves done {time.time() - t0:.0f}s  has_resp={int(has_resp.sum())}/{n_store_ids}(有store)"
        f"  has_rad={int(has_rad.sum())}  對不到(本次重讀)={n_unmatch}  壞樣本={n_bad}")
    log(f"  變體曲線: {len(variants)} 筆(resp {n_v_resp} / rad {n_v_rad})")
    for f in ("resp.npz", "rad.npz", "variant_resp.json"):
        p = os.path.join(out_dir, f)
        if os.path.exists(p):
            log(f"  {f}: {os.path.getsize(p) / 1e6:.1f} MB")
    return has_resp, has_rad


def main():
    ap = argparse.ArgumentParser(description="Pattern Browser 資料索引產線(契約=SPEC.md)")
    ap.add_argument("--ds", default=DS_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--full-rescan", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--skip-curves", action="store_true",
                    help="跳過曲線層 NAS 重讀(舊 resp/rad/variant 對齊新 ids 重寫)")
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

    has_resp, has_rad = build_curves(args.ds, args.out, ids, packed_rows, meta,
                                     res_index, pt_index, args.rebuild, args.skip_curves)
    for k, r in enumerate(meta):
        r["has_resp"] = bool(has_resp[k])
        r["has_rad"] = bool(has_rad[k])

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
