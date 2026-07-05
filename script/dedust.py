# -*- coding: utf-8 -*-
"""
script/dedust.py — Round-07 除塵驗證 (de-dust)：harvest 池達標 pattern 的粉塵是不是 load-bearing？

背景（見 docs/log/round-07-dedust.md / docs/discuss/scratch.md 2026-07-03 戰略討論塊）：
池內 18 筆達標 pattern 全是「3-8 塊大銅片＋10-18 顆 1-3px 粉塵」（~3 個設計家族），不符可製造性
（裁切製程：允許不連通、但不要很多 1×1 碎片）。本工具驗證「拔掉粉塵后 margin 撐不撐得住」，
每次 HFSS solve 順帶方向圖（±45° 覆蓋驗證 + Stage-3 rad 冷啟動資料）。

流程：
    開發機:  python -m script.dedust select      # pool 快取挑家族代表+近標者、產除塵變體 → NAS dedust_r7_input/
             python -m script.dedust sm-screen   # sm_harvest.pth 預測預篩（零 HFSS；Δ 只當方向訊號）
    正式機:  python -m script.dedust run         # HFSS 驗原版+除塵版（順收 rad）→ NAS dedust_r7/（可中斷續跑）
    任一機:  python -m script.dedust report      # 匯總表（貼 round-07 §4）

margin 與 analysis-01 / round-06 同一把尺（`antenna.losses.worst_margin` + 現行 targets）。
select 依賴 `tmp/pattern_anatomy/pool.npz`（沒有先跑 `python -m script.pattern_anatomy collect-pool`）。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from antenna.utils import config as _config, DATASET_PATH
_config.device = "cpu"
import torch

from antenna.training import load_config, PORT_SPECS
from antenna.losses import worst_margin

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL_NPZ = os.path.join(REPO, "tmp", "pattern_anatomy", "pool.npz")
DEFAULT_INPUT = "dedust_r7_input"                      # 輸入夾名（DATASET_PATH 下；--input 換 round）
DEFAULT_STORE = "dedust_r7"                            # 結果夾名（SampleStore + rad/ + results.json）
DEFAULT_CFG = os.path.join(REPO, "configs", "single_r5_explore.yaml")   # targets/radiation 與現行分析同尺

FEED = (24, 12)                                        # single feed 像素（對齊 FeedReachability.single_feed）
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)   # 4-連通十字


# ---------------------------------------------------------------- 純函式（可單測）
def strip_small(p, min_size: int, feed=FEED):
    """移除 4-連通尺寸 < min_size 的金屬碎片；含 feed 的組永遠保留。回 (新 pattern bool(25,25), 移除像素數)。"""
    from scipy.ndimage import label
    p = np.asarray(p).reshape(25, 25) > 0.5
    lab, n = label(p, structure=_CROSS)
    if n == 0:
        return p.copy(), 0
    sizes = np.bincount(lab.ravel())[1:]
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_size
    feed_id = int(lab[feed])
    if feed_id > 0:
        keep[feed_id] = True    # feed 組不可拔（管線強制 feed 為金屬）
    out = keep[lab]
    return out, int(p.sum() - out.sum())


def piece_stats(p) -> dict:
    """碎片組成：組數 / 1px 數 / 2-3px 數 / 最大組像素 / 金屬像素。"""
    from scipy.ndimage import label
    p = np.asarray(p).reshape(25, 25) > 0.5
    lab, n = label(p, structure=_CROSS)
    if n == 0:
        return dict(n_comp=0, n_1px=0, n_2_3px=0, main_px=0, metal_px=0)
    sizes = np.bincount(lab.ravel())[1:]
    return dict(n_comp=int(n), n_1px=int((sizes == 1).sum()),
                n_2_3px=int(((sizes >= 2) & (sizes <= 3)).sum()),
                main_px=int(sizes.max()), metal_px=int(p.sum()))


def cluster_families(patterns, max_dist: int = 100):
    """greedy leader clustering（Hamming ≤ max_dist 併同家族）。呼叫端先按優先序（wm 降冪）排好。
    patterns: (n, 625) 或 (n,25,25) 布林。回 labels (n,)。"""
    pats = [np.asarray(p).reshape(-1) > 0.5 for p in patterns]
    leaders, labels = [], np.zeros(len(pats), int)
    for i, p in enumerate(pats):
        for j, ld in enumerate(leaders):
            if int(np.count_nonzero(p != ld)) <= max_dist:
                labels[i] = j
                break
        else:
            labels[i] = len(leaders)
            leaders.append(p)
    return labels


def rad_window_margin(theta, gain, window_deg: float = 45.0, floor_db: float = 3.0) -> float:
    """±window 內覆蓋餘裕 (dB)：min(gain[窗內]) − (G(θ≈0) − floor)。正＝窗內都夠高（對齊 beam_coverage 語意）。"""
    theta = np.asarray(theta, float).reshape(-1)
    gain = np.asarray(gain, float).reshape(-1)
    g0 = gain[int(np.argmin(np.abs(theta)))]
    win = np.abs(theta) <= window_deg
    return float(gain[win].min() - (g0 - floor_db))


def close_holes(p):
    """補洞：被金屬包住（不觸邊）的介質區全部填成金屬。回 (新 pattern, 填入像素數)。
    analysis-01「Gain←少洞」的因果檢驗編輯（R8 B 臂）。"""
    from scipy.ndimage import label
    p = np.asarray(p).reshape(25, 25) > 0.5
    inv, m = label(~p, structure=_CROSS)
    border = set(np.unique(np.concatenate(
        [inv[0, :], inv[-1, :], inv[:, 0], inv[:, -1]])).tolist())
    out = p.copy()
    for h in set(range(1, m + 1)) - border:
        out[inv == h] = True
    return out, int(out.sum() - p.sum())


def _ensure_feed_pad(p, min_size: int = 4, feed=FEED):
    """修復後 feed 組若 < min_size（被翻/blob 沒長到 → 孤立小 feed）→ 蓋 3×3 貼底 pad，
    保證饋電件本身可裁。pad ≥ min_size 故後續不會被 strip。"""
    from scipy.ndimage import label
    lab, n = label(p, structure=_CROSS)
    fid = int(lab[feed])
    if fid > 0 and int((lab == fid).sum()) >= min_size:
        return p
    out = p.copy()
    r, c = feed
    out[max(0, r - 2):r + 1, max(0, c - 1):c + 2] = True
    return out


def perturb_repair(p, k: int, seed: int, min_size: int = 4, feed=FEED):
    """翻 k 個隨機像素後「無粉塵修復」（strip <min_size、feed 強制金屬＋pad 保底）——
    產生乾淨子空間內的鄰域樣本（R8 C 臂：SM 校準採樣）。決定性（seed）。"""
    rng = np.random.default_rng(seed)
    p = (np.asarray(p).reshape(25, 25) > 0.5).copy()
    flat = rng.choice(625, size=k, replace=False)
    p.ravel()[flat] = ~p.ravel()[flat]
    p[feed] = True
    out, _ = strip_small(p, min_size, feed=feed)
    return _ensure_feed_pad(out, min_size, feed=feed)


def smooth_blob(seed: int, metal_frac: float = 0.5, sigma: float = 2.5, min_size: int = 4, feed=FEED):
    """平滑隨機 blob：高斯濾波雜訊取閾值 → 天然整塊、無粉塵（修復＋feed pad 保險）。
    乾淨子空間的廣域覆蓋樣本（R8 C 臂）。決定性（seed）。"""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.random((25, 25)), sigma=sigma)
    thr = np.quantile(field, 1.0 - metal_frac)
    p = field >= thr
    p[feed] = True
    out, _ = strip_small(p, min_size, feed=feed)
    return _ensure_feed_pad(out, min_size, feed=feed)


# ---------------------------------------------------------------- 小工具
def _r(x, nd=2):
    return round(float(x), nd)


def _dir(name):
    """輸入/結果夾：DATASET_PATH 下的名字（跨機共享靠 NAS）。"""
    return DATASET_PATH.joinpath(name)


def _load_manifest(input_dir):
    path = input_dir.joinpath("manifest.json")
    if not path.exists():
        raise SystemExit(f"找不到 {path} —— 先在開發機跑 select / select-r8。")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(manifest, input_dir):
    path = input_dir.joinpath("manifest.json")
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------- select（開發機，零 HFSS）
def select(args):
    if not os.path.exists(POOL_NPZ):
        raise SystemExit(f"缺 {POOL_NPZ} —— 先跑 `python -m script.pattern_anatomy collect-pool`。")
    d = np.load(POOL_NPZ)
    ok = ~np.isnan(d["wm"][:, 2])
    wm = d["wm"][ok]
    pats = np.unpackbits(d["packed"][ok], axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
    worst = wm[:, 2]

    # 1) 達標家族代表（wm ≥ pass_thr，Hamming ≤ max_dist 併家族、每家族取 wm 最高者）
    idx = np.where(worst >= args.pass_thr)[0]
    idx = idx[np.argsort(worst[idx])[::-1]]
    fams = cluster_families(pats[idx], max_dist=args.max_dist)
    picked = []                                   # [(pool_idx, tag)]
    for f in sorted(set(fams.tolist())):
        i = int(idx[fams == f][0])                # 家族內 wm 最高（idx 已降冪）
        picked.append((i, f"F{f}"))
    print(f"達標 {len(idx)} 筆 → {len(picked)} 個家族代表")

    # 2) 近標補充（wm ≥ near_thr、與所有已選 Hamming > max_dist → 結構上真的不同的備援起點）
    near = np.where((worst >= args.near_thr) & (worst < args.pass_thr))[0]
    near = near[np.argsort(worst[near])[::-1]]
    for i in near:
        if len(picked) >= len(set(fams.tolist())) + args.extras:
            break
        if all(np.count_nonzero(pats[i] != pats[j]) > args.max_dist for j, _ in picked):
            picked.append((int(i), "near"))

    # 3) 產變體 + 落地（d1=拔 1px、d3=拔 1-3px；與前一級相同就略過）
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k, (pool_i, tag) in enumerate(picked):
        p = pats[pool_i]
        pool_margin = [_r(wm[pool_i, 0]), _r(wm[pool_i, 1]), _r(wm[pool_i, 2])]
        variants = [("orig", p, 0)]
        d1, r1 = strip_small(p, 2)
        if r1 > 0:
            variants.append(("d1", d1, r1))
        d3, r3 = strip_small(p, 4)
        if r3 > r1:
            variants.append(("d3", d3, r3))
        for kind, pat, removed in variants:
            pid = f"p{k:02d}_{kind}"
            torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))
            manifest.append(dict(id=pid, kind=kind, family=tag, pool_idx=pool_i,
                                 removed_px=removed, pool_wm=pool_margin, **piece_stats(pat)))
    _save_manifest(manifest, input_dir)

    n_sims = len(manifest)
    print(f"\n選出 {len(picked)} 個原版、共 {n_sims} 筆待模擬 → {input_dir}")
    print("\n| id | 家族 | 池wm(S11/Gain/worst) | 拔掉px | n_comp | 1px | 2-3px | 主件px |")
    print("|---|---|---|---|---|---|---|---|")
    for m in manifest:
        print(f"| {m['id']} | {m['family']} | {m['pool_wm'][0]:+.2f}/{m['pool_wm'][1]:+.2f}/{m['pool_wm'][2]:+.2f} "
              f"| {m['removed_px']} | {m['n_comp']} | {m['n_1px']} | {m['n_2_3px']} | {m['main_px']} |")


# ---------------------------------------------------------------- select-r8（開發機，零 HFSS）
def select_r8(args):
    """R8「乾淨子空間測繪」四臂輸入生成（詳見 docs/log/round-08）：
    A 乾淨前緣（main_frac≥0.9）top-K 原版+d3 —— 「整塊型可拔」通則檢驗＋前緣 HFSS 真值
    B 規則編輯 —— p03_d3 原樣重跑（跨 round 噪聲地板）＋補洞變體（Gain←少洞 因果檢驗）
    C SM 校準採樣 —— 錨點擾動修復＋平滑 blob（把 SM 在即將搜尋的乾淨區餵亮；順收 rad）
    D 真 uniform random —— 補 R6 誠實缺口（池≠隨機 的量化基線）
    """
    if not os.path.exists(POOL_NPZ):
        raise SystemExit(f"缺 {POOL_NPZ} —— 先跑 `python -m script.pattern_anatomy collect-pool`。")
    d = np.load(POOL_NPZ)
    ok = ~np.isnan(d["wm"][:, 2])
    wm = d["wm"][ok]
    feats = d["feats"][ok]
    pats = np.unpackbits(d["packed"][ok], axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
    worst, main_frac = wm[:, 2], feats[:, 1]

    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, removed=0, pool_i=None, base_id=None):
        row = dict(id=pid, kind=kind, family=family, removed_px=int(removed), **piece_stats(pat))
        if pool_i is not None:
            row["pool_idx"] = int(pool_i)
            row["pool_wm"] = [_r(wm[pool_i, 0]), _r(wm[pool_i, 1]), _r(wm[pool_i, 2])]
        if base_id:
            row["base_id"] = base_id
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))
        manifest.append(row)

    # -- A: 乾淨前緣 top-K（wm 降冪、互相 Hamming>60 保多樣）×（orig + d3）
    cand = np.where(main_frac >= 0.9)[0]
    cand = cand[np.argsort(worst[cand])[::-1]]
    picked = []
    for i in cand:
        if len(picked) >= args.frontier:
            break
        if all(np.count_nonzero(pats[i] != pats[j]) > 60 for j in picked):
            picked.append(int(i))
    a_d3 = {}                                        # k → (id, pattern)：B/C 臂的錨點來源
    for k, i in enumerate(picked):
        p = pats[i]
        emit(f"a{k:02d}_orig", "orig", f"A{k}", p, pool_i=i)
        d3, r3 = strip_small(p, 4)
        if r3 > 0:
            emit(f"a{k:02d}_d3", "d3", f"A{k}", d3, removed=r3, pool_i=i)
            a_d3[k] = (f"a{k:02d}_d3", d3)
        else:
            a_d3[k] = (f"a{k:02d}_orig", p)          # 本來就乾淨 → orig 即錨點

    # -- B: p03_d3 重跑 + 補洞；A 臂前 3 名 d3 補洞（Δ 基準走 base_id）
    p03_path = _dir(args.r7_input).joinpath("p03_d3.pt")
    if not p03_path.exists():
        raise SystemExit(f"缺 {p03_path}（R7 輸入）—— B/C 臂要用 p03_d3 當錨點。")
    p03 = np.asarray(torch.load(str(p03_path), weights_only=True)).reshape(25, 25) > 0.5
    emit("b00_ref", "orig", "B0", p03)               # 與 R7 同 pattern 重跑一次＝HFSS 重複性噪聲地板
    h, filled = close_holes(p03)
    if filled:
        emit("b00_holes", "holes", "B0", h, removed=-filled, base_id="b00_ref")
    for k in range(min(3, len(picked))):
        bid, bp = a_d3[k]
        h, filled = close_holes(bp)
        if filled:
            emit(f"b{k + 1:02d}_holes", "holes", f"B{k + 1}", h, removed=-filled, base_id=bid)

    # -- C: 錨點（p03_d3 + A 臂前 7 名 d3）× 翻 {8,32}px 修復 ×2 seed + 平滑 blob
    anchors = [p03] + [a_d3[k][1] for k in range(min(7, len(picked)))]
    n = 0
    for ap in anchors:
        for kk in (8, 32):
            for _ in range(2):
                emit(f"c{n:02d}_probe", "probe", "C", perturb_repair(ap, kk, seed=n))
                n += 1
    for s in range(args.blobs):
        emit(f"c{n:02d}_blob", "blob", "C", smooth_blob(seed=1000 + s))
        n += 1

    # -- D: 真 uniform random（iid p=0.5、feed 強制金屬、不修復——就是要真隨機）
    rng = np.random.default_rng(2026)
    for k in range(args.rand):
        p = rng.random((25, 25)) < 0.5
        p[FEED] = True
        emit(f"d{k:02d}_rand", "rand", "D", p)

    _save_manifest(manifest, input_dir)
    counts = {}
    for m in manifest:
        counts[m["id"][0]] = counts.get(m["id"][0], 0) + 1
    print(f"R8 輸入完成 → {input_dir}")
    print(f"各臂筆數: A={counts.get('a', 0)}  B={counts.get('b', 0)}  C={counts.get('c', 0)}  "
          f"D={counts.get('d', 0)}  共 {len(manifest)}（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


# ---------------------------------------------------------------- select-r9（開發機，零 HFSS）
def spread_idx(n_total: int, n_pick: int):
    """0..n_total-1 依序均勻取 n_pick 個索引（含頭尾;決定性,無隨機）。分層抽樣用。"""
    if n_total <= n_pick:
        return list(range(n_total))
    return sorted({int(round(x)) for x in np.linspace(0, n_total - 1, n_pick)})


def select_r9(args):
    """R9「池頂端重驗」輸入生成（詳見 docs/log/round-09）：
    R8 A 臂實錘池值系統性樂觀（14/15 向下、中位 −0.52、重跑噪聲地板 0.00）→ 一個批次答兩題：
      T 帳面達標（wm ≥ pass-thr）**全數**重驗 —— 這個 spec 在現行設定下有沒有已知解（R6 oracle 裁決）
      N 近標帶（near-lo ≤ wm < pass-thr）依 rank 分層抽樣 —— 「池值→現行值」校正曲線的取樣點
    原樣重跑、零編輯（report 的 Δ 欄無基準=「—」屬預期）。
    """
    if not os.path.exists(POOL_NPZ):
        raise SystemExit(f"缺 {POOL_NPZ} —— 先跑 `python -m script.pattern_anatomy collect-pool`。")
    d = np.load(POOL_NPZ)
    ok = ~np.isnan(d["wm"][:, 2])
    wm = d["wm"][ok]
    pats = np.unpackbits(d["packed"][ok], axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
    worst = wm[:, 2]

    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pool_i):
        pat = pats[pool_i]
        row = dict(id=pid, kind=kind, family=family, removed_px=0, pool_idx=int(pool_i),
                   pool_wm=[_r(wm[pool_i, 0]), _r(wm[pool_i, 1]), _r(wm[pool_i, 2])],
                   **piece_stats(pat))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))
        manifest.append(row)

    top = np.where(worst >= args.pass_thr)[0]
    top = top[np.argsort(worst[top])[::-1]]
    for k, i in enumerate(top):
        emit(f"t{k:02d}_top", "top", f"T{k}", i)

    band = np.where((worst >= args.near_lo) & (worst < args.pass_thr))[0]
    band = band[np.argsort(worst[band])[::-1]]           # 降冪＝rank 由好到差
    n_band = len(band)
    band = band[spread_idx(n_band, args.near)]
    for k, i in enumerate(band):
        emit(f"n{k:02d}_near", "near", f"N{k}", i)

    _save_manifest(manifest, input_dir)
    print(f"R9 輸入完成 → {input_dir}")
    print(f"T 帳面達標(≥{args.pass_thr:g}) 全數 {len(top)} 筆；N 近標帶[{args.near_lo:g},{args.pass_thr:g}) "
          f"{n_band} 筆取 {len(band)}；共 {len(manifest)}（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")
    print("\n| id | 池wm(S11/Gain/worst) | n_comp | 1px | 主件px |")
    print("|---|---|---|---|---|")
    for m in manifest:
        print(f"| {m['id']} | {m['pool_wm'][0]:+.2f}/{m['pool_wm'][1]:+.2f}/{m['pool_wm'][2]:+.2f} "
              f"| {m['n_comp']} | {m['n_1px']} | {m['main_px']} |")


# ---------------------------------------------------------------- sm-screen（開發機，零 HFSS）
def sm_screen(args):
    cfg = load_config(args.config)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    from antenna.zoo import SURROGATES
    cache = os.path.join(REPO, "tmp", "dedust")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
    sm.pre_load_model(DATASET_PATH.joinpath("sm_harvest.pth"), strict=True)
    sm.model.eval()

    input_dir = _dir(args.input)
    manifest = _load_manifest(input_dir)
    for m in manifest:
        p = torch.load(str(input_dir.joinpath(f"{m['id']}.pt")), weights_only=True)
        with torch.no_grad():
            pred = sm.model(p.flatten())
        w, per = worst_margin(pred, labels, cfg.targets)
        m["sm_wm"] = [_r(per[labels[0]]), _r(per[labels[1]]), _r(w)]
    _save_manifest(manifest, input_dir)

    orig = {m["id"].split("_", 1)[0]: m for m in manifest if m["kind"] == "orig"}   # 鍵=pXX（family near 不唯一）
    print("| id | 池wm | SM wm(S11/Gain/worst) | SM Δworst vs orig |")
    print("|---|---|---|---|")
    by_id = {m["id"]: m for m in manifest}
    for m in manifest:
        base = by_id.get(m.get("base_id")) or orig.get(m["id"].split("_", 1)[0])   # 明示基準優先（B 臂）
        dv = f"{m['sm_wm'][2] - base['sm_wm'][2]:+.2f}" if base else "—"
        pool = f"{m['pool_wm'][2]:+.2f}" if m.get("pool_wm") else "—"
        print(f"| {m['id']} | {pool} | {m['sm_wm'][0]:+.2f}/{m['sm_wm'][1]:+.2f}/{m['sm_wm'][2]:+.2f} "
              f"| {dv} |")
    print("\n⚠ SM 是 harvest 池上訓的（碎 pattern 主導），對除塵版屬輕度外插——Δ 只當方向訊號、以 HFSS 為準。")


# ---------------------------------------------------------------- run（正式機，燒 HFSS；可中斷續跑）
def run(args):
    from antenna.patch import SinglePortRadSimulator          # lazy：開發機/CI 不 import COM 相依
    from antenna.utils.store import SampleStore
    from antenna.utils.utils import Path

    cfg = load_config(args.config)
    labels = PORT_SPECS[cfg.port]["labels"]
    window = cfg.radiation.get("window_deg", 45)
    floor = cfg.radiation.get("floor_db", 3)
    input_dir = _dir(args.input)
    store_dir = _dir(args.store)
    manifest = _load_manifest(input_dir)

    store_dir.mkdir(parents=True, exist_ok=True)
    rad_dir = store_dir.joinpath("rad")
    rad_dir.mkdir(parents=True, exist_ok=True)
    results_path = store_dir.joinpath("results.json")
    results = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)

    def _flush():
        tmp = str(results_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        os.replace(tmp, results_path)

    store = SampleStore(store_dir)
    #? 續跑規則：成功的跳過、error 的重試（COM 偶發例外佔比 ~15%,R8 實測）
    todo = [(n, m) for n, m in enumerate(manifest)
            if m["id"] not in results or "error" in results[m["id"]]]
    print(f"待模擬 {len(todo)}/{len(manifest)} 筆（成功跳過、error 重試；中斷再跑即續）")

    out = Path(args.out).resolve()     # HFSS SaveAs 用自己的工作目錄解析相對路徑 → 必須絕對路徑
    sim = SinglePortRadSimulator(record_path=str(out))
    sim.open()
    try:
        for num, m in todo:
            p = torch.load(str(input_dir.joinpath(f"{m['id']}.pt")), weights_only=True)
            print(f"[{m['id']}] 模擬中… ({num + 1}/{len(manifest)})")
            try:
                sim.start(num)
                result = sim(p)
                elapsed = sim.end()
            except Exception as e:                       #! 單筆失敗不炸整批：記 error、下一筆（比照線上 skip）
                results[m["id"]] = {"error": str(e)}
                _flush()
                print(f"  ✗ {e}")
                continue

            resp = torch.stack([torch.as_tensor(result[l]).float().reshape(-1) for l in labels])
            w, per = worst_margin(resp, labels, cfg.targets)
            entry = {"wm": [_r(per[labels[0]]), _r(per[labels[1]]), _r(w)], "time_s": _r(elapsed, 1)}

            rad = sim.last_radiation                     # 方向圖順手收：±window 覆蓋餘裕 + 原始資料落檔
            if isinstance(rad, dict) and rad.get("theta") is not None:
                torch.save(rad, str(rad_dir.joinpath(f"{m['id']}.pt")))
                cuts = {f"phi{phi}": _r(rad_window_margin(rad["theta"], rad[f"phi{phi}"], window, floor))
                        for phi in (0, 90) if rad.get(f"phi{phi}") is not None}
                if cuts:
                    entry["rad"] = cuts
                    entry["rad_margin"] = min(cuts.values())
            store.add(p, resp)                           # (pattern, 真響應) 入庫：可再餵 SM 重錨/Stage-3
            results[m["id"]] = entry
            _flush()
            print(f"  ✓ wm={entry['wm']}  rad_margin={entry.get('rad_margin', '—')}  {entry['time_s']}s")
    finally:
        sim.quit()
    print(f"\n完成。結果：{results_path}；報表：python -m script.dedust report")


# ---------------------------------------------------------------- report（匯總表，貼 round 檔 §4）
def report(args):
    input_dir = _dir(args.input)
    manifest = _load_manifest(input_dir)
    results_path = _dir(args.store).joinpath("results.json")
    results = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            results = json.load(f)

    hfss_orig = {m["id"].split("_", 1)[0]: results.get(m["id"], {}).get("wm")
                 for m in manifest if m["kind"] == "orig"}
    print("| id | 拔px | 池wm | SM wm | HFSS S11/Gain/worst | Δworst vs 基準 | rad餘裕 | 分 |")
    print("|---|---|---|---|---|---|---|---|")
    for m in manifest:
        r = results.get(m["id"], {})
        sm = f"{m['sm_wm'][2]:+.2f}" if "sm_wm" in m else "—"
        if "wm" in r:
            wmtx = f"{r['wm'][0]:+.2f}/{r['wm'][1]:+.2f}/{r['wm'][2]:+.2f}"
            #? Δ 基準：明示 base_id（跨組編輯,如 B 臂補洞）優先,否則同組 orig
            base = results.get(m["base_id"], {}).get("wm") if m.get("base_id") \
                else hfss_orig.get(m["id"].split("_", 1)[0])
            dv = f"{r['wm'][2] - base[2]:+.2f}" if (base and m["kind"] != "orig") else "—"
            radm = f"{r['rad_margin']:+.2f}" if "rad_margin" in r else "—"
            t = f"{r.get('time_s', 0) / 60:.0f}m"
        elif "error" in r:
            wmtx, dv, radm, t = f"✗ {r['error'][:30]}", "—", "—", "—"
        else:
            wmtx, dv, radm, t = "（待跑）", "—", "—", "—"
        pool = f"{m['pool_wm'][2]:+.2f}" if m.get("pool_wm") else "—"
        print(f"| {m['id']} | {m.get('removed_px', 0)} | {pool} | {sm} | {wmtx} | {dv} | {radm} | {t} |")

    done = [r for r in results.values() if "wm" in r]
    if done:
        both = [r for r in done if r["wm"][2] >= 0 and r.get("rad_margin", -1) >= 0]
        print(f"\n已完成 {len(done)}/{len(manifest)}；三標全過（wm≥0 且 rad≥0）：{len(both)} 筆。")


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Round-07 除塵驗證工具（select/sm-screen 開發機、run 正式機）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="R7：挑家族代表+近標者、產除塵變體")
    s.add_argument("--input", default=DEFAULT_INPUT, help="輸入夾名（DATASET_PATH 下）")
    s.add_argument("--pass-thr", type=float, default=0.0, help="達標門檻 (預設 0)")
    s.add_argument("--near-thr", type=float, default=-1.0, help="近標補充門檻 (預設 -1)")
    s.add_argument("--extras", type=int, default=3, help="近標補充數上限 (預設 3)")
    s.add_argument("--max-dist", type=int, default=100, help="家族 Hamming 門檻 (預設 100)")
    s.set_defaults(fn=select)

    s = sub.add_parser("select-r8", help="R8：乾淨子空間測繪四臂輸入（A 前緣/B 規則編輯/C SM 校準/D 隨機基線）")
    s.add_argument("--input", default="dedust_r8_input")
    s.add_argument("--r7-input", default=DEFAULT_INPUT, help="R7 輸入夾（取 p03_d3 錨點）")
    s.add_argument("--frontier", type=int, default=15, help="A 臂前緣原版數")
    s.add_argument("--blobs", type=int, default=20, help="C 臂 blob 數")
    s.add_argument("--rand", type=int, default=10, help="D 臂 uniform random 數")
    s.set_defaults(fn=select_r8)

    s = sub.add_parser("select-r9", help="R9：池頂端重驗（T 帳面達標全數＋N 近標帶分層抽樣;原樣零編輯）")
    s.add_argument("--input", default="dedust_r9_input")
    s.add_argument("--pass-thr", type=float, default=0.0, help="帳面達標門檻 (預設 0)")
    s.add_argument("--near-lo", type=float, default=-1.0, help="近標帶下緣 (預設 -1)")
    s.add_argument("--near", type=int, default=12, help="近標帶抽樣數 (預設 12)")
    s.set_defaults(fn=select_r9)

    s = sub.add_parser("sm-screen", help="sm_harvest.pth 預測預篩（零 HFSS）")
    s.add_argument("--input", default=DEFAULT_INPUT)
    s.add_argument("--config", default=DEFAULT_CFG)
    s.set_defaults(fn=sm_screen)

    s = sub.add_parser("run", help="正式機：HFSS 驗證 manifest 所有 pattern（可中斷續跑）")
    s.add_argument("--input", default=DEFAULT_INPUT)
    s.add_argument("--store", default=DEFAULT_STORE, help="結果夾名（DATASET_PATH 下）")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--out", default="_dedust", help="HFSS 工作目錄（正式機本地碟）")
    s.set_defaults(fn=run)

    s = sub.add_parser("report", help="匯總表（貼 round 檔 §4）")
    s.add_argument("--input", default=DEFAULT_INPUT)
    s.add_argument("--store", default=DEFAULT_STORE)
    s.set_defaults(fn=report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
