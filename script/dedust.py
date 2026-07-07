# -*- coding: utf-8 -*-
"""
script/dedust.py — 批次 HFSS 驗證線（R7 起的研究主力）：開發機 select-* 生輸入 → 正式機 run 燒
HFSS → 任一機 report 看結果。輸入/結果都在 NAS（`DATASET_PATH/<name>_input/` 與 `<name>/`），
跨機共享；run 可中斷續跑（成功跳過、error 重試）；每筆 solve 順收方向圖（rad/ 夾）。

通用流程：
    開發機:  python -m script.dedust <select-指令> [--input X_input ...]   # 生 pattern+manifest 上 NAS
             python -m script.dedust sm-screen --input X_input             # SM 預篩（零 HFSS,選用）
    正式機:  python -m script.dedust run --input X_input --store X         # 燒 HFSS（可中斷續跑）
    任一機:  python -m script.dedust report --input X_input --store X      # 進度/結果表（貼 round 檔 §4）

select 子命令（各 round 的輸入生成器；歷史見 docs/log/round-NN 檔）：
    select          R7  除塵驗證（家族代表+近標者 × 除塵變體）
    select-r8       R8  乾淨子空間測繪（前緣/補洞因果/SM 校準/uniform random 基線）
    select-r9       R9  池頂端重驗（oracle 裁決+校正曲線）＋跨家族乾淨投影探索（E/G/S）
    select-refine1  R10 精修盲階段（s05 保對稱鄰域/對稱化救援推廣/g24 鄰域）→ 出了 w17 三標全過
    select-refine2  R10 精修知情階段（w17 密掃/承重圖知情編輯/重錨 SM 導引/y05 線）
    select-occlude  R10 物理遮蔽掃描（5×5 區塊逐一清空 → 真空間重要度圖）
    select-repeat   同 pattern × N 次（HFSS 可重複性；已公證雜訊地板 ≈0、跨機 bit 級一致,見 R9 §4 附錄）

慣例：margin/rad 全走同一把尺（`antenna.losses.worst_margin` + `configs/single_r5_explore.yaml` targets、
`rad_window_margin` ±45°/3dB）；生成全決定性（seed 進 manifest,各 select 的 seed 域不重疊）；
「可製造」= 全碎片 ≥4px＋feed pad。select/select-r8/r9 依賴 `tmp/pattern_anatomy/pool.npz`
（先跑 `python -m script.pattern_anatomy collect-pool`）；select-refine2 依賴 NAS 的 `sm_reanchor.pth`
（`script/sm_reanchor.py` 產出）。
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


def symmetrize(p, half: int, min_size: int = 4, feed=FEED):
    """左外側 `half` 欄鏡射到右外側（中央 25−2·half 欄保留原樣）→ 對稱先驗變體。
    half=12 ＝全鏡射（只留中央 1 欄自由）；half=10 ＝ 10-5-10 部分對稱（中央 5 欄自由，
    ONGOING「把對稱做對」候選的結構切法）。決定性。

    順序＝鏡射→除塵→**再鏡射**→feed pad：除塵保 feed 組件會單邊留件、破壞對稱，故除塵後
    重新鏡射（pad 蓋 rows22-24×cols11-13,本身左右對稱）。極罕見 merge 差異可能殘留 <4px
    碎片,manifest 的 n_1px 會現形。"""
    p = (np.asarray(p).reshape(25, 25) > 0.5).copy()
    for j in range(half):
        p[:, 24 - j] = p[:, j]
    p[feed] = True
    p, _ = strip_small(p, min_size, feed=feed)
    for j in range(half):
        p[:, 24 - j] = p[:, j]
    p[feed] = True
    return _ensure_feed_pad(p, min_size, feed=feed)


def add_block(p, r: int, c: int, h: int, w: int, gap: int = 1, feed=FEED):
    """加塊：在 (r,c) 蓋 h×w 金屬矩形成為**獨立新組件**——與既有金屬保持 ≥gap 圈空隙
    （否則 4-連通併件、組數不變）;出界或放不下回 None 讓呼叫端跳過。純幾何、決定性。
    組數階梯探索用（Ricky 2026-07-07:冠軍全 3 塊 → 試 4/5 塊;4=上3下1 或 2+2 皆可）。
    對稱放法由呼叫端組合:外側帶（c+w-1 ≤ 9）蓋完再 symmetrize(10) 得鏡射對（+2 件）;
    中央帶跨 col 12 對稱蓋（c=12-(w-1)//2, w 奇數）得單件（+1 件）。h·w ≥ 4 才過可製造。"""
    from scipy.ndimage import binary_dilation
    p = (np.asarray(p).reshape(25, 25) > 0.5).copy()
    if r < 0 or c < 0 or r + h > 25 or c + w > 25:
        return None
    foot = np.zeros((25, 25), bool)
    foot[r:r + h, c:c + w] = True
    if (binary_dilation(foot, structure=_CROSS, iterations=gap) & p).any():
        return None
    return p | foot


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


def oob_metrics(resp, n_side: int = 4) -> dict:
    """帶外選擇性指標（帶外要與帶內**反向**:S11 貼 0=全反射、Gain 越負=不輻射;Ricky 定義 2026-07-07）。
    遠帶外=兩側各 n_side 點（預設 4=24-25.5/30.5-32GHz,排除緊貼帶緣的過渡點 26.0/30.0）。
    判準仍只用 oob_bad（=gain_max−s11_min,綜合惡度越低越好）;其餘為壓帶外戰役的追蹤欄
    （2026-07-07 加,Ricky:「壓低的也要加更多東西 track」）:
      分側 _lo/_hi（哪側在漏——低頻裙擺=已知主破口）、rolloff_lo/hi（帶緣→遠帶外的 Gain 落差,
      越大=滾降越陡）、oob_gain_argmax（最壞 Gain 的頻點 GHz,診斷用）。17 點 24-32GHz 尺專用。"""
    r = np.asarray(resp, dtype=float).reshape(2, -1)
    n = r.shape[1]
    lo = list(range(n_side))
    hi = list(range(n - n_side, n))
    far = lo + hi
    s11_min = float(r[0][far].min())
    gain_max = float(r[1][far].max())
    freqs = 26.5 + (np.arange(n) - 5) * 0.5
    edge_lo, edge_hi = float(r[1][5]), float(r[1][11])       # 帶緣 Gain（26.5/29.5）
    return dict(oob_s11_min=round(s11_min, 2), oob_gain_max=round(gain_max, 2),
                oob_bad=round(gain_max - s11_min, 2),
                oob_gain_max_lo=round(float(r[1][lo].max()), 2),
                oob_gain_max_hi=round(float(r[1][hi].max()), 2),
                oob_s11_min_lo=round(float(r[0][lo].min()), 2),
                oob_s11_min_hi=round(float(r[0][hi].min()), 2),
                rolloff_lo=round(edge_lo - float(r[1][lo].max()), 2),
                rolloff_hi=round(edge_hi - float(r[1][hi].max()), 2),
                oob_gain_argmax=float(freqs[far][int(np.argmax(r[1][far]))]))


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
    """R9「池頂端重驗＋乾淨前緣探索」過夜批次輸入生成（詳見 docs/log/round-09）：
    重驗（R8 池值漂移警訊的直接後續）：
      T 帳面達標（wm ≥ pass-thr）**全數**重驗 —— 現行設定有沒有已知解（R6 oracle 裁決）
      N 近標帶 [near-lo, pass-thr) 分層抽樣 —— 「池值→現行值」校正曲線（0 附近加密）
      M 深帶 [cal-lo, near-lo) 分層抽樣 —— 校正曲線往搜尋工作區延伸
    探索（找有效 pattern；錨點＝池 top-300 greedy 家族聚類的**跨家族代表**——R9 普查實錘 R8 乾淨前緣
    只是邊緣家族（上下兩分型 F13）,top 家族是全面散布碎片雲,錨點不跨家族＝探索假多樣,見
    assets/round-09/pool_families.png;perturb_repair 內建除塵 → 探的是各家族的「乾淨投影」鄰域）：
      E 鄰域測繪 —— 每錨點先出 k=0（純除塵=家族乾淨投影真值）,再 k∈{4,8,16,32} × explore-seeds seed
      G SM 導引 —— 大量候選（同錨點,k 到 48）先 SM 篩,取分數最高＋互異的 guided 筆（批次版 guided 搜尋）
      S 對稱先驗 —— 錨點 × {全鏡射, 10-5-10 部分對稱}（ONGOING「把對稱做對」候選的初測）
    report 的 Δ 欄無基準=「—」屬預期;錨點分析離線做（manifest 帶 anchor_pool_wm）。
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

    # -- M: 深帶 [cal-lo, near-lo) 校正曲線延伸（覆蓋搜尋工作區 −3~−1）
    band2 = np.where((worst >= args.cal_lo) & (worst < args.near_lo))[0]
    band2 = band2[np.argsort(worst[band2])[::-1]]
    band2 = band2[spread_idx(len(band2), args.cal)]
    for k, i in enumerate(band2):
        emit(f"m{k:02d}_near", "near", f"M{k}", i)

    def emit_gen(pid, kind, family, pat, extra=None):
        row = dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat))
        if extra:
            row.update(extra)
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))
        manifest.append(row)

    # -- 探索臂錨點：池 top-300 greedy 家族聚類（Hamming>100,掃描序=wm 降冪 → leader 即家族最佳）
    top300 = np.argsort(worst)[::-1][:300]
    leaders = []
    for i in top300:
        if all(np.count_nonzero(pats[i] != pats[L]) > 100 for L in leaders):
            leaders.append(int(i))
    anchors = {f"F{k}": pats[L] for k, L in enumerate(leaders[:args.anchors])}
    a_info = {f"F{k}": dict(anchor_pool_idx=int(L), anchor_pool_wm=_r(worst[L]))
              for k, L in enumerate(leaders[:args.anchors])}

    # -- E: 鄰域測繪 —— 每錨點 k=0（純除塵=乾淨投影真值）+ k∈{4,8,16,32} × explore-seeds
    n, seed = 0, 0
    for a, ap in anchors.items():
        emit_gen(f"e{n:02d}_x0", "explore", f"E_{a}", perturb_repair(ap, 0, seed=seed),
                 extra=dict(anchor=a, flip_k=0, seed=seed, **a_info[a]))
        n += 1
        seed += 1
        for kk in (4, 8, 16, 32):
            for _ in range(args.explore_seeds):
                emit_gen(f"e{n:02d}_x{kk}", "explore", f"E_{a}", perturb_repair(ap, kk, seed=seed),
                         extra=dict(anchor=a, flip_k=kk, seed=seed, **a_info[a]))
                n += 1
                seed += 1

    # -- G: SM 導引 —— 960 候選（錨點×k{8,16,32,48}×40 seed,seed 1000+ 與 E 不重疊）→ SM 排序＋互異取前 guided 筆
    cfg = load_config(DEFAULT_CFG)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    from antenna.zoo import SURROGATES
    cache = os.path.join(REPO, "tmp", "dedust")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
    sm.pre_load_model(DATASET_PATH.joinpath("sm_harvest.pth"), strict=True)
    sm.model.eval()
    cands = []
    gseed = 1000
    for a, ap in anchors.items():
        for kk in (8, 16, 32, 48):
            for _ in range(40):
                pat = perturb_repair(ap, kk, seed=gseed)
                with torch.no_grad():
                    pred = sm.model(torch.tensor(pat, dtype=torch.float32).flatten())
                w, _per = worst_margin(pred, labels, cfg.targets)
                cands.append((float(w), a, kk, gseed, pat))
                gseed += 1
    cands.sort(key=lambda c: -c[0])                      # SM 分數降冪
    picked_pats, n = [], 0
    for w, a, kk, s, pat in cands:
        if n >= args.guided:
            break
        if all(np.count_nonzero(pat != q) > 30 for q in picked_pats):   # 互異守門（Hamming>30）
            emit_gen(f"g{n:02d}_sm", "guided", f"G_{a}", pat,
                     extra=dict(anchor=a, flip_k=kk, seed=s, sm_pick_wm=_r(w)))
            picked_pats.append(pat)
            n += 1

    # -- S: 對稱先驗 —— 錨點前 5 × {全鏡射 half=12, 10-5-10 half=10}
    n = 0
    for a, ap in list(anchors.items())[:5]:
        for tag, half in (("full", 12), ("1050", 10)):
            emit_gen(f"s{n:02d}_{tag}", "sym", f"S_{a}", symmetrize(ap, half),
                     extra=dict(anchor=a, sym=tag, **a_info[a]))
            n += 1

    _save_manifest(manifest, input_dir)
    cnt = {}
    for m in manifest:
        cnt[m["id"][0]] = cnt.get(m["id"][0], 0) + 1
    print(f"R9 輸入完成 → {input_dir}")
    print(f"T 達標全數={cnt.get('t', 0)}  N 近標[{args.near_lo:g},{args.pass_thr:g})={cnt.get('n', 0)}  "
          f"M 深帶[{args.cal_lo:g},{args.near_lo:g})={cnt.get('m', 0)}  E 鄰域={cnt.get('e', 0)}  "
          f"G 導引={cnt.get('g', 0)}  S 對稱={cnt.get('s', 0)}")
    print(f"共 {len(manifest)} 筆（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr @3分/筆；"
          f"COM 偶發 error 會跳過續跑、回來重跑同指令即重試）")


# ---------------------------------------------------------------- select-refine1（開發機，零 HFSS）
def select_refine1(args):
    """精修 phase-1（盲階段——不依賴 SM 重錨/遮蔽圖,那些進今晚 phase-2）：
      W s05 保對稱鄰域 —— perturb_repair(k) 後再 10-5-10 對稱化（翻轉被鏡射成對,活在 s05 同一子空間）
      X 對稱化救援推廣 —— 冠軍 g15/g24/e73/e39 各做 10-5-10（規則測試;R9 說救爛毀好 → 預測=變差,可證偽）
      Y g24 鄰域 —— rad 已過的種子做小步 perturb_repair,找「補 wm 不丟 rad」
    種子/編輯全決定性;margin 同一把尺。"""
    src = _dir(args.source_input)

    def load(pid):
        f = src.joinpath(f"{pid}.pt")
        if not f.exists():
            raise SystemExit(f"找不到 {f}")
        return np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5

    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra):
        row = dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra)
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))
        manifest.append(row)

    s05 = load("s05_1050")
    n, seed = 0, 5000                                    # seed 域與 R9 (0-/1000-) 不重疊
    for k in (2, 4, 8):
        for _ in range(args.seeds):
            emit(f"w{n:02d}_k{k}", "refine", "W_s05", symmetrize(perturb_repair(s05, k, seed=seed), 10),
                 dict(anchor="s05_1050", flip_k=k, seed=seed))
            n += 1
            seed += 1
    for m, cid in enumerate(("g15_sm", "g24_sm", "e73_x16", "e39_x0")):
        emit(f"x{m:02d}_symres", "symres", f"X_{cid}", symmetrize(load(cid), 10), dict(anchor=cid, sym="1050"))
    g24 = load("g24_sm")
    n = 0
    for k in (2, 4, 8):
        for _ in range(args.seeds):
            emit(f"y{n:02d}_k{k}", "refine", "Y_g24", perturb_repair(g24, k, seed=seed),
                 dict(anchor="g24_sm", flip_k=k, seed=seed))
            n += 1
            seed += 1
    _save_manifest(manifest, input_dir)
    print(f"精修 phase-1 輸入完成 → {input_dir}：W {3 * args.seeds}＋X 4＋Y {3 * args.seeds}＝{len(manifest)} 筆"
          f"（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


# ---------------------------------------------------------------- select-refine2（開發機，零 HFSS）
def perturb_blocks(p, k: int, seed: int, blocks, bs: int = 5, min_size: int = 4, feed=FEED):
    """只在指定 5×5 區塊集合內翻 k 像素，其餘同 perturb_repair（無粉塵修復＋feed 保底）。
    遮蔽圖知情編輯用（在低承重/rad 正效區內動刀）。決定性。純函式。"""
    mask = np.zeros((25, 25), bool)
    for br, bc in blocks:
        mask[br * bs:(br + 1) * bs, bc * bs:(bc + 1) * bs] = True
    idx = np.flatnonzero(mask.reshape(-1))
    rng = np.random.default_rng(seed)
    p = (np.asarray(p).reshape(25, 25) > 0.5).copy()
    flat = rng.choice(idx, size=min(k, len(idx)), replace=False)
    p.ravel()[flat] = ~p.ravel()[flat]
    p[feed] = True
    out, _ = strip_small(p, min_size, feed=feed)
    return _ensure_feed_pad(out, min_size, feed=feed)


def select_refine2(args):
    """精修 phase-2（知情階段,圍繞 w17）——目標=把 S11/Gain 帶緣餘裕再往上推、rad 不丟：
      A w17 保對稱鄰域 —— perturb→10-5-10 再對稱化,k∈{2,4,8,16}×seeds（w17 高地密掃）
      B 遮蔽圖知情編輯 —— 只在 s05 承重圖的「低 wm 成本/rad 正效」區塊（(0,1)(0,2)(3,1)(3,2),
        左半+中央,鏡射自動補右半）內翻,再對稱化——規則的因果使用
      C 重錨 SM 導引 —— 640 候選（w17 鄰域,seed 2000+）用 sm_reanchor.pth 排序,取互異 top-K
      D y05 線續推 —— ref1 Y 臂最佳（−0.97/rad+0.48）的小步鄰域
    全決定性;錨點真值已公證（雜訊地板≈0）。"""
    w17 = np.asarray(torch.load(str(_dir(args.ref1_input).joinpath("w17_k8.pt")), weights_only=True)).reshape(25, 25) > 0.5
    y05 = np.asarray(torch.load(str(_dir(args.ref1_input).joinpath("y05_k2.pt")), weights_only=True)).reshape(25, 25) > 0.5
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra):
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))

    n, seed = 0, 6000
    for k in (2, 4, 8, 16):
        for _ in range(args.seeds):
            emit(f"a{n:02d}_k{k}", "refine", "A_w17", symmetrize(perturb_repair(w17, k, seed=seed), 10),
                 dict(anchor="w17_k8", flip_k=k, seed=seed))
            n += 1
            seed += 1

    FREE_BLOCKS = ((0, 1), (0, 2), (3, 1), (3, 2))       # s05 承重圖: |Δwm|≤0.6 且含 rad 正效塊 (0,2)
    n = 0
    for k in (2, 4, 8):
        for _ in range(args.seeds):
            emit(f"b{n:02d}_k{k}", "blocked", "B_w17", symmetrize(perturb_blocks(w17, k, seed, FREE_BLOCKS), 10),
                 dict(anchor="w17_k8", flip_k=k, seed=seed, blocks=list(map(list, FREE_BLOCKS))))
            n += 1
            seed += 1

    # C: 重錨 SM 導引
    cfg = load_config(DEFAULT_CFG)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    from antenna.zoo import SURROGATES
    cache = os.path.join(REPO, "tmp", "dedust")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
    sm.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
    sm.model.eval()
    cands = []
    gseed = 7000
    for k in (4, 8, 16, 32):
        for _ in range(160):
            pat = symmetrize(perturb_repair(w17, k, seed=gseed), 10)
            with torch.no_grad():
                pred = sm.model(torch.tensor(pat, dtype=torch.float32).flatten())
            w, _per = worst_margin(pred, labels, cfg.targets)
            cands.append((float(w), k, gseed, pat))
            gseed += 1
    cands.sort(key=lambda c: -c[0])
    picked, n = [], 0
    for w, k, s, pat in cands:
        if n >= args.guided:
            break
        if all(np.count_nonzero(pat != q) > 20 for q in picked):
            emit(f"c{n:02d}_sm", "guided", "C_w17", pat, dict(anchor="w17_k8", flip_k=k, seed=s, sm_pick_wm=_r(w)))
            picked.append(pat)
            n += 1

    n = 0
    for k in (2, 4):
        for _ in range(3):
            emit(f"d{n:02d}_k{k}", "refine", "D_y05", perturb_repair(y05, k, seed=seed),
                 dict(anchor="y05_k2", flip_k=k, seed=seed))
            n += 1
            seed += 1

    _save_manifest(manifest, input_dir)
    cnt = {}
    for m in manifest:
        cnt[m["id"][0]] = cnt.get(m["id"][0], 0) + 1
    print(f"精修 phase-2 輸入完成 → {input_dir}：A={cnt.get('a', 0)} B={cnt.get('b', 0)} "
          f"C={cnt.get('c', 0)} D={cnt.get('d', 0)} 共 {len(manifest)}"
          f"（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


def select_refine3(args):
    """R11 ref3（過夜主力）——穩健化精修 × 帶外選擇性 × 組數階梯,錨點 c21/a15（certified）。
    字典序目標（decisions 2026-07-07）:①硬約束（三標;穩健=局部缺陷存活,tol 實測整面蝕刻無解）
    → ②帶內 min-margin↑ → ③帶外惡度 oob_bad↓（加分,永不換帶內）。
      A 穩健盲掃 —— occl2 低成本區塊內翻（LOW_BLOCKS 左半代表,鏡射自動補右）+ 再對稱化
      B SM 導引（sm_reanchor3）—— 兩層排序:預測頂帶（max−0.36=作戰區誤差內）按 oob_bad 升冪,
        其餘按預測 wm 降冪（頂帶內排 wm=排雜訊,排 oob 才有資訊）
      C add_block 組數階梯 —— 翼對(5塊)/頂中央單塊(4塊=上3下1)/破對稱 2+2（Ricky 2026-07-07 定)
    全決定性;判準先於發車寫死於 round-11。"""
    ref2 = _dir(args.ref2_input)
    anchors = {}
    for aid in ("c21_sm", "a15_k4"):
        anchors[aid] = np.asarray(torch.load(str(ref2.joinpath(f"{aid}.pt")), weights_only=True)).reshape(25, 25) > 0.5
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra):
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))

    # A 穩健盲掃:occl2 低成本區（頂部;左半代表塊,對稱化自動補鏡射側）
    LOW_BLOCKS = ((0, 1), (0, 2), (1, 1))
    n, seed = 0, 8000
    for aid, pat0 in anchors.items():
        for k in (2, 4):
            for _ in range(args.seeds):
                emit(f"a{n:02d}_{aid[:3]}k{k}", "robust", f"A_{aid}",
                     symmetrize(perturb_blocks(pat0, k, seed, LOW_BLOCKS), 10),
                     dict(anchor=aid, flip_k=k, seed=seed, blocks=list(map(list, LOW_BLOCKS))))
                n += 1
                seed += 1

    # B SM 導引（字典序:預測達標 → oob_bad 升冪;未達標 → 預測 wm 降冪）
    cfg = load_config(DEFAULT_CFG)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    from antenna.zoo import SURROGATES
    cache = os.path.join(REPO, "tmp", "dedust")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
    sm.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
    sm.model.eval()
    n = 0
    for aid, pat0 in anchors.items():
        cands, gseed = [], 9000
        for k in (4, 8, 16, 32):
            for _ in range(200):
                pat = symmetrize(perturb_repair(pat0, k, seed=gseed), 10)
                with torch.no_grad():
                    pred = sm.model(torch.tensor(pat, dtype=torch.float32).flatten())
                w, _per = worst_margin(pred, labels, cfg.targets)
                ob = oob_metrics(pred.detach().cpu().numpy())["oob_bad"]
                cands.append((float(w), ob, k, gseed, pat))
                gseed += 1
        top = max(c[0] for c in cands) - 0.36                 # 頂帶=SM 作戰區誤差內（0.36dB,R10 量化）
        cands.sort(key=lambda c: (0, c[1]) if c[0] >= top else (1, -c[0]))
        picked, m = [], 0
        for w, ob, k, sd, pat in cands:
            if m >= args.guided:
                break
            if all(np.count_nonzero(pat != q) > 12 for q in picked):
                emit(f"b{n:02d}_{aid[:3]}sm", "guided", f"B_{aid}", pat,
                     dict(anchor=aid, flip_k=k, seed=sd, sm_pick_wm=_r(w), sm_oob=_r(ob)))
                picked.append(pat)
                n += 1
                m += 1

    # C add_block 組數階梯:翼對(+2件=5塊)/頂中央(+1件=4塊 上3下1)/破對稱 2+2
    def scan_positions(pat0, h, w, cols, want_comp, mirror):
        """row-major 掃全部可放位（決定性）;回 (r,c,拓撲驗證後的 pattern) 清單。"""
        outs = []
        for r in range(0, 26 - h):
            for c in cols:
                q = add_block(pat0, r, c, h, w)
                if q is None:
                    continue
                qq = symmetrize(q, 10) if mirror else q
                if piece_stats(qq)["n_comp"] == want_comp:
                    outs.append((r, c, qq))
        return outs

    def spaced(cands, k):
        """從掃位結果均勻抽 k 個（頭尾覆蓋,決定性）。"""
        if len(cands) <= k:
            return cands
        idx = np.linspace(0, len(cands) - 1, k).round().astype(int)
        return [cands[i] for i in idx]

    def scan_multi(pat0, sizes, cols_of, want_comp, mirror):
        """多尺寸掃位串接＋按 pattern bytes 去重（不同尺寸可能落同一格局）。"""
        outs, seen = [], set()
        for h, w in sizes:
            for r, c, q in scan_positions(pat0, h, w, cols_of(w), want_comp, mirror):
                key = q.tobytes()
                if key not in seen:
                    seen.add(key)
                    outs.append((r, c, h, w, q))
        return outs

    n = 0
    SIZES = ((2, 2), (2, 3), (3, 3), (2, 4))
    for aid, pat0 in anchors.items():
        picks = []
        picks += [("5=3+wing_pair", t) for t in spaced(
            scan_multi(pat0, SIZES, lambda w: range(0, 10 - w), 5, True), args.wing)]
        picks += [("4=3+top_center", t) for t in spaced(
            scan_multi(pat0, SIZES, lambda w: (12 - (w - 1) // 2,), 4, False), args.center)]
        picks += [("4=asym", t) for t in spaced(
            scan_multi(pat0, SIZES, lambda w: range(0, 26 - w), 4, False), args.asym)]
        for topo, (r, c, h, w, q) in picks:
            tag = {"5=3+wing_pair": "w", "4=3+top_center": "m", "4=asym": "x"}[topo]
            emit(f"c{n:02d}_{aid[:3]}{tag}{r}_{c}_{h}{w}", "addblock", f"C_{aid}", q,
                 dict(anchor=aid, topo=topo, block_at=[r, c, h, w]))
            n += 1
        # 6 塊試點:先蓋中央塊(4塊) → 再放翼對(6塊),各錨點取前 2 個可行組合
        got = 0
        for _rc, _cc, hh, ww, q4 in scan_multi(pat0, ((3, 3), (2, 3)), lambda w: (12 - (w - 1) // 2,), 4, False):
            if got >= args.six:
                break
            hits = scan_multi(q4, ((2, 2), (3, 3)), lambda w: range(0, 10 - w), 6, True)
            if not hits:
                continue
            r, c, h, w, q6 = hits[0]
            emit(f"c{n:02d}_{aid[:3]}s{r}_{c}_{h}{w}", "addblock", f"C_{aid}", q6,
                 dict(anchor=aid, topo="6=3+center+pair", block_at=[r, c, h, w],
                      center_at=[_rc, _cc, hh, ww]))
            n += 1
            got += 1

    _save_manifest(manifest, input_dir)
    cnt = {}
    for m in manifest:
        cnt[m["id"][0]] = cnt.get(m["id"][0], 0) + 1
    print(f"ref3 輸入完成 → {input_dir}：A={cnt.get('a', 0)} B={cnt.get('b', 0)} C={cnt.get('c', 0)} "
          f"共 {len(manifest)}（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


def select_wide(args):
    """R11 wide（218 過夜）——冠軍高原測繪,與 37 的 ref3（近距精修）互補。錨點=榜首+代表
    （c21/a00/b11/a15,certified）。三臂:
      W 遠距對稱擾動 —— k∈{48,64,96,128}+再對稱化:量 w17 高原「半徑」（R6:進步靠躍遷）
      X 對稱必要性探針 —— k∈{2,4,8} **不再對稱化**:冠軍級的對稱是否承重,直接對答案
      Y SM 遠距導引 —— k∈{48,96} 候選字典序排序（同 ref3 B）:測導航儀有效射程
    全決定性（seed 12000+,不與 ref1/2/3 重疊）。"""
    ref2 = _dir(args.ref2_input)
    anchors = {}
    for aid in ("c21_sm", "a00_k2", "b11_k2", "a15_k4"):
        anchors[aid] = np.asarray(torch.load(str(ref2.joinpath(f"{aid}.pt")), weights_only=True)).reshape(25, 25) > 0.5
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra):
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))

    n, seed = 0, 12000
    for aid, pat0 in anchors.items():
        for k in (48, 64, 96, 128):
            for _ in range(args.seeds):
                emit(f"w{n:02d}_{aid[:3]}k{k}", "wide", f"W_{aid}",
                     symmetrize(perturb_repair(pat0, k, seed=seed), 10),
                     dict(anchor=aid, flip_k=k, seed=seed))
                n += 1
                seed += 1
    n = 0
    for aid, pat0 in anchors.items():
        for k in (2, 4, 8):
            for _ in range(args.seeds):
                emit(f"x{n:02d}_{aid[:3]}k{k}", "asym", f"X_{aid}",
                     perturb_repair(pat0, k, seed=seed),
                     dict(anchor=aid, flip_k=k, seed=seed))
                n += 1
                seed += 1

    cfg = load_config(DEFAULT_CFG)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    from antenna.zoo import SURROGATES
    cache = os.path.join(REPO, "tmp", "dedust")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
    sm.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
    sm.model.eval()
    n = 0
    for aid in ("c21_sm", "a15_k4"):
        pat0 = anchors[aid]
        cands, gseed = [], 14000
        for k in (48, 96):
            for _ in range(200):
                pat = symmetrize(perturb_repair(pat0, k, seed=gseed), 10)
                with torch.no_grad():
                    pred = sm.model(torch.tensor(pat, dtype=torch.float32).flatten())
                w, _per = worst_margin(pred, labels, cfg.targets)
                ob = oob_metrics(pred.detach().cpu().numpy())["oob_bad"]
                cands.append((float(w), ob, k, gseed, pat))
                gseed += 1
        top = max(c[0] for c in cands) - 0.36
        cands.sort(key=lambda c: (0, c[1]) if c[0] >= top else (1, -c[0]))
        picked, m = [], 0
        for w, ob, k, sd, pat in cands:
            if m >= args.guided:
                break
            if all(np.count_nonzero(pat != q) > 20 for q in picked):
                emit(f"y{n:02d}_{aid[:3]}sm", "guided_wide", f"Y_{aid}", pat,
                     dict(anchor=aid, flip_k=k, seed=sd, sm_pick_wm=_r(w), sm_oob=_r(ob)))
                picked.append(pat)
                n += 1
                m += 1

    _save_manifest(manifest, input_dir)
    cnt = {}
    for m in manifest:
        cnt[m["id"][0]] = cnt.get(m["id"][0], 0) + 1
    print(f"wide 輸入完成 → {input_dir}：W={cnt.get('w', 0)} X={cnt.get('x', 0)} Y={cnt.get('y', 0)} "
          f"共 {len(manifest)}（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


HISTORY_INPUTS = ("dedust_r7_input", "dedust_r8_input", "dedust_r9_input", "dedust_ref1_input",
                  "dedust_ref2_input", "dedust_occl_input", "dedust_occl2_input", "dedust_tol_input",
                  "dedust_w17rep_input", "dedust_verify_input", "dedust_ref2v_input",
                  "dedust_champ_input", "dedust_ref3_input", "dedust_wide_input")


def check_dup(args):
    """發車前查重（教訓 2026-07-07:ref3 出現 4/319 重複——掃位跨拓撲撞位、k=2 被再對稱化蓋回錨點）。
    查 --input 批內重複＋與歷史輸入夾（HISTORY_INPUTS 中既存者,排除自身）的交叉重複。exit 1=有重複。"""
    def load_folder(folder):
        d = DATASET_PATH.joinpath(folder)
        man = json.load(open(str(d.joinpath("manifest.json")), encoding="utf-8"))
        return {m["id"]: np.asarray(torch.load(str(d.joinpath(f"{m['id']}.pt")), weights_only=True)
                                    ).reshape(-1).__gt__(0.5).tobytes()
                for m in man if d.joinpath(f"{m['id']}.pt").exists()}

    new = load_folder(args.input)
    seen, bad = {}, 0
    for k, v in new.items():
        if v in seen:
            print(f"批內重複: {k} == {seen[v]}")
            bad += 1
        else:
            seen[v] = k
    hist = {}
    for fol in HISTORY_INPUTS:
        if fol == args.input or not DATASET_PATH.joinpath(fol, "manifest.json").exists():
            continue
        for k, v in load_folder(fol).items():
            hist.setdefault(v, f"{fol}:{k}")
    for k, v in new.items():
        if v in hist:
            print(f"與歷史重複: {k} == {hist[v]}")
            bad += 1
    print(f"{args.input}: {len(new)} 筆,重複 {bad}")
    if bad:
        raise SystemExit(1)


# ---------------------------------------------------------------- select-occlude（開發機，零 HFSS）
def occlude_block(p, br: int, bc: int, bs: int = 5, feed=FEED):
    """把第 (br,bc) 個 bs×bs 區塊的金屬清空（feed 像素永遠保留）。回 (新 pattern, 移除像素數)。
    **手術式、不修復**——量「拔掉這一塊」的因果效應,可能留 <4px 孤件（測量探針,非候選）。純函式。"""
    p = (np.asarray(p).reshape(25, 25) > 0.5).copy()
    r0, c0 = br * bs, bc * bs
    before = int(p.sum())
    p[r0:r0 + bs, c0:c0 + bs] = False
    p[feed] = True
    return p, before - int(p.sum())


def select_occlude(args):
    """R10 Stage B（前半）：物理遮蔽掃描——錨點 pattern 的 5×5 區塊逐一清空 → HFSS 直接給出
    **真·空間重要度圖**（哪塊承重/哪塊死區）,並校驗 SM 歸因的可信度。空區塊（無金屬可拔）自動跳過。
    Δ 判讀離線做（各錨點 base 值已在 dedust_r9 公證,雜訊地板≈0 → 每個 Δ 都是真效果）。"""
    src = _dir(args.source_input)
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    n = 0
    for sid in args.ids.split(","):
        f = src.joinpath(f"{sid}.pt")
        if not f.exists():
            raise SystemExit(f"找不到 {f}")
        pat = np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5
        kept = 0
        for br in range(5):
            for bc in range(5):
                q, removed = occlude_block(pat, br, bc)
                if removed <= 0:
                    continue                      # 空區塊=無資訊
                pid = f"o{n:02d}_{sid.split('_')[0]}b{br}{bc}"
                torch.save(torch.tensor(q, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))
                manifest.append(dict(id=pid, kind="occlude", family=f"O_{sid}", removed_px=removed,
                                     source_id=sid, block=[br, bc], **piece_stats(q)))
                n += 1
                kept += 1
        print(f"  {sid}: {kept}/25 區塊有金屬（其餘跳過）")
    _save_manifest(manifest, input_dir)
    print(f"遮蔽掃描輸入完成 → {input_dir}：共 {len(manifest)} 筆"
          f"（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


# ---------------------------------------------------------------- select-tolerance（開發機，零 HFSS）
def edge_sets(p, feed=FEED):
    """回 (金屬邊緣像素, 貼金屬的介質像素) 兩個布林圖——製造公差擾動的合法位置。純函式。"""
    from scipy.ndimage import binary_erosion, binary_dilation
    p = np.asarray(p).reshape(25, 25) > 0.5
    edge_metal = p & ~binary_erosion(p, structure=_CROSS)
    edge_diel = binary_dilation(p, structure=_CROSS) & ~p
    edge_metal[feed] = False                     # feed 像素不可動
    return edge_metal, edge_diel


def select_tolerance(args):
    """製造公差掃描：模擬蝕刻誤差對冠軍 pattern 的影響（**手術式,不修復**——真實製造缺陷不會被演算法修）：
      erode1 / dilate1 —— 全邊界收/漲 1px（系統性 under/over-etch 的極端）
      k∈{1,2,4} × seeds —— 邊緣隨機翻 k 像素（局部缺陷;只翻邊緣=物理上合理的誤差位置）
    判讀：margin 對公差的敏感度＝冠軍的「工程餘裕」;若 erode/dilate 就崩 → 需要更胖的 margin。"""
    from scipy.ndimage import binary_erosion, binary_dilation
    src = _dir(args.source_input)
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra):
        pat = pat.copy()
        pat[FEED] = True
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))

    seed = 9000
    for sid in args.ids.split(","):
        f = src.joinpath(f"{sid}.pt")
        if not f.exists():
            raise SystemExit(f"找不到 {f}")
        p = np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5
        tag = sid.split("_")[0]
        emit(f"t_{tag}_erode1", "tol", f"T_{sid}", binary_erosion(p, structure=_CROSS), dict(source_id=sid, mode="erode1"))
        emit(f"t_{tag}_dilate1", "tol", f"T_{sid}", binary_dilation(p, structure=_CROSS), dict(source_id=sid, mode="dilate1"))
        em, ed = edge_sets(p)
        pool = np.flatnonzero((em | ed).reshape(-1))
        for k in (1, 2, 4):
            for j in range(args.seeds):
                rng = np.random.default_rng(seed)
                q = p.copy()
                flat = rng.choice(pool, size=min(k, len(pool)), replace=False)
                q.ravel()[flat] = ~q.ravel()[flat]
                emit(f"t_{tag}_k{k}s{j}", "tol", f"T_{sid}", q, dict(source_id=sid, flip_k=k, seed=seed))
                seed += 1
    _save_manifest(manifest, input_dir)
    print(f"公差掃描輸入完成 → {input_dir}：{len(manifest)} 筆"
          f"（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


# ---------------------------------------------------------------- select-pick（開發機，零 HFSS）
def select_pick(args):
    """從既有輸入夾挑指定 pattern 組成新批次（id 原樣保留）——交叉驗證/重驗用。
    --items "來源夾:id,來源夾:id,..."（來源夾=DATASET_PATH 下的 *_input 夾名）。"""
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in args.items.split(","):
        src_name, pid = item.strip().split(":")
        f = _dir(src_name).joinpath(f"{pid}.pt")
        if not f.exists():
            raise SystemExit(f"找不到 {f}")
        pat = np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))
        manifest.append(dict(id=pid, kind="verify", family=src_name, removed_px=0,
                             source_input=src_name, **piece_stats(pat)))
    _save_manifest(manifest, input_dir)
    print(f"驗證批次完成 → {input_dir}：{len(manifest)} 筆（id 原樣保留,report 可直接對照原 store）")


# ---------------------------------------------------------------- select-repeat（開發機，零 HFSS）
def select_repeat(args):
    """同一 pattern 重複模擬 N 次 → 量 HFSS 可重複性/隨機性（模擬雜訊分布）。
    背景：跨 session 單點證據＝完全決定性（R8 b00_ref ≡ R7 p03_d3,diff 0.00）;本批次量
    「同一 HFSS instance 內連續 N 解」的分布（mesh/自適應收斂有無抖動）。⚠ 本批共享一個
    instance;要量跨 instance 變異,改 --input/--store 再跑一批即可（兩批各自開新 HFSS）。"""
    src = _dir(args.source_input)
    src_pt = src.joinpath(f"{args.id}.pt")
    if not src_pt.exists():
        raise SystemExit(f"找不到 {src_pt}——確認 --source-input / --id。")
    pat = np.asarray(torch.load(str(src_pt), weights_only=True)).reshape(25, 25) > 0.5
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k in range(args.n):
        pid = f"r{k:02d}_rep"
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))
        manifest.append(dict(id=pid, kind="repeat", family=args.id, removed_px=0,
                             source_id=args.id, **piece_stats(pat)))
    _save_manifest(manifest, input_dir)
    print(f"repeat 批次完成 → {input_dir}：{args.id} × {args.n} 次"
          f"（估 {args.n * 3} 分 ≈ {args.n * 3 / 60:.1f} hr；report 看各次 wm/rad 的散布）")


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

    out = Path(args.out if args.out else f"_dedust_{args.store}").resolve()   # 預設每 store 一個工作目錄
    #! 隔離理由 (2026-07-06)：CSV 檔名只含批內編號,跨批共用目錄+匯出 silently 失敗=讀到上一批殘留
    #  (verify-discrete 實際踩到);連同 single_port.py 的「匯出前刪舊檔」雙保險。
    sim = SinglePortRadSimulator(record_path=str(out), sweep_type=args.sweep)
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
            entry.update(oob_metrics(resp))               # 帶外選擇性 (2026-07-07 起隨批入檔)
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
    print("| id | 拔px | 池wm | SM wm | HFSS S11/Gain/worst | Δworst vs 基準 | rad餘裕 | 帶外惡度 | 分 |")
    print("|---|---|---|---|---|---|---|---|---|")
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
        ob = f"{r['oob_bad']:.1f}" if "oob_bad" in r else "—"
        print(f"| {m['id']} | {m.get('removed_px', 0)} | {pool} | {sm} | {wmtx} | {dv} | {radm} | {ob} | {t} |")

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

    s = sub.add_parser("select-r9", help="R9 過夜批次：池頂端重驗(T/N/M)＋乾淨前緣探索(E 鄰域/G SM導引/S 對稱)")
    s.add_argument("--input", default="dedust_r9_input")
    s.add_argument("--pass-thr", type=float, default=0.0, help="帳面達標門檻 (預設 0)")
    s.add_argument("--near-lo", type=float, default=-1.0, help="近標帶下緣 (預設 -1)")
    s.add_argument("--near", type=int, default=12, help="近標帶抽樣數 (預設 12)")
    s.add_argument("--cal-lo", type=float, default=-3.0, help="校正深帶下緣 (預設 -3)")
    s.add_argument("--cal", type=int, default=12, help="校正深帶抽樣數 (預設 12)")
    s.add_argument("--anchors", type=int, default=6, help="探索錨點數=top-300 家族代表前 N (預設 6)")
    s.add_argument("--explore-seeds", type=int, default=3, help="E 臂每(錨點,k)組合 seed 數 (預設 3 → 6×4×3=72)")
    s.add_argument("--guided", type=int, default=32, help="G 臂 SM 導引取樣數 (預設 32)")
    s.set_defaults(fn=select_r9)

    s = sub.add_parser("select-refine1", help="精修 phase-1：s05 保對稱鄰域 + 對稱化救援推廣 + g24 鄰域 (盲階段)")
    s.add_argument("--source-input", default="dedust_r9_input")
    s.add_argument("--seeds", type=int, default=6, help="W/Y 臂每 k 的 seed 數 (預設 6 → 18+4+18=40 筆)")
    s.add_argument("--input", default="dedust_ref1_input")
    s.set_defaults(fn=select_refine1)

    s = sub.add_parser("select-refine2", help="精修 phase-2：w17 密掃 + 遮蔽圖知情編輯 + 重錨 SM 導引 + y05 線")
    s.add_argument("--ref1-input", default="dedust_ref1_input", help="取 w17_k8/y05_k2 錨點")
    s.add_argument("--sm", default="sm_reanchor.pth", help="C 臂導引用的 SM 權重 (DATASET_PATH 下)")
    s.add_argument("--seeds", type=int, default=12, help="A/B 臂每 k 的 seed 數 (預設 12 → A48+B36)")
    s.add_argument("--guided", type=int, default=32, help="C 臂取樣數")
    s.add_argument("--input", default="dedust_ref2_input")
    s.set_defaults(fn=select_refine2)

    s = sub.add_parser("select-refine3", help="R11 ref3：穩健盲掃(occl2低成本區) + 字典序SM導引(含帶外) + add_block 組數階梯")
    s.add_argument("--input", default="dedust_ref3_input")
    s.add_argument("--ref2-input", default="dedust_ref2_input", help="錨點來源 (c21_sm/a15_k4)")
    s.add_argument("--sm", default="sm_reanchor3.pth", help="B 臂導引權重 (DATASET_PATH 下)")
    s.add_argument("--seeds", type=int, default=12, help="A 臂每 (錨點,k) 幾個 seed")
    s.add_argument("--guided", type=int, default=32, help="B 臂每錨點取幾個")
    s.add_argument("--wing", type=int, default=10, help="C 臂每錨點翼對(5塊)幾個")
    s.add_argument("--center", type=int, default=8, help="C 臂每錨點頂中央(4塊)幾個")
    s.add_argument("--asym", type=int, default=5, help="C 臂每錨點破對稱(2+2)幾個")
    s.add_argument("--six", type=int, default=2, help="C 臂每錨點 6 塊試點幾個")
    s.set_defaults(fn=select_refine3)

    s = sub.add_parser("select-wide", help="R11 wide：冠軍高原半徑測繪(遠距k) + 對稱必要性(不再對稱化) + SM 遠距導引")
    s.add_argument("--input", default="dedust_wide_input")
    s.add_argument("--ref2-input", default="dedust_ref2_input")
    s.add_argument("--sm", default="sm_reanchor3.pth")
    s.add_argument("--seeds", type=int, default=4, help="W/X 臂每 (錨點,k) 幾個 seed")
    s.add_argument("--guided", type=int, default=24, help="Y 臂每錨點取幾個")
    s.set_defaults(fn=select_wide)

    s = sub.add_parser("check-dup", help="發車前查重：批內 + 對歷史輸入夾交叉（exit 1=有重複）")
    s.add_argument("--input", required=True)
    s.set_defaults(fn=check_dup)

    s = sub.add_parser("select-occlude", help="物理遮蔽掃描：錨點 5×5 區塊逐一清空 → 真空間重要度圖 (R10)")
    s.add_argument("--source-input", default="dedust_r9_input", help="來源輸入夾（取 --ids 的 .pt）")
    s.add_argument("--ids", default="s05_1050,g24_sm", help="錨點 id,逗號分隔")
    s.add_argument("--input", default="dedust_occl_input")
    s.set_defaults(fn=select_occlude)

    s = sub.add_parser("select-tolerance", help="製造公差掃描：erode/dilate + 邊緣隨機缺陷 (冠軍工程餘裕)")
    s.add_argument("--ids", default="c21_sm,w17_k8", help="冠軍 id,逗號分隔")
    s.add_argument("--source-input", default="dedust_ref2_input", help="來源輸入夾 (w17 要用 dedust_ref1_input 時逐 id 不支援,放同夾或先 select-pick)")
    s.add_argument("--seeds", type=int, default=6, help="每 k 的 seed 數 (預設 6 → 每冠軍 2+18=20 筆)")
    s.add_argument("--input", default="dedust_tol_input")
    s.set_defaults(fn=select_tolerance)

    s = sub.add_parser("select-pick", help="從既有輸入夾挑指定 pattern 組新批次（交叉驗證/重驗用）")
    s.add_argument("--items", required=True, help='"來源夾:id,來源夾:id,..."')
    s.add_argument("--input", default="dedust_verify_input")
    s.set_defaults(fn=select_pick)

    s = sub.add_parser("select-repeat", help="同一 pattern 重複 N 次 → 量 HFSS 可重複性（模擬雜訊分布）")
    s.add_argument("--source-input", default="dedust_r9_input", help="來源輸入夾（取 --id 的 .pt）")
    s.add_argument("--id", default="s05_1050", help="要重複的 pattern id（預設可製造紀錄 s05）")
    s.add_argument("--n", type=int, default=30, help="重複次數 (預設 30 ≈ 1.5hr)")
    s.add_argument("--input", default="dedust_repeat_input")
    s.set_defaults(fn=select_repeat)

    s = sub.add_parser("sm-screen", help="sm_harvest.pth 預測預篩（零 HFSS）")
    s.add_argument("--input", default=DEFAULT_INPUT)
    s.add_argument("--config", default=DEFAULT_CFG)
    s.set_defaults(fn=sm_screen)

    s = sub.add_parser("run", help="正式機：HFSS 驗證 manifest 所有 pattern（可中斷續跑）")
    s.add_argument("--input", default=DEFAULT_INPUT)
    s.add_argument("--store", default=DEFAULT_STORE, help="結果夾名（DATASET_PATH 下）")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--out", default=None, help="HFSS 工作目錄（正式機本地碟;預設 _dedust_<store>,批次間隔離防殘留污染）")
    s.add_argument("--sweep", default="Interpolating", choices=["Interpolating", "Discrete", "Fast"],
                   help="掃頻演算法（Discrete=17 點逐點硬解,慢但每點真解;掃頻法交叉驗證用）")
    s.set_defaults(fn=run)

    s = sub.add_parser("report", help="匯總表（貼 round 檔 §4）")
    s.add_argument("--input", default=DEFAULT_INPUT)
    s.add_argument("--store", default=DEFAULT_STORE)
    s.set_defaults(fn=report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
