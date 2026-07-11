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


def add_bridge(p, comp_rank: int = 1, pair_rank: int = 0, feed=FEED):
    """搭橋：把第 comp_rank 大的「非饋電」組件用 L 形 1px 金屬橋接到饋電主件（材料只增不減）。
    pair_rank 選第 n 近的像素對（0=最短橋）——懸浮件功能性因果測試（probes 批②）。
    無該組件回 None。決定性（掃描序定序）。"""
    from scipy.ndimage import label
    p = (np.asarray(p).reshape(25, 25) > 0.5).copy()
    lab, n = label(p, structure=_CROSS)
    fid = int(lab[feed])
    sizes = [(int((lab == k).sum()), k) for k in range(1, n + 1) if k != fid]
    sizes.sort(reverse=True)
    if comp_rank > len(sizes):
        return None
    tid = sizes[comp_rank - 1][1]
    A = np.argwhere(lab == fid)
    B = np.argwhere(lab == tid)
    pairs = sorted(((abs(a[0] - b[0]) + abs(a[1] - b[1]), tuple(a), tuple(b))
                    for a in A for b in B))
    if pair_rank >= len(pairs):
        return None
    _d, (ar, ac), (br, bc) = pairs[pair_rank]
    r, c = br, bc
    while r != ar:                                # 先走列
        r += 1 if ar > r else -1
        p[r, c] = True
    while c != ac:                                # 再走欄
        c += 1 if ac > c else -1
        p[r, c] = True
    return p


def resize_component(p, which: str, delta: int, min_size: int = 4, feed=FEED):
    """組件級大小調整（Ricky「像素級→組件級」方向,R14 起的主力算子）：
    which="main"（feed 主件）或 "wings"（全部非 feed 組件,成組縮放保對稱）;
    delta=±N（邊界 grow/shrink N 圈,4-連通形態學）。grow 保持組件獨立（不併件,留 ≥1px 間隙）;
    shrink 後任一組件 <min_size 或 feed 丟失 → 回 None。純形態學、決定性。"""
    from scipy.ndimage import label, binary_dilation, binary_erosion
    p = (np.asarray(p).reshape(25, 25) > 0.5).copy()
    lab, n = label(p, structure=_CROSS)
    fid = int(lab[feed])
    if fid == 0:
        return None
    target = (lab == fid) if which == "main" else (p & (lab != fid))
    others = p & ~target
    if delta > 0:
        for _ in range(delta):
            grown = binary_dilation(target, structure=_CROSS)
            forbidden = binary_dilation(others, structure=_CROSS)   # 與其他組件保 ≥1px 間隙
            target = grown & ~forbidden
    else:
        for _ in range(-delta):
            target = binary_erosion(target, structure=_CROSS)
    out = others | target
    out[feed] = True
    lab2, n2 = label(out, structure=_CROSS)
    sizes = np.bincount(lab2.ravel())[1:]
    if n2 == 0 or (sizes < min_size).any():
        return None                                   # 縮出碎片/消失 → 無效
    if which == "wings" and n2 != n:
        return None                                   # 翼消失或併件 → 拓撲變了,不是尺寸調整
    return out


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


def select_probes(args):
    """probes＋帶外批（判準預註冊於 scratch「下一批規格」塊,2026-07-07）：
      N c25 公證 ×6 —— ref3 新王候選（+0.22/+0.34,5 塊翼對）紀錄級鐵則重複驗證
      P1 全對稱冠軍 ×8 —— symmetrize(12):「中央帶=匹配旋鈕、對稱度=rad/帶外旋鈕」因果測試
      P2 搭橋 ×6 —— 懸浮件串接主件（材料只增）:「懸浮=功能非缺陷」反向因果（孤島論點壓陣）
      P3 t07 構造化 ×5 —— 論文圖 4-4 pattern 的對稱化/除塵改造（第二山頭?）
      P4 底緣精修 ×32 —— 壓帶外機理主臂:只翻底緣（row≥20,左半+中央）1-2px vs 全域邊緣對照"""
    ref2 = _dir(args.ref2_input)
    champs = ["c21_sm", "a00_k2", "b11_k2", "c10_sm", "c18_sm", "c17_sm", "a15_k4", "a11_k2"]
    P = {cid: np.asarray(torch.load(str(ref2.joinpath(f"{cid}.pt")), weights_only=True)).reshape(25, 25) > 0.5
         for cid in champs}
    c25 = np.asarray(torch.load(str(_dir(args.ref3_input).joinpath("c25_a15w10_2_22.pt")),
                                weights_only=True)).reshape(25, 25) > 0.5
    t07 = np.asarray(torch.load(str(_dir("dedust_r9_input").joinpath("t07_top.pt")),
                                weights_only=True)).reshape(25, 25) > 0.5
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra):
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))

    for k in range(6):                            # N 公證
        emit(f"n{k:02d}_c25rep", "notarize", "N_c25", c25, dict(source_id="c25_a15w10_2_22"))
    for cid in champs:                            # P1 全對稱
        emit(f"p{cid[:3]}_full", "fullsym", "P1_fullsym", symmetrize(P[cid], 12),
             dict(anchor=cid, sym="full12"))
    n = 0                                         # P2 搭橋
    for src, pat0, ranks in (("c21_sm", P["c21_sm"], ((1, 0), (1, 40))),
                             ("b11_k2", P["b11_k2"], ((1, 0), (1, 40))),
                             ("t07_top", t07, ((1, 0), (2, 0)))):
        for cr, pr in ranks:
            q = add_bridge(pat0, comp_rank=cr, pair_rank=pr)
            if q is None:
                continue
            emit(f"q{n:02d}_{src[:3]}br", "bridge", "P2_bridge", q,
                 dict(anchor=src, comp_rank=cr, pair_rank=pr))
            n += 1
    variants = [("sym1050", symmetrize(t07, 10)), ("sym12", symmetrize(t07, 12))]
    dd, _ = strip_small(t07, 4)                   # P3 t07 構造化
    variants += [("strip4", _ensure_feed_pad(dd)), ("strip_sym1050", symmetrize(dd, 10))]
    # strip_sym12 與 sym12 重複（全鏡射把除塵差異蓋掉）——check-dup 實抓,刪

    for tag, q in variants:
        emit(f"t_{tag}", "construct", "P3_t07", q, dict(anchor="t07_top", variant=tag))
    n, seed = 0, 16000                            # P4 底緣精修 vs 全域邊緣對照
    for aid in ("c21_sm", "a15_k4"):
        pat0 = P[aid]
        em, ed = edge_sets(pat0)
        half = np.zeros((25, 25), bool)
        half[:, :13] = True                       # 左半+中央（再對稱化不會蓋掉編輯）
        lo = (em | ed) & half
        lo[:20] = False                           # 底緣帶 row>=20
        gl = (em | ed) & half
        for zone, mask, ks, seeds in (("edgelo", lo, (1, 2), 5), ("edgeglob", gl, (1, 2), 3)):
            cand = np.argwhere(mask)
            for k in ks:
                for _ in range(seeds):
                    rng = np.random.default_rng(seed)
                    q = pat0.copy()
                    for r, c in cand[rng.choice(len(cand), size=min(k, len(cand)), replace=False)]:
                        q[r, c] = ~q[r, c]
                    q[FEED] = True
                    q = symmetrize(q, 10)
                    emit(f"e{n:02d}_{aid[:3]}{zone[4:]}k{k}", zone, f"P4_{aid}", q,
                         dict(anchor=aid, flip_k=k, seed=seed, zone=zone))
                    n += 1
                    seed += 1

    _save_manifest(manifest, input_dir)
    cnt = {}
    for m in manifest:
        cnt[m["kind"]] = cnt.get(m["kind"], 0) + 1
    print(f"probes 輸入完成 → {input_dir}：{cnt} 共 {len(manifest)}"
          f"（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


def select_ablate(args):
    """元件消融實驗（Ricky）——量每個組件的貢獻:full → 去掉各非饋電組件 → 累積組合。
    feed 組件(下主件)永遠保留(饋電必需);對每個冠軍列出「去掉哪些翼」的所有組合 HFSS 實測。
    判準:單翼移除 Δwm/Δrad/Δoob → 各翼貢獻量;與搭橋實驗(加材料連接)互補=移除材料的因果。
    --items "來源夾:id,...";公證 ×2(消融解是新 pattern,鐵則)。"""
    from scipy.ndimage import label
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra):
        pat = (np.asarray(pat).reshape(25, 25) > 0.5).copy()
        pat[FEED] = True
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))

    for item in args.items.split(","):
        src_name, pid = item.strip().split(":")
        p = np.asarray(torch.load(str(_dir(src_name).joinpath(f"{pid}.pt")), weights_only=True)).reshape(25, 25) > 0.5
        lab, n = label(p, structure=_CROSS)
        fid = int(lab[FEED])
        wings = sorted((k for k in range(1, n + 1) if k != fid),
                       key=lambda k: -int((lab == k).sum()))        # 大翼在前,序穩定
        tag = pid[:4]
        for r in range(2):                                          # full 公證基準
            emit(f"z{tag}_full{r}", "notarize", f"AB_{pid}", p, dict(source_id=pid, ablate="none"))
        emit(f"z{tag}_botonly", "ablate", f"AB_{pid}", (lab == fid),
             dict(source_id=pid, ablate="wings_removed", kept="feed_only"))
        for wi, wk in enumerate(wings):                             # 下主件+單翼（各翼獨立貢獻）
            emit(f"z{tag}_botw{wi}", "ablate", f"AB_{pid}", (lab == fid) | (lab == wk),
                 dict(source_id=pid, ablate=f"keep_wing{wi}", wing_px=int((lab == wk).sum())))
    _save_manifest(manifest, input_dir)
    print(f"消融輸入完成 → {input_dir}：{len(manifest)} 筆"
          f"（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


def select_resize(args):
    """R14 組件尺寸掃描——各錨點 × {main,wings} × delta∈{−2,−1,+1,+2}:量「組件尺寸→三標/帶外」
    的響應曲線（組件級軸的第一批;與消融互補:消融=有無,這裡=大小）。純形態學決定性、歷史排除。"""
    ref = {"x00_c21k2": "dedust_wide_input", "c21_sm": "dedust_ref2_input",
           "a15_k4": "dedust_ref2_input", "c25_a15w10_2_22": "dedust_ref3_input"}
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    hist = set()
    for fol in HISTORY_INPUTS:
        d = DATASET_PATH.joinpath(fol)
        if not d.joinpath("manifest.json").exists():
            continue
        for m in json.load(open(str(d.joinpath("manifest.json")), encoding="utf-8")):
            f = d.joinpath(f"{m['id']}.pt")
            if f.exists():
                hist.add((np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes())
    n_skip = 0
    for pid, fol in ref.items():
        p = np.asarray(torch.load(str(_dir(fol).joinpath(f"{pid}.pt")), weights_only=True)).reshape(25, 25) > 0.5
        tag = pid[:4]
        for which in ("main", "wings"):
            for delta in (-2, -1, 1, 2):
                q = resize_component(p, which, delta)
                if q is None:
                    continue
                if q.tobytes() in hist:
                    n_skip += 1
                    continue
                st = piece_stats(q)
                if st["n_1px"] > 0:
                    continue
                sign = "p" if delta > 0 else "m"
                manifest.append(dict(id=f"r{tag}_{which[0]}{sign}{abs(delta)}", kind="resize",
                                     family=f"R_{pid}", removed_px=0, **st,
                                     source_id=pid, which=which, delta=delta))
                torch.save(torch.tensor(q, dtype=torch.float32),
                           str(input_dir.joinpath(f"{manifest[-1]['id']}.pt")))
    _save_manifest(manifest, input_dir)
    print(f"resize 輸入完成 → {input_dir}：{len(manifest)} 筆（跳過歷史 {n_skip};"
          f"估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


def select_blocks(args):
    """R13 組數階梯系統對比——固定錨點,系統掃 3/4/5/6 塊拓撲,答「組數 vs 三標+選擇性」。
    錨點=c21/a15（3 塊冠軍,c25 母體）;每目標組數用 add_block 掃位×尺寸 → 對稱化 → SM 篩 top-K。
    3 塊=錨點本身（baseline,含缺陷 k1×4 供穩健對照）。判準:各組數 best wm/rad/oob 曲線 + 邊際報酬。"""
    ref2 = _dir(args.ref2_input)
    anchors = {aid: np.asarray(torch.load(str(ref2.joinpath(f"{aid}.pt")), weights_only=True)).reshape(25, 25) > 0.5
               for aid in ("c21_sm", "a15_k4")}
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    hist = set()                                          # 已跑過的 pattern(避免與 ref3 C 臂等重跑)
    for fol in ("dedust_ref3_input", "dedust_wide_input", "dedust_ref2_input"):
        d = DATASET_PATH.joinpath(fol)
        if not d.joinpath("manifest.json").exists():
            continue
        for m in json.load(open(str(d.joinpath("manifest.json")), encoding="utf-8")):
            f = d.joinpath(f"{m['id']}.pt")
            if f.exists():
                hist.add((np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes())
    skipped = [0]

    def emit(pid, kind, family, pat, extra):
        if (np.asarray(pat).reshape(-1) > 0.5).tobytes() in hist:
            skipped[0] += 1
            return False                                  # 歷史已跑,跳過(分析時從原 store 併)
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))
        return True

    cfg = load_config(DEFAULT_CFG)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    from antenna.zoo import SURROGATES
    cache = os.path.join(REPO, "tmp", "dedust")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
    sm.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
    sm.model.eval()

    def sm_wm(pat):
        with torch.no_grad():
            pred = sm.model(torch.tensor(pat, dtype=torch.float32).flatten())
        return float(worst_margin(pred, labels, cfg.targets)[0])

    def scan(pat0, want_comp, mirror, cols, sizes):
        """掃可放位×尺寸 → 拓撲驗證後的互異 pattern 清單（按 pattern bytes 去重）。"""
        outs, seen = [], set()
        for h, w in sizes:
            colrange = cols(w)
            for r in range(0, 26 - h):
                for c in colrange:
                    q = add_block(pat0, r, c, h, w)
                    if q is None:
                        continue
                    qq = symmetrize(q, 10) if mirror else q
                    if piece_stats(qq)["n_comp"] != want_comp or piece_stats(qq)["n_1px"] > 0:
                        continue
                    key = qq.tobytes()
                    if key not in seen:
                        seen.add(key)
                        outs.append(qq)
        return outs

    SIZES = ((2, 2), (2, 3), (3, 3), (2, 4))
    n = 0
    for aid, pat0 in anchors.items():
        tag = aid[:3]
        # 3 塊 baseline（錨點本人,穩健對照從 crown/tol 併,這裡只補缺陷新樣本）
        n += emit(f"g{n:02d}_{tag}_3base", "block3", f"B_{aid}", pat0, dict(anchor=aid, ncomp_target=3))
        em, ed = edge_sets(pat0)
        epool = np.flatnonzero((em | ed).reshape(-1))
        seed = 40000 + hash(aid) % 1000
        for j in range(4):
            rng = np.random.default_rng(seed)
            q = pat0.copy()
            q.ravel()[rng.choice(epool, size=1, replace=False)] ^= True
            n += emit(f"g{n:02d}_{tag}_3d{j}", "block3", f"B_{aid}", q, dict(anchor=aid, ncomp_target=3, flip_k=1, seed=seed))
            seed += 1
        # 4/5/6 塊：掃位 → SM 篩 top-K
        for ncomp, mirror, cols in ((4, False, lambda w: (12 - (w - 1) // 2,)),
                                    (5, True, lambda w: range(0, 10 - w)),
                                    (6, True, lambda w: range(0, 10 - w))):
            cands = scan(pat0, ncomp, mirror, cols, SIZES)
            if ncomp == 6:                                    # 6 塊=先蓋中央塊再翼對
                base4 = scan(pat0, 4, False, lambda w: (12 - (w - 1) // 2,), ((3, 3), (2, 3)))
                cands = []
                for b4 in base4[:3]:
                    cands += scan(b4, 6, True, lambda w: range(0, 10 - w), ((2, 2), (3, 3)))
            ranked = sorted(((sm_wm(q), q) for q in cands), key=lambda t: -t[0])
            got = 0
            for _w, q in ranked:
                if got >= args.per_topo:
                    break
                if emit(f"g{n:02d}_{tag}_{ncomp}b", "blockN", f"B_{aid}", q,
                        dict(anchor=aid, ncomp_target=ncomp, sm_pick_wm=_r(_w))):
                    n += 1
                    got += 1

    _save_manifest(manifest, input_dir)
    cnt = {}
    for m in manifest:
        cnt[m["ncomp_target"]] = cnt.get(m["ncomp_target"], 0) + 1
    print(f"blocks 輸入完成 → {input_dir}：組數分布 {cnt} 共 {len(manifest)}"
          f"（跳過歷史已跑 {skipped[0]} 筆;估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


def select_r15(args):
    """R15 對照組實驗（push-button vs 工具箱,同一組件空間、同 SM v4、同 HFSS 驗證預算）：
      G 臂=GA（MWSCAS 式全代理演化,fitness=SM 預測 wm − 0.02·oob,盲搜）→ 互異 top-50 驗證
      I 臂=知情（區調整圖低成本區放塊+R13 甜蜜尺寸+SM 字典序頂帶按 oob）→ 50 驗證
      N 臂=空間內均勻隨機 20（隔離「空間 vs 搜尋」的貢獻）
    組件空間（三臂同一個）:錨點∈{x00,c21,a15,c25} × 0-3 塊(r∈0..22,c∈0..11,h,w∈{2,3}) → symmetrize(10)。
    判準預註冊於 round-15;歷史已跑自動排除。輸出兩個輸入夾（G+N→37 / I→218）。"""
    ANCH = {"x00": ("dedust_wide_input", "x00_c21k2"), "c21": ("dedust_ref2_input", "c21_sm"),
            "a15": ("dedust_ref2_input", "a15_k4"), "c25": ("dedust_ref3_input", "c25_a15w10_2_22")}
    A = {k: np.asarray(torch.load(str(_dir(f).joinpath(i + ".pt")), weights_only=True)).reshape(25, 25) > 0.5
         for k, (f, i) in ANCH.items()}
    hist = set()
    for fol in HISTORY_INPUTS:
        d = DATASET_PATH.joinpath(fol)
        if not d.joinpath("manifest.json").exists():
            continue
        for m in json.load(open(str(d.joinpath("manifest.json")), encoding="utf-8")):
            f = d.joinpath(m["id"] + ".pt")
            if f.exists():
                hist.add((np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes())

    cfg = load_config(DEFAULT_CFG)
    labels = PORT_SPECS[cfg.port]["labels"]
    from antenna.zoo import SURROGATES
    cache = os.path.join(REPO, "tmp", "dedust")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 625, (len(labels), sum(cfg.targets[labels[0]]["width"])))
    sm.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
    sm.model.eval()

    # 每錨點預掃合法放塊位置（單塊可放;組合衝突於 build 時跳過）——避免隨機基因塌回錨點本體
    VALID = {}
    for aid, base in A.items():
        vs = []
        for h in (2, 3, 4):
            for w in (2, 3, 4):
                for r in range(0, 23 - h + 1):
                    for c in range(0, 12):
                        if add_block(base, r, c, h, w) is not None:
                            vs.append((r, c, h, w))
        VALID[aid] = vs
    print("合法放塊位:", {k: len(v) for k, v in VALID.items()})

    def build(gen):
        aid, idxs = gen
        p = A[aid]
        for ix in idxs:
            r, c, h, w = VALID[aid][ix % len(VALID[aid])]
            q = add_block(p, r, c, h, w)
            if q is not None:
                p = q
        p = symmetrize(p, 10)
        return p if piece_stats(p)["n_1px"] == 0 else None

    _score_cache = {}

    def score(pat):
        key = pat.tobytes()
        if key not in _score_cache:
            with torch.no_grad():
                pred = sm.model(torch.tensor(pat, dtype=torch.float32).flatten())
            w, _ = worst_margin(pred, labels, cfg.targets)
            _score_cache[key] = (float(w), oob_metrics(pred.detach().cpu().numpy())["oob_bad"])
        return _score_cache[key]

    def rnd_genome(rng):
        aid = list(A)[int(rng.integers(0, 4))]
        return (aid, [int(rng.integers(0, len(VALID[aid]))) for _ in range(int(rng.integers(0, 4)))])

    # ---- G 臂：GA（決定性,全 SM）----
    rng = np.random.default_rng(15000)
    pop = [rnd_genome(rng) for _ in range(args.pop)]

    def fit(gen):
        pat = build(gen)
        if pat is None:
            return -99.0, None
        w, ob = score(pat)
        return w - 0.02 * ob, pat

    seen_best = {}
    for _g in range(args.gens):
        scored = []
        for gen in pop:
            f, pat = fit(gen)
            scored.append((f, gen))
            if pat is not None:
                seen_best[pat.tobytes()] = (f, pat)
        scored.sort(key=lambda t: -t[0])
        elite = [gen for _f, gen in scored[:8]]
        nxt = list(elite)
        while len(nxt) < args.pop:
            def pick():
                cand = [scored[int(rng.integers(0, len(scored)))] for _ in range(3)]
                return max(cand, key=lambda t: t[0])[1]
            pa, pb = pick(), pick()
            blocks = [b for b in (pa[1] + pb[1]) if rng.random() < 0.5][:3]
            child = (pa[0] if rng.random() < 0.8 else pb[0], blocks)
            if rng.random() < 0.6:
                m = rng.random()
                bl = list(child[1])
                nv = len(VALID[child[0]])
                if m < 0.3 and bl:
                    i = int(rng.integers(0, len(bl)))
                    bl[i] = int(np.clip(bl[i] + rng.integers(-40, 41), 0, nv - 1))
                elif m < 0.5 and len(bl) < 3:
                    bl.append(int(rng.integers(0, nv)))
                elif m < 0.7 and bl:
                    bl.pop(int(rng.integers(0, len(bl))))
                elif m < 0.85:
                    child = (list(A)[int(rng.integers(0, 4))], bl)
                child = (child[0], bl)
            nxt.append(child)
        pop = nxt
    ga_rank = sorted(seen_best.values(), key=lambda t: -t[0])
    print("GA: 演化評估互異 pattern " + str(len(seen_best)) + ", 最高 fitness " + format(ga_rank[0][0], "+.2f"))

    def diverse_pick(ranked_pats, k):
        out = []
        for pat in ranked_pats:
            if len(out) >= k:
                break
            if pat.tobytes() in hist:
                continue
            if all(np.count_nonzero(pat != q) > 8 for q in out):
                out.append(pat)
        return out

    ga_pick = diverse_pick([p for _f, p in ga_rank], args.verify)

    # ---- I 臂：知情 ----
    cands = []
    for aid in ("x00", "c25", "c21", "a15"):
        g2 = np.random.default_rng(16000 + sum(ord(ch) for ch in aid))
        pool = list(range(len(VALID[aid])))
        for ix in pool:                                      # 單塊全枚舉（全空間）
            pat = build((aid, [ix]))
            if pat is None:
                continue
            zone = 1 if VALID[aid][ix][0] <= 8 else 0        # 知識:低成本帶加一層
            w, ob = score(pat)
            cands.append((zone, w, ob, pat))
        for _n in range(400):                                # 雙/三塊抽樣
            idxs = [pool[int(g2.integers(0, len(pool)))] for _ in range(1 + int(g2.integers(1, 3)))]
            pat = build((aid, idxs))
            if pat is None:
                continue
            zone = 1 if all(VALID[aid][ix % len(VALID[aid])][0] <= 8 for ix in idxs) else 0
            w, ob = score(pat)
            cands.append((zone, w, ob, pat))
    top = max(c[1] for c in cands) - 0.36
    # 知識分層:①低成本帶 ②SM 頂帶(按 oob) ③其餘按 wm——字典序
    cands.sort(key=lambda c: (-c[0], (0, c[2]) if c[1] >= top else (1, -c[1])))
    inf_pick = diverse_pick([p for _z, _w, _o, p in cands], args.verify)
    print("知情臂: 候選 " + str(len(cands)) + ", 揀 " + str(len(inf_pick)))

    k = min(len(ga_pick), len(inf_pick))                     # 公平同額（判準要求同驗證預算）
    ga_pick, inf_pick = ga_pick[:k], inf_pick[:k]
    print("同額預算: 每臂 " + str(k) + " 筆")

    # ---- N 臂：空間內均勻隨機 ----
    rng2 = np.random.default_rng(17000)
    n_pick = []
    while len(n_pick) < args.random_n:
        pat = build(rnd_genome(rng2))
        if pat is None or pat.tobytes() in hist:
            continue
        if all(np.count_nonzero(pat != q) > 8 for q in n_pick):
            n_pick.append(pat)

    for fol, groups in ((args.input_ga, (("g", ga_pick), ("n", n_pick))),
                        (args.input_inf, (("i", inf_pick),))):
        d = _dir(fol)
        d.mkdir(parents=True, exist_ok=True)
        man = []
        for tag, picks in groups:
            for k, pat in enumerate(picks):
                w, ob = score(pat)
                pid = tag + format(k, "02d") + "_r15"
                kindmap = {"g": "ga", "i": "informed", "n": "randspace"}
                man.append(dict(id=pid, kind=kindmap[tag], family="R15_" + tag.upper(),
                                removed_px=0, **piece_stats(pat), sm_pick_wm=_r(w), sm_oob=_r(ob)))
                torch.save(torch.tensor(pat, dtype=torch.float32), str(d.joinpath(pid + ".pt")))
        _save_manifest(man, d)
        print("→ " + str(d) + ": " + str(len(man)) + " 筆")


def select_r15v(args):
    """R15 收尾批（37）：新王認證＋rad 救援＋理論模板探針。
      V 公證/缺陷 —— i02(+0.29 新王候選)×[公證3+缺陷4]、g16×[公證2+缺陷2]、g14×公證2（鐵則）
      R rad 救援 —— g14(+0.40/rad −0.39,史上最高帶內 margin)加第 4 塊 ×6 位置（試把 rad 拉回 0）
      T 理論模板 —— 實心矩形貼片（非池衍生種子,局部最佳討論的逃逸探針）h∈{8,10,12}×w∈{13,17} 貼底置中
    判準:V=公證一致+缺陷存活;R=任一筆 rad≥0 且 wm≥0.2=救援成功;T=任一筆 wm≥−1=新盆地線索。"""
    W = {"i02_r15": "dedust_r15inf_input", "g16_r15": "dedust_r15ga_input", "g14_r15": "dedust_r15ga_input"}
    P = {k: np.asarray(torch.load(str(_dir(f).joinpath(k + ".pt")), weights_only=True)).reshape(25, 25) > 0.5
         for k, f in W.items()}
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra):
        pat = (np.asarray(pat).reshape(25, 25) > 0.5).copy()
        pat[FEED] = True
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(pid + ".pt")))

    seed = 50000
    for pid, (n_not, n_def) in (("i02_r15", (3, 4)), ("g16_r15", (2, 2)), ("g14_r15", (2, 0))):
        p = P[pid]
        for r in range(n_not):
            emit("v" + pid[:3] + "_n" + str(r), "notarize", "V_" + pid, p, dict(source_id=pid))
        if n_def:
            em, ed = edge_sets(p)
            epool = np.flatnonzero((em | ed).reshape(-1))
            for j in range(n_def):
                rng = np.random.default_rng(seed)
                q = p.copy()
                q.ravel()[rng.choice(epool, size=1, replace=False)] ^= True
                emit("v" + pid[:3] + "_d" + str(j), "tol", "V_" + pid, q, dict(source_id=pid, flip_k=1, seed=seed))
                seed += 1

    # R: g14 rad 救援——掃第 4 塊可放位,均勻取 6
    g14 = P["g14_r15"]
    spots = []
    for h in (2, 3):
        for w in (2, 3):
            for r in range(0, 23):
                for c in range(0, 12):
                    if add_block(g14, r, c, h, w) is not None:
                        spots.append((r, c, h, w))
    idx = np.linspace(0, len(spots) - 1, min(6, len(spots))).round().astype(int)
    for k in idx:
        r, c, h, w = spots[int(k)]
        q = symmetrize(add_block(g14, r, c, h, w), 10)
        if piece_stats(q)["n_1px"] > 0:
            continue
        emit("rg14_" + str(r) + "_" + str(c), "rescue", "R_g14", q, dict(source_id="g14_r15", block_at=[r, c, h, w]))

    # T: 理論模板——實心矩形貼片貼底置中（非池衍生）
    for h in (8, 10, 12):
        for w in (13, 17):
            q = np.zeros((25, 25), bool)
            c0 = (25 - w) // 2
            q[25 - h:25, c0:c0 + w] = True
            emit("t_rect" + str(h) + "x" + str(w), "theory", "T_rect", q, dict(rect=[h, w]))
    _save_manifest(manifest, input_dir)
    print("r15v 輸入完成 → " + str(input_dir) + ": " + str(len(manifest)) + " 筆")


def select_addmap(args):
    """R16 機理臂（218）：「添加收益圖」——治 R15 的「移除成本圖≠添加收益圖」教訓。
      A 單塊全掃 —— x00 全部合法位（2×2）逐一加塊 → 每個位置的 Δwm/Δrad/Δoob 真值圖
      B 贏家塊歸因 —— g14/i02/g16 的加料組件逐一移除（含鏡射夥伴）→ 各塊貢獻
    判準:A=位置-收益圖與遮蔽圖的相關性（同/不同=知識遷移邊界的定量答案）;B=贏家是靠哪塊贏的。"""
    from scipy.ndimage import label as _label
    x00 = np.asarray(torch.load(str(_dir("dedust_wide_input").joinpath("x00_c21k2.pt")), weights_only=True)).reshape(25, 25) > 0.5
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    hist = set()
    for fol in HISTORY_INPUTS + ("dedust_r15ga_input", "dedust_r15inf_input"):
        d = DATASET_PATH.joinpath(fol)
        if not d.joinpath("manifest.json").exists():
            continue
        for m in json.load(open(str(d.joinpath("manifest.json")), encoding="utf-8")):
            f = d.joinpath(m["id"] + ".pt")
            if f.exists():
                hist.add((np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes())

    def emit(pid, kind, family, pat, extra):
        pat = (np.asarray(pat).reshape(25, 25) > 0.5).copy()
        pat[FEED] = True
        if pat.tobytes() in hist:
            return False
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(pid + ".pt")))
        return True

    c25 = np.asarray(torch.load(str(_dir("dedust_ref3_input").joinpath("c25_a15w10_2_22.pt")),
                                 weights_only=True)).reshape(25, 25) > 0.5
    ANCHORS_A = {"x00": x00, "c25": c25}
    n = 0
    single_ok = {}                                       # (aid,r,c,h) → pattern（供可加性探針配對）
    for aid, base in ANCHORS_A.items():
        for h in (2, 3):
            for r in range(0, 23 - h + 1):
                for c in range(0, 12):
                    q = add_block(base, r, c, h, h)
                    if q is None:
                        continue
                    q = symmetrize(q, 10)
                    if piece_stats(q)["n_1px"] > 0:
                        continue
                    single_ok[(aid, r, c, h)] = q
                    if emit("a" + format(n, "03d") + "_" + aid + "r" + str(r) + "c" + str(c) + "s" + str(h),
                            "addmap", "A_" + aid, q, dict(source_id=aid, block_at=[r, c, h, h])):
                        n += 1
    # C 臂:可加性探針——中帶(rows 4-11)成對放塊,測 Δ(pair) ?= Δa+Δb
    rngc = np.random.default_rng(60000)
    mid = [k for k in single_ok if 4 <= k[1] <= 11 and k[3] == 2]
    m = 0
    tried = 0
    while m < 10 and tried < 200:
        tried += 1
        ka = mid[int(rngc.integers(0, len(mid)))]
        kb = mid[int(rngc.integers(0, len(mid)))]
        if ka[0] != kb[0] or ka == kb:
            continue
        aid = ka[0]
        base = ANCHORS_A[aid]
        q = add_block(base, ka[1], ka[2], 2, 2)
        if q is None:
            continue
        q = add_block(q, kb[1], kb[2], 2, 2)
        if q is None:
            continue
        q = symmetrize(q, 10)
        if piece_stats(q)["n_1px"] > 0:
            continue
        if emit("c" + format(m, "02d") + "_" + aid + "pair", "addpair", "C_" + aid, q,
                dict(source_id=aid, block_a=list(ka[1:]), block_b=list(kb[1:]))):
            m += 1

    ANCH = {"g14_r15": ("dedust_r15ga_input", "x00_c21k2", "dedust_wide_input"),
            "g16_r15": ("dedust_r15ga_input", "c21_sm", "dedust_ref2_input"),
            "i02_r15": ("dedust_r15inf_input", "c25_a15w10_2_22", "dedust_ref3_input")}
    m = 0
    for pid, (fol, aid, afol) in ANCH.items():
        p = np.asarray(torch.load(str(_dir(fol).joinpath(pid + ".pt")), weights_only=True)).reshape(25, 25) > 0.5
        base = np.asarray(torch.load(str(_dir(afol).joinpath(aid + ".pt")), weights_only=True)).reshape(25, 25) > 0.5
        diff = p & ~base
        lab, k = _label(diff, structure=_CROSS)
        groups = {}
        for g in range(1, k + 1):                       # 鏡射夥伴併一組（col→24−col）
            cells = np.argwhere(lab == g)
            key = tuple(sorted(map(tuple, np.vstack([cells, np.column_stack([cells[:, 0], 24 - cells[:, 1]])]).tolist())))
            groups.setdefault(key, []).append(g)
        for gi, gs in enumerate(groups.values()):
            q = p.copy()
            for g in gs:
                q[lab == g] = False
            if emit("b" + format(m, "02d") + "_" + pid[:3] + "g" + str(gi), "blockablate", "B_" + pid,
                    q, dict(source_id=pid, group=gi, removed=int(sum((lab == g).sum() for g in gs)))):
                m += 1
    _save_manifest(manifest, input_dir)
    print("addmap 輸入完成 → " + str(input_dir) + ": " + str(len(manifest)) + " 筆")


def _trim_wings(p, k, feed=FEED):
    """翼修邊 k px（組件內細粒度縮放,analysis-02 探針）：左翼邊緣像素按 (row,col) 序移除 k 個
    → symmetrize(10) 鏡射到右翼＝雙翼對稱修邊。回 None 若翼太小。純函式決定性。"""
    from scipy.ndimage import label as _label, binary_erosion
    p = (np.asarray(p).reshape(25, 25) > 0.5).copy()
    lab, n = _label(p, structure=_CROSS)
    fid = int(lab[feed])
    wings = p & (lab != fid)
    left = wings.copy()
    left[:, 13:] = False                                  # 只動左半,鏡射補右
    edge = left & ~binary_erosion(left, structure=_CROSS)
    cells = np.argwhere(edge)
    if len(cells) < k:
        return None
    for r, c in cells[:k]:
        p[r, c] = False
    q = symmetrize(p, 10)
    st = piece_stats(q)
    return q if (st["n_1px"] == 0 and st["n_comp"] == n) else None


def _realloc(p, k, feed=FEED):
    """等金屬量再分配（analysis-02 探針）：主件底緣修 k px＋翼長 k px（金屬總量近守恆）。
    主件修=底排(row≥20)邊緣像素 row-major 前 k 個（避開 feed pad 周邊 ±2 欄）;
    翼長=左翼相鄰空位 row-major 前 k 個。回 None 若失敗。決定性。"""
    from scipy.ndimage import label as _label, binary_erosion, binary_dilation
    p = (np.asarray(p).reshape(25, 25) > 0.5).copy()
    lab, n = _label(p, structure=_CROSS)
    fid = int(lab[feed])
    main = (lab == fid)
    edge = main & ~binary_erosion(main, structure=_CROSS)
    edge[:20] = False
    edge[:, feed[1] - 2:feed[1] + 3] = False              # 避開 feed 周邊
    edge[:, 13:] = False                                  # 左半,鏡射補右
    cells = np.argwhere(edge)
    if len(cells) < k:
        return None
    for r, c in cells[:k]:
        p[r, c] = False
    wings = (np.asarray(p).reshape(25, 25) > 0.5) & (lab != fid) & (lab > 0)
    grow = binary_dilation(wings, structure=_CROSS) & ~p
    grow[:, 13:] = False
    gcells = np.argwhere(grow)
    if len(gcells) < k:
        return None
    for r, c in gcells[:k]:
        p[r, c] = True
    q = symmetrize(p, 10)
    st = piece_stats(q)
    return q if (st["n_1px"] == 0 and st["n_comp"] == n) else None


def select_r16b(args):
    """R16 續批（37,r15v 收完接跑）——analysis-02 兩個因果探針＋添加收益圖擴錨:
      W 翼修邊 —— {c21,a15,x00,c25} × k∈{2,4,6}:「縮小翼」的直接因果（analysis-02 cv 相關性升級檢驗）
      M 等金屬再分配 —— {c21,x00} × k∈{4,8}:主件−k+翼+k,「分配>堆料」假說的乾淨對照
      A 添加收益圖擴錨 —— c21/a15 單塊全掃（2×2+3×3,歷史排除）:與 218 批的 x00/c25 拼成四錨點收益圖
    判準:W=修邊後 wm/rad/oob 的劑量反應曲線;M=再分配後 wm 升=分配假說過因果關;A=收益圖跨錨點相關。"""
    SRC = {"c21": ("dedust_ref2_input", "c21_sm"), "a15": ("dedust_ref2_input", "a15_k4"),
           "x00": ("dedust_wide_input", "x00_c21k2"), "c25": ("dedust_ref3_input", "c25_a15w10_2_22")}
    P = {k: np.asarray(torch.load(str(_dir(f).joinpath(i + ".pt")), weights_only=True)).reshape(25, 25) > 0.5
         for k, (f, i) in SRC.items()}
    hist = set()
    for fol in HISTORY_INPUTS + ("dedust_r15ga_input", "dedust_r15inf_input", "dedust_r15v_input",
                                 "dedust_addmap_input"):
        d = DATASET_PATH.joinpath(fol)
        if not d.joinpath("manifest.json").exists():
            continue
        for m in json.load(open(str(d.joinpath("manifest.json")), encoding="utf-8")):
            f = d.joinpath(m["id"] + ".pt")
            if f.exists():
                hist.add((np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes())
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra):
        if pat is None or pat.tobytes() in hist:
            return False
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(pid + ".pt")))
        return True

    for aid, p in P.items():
        for k in (2, 4, 6):
            emit("w" + aid + "_t" + str(k), "wingtrim", "W_" + aid, _trim_wings(p, k),
                 dict(source_id=aid, trim_px=k))
    for aid in ("c21", "x00"):
        for k in (4, 8):
            emit("m" + aid + "_k" + str(k), "realloc", "M_" + aid, _realloc(P[aid], k),
                 dict(source_id=aid, realloc_px=k))
    n = 0
    for aid in ("c21", "a15"):
        for h in (2, 3):
            for r in range(0, 23 - h + 1):
                for c in range(0, 12):
                    q = add_block(P[aid], r, c, h, h)
                    if q is None:
                        continue
                    q = symmetrize(q, 10)
                    if piece_stats(q)["n_1px"] > 0:
                        continue
                    if emit("a" + format(n, "03d") + "_" + aid + "r" + str(r) + "c" + str(c) + "s" + str(h),
                            "addmap", "A_" + aid, q, dict(source_id=aid, block_at=[r, c, h, h])):
                        n += 1
    _save_manifest(manifest, input_dir)
    kinds = {}
    for m in manifest:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    print("r16b 輸入完成 → " + str(input_dir) + ": " + str(kinds) + " 共 " + str(len(manifest)) + " 筆")


def _surgery(p, op, k):
    """帶外手術算子（R17）。冠軍家族實形＝上半碎片雲＋下主件（無平移/加寄生條空間），
    帶外攻堅改走「切削改諧振」——slot 陷波是貼片天線經典手法:
      hslot k=列 → 整列清空（雲區 1px 水平槽,切垂直電流路徑;整列自然對稱）
      vslot k=欄 → 該欄+鏡射欄 rows0-12 清空（切雲的水平電流路徑→低側諧振上推,主攻低側）
      rowcut k → 頂部 k 列全清（雲高方向性縮減;≠R14 形態學整圈 erode）
      colcut k → 最外 k 欄+鏡射欄 rows0-12 清空（雲寬縮減,同樣主攻水平諧振）
      mslot k=列 → 主件該列 cols0-9+鏡射段清空（主件陷波槽,留中央橋;避開 feed pad 列）
    回未除塵 pattern;呼叫端 symmetrize(10) 收尾（除塵+保對稱）。決定性。"""
    p = (np.asarray(p).reshape(25, 25) > 0.5).copy()
    if op == "hslot":
        p[k, :] = False
    elif op == "vslot":
        p[0:13, k] = False
        p[0:13, 24 - k] = False
    elif op == "rowcut":
        p[0:k, :] = False
    elif op == "colcut":
        p[0:13, 0:k] = False
        p[0:13, 25 - k:25] = False
    elif op == "mslot":
        p[k, 0:10] = False
    return p


def select_r17(args):
    """R17 帶外主目標批（Ricky 2026-07-09「當作主目標探索幾輪」）——判準寫死於 round-17 檔。
    問題核心:「低側裙擺（24-25.5,全家族地板 ≈9.2）是體質還是可壓?」三臂:
      T 帶外手術 —— slot/rowcut/colcut/mslot 切削掃描（_surgery;雲形態下的頻率手術,
        vslot/colcut 主攻低側水平諧振、mslot=主件陷波、hslot/rowcut=劑量對照）
      C 雙中央塊 —— c25/x00 中央帶疊兩塊（addmap 已證單塊=高側旋鈕,測疊加+可加性）
      N 公證 —— a024(+0.35 未公證,挑戰 i02)×3＋a017(−0.01 卡線)×2＋a022×1（鐵則）"""
    SRC = {"c25": ("dedust_ref3_input", "c25_a15w10_2_22"), "x00": ("dedust_wide_input", "x00_c21k2"),
           "i02": ("dedust_r15inf_input", "i02_r15"),
           "a024": ("dedust_addmap_input", "a024_c25r9c11s3"),
           "a017": ("dedust_addmap_input", "a017_c25r8c11s2"),
           "a022": ("dedust_addmap_input", "a022_c25r7c11s3"),
           "i12": ("dedust_r15inf_input", "i12_r15")}
    P = {k: np.asarray(torch.load(str(_dir(f).joinpath(i + ".pt")), weights_only=True)).reshape(25, 25) > 0.5
         for k, (f, i) in SRC.items()}
    hist = set()
    for fol in HISTORY_INPUTS:
        if fol == args.input:                     # 自身夾跳過,否則重跑 select 全被查重擋空
            continue
        d = DATASET_PATH.joinpath(fol)
        if not d.joinpath("manifest.json").exists():
            continue
        for m in json.load(open(str(d.joinpath("manifest.json")), encoding="utf-8")):
            f = d.joinpath(m["id"] + ".pt")
            if f.exists():
                hist.add((np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes())
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra, dedup=True):
        if pat is None:
            return False
        pat = (np.asarray(pat).reshape(25, 25) > 0.5).copy()
        pat[FEED] = True
        if dedup and pat.tobytes() in hist:
            return False
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(pid + ".pt")))
        return True

    def _finish(q, aid):
        """收尾除塵。⚠ x00 破對稱錨點（翻轉含 (4,18),symmetrize 會蓋回=毀身分——addmap/r16b
        的 x00 條目已中招,見 round-16 caveat）→ 只除塵不對稱化;其餘錨點走 symmetrize 慣例。"""
        if q is None:
            return None
        if aid == "x00":
            q, _ = strip_small((np.asarray(q).reshape(25, 25) > 0.5).copy(), 4)
            return _ensure_feed_pad(q, 4)
        return symmetrize(q, 10)

    # T: 帶外手術掃描（切削改諧振;_finish 收尾=除塵+保對稱/保 x00 身分）
    SURG = (("c25", "hslot", (1, 3, 5, 7)), ("x00", "hslot", (2, 5)),
            ("c25", "vslot", (2, 5, 8)), ("x00", "vslot", (3, 7)),
            ("c25", "rowcut", (1, 2, 3)), ("c25", "colcut", (1, 2)), ("x00", "colcut", (1,)),
            ("c25", "mslot", (15, 17, 19)))
    for aid, op, ks in SURG:
        for k in ks:
            q = _finish(_surgery(P[aid], op, k), aid)
            if piece_stats(q)["n_1px"] == 0:
                emit("%s_%s_k%d" % (op, aid, k), "surgery", "T_%s_%s" % (op, aid), q,
                     dict(source_id=aid, op=op, k=k))

    # C: 雙中央塊（中央帶 col11 起 w=3;候選對寫死,不合法自動跳過）
    PAIRS = ((5, 2, 8, 3), (5, 2, 9, 3), (6, 2, 9, 3), (5, 3, 9, 3), (6, 2, 9, 2), (5, 2, 10, 2))
    for aid in ("c25", "x00"):
        for r1, s1, r2, s2 in PAIRS:
            q1 = add_block(P[aid], r1, 11, s1, 3)
            q2 = add_block(q1, r2, 11, s2, 3) if q1 is not None else None
            q = _finish(q2, aid)
            if q is not None and piece_stats(q)["n_1px"] == 0:
                emit("cc_%s_r%ds%d_r%ds%d" % (aid, r1, s1, r2, s2), "dblcentral", "C_%s" % aid, q,
                     dict(source_id=aid, blocks=[[r1, 11, s1, 3], [r2, 11, s2, 3]]))

    # S: 分組/尺寸軸（Ricky 2026-07-09「只加 2×2 不算測分組」）——中央帶等金屬 12px 三態＋尺寸階梯。
    #    等金屬分組: 1×(4×3)＝階梯 h4 兼任 vs 2×(2×3) vs 3×(2×2)——同金屬同區域,唯一變因=分組
    GROUPS = (("g2", ((5, 11, 2, 3), (9, 11, 2, 3))),
              ("g3", ((5, 11, 2, 2), (8, 11, 2, 2), (11, 12, 2, 2))))
    for aid in ("c25", "x00"):
        for gid, blocks in GROUPS:
            q = P[aid]
            for r, c, h, w in blocks:
                q = add_block(q, r, c, h, w) if q is not None else None
            q = _finish(q, aid)
            if q is not None and piece_stats(q)["n_1px"] == 0:
                emit("eq12_%s_%s" % (aid, gid), "grouping", "S_%s" % aid, q,
                     dict(source_id=aid, blocks=[list(b) for b in blocks]))
        # 尺寸階梯（同欄 col11 w3,下緣對齊 row11;3×3@r9=addmap a024 已測當第一階）
        for r, h in ((8, 4), (7, 5)):
            q = _finish(add_block(P[aid], r, 11, h, 3), aid)
            if q is not None and piece_stats(q)["n_1px"] == 0:
                emit("lad_%s_h%dr%d" % (aid, h, r), "sizeladder", "S_%s" % aid, q,
                     dict(source_id=aid, block_at=[r, 11, h, 3]))

    # N: 公證（鐵則:紀錄級一律重複量測;蓄意重複,不查重）
    # i12=c25 雙中央塊(R15 知情臂舊藏):wm +0.32/rad −0.01/oob 9.95——rad 卡線,翻正=三帶皆冠
    for pid, reps in (("a024", 3), ("a017", 2), ("a022", 1), ("i12", 2)):
        for j in range(reps):
            emit("n_%s_%d" % (pid, j), "notarize", "N_%s" % pid, P[pid],
                 dict(source_id=SRC[pid][1]), dedup=False)
    _save_manifest(manifest, input_dir)
    kinds = {}
    for m in manifest:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    print("r17 輸入完成 → %s: %s 共 %d 筆" % (input_dir, kinds, len(manifest)))


def select_r18(args):
    """R18 帶外戰役第二批（218;歷史挖礦 analysis-03 三方向落地,判準寫死於 round-18 檔）:
      V 舊藏公證 —— 挖礦出土的未公證高價值:b20_k4(+0.32/oob 9.56,margin+帶外雙挑戰)×3、
        vpc18_f_d2(+0.27/9.53)×2、vb43_a1_d0(oob 9.03,帶外紀錄挑戰)×2、x20_a00k8(9.15)×1
      S 低側家族構造化救援 —— 池頂低側家族(lo −1.7~−4.5=唯一破 w17 低側地板的族群,
        但 rad/製造全滅)×{sym10,sym12,純除塵}——s05 劇本換目標:救「低側乾淨」而非 wm;
        生還者(wm≥−1 且 lo≤+1)進 R19 rad 救援,全滅=低側優勢係粉塵諧振本體(R7 定律帶外版)
      T 手術擴錨 —— c18_sm(三標內帶外紀錄 9.04,跨店雙響應公證)上 vslot/colcut/hslot"""
    SRC = {"b20": ("dedust_ref2_input", "b20_k4"), "vpd2": ("dedust_crown_input", "vpc18_f_d2"),
           "vb43": ("dedust_crown_input", "vb43_a1_d0"), "x20": ("dedust_wide_input", "x20_a00k8"),
           "c18": ("dedust_ref2_input", "c18_sm")}
    LOW = {pid: "dedust_r9_input" for pid in ("t09_top", "t03_top", "n09_near", "t08_top",
                                              "t07_top", "t04_top", "t11_top", "t14_top")}
    LOW["p00_orig"] = "dedust_r7_input"
    P = {k: np.asarray(torch.load(str(_dir(f).joinpath(i + ".pt")), weights_only=True)).reshape(25, 25) > 0.5
         for k, (f, i) in SRC.items()}
    hist = set()
    for fol in HISTORY_INPUTS:
        if fol == args.input:
            continue
        d = DATASET_PATH.joinpath(fol)
        if not d.joinpath("manifest.json").exists():
            continue
        for m in json.load(open(str(d.joinpath("manifest.json")), encoding="utf-8")):
            f = d.joinpath(m["id"] + ".pt")
            if f.exists():
                hist.add((np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes())
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra, dedup=True):
        if pat is None:
            return False
        pat = (np.asarray(pat).reshape(25, 25) > 0.5).copy()
        pat[FEED] = True
        if dedup and pat.tobytes() in hist:
            return False
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(pid + ".pt")))
        return True

    # V: 舊藏公證（蓄意重複,不查重）
    for pid, reps in (("b20", 3), ("vpd2", 2), ("vb43", 2), ("x20", 1)):
        for j in range(reps):
            emit("n_%s_%d" % (pid, j), "notarize", "N_%s" % pid, P[pid],
                 dict(source_id=SRC[pid][1]), dedup=False)

    # S: 低側家族構造化救援（sym10/sym12=對稱化+除塵;dust=純除塵不對稱——隔離「對稱」與「除塵」）
    for pid, fol in LOW.items():
        p = np.asarray(torch.load(str(_dir(fol).joinpath(pid + ".pt")), weights_only=True)).reshape(25, 25) > 0.5
        short = pid.split("_")[0]
        dq, _ = strip_small(p.copy(), 4)
        for tag, q in (("s10", symmetrize(p, 10)), ("s12", symmetrize(p, 12)),
                       ("dust", _ensure_feed_pad(dq, 4))):
            if piece_stats(q)["n_1px"] == 0:
                emit("lw_%s_%s" % (short, tag), "lowrescue", "S_%s" % short, q,
                     dict(source_id=pid, variant=tag))

    # T: 手術擴錨到帶外紀錄保持者 c18_sm（對稱性先驗證,非不變則走純除塵收尾）
    c18 = P["c18"]
    sym_ok = bool((symmetrize(c18.copy(), 10) == c18).all())
    for op, ks in (("vslot", (2, 5, 8)), ("colcut", (1, 2)), ("hslot", (3,))):
        for k in ks:
            q = _surgery(c18, op, k)
            if sym_ok:
                q = symmetrize(q, 10)
            else:
                q, _ = strip_small(q, 4)
                q = _ensure_feed_pad(q, 4)
            if piece_stats(q)["n_1px"] == 0:
                emit("%s_c18_k%d" % (op, k), "surgery", "T_%s_c18" % op, q,
                     dict(source_id="c18_sm", op=op, k=k))
    _save_manifest(manifest, input_dir)
    kinds = {}
    for m in manifest:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    print("r18 輸入完成 → %s: %s 共 %d 筆 (c18 對稱不變=%s)" % (input_dir, kinds, len(manifest), sym_ok))


def select_r19data(args):
    """R19 模型線資料批（Ricky 2026-07-09:「大量基於現有王結構做組件級隨機 variation,
    37/218 各收 200 組不重複,為訓練輪作準備」）——SM v5 的新區域覆蓋批:
      九個王錨點（加權:margin/製造/帶外/rad 王系）× 組件級算子隨機鏈（1-3 個:加塊/移塊/手術/
      修翼/再分配/邊緣翻/縮放;全 seeded 決定性）→ 兩夾各 --n 筆、全域去重（批內+27 夾歷史）。
      標籤自然涵蓋好壞全譜（正負樣本都是訓練訊號）;r19a 另搭載 R17 三筆單次紀錄級公證（鐵則）。"""
    ANCH = (("a024", "dedust_addmap_input", "a024_c25r9c11s3", .20),
            ("c25", "dedust_ref3_input", "c25_a15w10_2_22", .15),
            ("x00", "dedust_wide_input", "x00_c21k2", .15),
            ("c21", "dedust_ref2_input", "c21_sm", .12),
            ("a15", "dedust_ref2_input", "a15_k4", .10),
            ("i02", "dedust_r15inf_input", "i02_r15", .10),
            ("g16", "dedust_r15ga_input", "g16_r15", .08),
            ("c18", "dedust_ref2_input", "c18_sm", .06),
            ("g14", "dedust_r15ga_input", "g14_r15", .04))
    P = {a: np.asarray(torch.load(str(_dir(f).joinpath(i + ".pt")), weights_only=True)).reshape(25, 25) > 0.5
         for a, f, i, _ in ANCH}
    aw = np.array([w for *_, w in ANCH]); aw /= aw.sum()
    hist = set()
    for fol in _all_input_folders():              # 自動掃描(2026-07-11:vgen3 撞 vgen2 教訓,舊清單漏新夾)
        if fol in (args.input_a, args.input_b):
            continue
        d = DATASET_PATH.joinpath(fol)
        if not d.joinpath("manifest.json").exists():
            continue
        for m in json.load(open(str(d.joinpath("manifest.json")), encoding="utf-8")):
            f = d.joinpath(m["id"] + ".pt")
            if f.exists():
                hist.add((np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes())
    rng = np.random.default_rng(args.seed)
    from scipy.ndimage import label as _label

    def op_addblock(p):
        for _ in range(10):
            r, c = int(rng.integers(0, 22)), int(rng.integers(0, 23))
            h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
            q = add_block(p, r, c, h, w)
            if q is not None:
                return q, ["addblock", r, c, h, w]
        return None, None

    def op_rmblock(p):
        lab, n = _label(p, structure=_CROSS)
        fid = int(lab[FEED])
        small = [k for k in range(1, n + 1) if k != fid and (lab == k).sum() <= 20]
        if not small:
            return None, None
        k = small[int(rng.integers(0, len(small)))]
        q = p.copy(); q[lab == k] = False
        return q, ["rmblock", int((lab == k).sum())]

    def op_surgery(p):
        op = ("hslot", "vslot", "rowcut", "colcut", "mslot")[int(rng.integers(0, 5))]
        k = {"hslot": int(rng.integers(0, 13)), "vslot": int(rng.integers(1, 12)),
             "rowcut": int(rng.integers(1, 4)), "colcut": int(rng.integers(1, 3)),
             "mslot": int(rng.integers(14, 21))}[op]
        return _surgery(p, op, k), ["surgery", op, k]

    def op_wingtrim(p):
        k = int(rng.integers(1, 7))
        return _trim_wings(p, k), ["wingtrim", k]

    def op_realloc(p):
        k = int(rng.integers(2, 9))
        return _realloc(p, k), ["realloc", k]

    def op_flips(p):
        em, ed = edge_sets(p)
        pool = np.flatnonzero((em | ed).reshape(-1))
        k = int(rng.integers(1, 9))
        q = p.copy()
        q.ravel()[rng.choice(pool, size=min(k, len(pool)), replace=False)] ^= True
        return q, ["flips", k]

    def op_resize(p):
        which = ("main", "wings")[int(rng.integers(0, 2))]
        delta = (-1, 1)[int(rng.integers(0, 2))]
        return resize_component(p, which, delta), ["resize", which, delta]

    OPS = ((op_addblock, .28), (op_rmblock, .10), (op_surgery, .20), (op_wingtrim, .08),
           (op_realloc, .07), (op_flips, .17), (op_resize, .10))
    ow = np.array([w for _, w in OPS]); ow /= ow.sum()
    dirs = [_dir(args.input_a), _dir(args.input_b)]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    manifests = [[], []]
    total, attempts = 2 * args.n, 0
    # 擾動幅度分層配額（Ricky:「不一樣的 pixel 要有合理的分布」）——diff_px 對錨點,五帶強制
    BANDS = ((1, 3, .15), (4, 10, .30), (11, 25, .30), (26, 60, .18), (61, 120, .07))
    quota = [int(round(fr * total)) for _, _, fr in BANDS]
    quota[1] += total - sum(quota)                # 湊整差額進次小帶
    filled = [0] * len(BANDS)
    made = 0
    while made < total and attempts < total * 200:
        attempts += 1
        ai = int(rng.choice(len(ANCH), p=aw))
        aid = ANCH[ai][0]
        q = P[aid]
        chain = []
        for _ in range(int(rng.choice((1, 2, 3), p=(.5, .35, .15)))):
            fn = OPS[int(rng.choice(len(OPS), p=ow))][0]
            q, desc = fn(q)
            if q is None:
                break
            chain.append(desc)
        if q is None or not chain:
            continue
        q, _n = strip_small((np.asarray(q).reshape(25, 25) > 0.5).copy(), 4)
        q = _ensure_feed_pad(q, 4)
        st = piece_stats(q)
        if st["n_1px"] > 0 or not (250 <= st["metal_px"] <= 520):
            continue
        d = int((q != P[aid]).sum())
        band = next((i for i, (lo, hi, _) in enumerate(BANDS) if lo <= d <= hi), None)
        if band is None or filled[band] >= quota[band]:
            continue
        key = q.tobytes()
        if key in hist:
            continue
        hist.add(key)
        filled[band] += 1
        b = made % 2                              # 交錯分夾=兩機分布一致
        pid = "vg%04d_%s" % (made, aid)
        manifests[b].append(dict(id=pid, kind="vargen", family="V_" + aid, removed_px=0,
                                 **st, source_id=aid, ops=chain, diff_px=d))
        torch.save(torch.tensor(q, dtype=torch.float32), str(dirs[b].joinpath(pid + ".pt")))
        made += 1
    print("diff_px 各帶(配額/實收):", ["%d-%d:%d/%d" % (lo, hi, f, qt)
                                       for (lo, hi, _), f, qt in zip(BANDS, filled, quota)])
    # r19a 搭載:R17 三筆單次紀錄級公證（cc margin/平衡/rad 王挑戰者,各 ×2;鐵則）
    for pid in ("cc_c25_r6s2_r9s3", "cc_x00_r5s2_r8s3", "cc_c25_r6s2_r9s2"):
        p = np.asarray(torch.load(str(_dir("dedust_r17_input").joinpath(pid + ".pt")),
                                  weights_only=True)).reshape(25, 25) > 0.5
        for j in range(2):
            nid = "n_%s_%d" % (pid, j)
            manifests[0].append(dict(id=nid, kind="notarize", family="N_" + pid, removed_px=0,
                                     **piece_stats(p), source_id=pid))
            torch.save(torch.tensor(p, dtype=torch.float32), str(dirs[0].joinpath(nid + ".pt")))
    for man, d in zip(manifests, dirs):
        _save_manifest(man, d)
    kinds = {}
    for man in manifests:
        for m in man:
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    print("r19data 完成 → %s %d 筆 / %s %d 筆 (%s;嘗試 %d)" % (
        dirs[0], len(manifests[0]), dirs[1], len(manifests[1]), kinds, attempts))


def select_r20gen(args):
    """R20 一代選批（模型線終審＝真值在迴圈的 (μ+λ) 演化;判準寫死於 round-20 檔）:
      G GA 臂（--ga 60）—— 父母（現任榜＋上代真值贏家）× 算子鏈子代 λ≈5000 → SM 粗篩
        （帶狀 wm 排序＋帶外 tiebreak;R19 校準:粗排可用/細排不可用/帶外 ρ0.603 可信）＋多樣性 → 60
      N 隨機對照（--rand 60）—— 同一子代池、不經 SM、均勻抽＝模型線終審的對照臂
      F 碎片探索（--frag 30）—— 池頂碎片族（p00/t07/n09…,wm 正 rad 死）× 輕變異
        （flips/塊/hslot=rad 旋鈕/部分除塵）,不強制可製造（Ricky:「探索碎片化好 pattern 的空間」）
    全部樣本 manifest 記 pred_wm/pred_oob＝前瞻性驗證（每代自動一次模型考試,收檔算 ρ）。
    gen>1:父母自動＝上一代真值字典序 top＋現任榜 elitism。算子權重＝R19 真值校準
    （三標過率 flips20/addblock18/rmblock17% vs surgery1/resize0%）。輸出三夾（%3 交錯,三機並行）。"""
    from scipy.ndimage import label as _label
    from antenna.training import setup_responses
    from antenna.zoo import SURROGATES
    rng = np.random.default_rng(args.seed + args.gen)

    def loadp(fol, pid):
        return np.asarray(torch.load(str(_dir(fol).joinpath(pid + ".pt")), weights_only=True)).reshape(25, 25) > 0.5

    # ---- 父母池
    ELITE = [("a024", "dedust_addmap_input", "a024_c25r9c11s3"),
             ("ccr9s2", "dedust_r17_input", "cc_c25_r6s2_r9s2"),
             ("ccx00", "dedust_r17_input", "cc_x00_r5s2_r8s3")]
    if args.gen == 1:
        PAR = ELITE + [("ccr9s3", "dedust_r17_input", "cc_c25_r6s2_r9s3"),
                       ("i02", "dedust_r15inf_input", "i02_r15"),
                       ("c25", "dedust_ref3_input", "c25_a15w10_2_22"),
                       ("x00", "dedust_wide_input", "x00_c21k2"),
                       ("g16", "dedust_r15ga_input", "g16_r15"),
                       ("c18", "dedust_ref2_input", "c18_sm"),
                       ("vg0258", "dedust_r19a_input", "vg0258_c25"),
                       ("vg0338", "dedust_r19a_input", "vg0338_c18"),
                       ("vg0765", "dedust_r19b_input", "vg0765_a024")]
    else:                                             # 上一代真值字典序 top9 + elitism 3
        prev, seen_k = [], set()
        for suf in "abc":
            st = f"{args.input_prefix}{args.gen - 1}{suf}"
            rp = DATASET_PATH.joinpath(st, "results.json")
            if not rp.exists():
                raise SystemExit(f"上一代 {st} 無結果——先收完再開新代")
            res = json.load(open(str(rp), encoding="utf-8"))
            for i, r in res.items():
                if "wm" not in r or i.startswith("n_"):
                    continue
                p = loadp(st + "_input", i)
                k = p.tobytes()
                if k in seen_k:
                    continue
                seen_k.add(k)
                tri = r["wm"][2] >= 0 and (r.get("rad_margin") or -9) >= 0
                prev.append((tri, r["wm"][2], i, st + "_input"))
        prev.sort(reverse=True)
        PAR = ELITE + [(i, fol, i) for _, _, i, fol in prev[:9]]
    P = {}
    for name, fol, pid in PAR:
        if name not in P:
            P[name] = loadp(fol, pid)
    print(f"gen{args.gen} 父母 {len(P)}: {list(P)}")

    # ---- 算子庫（優化配方;權重=R19 真值三標過率校準）
    def op_flips(p):
        em, ed = edge_sets(p)
        pool = np.flatnonzero((em | ed).reshape(-1))
        k = int(rng.integers(1, 9))
        q = p.copy()
        q.ravel()[rng.choice(pool, size=min(k, len(pool)), replace=False)] ^= True
        return q, ["flips", k]

    def op_addblock(p):
        for _ in range(8):
            r, c = int(rng.integers(0, 22)), int(rng.integers(0, 23))
            h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
            q = add_block(p, r, c, h, w)
            if q is not None:
                return q, ["addblock", r, c, h, w]
        return None, None

    def op_rmblock(p):
        lab, n = _label(p, structure=_CROSS)
        fid = int(lab[FEED])
        small = [k for k in range(1, n + 1) if k != fid and (lab == k).sum() <= 20]
        if not small:
            return None, None
        k = small[int(rng.integers(0, len(small)))]
        q = p.copy(); q[lab == k] = False
        return q, ["rmblock", int((lab == k).sum())]

    def op_realloc(p):
        return _realloc(p, int(rng.integers(2, 7))), ["realloc"]

    def op_wtrim(p):
        return _trim_wings(p, int(rng.integers(1, 4))), ["wingtrim"]

    def op_hslot(p):
        k = int(rng.integers(0, 13))
        return _surgery(p, "hslot", k), ["hslot", k]

    def op_dustpart(p):                               # 部分除塵（F 臂:朝可製造走一步,不走到底）
        lab, n = _label(p, structure=_CROSS)
        fid = int(lab[FEED])
        sizes = [(int((lab == k).sum()), k) for k in range(1, n + 1) if k != fid]
        sizes.sort()
        take = [k for _, k in sizes[:int(rng.integers(1, 5))]]
        if not take:
            return None, None
        q = p.copy()
        for k in take:
            q[lab == k] = False
        return q, ["dustpart", len(take)]

    OPS_G = ((op_flips, .38), (op_addblock, .27), (op_rmblock, .21), (op_realloc, .07),
             (op_wtrim, .04), (op_hslot, .03))
    OPS_F = ((op_flips, .35), (op_addblock, .20), (op_rmblock, .15), (op_hslot, .20), (op_dustpart, .10))

    def gen_children(anchors, ops, per_anchor, dmax, manuf, hist):
        ws = np.array([w for _, w in ops]); ws /= ws.sum()
        out = []
        for aname, ap in anchors.items():
            made, tries = 0, 0
            while made < per_anchor and tries < per_anchor * 25:
                tries += 1
                q = ap
                chain = []
                for _ in range(int(rng.choice((1, 2), p=(.6, .4)))):
                    fn = ops[int(rng.choice(len(ops), p=ws))][0]
                    q, desc = fn(q)
                    if q is None:
                        break
                    chain.append(desc)
                if q is None or not chain:
                    continue
                q = (np.asarray(q).reshape(25, 25) > 0.5).copy()
                q[FEED] = True
                if manuf:
                    q, _n = strip_small(q, 4)
                    q = _ensure_feed_pad(q, 4)
                st = piece_stats(q)
                if manuf and st["n_1px"] > 0:
                    continue
                if not (230 <= st["metal_px"] <= 520):
                    continue
                d = int((q != ap).sum())
                if not (1 <= d <= dmax):
                    continue
                k = q.tobytes()
                if k in hist:
                    continue
                hist.add(k)
                out.append(dict(pat=q, parent=aname, ops=chain, d=d, stats=st))
                made += 1
        return out

    hist = set()
    for fol in _all_input_folders():
        d = DATASET_PATH.joinpath(fol)
        for m in json.load(open(str(d.joinpath("manifest.json")), encoding="utf-8")):
            f = d.joinpath(m["id"] + ".pt")
            if f.exists():
                hist.add((np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes())

    per_g = max(1, (args.ga + args.rand) * 40 // max(len(P), 1))
    cand_g = gen_children(P, OPS_G, per_g, 25, True, hist)
    FRAG = {n: loadp(f, i) for n, f, i in
            (("p00", "dedust_r7_input", "p00_orig"), ("t07", "dedust_r9_input", "t07_top"),
             ("t08", "dedust_r9_input", "t08_top"), ("n09", "dedust_r9_input", "n09_near"),
             ("t04", "dedust_r9_input", "t04_top"), ("t11", "dedust_r9_input", "t11_top"),
             ("t14", "dedust_r9_input", "t14_top"), ("t09", "dedust_r9_input", "t09_top"))}
    cand_f = gen_children(FRAG, OPS_F, max(1, args.frag * 40 // len(FRAG)), 40, False, hist)
    print(f"子代池: G {len(cand_g)} / F {len(cand_f)}")

    # ---- SM 粗篩＋前瞻性預測（全部候選都預測;帶狀排序=1dB 帶內用帶外 tiebreak）
    cfg = load_config(args.config)
    setup_responses(cfg)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    cache = os.path.join(REPO, "tmp", "r20_sm")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
    sm.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)

    def predict(cands):
        pats = torch.stack([torch.tensor(c["pat"], dtype=torch.float32).reshape(-1) for c in cands])
        with torch.no_grad():
            raw = sm.model(pats).reshape(len(cands), len(labels), n_pts)
        for k, c in enumerate(cands):
            w, _ = worst_margin(raw[k], labels, cfg.targets)
            c["pred_wm"] = _r(float(w))
            c["pred_oob"] = oob_metrics(raw[k].numpy())["oob_bad"]

    predict(cand_g)
    predict(cand_f)

    def pick(cands, n, top_frac, min_d=6):
        order = sorted(range(len(cands)),
                       key=lambda i: (-np.floor(cands[i]["pred_wm"]), cands[i]["pred_oob"]))
        order = order[:max(n * 3, int(len(cands) * top_frac))]
        picked = []
        for i in order:
            if len(picked) >= n:
                break
            if all(int((cands[i]["pat"] != cands[j]["pat"]).sum()) >= min_d for j in picked):
                picked.append(i)
        for i in order:                               # 多樣性湊不滿就放寬
            if len(picked) >= n:
                break
            if i not in picked:
                picked.append(i)
        return picked

    gi = pick(cand_g, args.ga, .4)
    rest = [i for i in range(len(cand_g)) if i not in gi]
    ni = list(rng.choice(rest, size=min(args.rand, len(rest)), replace=False))
    fi = pick(cand_f, args.frag, .5)

    entries = []
    for arm, idxs, cands in (("ga", gi, cand_g), ("rand", ni, cand_g), ("frag", fi, cand_f)):
        for j, i in enumerate(idxs):
            c = cands[i]
            entries.append(dict(id=f"{arm[0]}{args.gen}_{j:03d}_{c['parent']}", kind=arm,
                                family=f"{arm.upper()}_g{args.gen}", removed_px=0, **c["stats"],
                                source_id=c["parent"], ops=c["ops"], diff_px=c["d"],
                                pred_wm=c["pred_wm"], pred_oob=c["pred_oob"], _pat=c["pat"]))
    if args.gen == 1:                                 # 公證搭載（R19 單次紀錄級;鐵則）
        for pid, fol, reps in (("vg0338_c18", "dedust_r19a_input", 2), ("vg0396_c18", "dedust_r19a_input", 1),
                               ("vg0258_c25", "dedust_r19a_input", 2), ("vg0765_a024", "dedust_r19b_input", 1)):
            p = loadp(fol, pid)
            for j in range(reps):
                entries.append(dict(id=f"n_{pid}_{j}", kind="notarize", family="N_" + pid,
                                    removed_px=0, **piece_stats(p), source_id=pid, _pat=p))
    dirs = []
    for suf in "abc":
        d = _dir(f"{args.input_prefix}{args.gen}{suf}_input")
        d.mkdir(parents=True, exist_ok=True)
        dirs.append(d)
    manifests = [[], [], []]
    for k, e in enumerate(entries):
        pat = e.pop("_pat")
        b = k % 3
        manifests[b].append(e)
        torch.save(torch.tensor(pat, dtype=torch.float32), str(dirs[b].joinpath(e["id"] + ".pt")))
    for man, d in zip(manifests, dirs):
        _save_manifest(man, d)
    print(f"r20 gen{args.gen} 完成: GA {len(gi)}+隨機 {len(ni)}+碎片 {len(fi)}"
          f"{'+公證 6' if args.gen == 1 else ''} → 三夾 {[len(m) for m in manifests]}")


def select_r21harvest(args):
    """R21 收割管線（R20 贏家配方;Ricky 拍板 (a)＋探索稅修訂 2026-07-11 batch4 起）:
      O 帶外收割 —— pred_wm 砍尾 40% → pred_oob 升冪 → 多樣性（R20 ③ 19:10 的兌現配方）
      M margin 樂透 —— 純隨機不經 SM（margin 大獎全出自隨機;+0.39/+0.36 皆 d1）
      L 低側收割（--lo,預設 0;先決=lo-realized 排序回測過門檻）—— pred 低側 realized 峰升冪
      W 大跳彩票（--wild,預設 8）—— d 26-60,持續複驗「26px 死區」結論
    **錨點兩池抽樣（治支系馬太效應,Ricky 2026-07-11）**:王朝池（c18 血系）70% / 冷支池 30%,
    池內均勻;自動吸收前批贏家（按血系標記歸池）;毒名單排除。全樣本記 pred_*（前瞻驗證看無偏 M 臂）。"""
    from scipy.ndimage import label as _label
    from antenna.training import setup_responses
    from antenna.zoo import SURROGATES
    rng = np.random.default_rng(args.seed + args.batch)
    POISON = ("g2_029", "t14", "vg0795")              # HFSS 敵意血系（R20 gen3;不當錨點）
    DYN = ("c18", "vg0338", "vg0396", "g1_038", "g1_039", "r2_016", "r3_001")   # 王朝血系標記

    def loadp(fol, pid):
        f = _dir(fol).joinpath(pid + ".pt")
        if not f.exists():                            # %3 交錯分夾=夾號難記,fallback 全夾搜尋
            for alt in _all_input_folders():
                if _dir(alt).joinpath(pid + ".pt").exists():
                    f = _dir(alt).joinpath(pid + ".pt")
                    break
        return np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5

    ANCH = [("r2_016", "dedust_r20g2b_input", "r2_016_g1_039_vg0338"),
            ("r3_001", "dedust_r20g3b_input", "r3_001_r2_016_g1_039_vg0338"),
            ("vg0338", "dedust_r19a_input", "vg0338_c18"),
            ("vg0396", "dedust_r19a_input", "vg0396_c18"),
            ("g1_038", "dedust_r20g1b_input", "g1_038_vg0338"),
            ("c18", "dedust_ref2_input", "c18_sm"),
            # —— 冷支池固定名額（不被吸收機制稀釋）——
            ("a024", "dedust_addmap_input", "a024_c25r9c11s3"),
            ("ccr9s2", "dedust_r17_input", "cc_c25_r6s2_r9s2"),
            ("ccx00", "dedust_r17_input", "cc_x00_r5s2_r8s3"),
            ("vg0765", "dedust_r19b_input", "vg0765_a024"),
            ("c25", "dedust_ref3_input", "c25_a15w10_2_22"),
            ("x00", "dedust_wide_input", "x00_c21k2"),
            ("g16", "dedust_r15ga_input", "g16_r15"),
            ("i02", "dedust_r15inf_input", "i02_r15")]
    for b in range(1, args.batch):                    # 自動吸收前批三標贏家
        for suf in "abc":
            st = f"dedust_r21b{b}{suf}"
            rp = DATASET_PATH.joinpath(st, "results.json")
            if not rp.exists():
                continue
            res = json.load(open(str(rp), encoding="utf-8"))
            for i, r in res.items():
                if "wm" not in r or any(px in i for px in POISON):
                    continue
                tri = r["wm"][2] >= 0 and (r.get("rad_margin") or -9) >= 0
                if tri and (r["wm"][2] >= 0.15 or (r.get("oob_bad") or 99) < 9.5):
                    ANCH.append((i, st + "_input", i))
    P = {}
    for name, fol, pid in ANCH:
        if name not in P and not any(px in name for px in POISON):
            P[name] = loadp(fol, pid)
    dyn_names = [n for n in P if any(m in n for m in DYN)]
    cold_names = [n for n in P if n not in dyn_names]
    print(f"batch{args.batch} 錨點 {len(P)}（王朝池 {len(dyn_names)} / 冷支池 {len(cold_names)};抽樣 70/30）")

    def pick_anchor():
        pool = dyn_names if (rng.random() < 0.7 and dyn_names) else cold_names
        if not pool:
            pool = list(P)
        return pool[int(rng.integers(0, len(pool)))]

    def op_flips(p):
        em, ed = edge_sets(p)
        pool = np.flatnonzero((em | ed).reshape(-1))
        k = int(rng.integers(1, 9))
        q = p.copy()
        q.ravel()[rng.choice(pool, size=min(k, len(pool)), replace=False)] ^= True
        return q, ["flips", k]

    def op_addblock(p):
        for _ in range(8):
            r, c = int(rng.integers(0, 22)), int(rng.integers(0, 23))
            h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
            q = add_block(p, r, c, h, w)
            if q is not None:
                return q, ["addblock", r, c, h, w]
        return None, None

    def op_rmblock(p):
        lab, n = _label(p, structure=_CROSS)
        fid = int(lab[FEED])
        small = [k for k in range(1, n + 1) if k != fid and (lab == k).sum() <= 20]
        if not small:
            return None, None
        k = small[int(rng.integers(0, len(small)))]
        q = p.copy(); q[lab == k] = False
        return q, ["rmblock", int((lab == k).sum())]

    OPS = ((op_flips, .50), (op_addblock, .28), (op_rmblock, .22))
    ws = np.array([w for _, w in OPS]); ws /= ws.sum()
    hist = set()
    for fol in _all_input_folders():
        dd = DATASET_PATH.joinpath(fol)
        for m in json.load(open(str(dd.joinpath("manifest.json")), encoding="utf-8")):
            f = dd.joinpath(m["id"] + ".pt")
            if f.exists():
                hist.add((np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes())
    cands, wilds, tries = [], [], 0
    target_pool = args.n * 35
    wild_target = max(args.wild, 1) * 40
    while (len(cands) < target_pool or len(wilds) < wild_target) and tries < target_pool * 25:
        tries += 1
        want_wild = args.wild > 0 and len(wilds) < wild_target and rng.random() < 0.18
        aname = pick_anchor()
        q = P[aname]
        chain = []
        n_ops = int(rng.choice((2, 3, 4))) if want_wild else int(rng.choice((1, 2), p=(.65, .35)))
        for _ in range(n_ops):
            fn = OPS[int(rng.choice(len(OPS), p=ws))][0]
            q, desc = fn(q)
            if q is None:
                break
            chain.append(desc)
        if q is None or not chain:
            continue
        q = (np.asarray(q).reshape(25, 25) > 0.5).copy()
        q[FEED] = True
        q, _n = strip_small(q, 4)
        q = _ensure_feed_pad(q, 4)
        st = piece_stats(q)
        if st["n_1px"] > 0 or not (230 <= st["metal_px"] <= 520):
            continue
        d = int((q != P[aname]).sum())
        k = q.tobytes()
        if k in hist:
            continue
        c = dict(pat=q, parent=aname, ops=chain, d=d, stats=st)
        if 26 <= d <= 60 and len(wilds) < wild_target:
            hist.add(k)
            wilds.append(c)
        elif 1 <= d <= 25 and len(cands) < target_pool:
            hist.add(k)
            cands.append(c)
    print(f"子代池 {len(cands)}＋彩票池 {len(wilds)}")
    cfg = load_config(args.config)
    setup_responses(cfg)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    cache = os.path.join(REPO, "tmp", "r21_sm")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
    sm.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
    allc = cands + wilds
    pats = torch.stack([torch.tensor(c["pat"], dtype=torch.float32).reshape(-1) for c in allc])
    with torch.no_grad():
        raw = sm.model(pats).reshape(len(allc), len(labels), n_pts)
    for k, c in enumerate(allc):
        w, _ = worst_margin(raw[k], labels, cfg.targets)
        c["pred_wm"] = _r(float(w))
        r = raw[k].numpy()
        c["pred_oob"] = oob_metrics(r)["oob_bad"]
        # 低側 realized 峰（Gain+失配損耗;壓左側的物理正確目標量,Ricky 2026-07-11 討論）
        ml = 10 * np.log10(np.clip(1 - 10 ** (r[0][:4] / 10), 1e-6, 1))
        c["pred_lor"] = _r(float((r[1][:4] + ml).max()))
    n_core = args.n - args.lo - args.wild
    half = n_core // 2

    def _diverse(pool, n, taken):
        out = []
        for i in pool:
            if len(out) >= n:
                break
            if i in taken:
                continue
            if all(int((cands[i]["pat"] != cands[j]["pat"]).sum()) >= 6 for j in out):
                out.append(i)
        for i in pool:
            if len(out) >= n:
                break
            if i not in out and i not in taken:
                out.append(i)
        return out

    order = sorted(range(len(cands)), key=lambda i: cands[i]["pred_wm"], reverse=True)
    trimmed = order[:int(len(order) * .6)]
    oi = _diverse(sorted(trimmed, key=lambda i: cands[i]["pred_oob"]), half, set())
    li = _diverse(sorted(trimmed, key=lambda i: cands[i]["pred_lor"]), args.lo, set(oi)) if args.lo else []
    rest = [i for i in range(len(cands)) if i not in oi and i not in li]
    mi = list(rng.choice(rest, size=min(args.n - args.lo - args.wild - len(oi), len(rest)), replace=False))
    wi = list(rng.choice(len(wilds), size=min(args.wild, len(wilds)), replace=False)) if wilds else []
    entries = []
    for arm, idxs, src in (("oobharv", oi, cands), ("loharv", li, cands),
                           ("mlotto", mi, cands), ("wild", wi, wilds)):
        for j, i in enumerate(idxs):
            c = src[i]
            entries.append(dict(id=f"{arm[0]}{args.batch}_{j:03d}_{c['parent'][:12]}", kind=arm,
                                family=f"{arm.upper()}_b{args.batch}", removed_px=0, **c["stats"],
                                source_id=c["parent"], ops=c["ops"], diff_px=c["d"],
                                pred_wm=c["pred_wm"], pred_oob=c["pred_oob"], pred_lor=c["pred_lor"],
                                _pat=c["pat"]))
    dirs = []
    for suf in "abc":
        dd = _dir(f"dedust_r21b{args.batch}{suf}_input")
        dd.mkdir(parents=True, exist_ok=True)
        dirs.append(dd)
    manifests = [[], [], []]
    for k, e in enumerate(entries):
        pat = e.pop("_pat")
        b = k % 3
        manifests[b].append(e)
        torch.save(torch.tensor(pat, dtype=torch.float32), str(dirs[b].joinpath(e["id"] + ".pt")))
    for man, dd in zip(manifests, dirs):
        _save_manifest(man, dd)
    print(f"r21 batch{args.batch}: 帶外收割 {len(oi)}+低側收割 {len(li)}+margin 樂透 {len(mi)}"
          f"+彩票 {len(wi)} → 三夾 {[len(m) for m in manifests]}")


HISTORY_INPUTS = ("dedust_r7_input", "dedust_r8_input", "dedust_r9_input", "dedust_ref1_input",
                  "dedust_ref2_input", "dedust_occl_input", "dedust_occl2_input", "dedust_tol_input",
                  "dedust_w17rep_input", "dedust_verify_input", "dedust_ref2v_input",
                  "dedust_champ_input", "dedust_ref3_input", "dedust_wide_input",
                  "dedust_repeat_input", "dedust_bakeoff_input", "dedust_crown_input",
                  "dedust_family2_input", "dedust_blocks_input", "dedust_probes_input",
                  "dedust_ablate_input", "dedust_resize_input", "dedust_r15ga_input",
                  "dedust_r15inf_input", "dedust_r15v_input", "dedust_addmap_input",
                  "dedust_r16b_input", "dedust_r17_input", "dedust_r18_input",
                  "dedust_r19a_input", "dedust_r19b_input")


def _all_input_folders():
    """查重全集＝NAS 上全部輸入夾（自動掃描;2026-07-10 起取代手動維護 HISTORY_INPUTS——
    漏補清單的教訓從制度上消滅）。HISTORY_INPUTS 保留給舊 select 的內建去重引用。"""
    return tuple(sorted(
        d for d in os.listdir(str(DATASET_PATH))
        if d.endswith("_input") and DATASET_PATH.joinpath(d, "manifest.json").exists()))


def check_dup(args):
    """發車前查重（教訓 2026-07-07:ref3 出現 4/319 重複——掃位跨拓撲撞位、k=2 被再對稱化蓋回錨點）。
    查 --input 批內重複＋與全部歷史輸入夾（自動掃描,排除自身）的交叉重複。exit 1=有重複。"""
    def load_folder(folder):
        d = DATASET_PATH.joinpath(folder)
        man = json.load(open(str(d.joinpath("manifest.json")), encoding="utf-8"))
        return {m["id"]: np.asarray(torch.load(str(d.joinpath(f"{m['id']}.pt")), weights_only=True)
                                    ).reshape(-1).__gt__(0.5).tobytes()
                for m in man if d.joinpath(f"{m['id']}.pt").exists()}

    man_kind = {m["id"]: m.get("kind", "") for m in json.load(
        open(str(DATASET_PATH.joinpath(args.input, "manifest.json")), encoding="utf-8"))}
    new = {k: v for k, v in load_folder(args.input).items()
           if man_kind.get(k) not in ("repeat", "notarize")}     # 蓄意重複(公證)不算違規
    seen, bad = {}, 0
    for k, v in new.items():
        if v in seen:
            print(f"批內重複: {k} == {seen[v]}")
            bad += 1
        else:
            seen[v] = k
    hist = {}
    for fol in _all_input_folders():
        if fol == args.input:
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


def select_crown(args):
    """R12 收斂線（公證＋穩健化,選要送製造的那一個）——top 三標候選逐一:
      公證 ×2（同 pattern 重跑,鐵則:紀錄級一律多維驗證,防 +0.48 假象）
      局部缺陷 k1 ×4（邊緣隨機翻 1px;candidate 間的區分度在此——c21 存活率遠高於其他薄冠軍）
    判準:公證值一致 且 局部缺陷存活率高 → 穩健冠軍。整面 erode/dilate 不跑（tol 已證全滅,低資訊）。
    --items "來源夾:id,..."（跨批候選;id 前綴保留供 report 對照）。"""
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    def emit(pid, kind, family, pat, extra):
        pat = (np.asarray(pat).reshape(25, 25) > 0.5).copy()
        pat[FEED] = True
        manifest.append(dict(id=pid, kind=kind, family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))

    seed = 20000
    for item in args.items.split(","):
        src_name, pid = item.strip().split(":")
        f = _dir(src_name).joinpath(f"{pid}.pt")
        if not f.exists():
            raise SystemExit(f"找不到 {f}")
        p = np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5
        tag = pid[:6]
        for r in range(args.notarize):
            emit(f"v{tag}_n{r}", "notarize", f"C_{pid}", p, dict(source_id=pid))
        em, ed = edge_sets(p)
        epool = np.flatnonzero((em | ed).reshape(-1))
        for j in range(args.defects):
            rng = np.random.default_rng(seed)
            q = p.copy()
            q.ravel()[rng.choice(epool, size=min(1, len(epool)), replace=False)] ^= True
            emit(f"v{tag}_d{j:02d}", "tol", f"C_{pid}", q, dict(source_id=pid, flip_k=1, seed=seed))
            seed += 1
    _save_manifest(manifest, input_dir)
    print(f"crown 輸入完成 → {input_dir}：{len(manifest)} 筆"
          f"（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr）")


def select_family2(args):
    """R12 破單一化線（找第二座山頭）——非 w17 家族深掘。pool top-300 聚類的最大非 w17 家族
    （F0/F1/F4,對 w17 Hamming>230）取前 N 成員,除塵 → 對稱化{10,12} → SM 篩選:
    測「w17/F2 是不是唯一能過三標的家族,還是只是我們沒深挖別家」。R9 每族只試 leader 一個,本批深掘。"""
    pool = np.load(os.path.join(REPO, "tmp", "pattern_anatomy", "pool.npz"))
    ok = ~np.isnan(pool["wm"][:, 2])
    worst = pool["wm"][ok][:, 2]
    pats = np.unpackbits(pool["packed"][ok], axis=1)[:, :625].reshape(-1, 25, 25).astype(bool)
    order = np.argsort(worst)[::-1][:300]
    leaders, members = [], {}
    for i in order:
        for L in leaders:
            if np.count_nonzero(pats[i] != pats[L]) <= 100:
                members[L].append(i)
                break
        else:
            leaders.append(i)
            members[i] = [i]
    fam_idx = [int(x) for x in args.families.split(",")]      # F 編號（leader wm 降冪序）

    cfg = load_config(DEFAULT_CFG)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    from antenna.zoo import SURROGATES
    cache = os.path.join(REPO, "tmp", "dedust")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
    sm.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
    sm.model.eval()

    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest, n = [], 0

    def emit(pid, family, pat, extra):
        manifest.append(dict(id=pid, kind="family2", family=family, removed_px=0, **piece_stats(pat), **extra))
        torch.save(torch.tensor(pat, dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))

    for fk in fam_idx:
        L = leaders[fk]
        ms = sorted(members[L], key=lambda i: -worst[i])[:args.per_family]
        cands, gseed = [], 30000 + fk * 1000
        for mi in ms:
            base = pats[mi]
            variants = [("clean", strip_small(base, 4)[0]),
                        ("sym1050", symmetrize(base, 10)),
                        ("sym12", symmetrize(base, 12))]
            for kk in (4, 8):                                 # 鄰域擾動採樣（增家族內多樣性,對稱化保可製造）
                variants.append((f"nb{kk}", symmetrize(perturb_repair(base, kk, seed=gseed), 10)))
                gseed += 1
            for tag, variant in variants:
                variant = _ensure_feed_pad(variant)
                if piece_stats(variant)["n_1px"] > 0:
                    continue                                  # 仍有粉塵→不可製造,跳過
                with torch.no_grad():
                    pred = sm.model(torch.tensor(variant, dtype=torch.float32).flatten())
                w, _ = worst_margin(pred, labels, cfg.targets)
                cands.append((float(w), fk, tag, float(worst[mi]), variant))
        cands.sort(key=lambda c: -c[0])                       # SM 預測 wm 降冪
        picked = []
        for w, fk2, tag, pv, variant in cands:
            if len(picked) >= args.top_per_family:
                break
            if all(np.count_nonzero(variant != q) > 20 for q in picked):
                emit(f"f{n:02d}_F{fk2}{tag[:3]}", f"F{fk2}", variant,
                     dict(pool_family=fk2, variant=tag, pool_wm=pv, sm_pick_wm=_r(w)))
                picked.append(variant)
                n += 1
    _save_manifest(manifest, input_dir)
    print(f"family2 輸入完成 → {input_dir}：{len(manifest)} 筆"
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
    import threading

    def _kill_hfss():
        try:
            from script.kill import kill as _kill
            _kill()
        except Exception as ke:
            print(f"  watchdog kill 失敗: {ke}")

    def _guard(fn, timeout, desc):
        """COM 呼叫留主執行緒（STA）,watchdog 執行緒超時殺 HFSS 讓呼叫拋錯——
        每一個會碰 COM 的環節都要有處決線（2026-07-11 教訓:重開卡死=永久凍結）。"""
        done_evt = threading.Event()

        def _wd():
            if not done_evt.wait(timeout):
                print(f"  ⏱ {desc} 超過 {timeout}s——殺 HFSS 解鎖")
                _kill_hfss()
        threading.Thread(target=_wd, daemon=True).start()
        try:
            return fn()
        finally:
            done_evt.set()

    def _open_sim():
        #! 開啟/重開會卡死（license/殭屍行程;218 實測 2026-07-11）→ watchdog + 3 試保險絲
        import time as _t
        last = None
        for attempt in range(3):
            def _mk():
                s = SinglePortRadSimulator(record_path=str(out), sweep_type=args.sweep)
                s.open()
                return s
            try:
                return _guard(_mk, 300, f"HFSS 開啟（第 {attempt + 1} 試）")
            except Exception as e:
                last = e
                print(f"  HFSS 開啟失敗（{attempt + 1}/3）: {e}")
                _kill_hfss()
                _t.sleep(15)
        raise SystemExit(f"HFSS 連續 3 次開不起來——疑似壞死,中止本批（{last}）")

    def _watchdog(done_evt, fired):
        #! 卡住偵測（資料工廠 2026-07-10）：COM 呼叫可以「無例外地永遠不回來」——單筆超時就殺
        #  HFSS 行程樹（script.kill,線上線容錯同款）,讓卡住的 COM 呼叫拋例外走 error 路徑。
        #  若殺完 COM 仍不返回（罕見）,第二層=外部 status 掃 results.json staleness 告警。
        if not done_evt.wait(getattr(args, "timeout", 900)):
            fired.set()
            print(f"  ⏱ 單筆超過 {getattr(args, 'timeout', 900)}s＝疑似卡住,強制終結 HFSS（watchdog）")
            try:
                from script.kill import kill as _kill
                _kill()
            except Exception as ke:
                print(f"  watchdog kill 失敗: {ke}")

    sim = _open_sim()
    fails = 0                                            # 連續失敗計數（HFSS 壞死保險絲）
    try:
        for k, (num, m) in enumerate(todo):
            p = torch.load(str(input_dir.joinpath(f"{m['id']}.pt")), weights_only=True)
            print(f"[{m['id']}] 模擬中… (本次第 {k + 1}/{len(todo)} 筆;manifest #{num + 1}/{len(manifest)})")
            done_evt, fired = threading.Event(), threading.Event()
            threading.Thread(target=_watchdog, args=(done_evt, fired), daemon=True).start()
            try:
                sim.start(num)
                result = sim(p)
                elapsed = sim.end()
            except Exception as e:                       #! 單筆失敗不炸整批：記 error、下一筆（比照線上 skip）
                done_evt.set()
                results[m["id"]] = {"error": ("watchdog_timeout: " if fired.is_set() else "") + str(e)}
                _flush()
                print(f"  ✗ {e}")
                fails += 1
                if fails >= getattr(args, "max_fail", 5):
                    raise SystemExit(f"連續 {fails} 筆失敗——HFSS 疑似壞死,中止本批"
                                     "（已完成部分已落檔,修復後重跑同指令即續）")
                _kill_hfss()                             # 先殺透再重開（quit 對殭屍 COM 會卡,跳過）
                sim = _open_sim()
                continue
            done_evt.set()
            fails = 0
            #? 接管防互踩:若本批綁 claim 且已被別台接管（stale takeover）→ 優雅退出,不再寫 store
            cp = getattr(args, "claim_path", None)
            if cp is not None:
                try:
                    owner = json.load(open(cp, encoding="utf-8")).get("machine")
                except Exception:
                    owner = None
                if owner != getattr(args, "claim_me", None):
                    print(f"⚠ claim 已被 {owner} 接管——本機停寫退出（避免 results.json 互踩）")
                    return

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
        try:
            _guard(sim.quit, 120, "HFSS 收尾關閉")
        except Exception:
            pass
    print(f"\n完成。結果：{results_path}；報表：python -m script.dedust report")


# ---------------------------------------------------------------- 資料工廠（NAS 派工,2026-07-10）
def _jobs_paths():
    """佇列=DATASET_PATH/jobs.json（人/agent 編輯）;狀態檔=jobs_state/<store>.{claim,done,fail}。"""
    sd = DATASET_PATH.joinpath("jobs_state")
    sd.mkdir(parents=True, exist_ok=True)
    return DATASET_PATH.joinpath("jobs.json"), sd


def jobs_add(args):
    """把一個批次加進 NAS 派工佇列（round 檔照常開、check-dup 照常跑——佇列只管「誰去燒」）。"""
    qp, _ = _jobs_paths()
    jobs = json.load(open(str(qp), encoding="utf-8")) if qp.exists() else []
    if any(j["store"] == args.store for j in jobs):
        raise SystemExit(f"{args.store} 已在佇列")
    if not DATASET_PATH.joinpath(args.input, "manifest.json").exists():
        raise SystemExit(f"{args.input} 無 manifest——先跑 select 與 check-dup")
    jobs.append(dict(input=args.input, store=args.store, prio=args.prio))
    tmp = str(qp) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=1)
    os.replace(tmp, qp)
    print(f"已入佇列: {args.store} (prio {args.prio});目前 {len(jobs)} 個 job")


def worker(args):
    """資料工廠 worker（Ricky 2026-07-10「三台都變資料收集系統,用 NAS 控制」）——正式機常駐:
    迴圈＝讀 jobs.json（prio 升冪）→ 跳過 done/被認領 → **原子認領**（jobs_state/<store>.claim,
    O_EXCL 建檔含機器 IP）→ run（斷點續跑＋單筆 watchdog＋連敗保險絲）→ 標 done → 下一個;
    佇列空 → 睡 --poll 秒再掃。**同 store 兩機並跑由 claim 檔擋掉**（results.json 互踩防護）。
    stale 接管:claim 存在但 store 無進度超過 --stale 分鐘（機器死了）→ 別台可接手續跑。
    停止:建 jobs_state/STOP（跑完當前 job 收工）或 Ctrl-C。run 觸發保險絲（連敗）→ 寫
    <store>.fail 並停機（HFSS 壞死要人工/告警介入;修復後刪 .fail+.claim 即可重派）。"""
    import time
    from antenna.utils.web import get_local_ip
    me = get_local_ip()
    qp, sd = _jobs_paths()
    print(f"worker 上線 @ {me}（poll {args.poll}s / 單筆 timeout {args.timeout}s / stale {args.stale}m）")
    while True:
        if sd.joinpath("STOP").exists():
            print("STOP 檔存在,worker 收工")
            break
        jobs = sorted(json.load(open(str(qp), encoding="utf-8")), key=lambda j: j.get("prio", 9)) \
            if qp.exists() else []
        picked = None
        for j in jobs:
            st = j["store"]
            if sd.joinpath(st + ".done").exists() or sd.joinpath(st + ".fail").exists():
                continue
            cp = sd.joinpath(st + ".claim")
            if cp.exists():
                try:
                    owner = json.load(open(str(cp), encoding="utf-8")).get("machine")
                except Exception:
                    owner = None
                if owner == me:                       # 自己的 claim=worker 重啟場景 → 直接續跑
                    print(f"↻ 續跑自己的 claim: {st}")
                    picked = j
                    break
                rp = DATASET_PATH.joinpath(st, "results.json")
                progressed = rp.exists() and (time.time() - os.path.getmtime(str(rp))) < args.stale * 60
                claim_fresh = (time.time() - os.path.getmtime(str(cp))) < args.stale * 60
                if progressed or claim_fresh:
                    continue
                print(f"接管 stale job {st}（無進度＞{args.stale} 分）")
                try:
                    os.remove(str(cp))
                except OSError:
                    continue
            try:
                fd = os.open(str(cp), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue                                  # 別台剛好搶到
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(dict(machine=me, at=time.strftime("%Y-%m-%d %H:%M:%S")), f)
            picked = j
            break
        if picked is None:
            if args.once:
                print("佇列空/無可認領,--once 收工")
                break
            print(f"({time.strftime('%H:%M:%S')}) 佇列無可認領,{args.poll}s 後再掃")   # 心跳:睡眠不裝死
            time.sleep(args.poll)
            continue
        st = picked["store"]
        print(f"▶ 認領 {st}（input {picked['input']}）")
        ns = argparse.Namespace(config=args.config, input=picked["input"], store=st, out=None,
                                sweep=args.sweep, timeout=args.timeout, max_fail=args.max_fail,
                                claim_path=str(sd.joinpath(st + ".claim")), claim_me=me)
        try:
            run(ns)
        except SystemExit as e:                          # 連敗保險絲:標 fail、停機等人工
            with open(str(sd.joinpath(st + ".fail")), "w", encoding="utf-8") as f:
                f.write(str(e))
            print(f"✗ {st} 中止:{e}\nworker 停機（修復 HFSS 後刪 jobs_state/{st}.fail 與 .claim 重派）")
            raise
        #! done 語義（2026-07-10 修）:「跑完」≠「全成功」——殘留 error 記進 .done 並顯性警告,
        #  哨兵 --factory 會列出;重派=刪 .done+.claim（三連敗同一筆=毒樣本嫌疑,人工判）。
        rp = DATASET_PATH.joinpath(st, "results.json")
        errs = []
        if rp.exists():
            _res = json.load(open(str(rp), encoding="utf-8"))
            errs = [i for i, v in _res.items() if "error" in v]
        with open(str(sd.joinpath(st + ".done")), "w", encoding="utf-8") as f:
            json.dump(dict(machine=me, at=time.strftime("%Y-%m-%d %H:%M:%S"),
                           errors=len(errs), error_ids=errs[:20]), f)
        print(f"✔ {st} 完成（殘留 error {len(errs)} 筆{': ' + ','.join(errs[:5]) if errs else ''}）")
        if args.once:
            break


def jobs_ls(args):
    """看佇列現況（人用;零 token）:每個 job 的 認領/進度/done/殘留 error。"""
    import time
    qp, sd = _jobs_paths()
    if not qp.exists():
        print("（無 jobs.json）")
        return
    for j in sorted(json.load(open(str(qp), encoding="utf-8")), key=lambda j: j.get("prio", 9)):
        st = j["store"]
        rp = DATASET_PATH.joinpath(st, "results.json")
        mp = DATASET_PATH.joinpath(j["input"], "manifest.json")
        total = len(json.load(open(str(mp), encoding="utf-8"))) if mp.exists() else "?"
        done_n = 0
        if rp.exists():
            res = json.load(open(str(rp), encoding="utf-8"))
            done_n = sum(1 for v in res.values() if "wm" in v)
        state = "排隊中"
        for tag in ("fail", "done", "claim"):
            fp = sd.joinpath(f"{st}.{tag}")
            if fp.exists():
                info = open(str(fp), encoding="utf-8").read()[:100]
                age = (time.time() - __import__("os").path.getmtime(str(fp))) / 60
                state = f"{tag}（{age:.0f} 分前）{info}"
                break
        print(f"[prio {j.get('prio', 9)}] {st}: {done_n}/{total} | {state}")


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

    s = sub.add_parser("select-r21harvest", help="R21 收割管線：帶外收割(SM過濾)+margin樂透(純隨機),血系加權+吸收贏家")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260711)
    s.add_argument("--sm", default="sm_reanchor7.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--lo", type=int, default=0, help="低側 realized 收割臂筆數（先決:排序回測過門檻才開）")
    s.add_argument("--wild", type=int, default=8, help="大跳彩票筆數（d 26-60,26px 死區持續複驗）")
    s.set_defaults(fn=select_r21harvest)

    s = sub.add_parser("select-r20gen", help="R20 一代選批：GA(SM粗篩)+隨機對照+碎片探索,三夾三機並行;gen>1 自動接代")
    s.add_argument("--gen", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260710)
    s.add_argument("--sm", default="sm_reanchor5.pth", help="粗篩用 SM 權重（NAS;每代重錨後換版本名）")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--ga", type=int, default=60)
    s.add_argument("--rand", type=int, default=60)
    s.add_argument("--frag", type=int, default=30)
    s.add_argument("--input-prefix", default="dedust_r20g", dest="input_prefix")
    s.set_defaults(fn=select_r20gen)

    s = sub.add_parser("select-r19data", help="R19 模型線資料批：王錨點×組件級算子隨機鏈,雙夾各 n 筆不重複(訓練用)")
    s.add_argument("--input-a", default="dedust_r19a_input", help="37 跑的夾(含 cc 公證 6 筆)")
    s.add_argument("--input-b", default="dedust_r19b_input", help="218 跑的夾")
    s.add_argument("--n", type=int, default=200, help="每夾 vargen 筆數")
    s.add_argument("--seed", type=int, default=20260709)
    s.set_defaults(fn=select_r19data)

    s = sub.add_parser("select-r18", help="R18 帶外二批：舊藏公證(b20/vpd2/vb43/x20)+低側家族構造化救援+c18手術")
    s.add_argument("--input", default="dedust_r18_input")
    s.set_defaults(fn=select_r18)

    s = sub.add_parser("select-r17", help="R17 帶外主目標：低側陷波條+翼幾何decouple+高側疊塊+公證a024")
    s.add_argument("--input", default="dedust_r17_input")
    s.set_defaults(fn=select_r17)

    s = sub.add_parser("select-r16b", help="R16 續批：翼修邊+等金屬再分配 (analysis-02 因果探針) + 添加收益圖擴錨")
    s.add_argument("--input", default="dedust_r16b_input")
    s.set_defaults(fn=select_r16b)

    s = sub.add_parser("select-r15v", help="R15 收尾：i02/g16/g14 公證+缺陷 + g14 rad 救援 + 理論模板探針")
    s.add_argument("--input", default="dedust_r15v_input")
    s.set_defaults(fn=select_r15v)

    s = sub.add_parser("select-addmap", help="R16 機理：添加收益圖(x00 單塊全掃) + 贏家塊歸因(逐塊移除)")
    s.add_argument("--input", default="dedust_addmap_input")
    s.set_defaults(fn=select_addmap)

    s = sub.add_parser("select-r15", help="R15 對照組：GA(push-button) vs 知情 vs 空間隨機,同組件空間+SM v4+同驗證預算")
    s.add_argument("--input-ga", default="dedust_r15ga_input", help="G+N 臂輸入夾（37）")
    s.add_argument("--input-inf", default="dedust_r15inf_input", help="I 臂輸入夾（218）")
    s.add_argument("--sm", default="sm_reanchor4.pth")
    s.add_argument("--pop", type=int, default=64)
    s.add_argument("--gens", type=int, default=40)
    s.add_argument("--verify", type=int, default=30, help="G/I 臂各驗證幾筆(實際=兩臂可湊的同額 min)")
    s.add_argument("--random-n", type=int, default=20)
    s.set_defaults(fn=select_r15)

    s = sub.add_parser("select-resize", help="R14 組件尺寸掃描：錨點 × {main,wings} × ±1,±2 圈 (組件級軸)")
    s.add_argument("--input", default="dedust_resize_input")
    s.set_defaults(fn=select_resize)

    s = sub.add_parser("select-ablate", help="元件消融(Ricky)：full→去各翼→累積,量每組件貢獻 (下主件永保留)")
    s.add_argument("--input", default="dedust_ablate_input")
    s.add_argument("--items", required=True, help="'來源夾:id,...' 要消融的冠軍")
    s.set_defaults(fn=select_ablate)

    s = sub.add_parser("select-blocks", help="R13 組數階梯系統對比：錨點固定,掃 3/4/5/6 塊拓撲 (add_block+SM篩)")
    s.add_argument("--input", default="dedust_blocks_input")
    s.add_argument("--ref2-input", default="dedust_ref2_input")
    s.add_argument("--sm", default="sm_reanchor3.pth")
    s.add_argument("--per-topo", type=int, default=12, help="每 (錨點,組數) SM 篩後留幾個")
    s.set_defaults(fn=select_blocks)

    s = sub.add_parser("select-crown", help="R12 收斂：top 候選公證×2 + 穩健 erode/dilate/缺陷 → 選穩健冠軍")
    s.add_argument("--input", default="dedust_crown_input")
    s.add_argument("--items", required=True, help="'來源夾:id,...' 跨批 top 三標候選")
    s.add_argument("--notarize", type=int, default=2, help="每候選公證重跑次數")
    s.add_argument("--defects", type=int, default=4, help="每候選局部缺陷 k1 樣本數(bake-off 調高)")
    s.set_defaults(fn=select_crown)

    s = sub.add_parser("select-family2", help="R12 破單一化：非 w17 pool 家族深掘（除塵+對稱化+SM篩）")
    s.add_argument("--input", default="dedust_family2_input")
    s.add_argument("--families", default="0,1,4", help="pool 家族編號（wm 降冪序,逗號分隔）")
    s.add_argument("--sm", default="sm_reanchor3.pth")
    s.add_argument("--per-family", type=int, default=12, help="每族取前幾個成員")
    s.add_argument("--top-per-family", type=int, default=14, help="每族 SM 篩後留幾個")
    s.set_defaults(fn=select_family2)

    s = sub.add_parser("select-probes", help="probes＋帶外批：c25公證+全對稱冠軍+搭橋+t07構造化+底緣精修")
    s.add_argument("--input", default="dedust_probes_input")
    s.add_argument("--ref2-input", default="dedust_ref2_input")
    s.add_argument("--ref3-input", default="dedust_ref3_input")
    s.set_defaults(fn=select_probes)

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
    s.add_argument("--timeout", type=int, default=900, help="單筆 watchdog 秒數（超時殺 HFSS 標 error 續跑;中位 160s/P90 176s）")
    s.add_argument("--max-fail", type=int, default=5, dest="max_fail", help="連續失敗幾筆判 HFSS 壞死中止")
    s.set_defaults(fn=run)

    s = sub.add_parser("worker", help="資料工廠 worker：常駐認領 NAS 佇列 job（jobs.json）自動跑批")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--sweep", default="Interpolating", choices=["Interpolating", "Discrete", "Fast"])
    s.add_argument("--poll", type=int, default=300, help="佇列空時幾秒掃一次")
    s.add_argument("--timeout", type=int, default=900)
    s.add_argument("--max-fail", type=int, default=5, dest="max_fail")
    s.add_argument("--stale", type=int, default=45, help="claim 無進度幾分鐘可被接管")
    s.add_argument("--once", action="store_true", help="只跑一個 job 就收工（測試用）")
    s.set_defaults(fn=worker)

    s = sub.add_parser("jobs-add", help="把批次加進派工佇列（select+check-dup 先跑完）")
    s.add_argument("--input", required=True)
    s.add_argument("--store", required=True)
    s.add_argument("--prio", type=int, default=5, help="小=先跑")
    s.set_defaults(fn=jobs_add)

    s = sub.add_parser("jobs-ls", help="看派工佇列現況（認領/進度/done/殘留 error）")
    s.set_defaults(fn=jobs_ls)

    s = sub.add_parser("report", help="匯總表（貼 round 檔 §4）")
    s.add_argument("--input", default=DEFAULT_INPUT)
    s.add_argument("--store", default=DEFAULT_STORE)
    s.set_defaults(fn=report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
