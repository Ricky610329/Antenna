# -*- coding: utf-8 -*-
"""
script/dedust.py — 批次 HFSS 驗證線（R7 起的研究主力）：開發機 select-* 生輸入 → 正式機 run 燒
HFSS → 任一機 report 看結果。輸入/結果都在 NAS（`DATASET_PATH/<name>_input/` 與 `<name>/`），
跨機共享；run 可中斷續跑（成功跳過、error 重試）＋批尾自動補測；每筆 solve 順收方向圖（rad/ 夾）。

現役流程（2026-07-12 起,資料工廠＋弱模型化;整鏈 runbook＝/batch-cycle skill）：
    開發機:  select-r25 --batch N --sm ... [--rad-key]     # 現役生成器（r22mix 機器;各 round 檔 §3=真相）
             check-dup --input X_input                     # 必跑,exit 1 不發車
             jobs-add --input X_input --store X --prio 3   # 入 NAS 佇列（填空池 prio 9）
             watch --stores X,Y,...                        # 收檔偵測（Monitor 直接掛）
             （判讀= python -m script.analyze batch --round R --batch N）
    正式機:  worker                                        # 常駐:認領佇列+watchdog+補測+tier-2 讓位
                                                           # +--selfgen 自產（佇列空也不停）
    公證:    select-repeat --source-input ... --id ... --n 2 → jobs-add --prio 2（判定=/notarize skill）

歷史 select 子命令（各 round 的輸入生成器,保留供重現;歷史見 docs/log/round-NN 檔）：
    select(R7 除塵)/select-r8(測繪)/select-r9(重驗)/select-refine1-3+wide(R10-11 精修)/
    select-occlude/tolerance/ablate/resize/blocks/crown/family2/probes(R11-16 因果批)/
    select-r15(GA vs 知情)/r16b/r17/r18(帶外戰役)/select-r19data(模型線)/select-r20gen(演化)/
    select-r21harvest(收割;--tag 填空池仍現役)/select-r22mix(分布組合,select-r23/r24/r25 的本體)

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


def diag_bridge(p) -> int:
    """對角橋數=4-連通組數 − 8-連通組數（analysis-05 口徑;僅對角接觸的接點數）。
    Ricky 2026-07-23:「不要對角線的那種比較重要」——對角黏著=壞徵兆（三標 14% vs 36%）,
    R36b3 起 select 記錄鍵,R37 進罰分;粉塵軸同日起忽略（研究期不作否決）。"""
    from scipy.ndimage import label
    p = np.asarray(p).reshape(25, 25) > 0.5
    _, k4 = label(p, structure=_CROSS)
    _, k8 = label(p, structure=np.ones((3, 3), bool))
    return int(k4 - k8)


def _grp_counts(p):
    """雙口徑組數 (4-連通, 8-連通)——組結構的指紋:合併會 −1、分裂會 +1、消滅會 −1。"""
    from scipy.ndimage import label as _lab
    p = np.asarray(p).reshape(25, 25) > 0.5
    return int(_lab(p, structure=_CROSS)[1]), int(_lab(p, structure=np.ones((3, 3), bool))[1])


def diag_clean(p0, k=99, mode="auto"):
    """對角清潔（組保持版;Ricky 2026-07-26「補實或移除都要符合組的規範,該組還是要維持住」）。

    背景=R41/R42 清潔階梯兩次全滅（fill2 即崩）:舊算子不分對角種類一律動,動到**橋接型**
    （8-連通靠它連、4-連通分兩塊）——補實=兩諧振器併一個/斷開=一組拆兩半,兩者都在改電路。
    本函式只做**雙口徑組數皆不變**的操作（4-conn 與 8-conn 組數同時守恆=無合併/分裂/消滅）:
      · 冗餘對角（兩端在 4-連通下已另有路徑）→ 可 fill（補正交格）或 brk（刪尖角）;
      · 橋接對角 → **一律不動**（結構性對角,是功能部件不是雜訊）。
    每步套用後複驗組數,不合規就跳過。回傳 (q, 已清數, 記錄 list)。
    mode: auto=優先 fill 失敗改 brk;fill/brk=只用該法。"""
    from scipy.ndimage import label as _lab
    p = (np.asarray(p0).reshape(25, 25) > 0.5).copy()
    base = _grp_counts(p)
    done, log = 0, []
    for r in range(24):
        if done >= k:
            break
        for c in range(24):
            if done >= k:
                break
            a, b, cx, d = p[r, c], p[r, c + 1], p[r + 1, c], p[r + 1, c + 1]
            if a and d and not b and not cx:
                ends, fills = [(r, c), (r + 1, c + 1)], [(r, c + 1), (r + 1, c)]
            elif b and cx and not a and not d:
                ends, fills = [(r, c + 1), (r + 1, c)], [(r, c), (r + 1, c + 1)]
            else:
                continue
            lab4 = _lab(p, structure=_CROSS)[0]
            if lab4[ends[0]] != lab4[ends[1]]:
                log.append((r, c, "bridge", "skip"))      # 橋接對角:動了必改組 → 保留
                continue
            cand = []
            if mode in ("auto", "fill"):
                cand += [("fill", f) for f in fills]
            if mode in ("auto", "brk"):
                cand += [("brk", e) for e in ends]
            for how, (rr, cc) in cand:
                if (rr, cc) == FEED:
                    continue
                q = p.copy()
                q[rr, cc] = (how == "fill")
                if _grp_counts(q) == base:                # 雙口徑組數守恆=組維持
                    p = q
                    done += 1
                    log.append((r, c, "redundant", how))
                    break
            else:
                log.append((r, c, "redundant", "nosafe"))
    p[FEED] = True
    return p, done, log


#? 組文法生成系統（Ricky 2026-07-26 拍板;decisions「組文法生成系統」）——資料驅動文法取代
#  手寫隨機。普查基礎（作戰區 7,011/合格 1,261）:間距鐵律=99.5% 貼 1 格縫;合格解 84% 零對角;
#  典型=主件~240px+中件1.4+小件1.2;質心熱圖離散（家族祖傳格點→G-A 用熱點,G-B 加溫脫格點）。
#  四候選:GA 忠實/GB 加溫/GC 可製造(零對角+碎片度旋鈕)/GD 左側仿生(骨架+星座,雙變體)。
#  KPI=資訊增益四尺（response 新穎度/誤差錨/苗子率/lo 解耦）,不用三標率評;判準=round-43 §1。
_GA_HOTSPOTS = [(3, 4), (4, 19), (6, 3), (18, 11), (13, 17), (14, 6)]   # qual 主件質心熱點（普查）
_GA2_SLOTS = [  # 組義字典（scratch 2026-07-26「組義字典」;質心 r,c／面積中位／出現率;前 3=三位一體）
    (18.4, 12.0, 241, 1.00), (3.9, 4.9, 79, 1.00), (4.0, 19.5, 73, 1.00),
    (9.5, 12.0, 6, 0.50), (6.2, 12.0, 4, 0.43), (10.5, 22.9, 6, 0.33),
    (10.5, 1.2, 6, 0.28), (10.1, 1.4, 3, 0.26), (10.5, 21.7, 4, 0.14),
]


def _place_rect(q, rng, h, w, r0=None, c0=None, gap=2, tries=60):
    """在 q 上找一個與既有金屬 Chebyshev 距離 ≥gap 的位置放 h×w 矩形;成功回 (r,c),失敗 None。
    gap=2=貼 1 格縫（間距鐵律）;r0/c0 給定=以該質心為中心 ±2 抖動（熱點模式）。"""
    from scipy.ndimage import binary_dilation
    forb = binary_dilation(q, structure=np.ones((2 * gap - 1, 2 * gap - 1), bool)) if q.any() else np.zeros_like(q)
    for _ in range(tries):
        if r0 is not None:
            rr = int(np.clip(r0 - h // 2 + rng.integers(-2, 3), 0, 25 - h))
            cc = int(np.clip(c0 - w // 2 + rng.integers(-2, 3), 0, 25 - w))
        else:
            rr, cc = int(rng.integers(0, 26 - h)), int(rng.integers(0, 26 - w))
        if not forb[rr:rr + h, cc:cc + w].any():
            return rr, cc
    return None


def _bite(q, rng, rr, cc, h, w, k):
    """矩形咬角 k 次（不破 4-連通:只咬角落格）——長出非矩形輪廓。"""
    for _ in range(k):
        corners = [(rr, cc), (rr, cc + w - 1), (rr + h - 1, cc), (rr + h - 1, cc + w - 1)]
        r_, c_ = corners[int(rng.integers(0, 4))]
        q[r_, c_] = False


def _rand_grammar(rng, gset="GA"):
    """組文法採樣一張（決定性=呼叫端給 seeded rng）。gset ∈ {GA, GB, GC, GD, GDd}。
    GA=忠實（主件熱點+合格解邊際分布）;GB=加溫（位置均勻）;GC=可製造（零對角保證+
    碎片度旋鈕+件≥2px 無粉塵）;GD=左側仿生零對角/GDd=帶對角變體（小件貼骨架對角接觸）。"""
    q = np.zeros((25, 25), bool)
    if gset == "GA2":
        #? GA v2 組義槽採樣（Ricky 2026-07-26 核准三升級③;R44 進槽）:字典逐槽——
        #  三位一體必放+調諧件按出現率擲骰;位置=簇質心 ±2 抖動;面積 ×U(0.7,1.3)、長寬比 U(0.7,1.4)。
        for (cr, cc0, area, pres) in _GA2_SLOTS:
            if rng.random() > pres:
                continue
            a_ = area * float(rng.uniform(0.7, 1.3))
            asp = float(rng.uniform(0.7, 1.4))
            h = int(np.clip(round(np.sqrt(a_ * asp)), 1, 20))
            w = int(np.clip(round(a_ / max(h, 1)), 1, 22))
            pos = _place_rect(q, rng, h, w, cr, cc0)
            if pos is None:
                if area > 100:                                        # 三位一體主件放不下=整張作廢
                    return None
                continue
            q[pos[0]:pos[0] + h, pos[1]:pos[1] + w] = True
            if area > 25:
                _bite(q, rng, pos[0], pos[1], h, w, int(rng.integers(0, 4)))
    elif gset in ("GA", "GB"):
        # 主件 150-300px:由 12-16×12-20 矩形咬角逼近
        h, w = int(rng.integers(12, 17)), int(rng.integers(12, 21))
        if gset == "GA":
            r0, c0 = _GA_HOTSPOTS[int(rng.integers(0, len(_GA_HOTSPOTS)))]
            pos = _place_rect(q, rng, h, w, r0, c0)
        else:
            pos = _place_rect(q, rng, h, w)
        if pos is None:
            return None
        q[pos[0]:pos[0] + h, pos[1]:pos[1] + w] = True
        _bite(q, rng, pos[0], pos[1], h, w, int(rng.integers(0, 4)))
        for _ in range(int(rng.integers(1, 3))):                      # 副大件 1-2（王朝 77/70 帶）
            h1, w1 = int(rng.integers(6, 11)), int(rng.integers(6, 13))
            p1 = _place_rect(q, rng, h1, w1)
            if p1:
                q[p1[0]:p1[0] + h1, p1[1]:p1[1] + w1] = True
                _bite(q, rng, p1[0], p1[1], h1, w1, int(rng.integers(0, 3)))
        for _ in range(int(rng.integers(1, 3))):                      # 中件 1-2
            h2, w2 = int(rng.integers(2, 6)), int(rng.integers(2, 6))
            p2 = _place_rect(q, rng, h2, w2)
            if p2:
                q[p2[0]:p2[0] + h2, p2[1]:p2[1] + w2] = True
        for _ in range(int(rng.integers(1, 3))):                      # 小件 1-2
            h3, w3 = int(rng.integers(1, 3)), int(rng.integers(1, 4))
            p3 = _place_rect(q, rng, h3, w3)
            if p3:
                q[p3[0]:p3[0] + h3, p3[1]:p3[1] + w3] = True
    elif gset == "GC":
        n = int(rng.integers(3, 13))                                  # 碎片度旋鈕
        budget = int(rng.integers(200, 420))
        h, w = int(rng.integers(8, 15)), int(rng.integers(8, 17))     # 主件縮小讓位碎片
        pos = _place_rect(q, rng, h, w)
        if pos is None:
            return None
        q[pos[0]:pos[0] + h, pos[1]:pos[1] + w] = True
        for _ in range(n - 1):
            if q.sum() >= budget:
                break
            h2, w2 = int(rng.integers(1, 6)), int(rng.integers(2, 7))
            if h2 * w2 < 2:
                w2 = 2                                                # 件 ≥2px（無粉塵）
            p2 = _place_rect(q, rng, h2, w2)
            if p2:
                q[p2[0]:p2[0] + h2, p2[1]:p2[1] + w2] = True
    elif gset in ("GD", "GDd"):
        # 骨架:主件 ~190+二件 ~80+三件 ~30。GD=實心矩形咬角（零對角）;
        # GDd=**對角塊鏈**（家族真形態:大件=斜接子塊鏈,diagb 13-16 來自骨架內部）——
        # 子塊 3-5×3-6 逐塊「角對角」斜接,一件 3-6 子塊 → 每件 2-5 個內部對角接點。
        for (hl, hu, wl, wu, bites, nsub) in [(13, 16, 13, 16, 5, 6), (8, 11, 8, 11, 3, 4), (5, 7, 5, 7, 1, 3)]:
            if gset == "GD":
                h2, w2 = int(rng.integers(hl, hu)), int(rng.integers(wl, wu))
                p2 = _place_rect(q, rng, h2, w2)
                if p2 is None:
                    return None
                q[p2[0]:p2[0] + h2, p2[1]:p2[1] + w2] = True
                _bite(q, rng, p2[0], p2[1], h2, w2, bites)
            else:
                hs, ws = int(rng.integers(4, 8)), int(rng.integers(4, 9))
                p2 = _place_rect(q, rng, hs, ws)
                if p2 is None:
                    return None
                r_, c_ = p2
                q[r_:r_ + hs, c_:c_ + ws] = True
                for _s in range(int(rng.integers(max(nsub - 2, 2), nsub + 1)) - 1):
                    h4, w4 = int(rng.integers(3, 7)), int(rng.integers(3, 8))
                    dr = 1 if rng.random() < 0.7 else -1              # 偏向下（家族住下半）
                    dc = 1 if rng.random() < 0.5 else -1
                    # 角對角斜接:新塊的角貼在現塊角的斜對面
                    r4 = r_ + (hs if dr > 0 else -h4)
                    c4 = c_ + (ws if dc > 0 else -w4)
                    if not (0 <= r4 <= 25 - h4 and 0 <= c4 <= 25 - w4):
                        break
                    blk = q[max(r4 - 1, 0):r4 + h4 + 1, max(c4 - 1, 0):c4 + w4 + 1]
                    if int(blk.sum()) != 1:                            # 只允許那顆斜接角
                        break
                    q[r4:r4 + h4, c4:c4 + w4] = True
                    r_, c_, hs, ws = r4, c4, h4, w4
        n_con = int(rng.integers(4, 7)) if gset == "GD" else int(rng.integers(7, 11))
        for _ in range(n_con):                                        # 星座（GDd 補金屬多帶幾顆）
            h3, w3 = int(rng.integers(1, 4)), int(rng.integers(2, 5))
            if gset == "GDd" and rng.random() < 0.6:
                # 帶對角變體:小件顯式「對角落位」——找金屬凸角,斜對角貼一件（正交鄰全空=真斜碰）
                placed = False
                for _t in range(40):
                    r_ = int(rng.integers(0, 24))
                    c_ = int(rng.integers(0, 24))
                    dr, dc = (1, 1) if rng.random() < 0.5 else (1, -1)
                    if not (0 <= c_ + dc * 1 <= 24 - (w3 - 1) * max(dc, 0)):
                        continue
                    r1, c1 = r_ + dr, c_ + dc
                    if not (0 <= r1 <= 25 - h3 and 0 <= c1 <= 25 - w3) or not q[r_, c_]:
                        continue
                    blk = q[max(r1 - 1, 0):r1 + h3 + 1, max(c1 - 1, 0):c1 + w3 + 1]
                    if int(blk.sum()) != 1:                            # 環域僅那顆凸角=純斜碰
                        continue
                    q[r1:r1 + h3, c1:c1 + w3] = True
                    placed = True
                    break
                if placed:
                    continue
            p3 = _place_rect(q, rng, h3, w3)
            if p3:
                q[p3[0]:p3[0] + h3, p3[1]:p3[1] + w3] = True
    else:
        raise ValueError(f"未知文法 {gset}")
    q[FEED] = True
    if not (140 <= int(q.sum()) <= 560):
        return None
    return q


def _group_mutate(p0, rng):
    """組級變異一式（R41 C 臂;Ricky 2026-07-25「組是變異單元」提案）:
    8-連通分組（9宮格含對角=一組）→ 骨架組(>25px)凍結;中件(5-25px)=修邊(長/縮)/平移;
    小件(≤4px)=平移/刪/複製/新增。另設「對角開關」——A 臂預跑發現 1px 對角接點=導通拓撲
    開關（4-conn 下骨架裂 24-25 組且合格個體間拓撲不同）,此算子**不受骨架凍結限制**。
    回傳 (q, op, diff_px) 或 None（該抽不可行,呼叫端重抽）。"""
    from scipy.ndimage import label as _lab, binary_dilation as _bd, binary_erosion as _be
    S8_ = np.ones((3, 3), bool)
    p0 = np.asarray(p0).reshape(25, 25) > 0.5
    lab, n = _lab(p0, structure=S8_)
    if n == 0:
        return None
    sizes = {i: int((lab == i).sum()) for i in range(1, n + 1)}
    mids = [i for i, s in sizes.items() if 5 <= s <= 25]
    smalls = [i for i, s in sizes.items() if s <= 4]
    q = p0.copy()

    def _fin(qq, op):
        qq[FEED] = True
        d = int((qq != p0).sum())
        return (qq, op + [d], d) if d else None

    #? 算子權重（analysis-07 2026-07-29:C/B 帶 E[Δwm⁺] 校準——shrink/del 安全但 best-of-N 下
    #  只值 grow 的 1/10,席位讓給 grow/move;組級包已由 chain 發車閘限錨 wm≤−2,故單一分布即 C/B 帶）
    op = str(rng.choice(["grow", "shrink", "move", "del", "dup", "spawn", "diag"],
                        p=[0.32, 0.08, 0.22, 0.06, 0.08, 0.10, 0.14]))
    if op in ("grow", "shrink") and mids:
        #? gap≤2（貼主件一步可橋）中件加權 ×2（analysis-07 探索性:E[best-of-18] +1.22→+1.50,未顯著）
        main_i = max(sizes, key=sizes.get)
        _mb = _bd(lab == main_i, structure=S8_)
        _wts = np.array([2.0 if gi != main_i and (_bd(lab == gi, structure=S8_) & _mb).any() else 1.0
                         for gi in mids])
        gi = int(rng.choice(mids, p=_wts / _wts.sum()))
        m = lab == gi
        cand = np.argwhere((_bd(m, structure=S8_) & ~p0) if op == "grow" else (m & ~_be(m)))
        if len(cand) == 0:
            return None
        if op == "grow":
            #? grow k 偏 2-3（analysis-07:k=1 勝率 16%/u=0.024 → k=2 30%/0.187 → k=3 39%/0.180）
            _km = min(3, len(cand))
            _kw = np.array([0.2, 0.4, 0.4][:_km])
            k = int(rng.choice(np.arange(1, _km + 1), p=_kw / _kw.sum()))
        else:
            k = int(rng.integers(1, min(3, len(cand)) + 1))
        for r, c in cand[rng.choice(len(cand), size=k, replace=False)]:
            q[r, c] = (op == "grow")
        return _fin(q, ["grp_" + op, sizes[gi]])
    if op == "move" and (smalls or mids):
        pool = smalls if (smalls and rng.random() < 0.7) else (mids or smalls)
        gi = int(rng.choice(pool))
        m = lab == gi
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)][int(rng.integers(0, 8))]
        rr, cc = np.argwhere(m).T
        nr, nc = rr + dr, cc + dc
        if nr.min() < 0 or nr.max() > 24 or nc.min() < 0 or nc.max() > 24 \
                or (p0 & ~m)[nr, nc].any():
            return None
        q[rr, cc] = False
        q[nr, nc] = True
        return _fin(q, ["grp_move", sizes[gi]])
    if op == "del" and smalls:
        gi = int(rng.choice(smalls))
        q[lab == gi] = False
        return _fin(q, ["grp_del", sizes[gi]])
    if op in ("dup", "spawn"):
        if op == "dup" and smalls:
            cells = np.argwhere(lab == int(rng.choice(smalls)))
            cells = cells - cells.min(axis=0)
        else:
            cells = np.argwhere(np.ones((int(rng.integers(1, 3)), int(rng.integers(1, 4))), bool))
        r0 = int(rng.integers(0, 25 - cells[:, 0].max()))
        c0 = int(rng.integers(0, 25 - cells[:, 1].max()))
        pad = np.zeros_like(p0)
        pad[cells[:, 0] + r0, cells[:, 1] + c0] = True
        if (_bd(pad, structure=S8_) & p0).any():       # 含 8 鄰淨空——保「獨立新組」語義
            return None
        return _fin(q | pad, ["grp_" + op, int(pad.sum())])
    if op == "diag":
        cand = []
        for r in range(24):
            for c in range(24):
                a, b, cx, d = p0[r, c], p0[r, c + 1], p0[r + 1, c], p0[r + 1, c + 1]
                if a and d and not b and not cx:
                    cand.append((r, c, c + 1))         # 主對角:make=填(r,c+1) break=清(r,c)
                elif b and cx and not a and not d:
                    cand.append((r, c + 1, c))         # 反對角:make=填(r,c) break=清(r,c+1)
        if not cand:
            return None
        r, c_br, c_mk = cand[int(rng.integers(0, len(cand)))]
        if rng.random() < 0.5:
            q[r, c_mk] = True                          # make:正交補橋=轉導通
        else:
            q[r, c_br] = False                         # break:斷對角一端
        return _fin(q, ["grp_diag", r * 25 + c_br])
    return None


def dyn_struct(p) -> bool:
    """王朝表型結構判（Ricky 2026-07-17 定案,decisions「王朝重定義」）:黑名單制只擋一種——
    「底部 1 大件（≥60px∧質心 row≥12）＋上半 ≥2 中件（≥12px∧質心 row<10）」;小碎塊(<12px)
    不進判定。驗證:王朝家族 100% 命中/功能判 lo>0 96%（全史八成困此結構=低側壓不下去主因）。
    用途:R33 起生成端/select 對無實測佐證的新樣本軟過濾（錨定臂豁免——親代有實測 lo 佐證）。"""
    from scipy.ndimage import gaussian_filter, label
    img = np.asarray(p).reshape(25, 25).astype(float)
    lab, n = label(gaussian_filter(img, 0.8) > 0.6)
    comps = []
    for i in range(1, n + 1):
        m = lab == i
        comps.append((int(m.sum()), float(np.argwhere(m)[:, 0].mean())))
    comps.sort(key=lambda c: -c[0])
    if not comps or comps[0][0] < 60 or comps[0][1] < 12:
        return False
    return len([c for c in comps[1:] if c[1] < 10 and c[0] >= 12]) >= 2


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
    下游契約（2026-07-29 補寫死;原「其餘皆追蹤欄」自左右側拆帳制〔2026-07-23〕起過時）:
      oob_bad（=gain_max−s11_min）=總帳判準;**oob_gain_max_lo/hi（絕對 Gain,分側）=
      紀錄鍵 usable_lo/hi 與鏈鍵 goal=lo/hi/tri 門檻所用**（見 decisions「左右側拆帳紀錄制」）;
      contrast_*/rolloff_*/oob_gain_argmax=追蹤欄,不進任何鍵。17 點 24-32GHz 尺專用。"""
    r = np.asarray(resp, dtype=float).reshape(2, -1)
    n = r.shape[1]
    lo = list(range(n_side))
    hi = list(range(n - n_side, n))
    far = lo + hi
    s11_min = float(r[0][far].min())
    gain_max = float(r[1][far].max())
    freqs = 26.5 + (np.arange(n) - 5) * 0.5
    edge_lo, edge_hi = float(r[1][5]), float(r[1][11])       # 帶緣 Gain（26.5/29.5）
    in_gain_min = float(r[1][5:12].min())                    # 帶內 Gain 最低點（26.5-29.5）
    return dict(oob_s11_min=round(s11_min, 2), oob_gain_max=round(gain_max, 2),
                oob_bad=round(gain_max - s11_min, 2),
                # 相對選擇性（Ricky 2026-07-12「帶內相對帶外高就可以」）:帶內min−帶外max,分側
                contrast_lo=round(in_gain_min - float(r[1][lo].max()), 2),
                contrast_hi=round(in_gain_min - float(r[1][hi].max()), 2),
                oob_gain_max_lo=round(float(r[1][lo].max()), 2),
                oob_gain_max_hi=round(float(r[1][hi].max()), 2),
                oob_s11_min_lo=round(float(r[0][lo].min()), 2),
                oob_s11_min_hi=round(float(r[0][hi].min()), 2),
                rolloff_lo=round(edge_lo - float(r[1][lo].max()), 2),
                rolloff_hi=round(edge_hi - float(r[1][hi].max()), 2),
                oob_gain_argmax=float(freqs[far][int(np.argmax(r[1][far]))]))


SEL_BUFFER, SEL_KAPPA = 0.15, 10.0                     # 可用解 wm buffer（R11 缺陷存活=margin 函數）/罰權


def sel_score(wm, rad, oob):
    """價值軸單一標量（Ricky 2026-07-12 定調,越低越好）:過線+buffer 後=純 oob_bad——
    帶內餘裕不再加分,只有壓帶外有收益;未過線罰 κ·缺口（保留梯度,修復臂可導引）。
    rad=None（無方向圖）視為未過。詳 decisions「價值軸修正」。"""
    pen = SEL_KAPPA * (max(0.0, SEL_BUFFER - wm) + max(0.0, -(rad if rad is not None else -1.0)))
    return round(float(oob + pen), 2)


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
                tri = _tri(r)
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
                tri = _tri(r)
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
    wild_target = args.wild * 40                      # wild=0 → 不生彩票池（舊 max(,1) 會空轉等不到）
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
    half = n_core // 2 if args.o < 0 else args.o      # --o 0=純樂透填空批（機器空檔用,不動判準）

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
    tag = args.tag or f"b{args.batch}"                # --tag=填空批命名（夾/id/family 全隔離,防撞正批）
    #? R23 命名規範:tag 直接帶 round 前綴（如 r22g1 → 夾 dedust_r22g1*、id m22g1_*）;無前綴=legacy r21
    fol = tag if tag.startswith("r2") else f"r21{tag}"
    idt = tag[1:] if tag.startswith("r2") else (args.tag or str(args.batch))
    for arm, idxs, src in (("oobharv", oi, cands), ("loharv", li, cands),
                           ("mlotto", mi, cands), ("wild", wi, wilds)):
        for j, i in enumerate(idxs):
            c = src[i]
            entries.append(dict(id=f"{arm[0]}{idt}_{j:03d}_{c['parent'][:12]}", kind=arm,
                                family=f"{arm.upper()}_{tag}", removed_px=0, **c["stats"],
                                source_id=c["parent"], ops=c["ops"], diff_px=c["d"],
                                pred_wm=c["pred_wm"], pred_oob=c["pred_oob"], pred_lor=c["pred_lor"],
                                _pat=c["pat"]))
    dirs = []
    #? 切片數=拖尾粒度（2026-07-11 Ricky「不浪費算力」）:夾多於機器,先跑完的機接下一夾,
    #  慢機只拖住自己那夾——batch5 起用 --shards 6（3 機 × 2）。
    for suf in "abcdefgh"[:args.shards]:
        dd = _dir(f"dedust_{fol}{suf}_input")
        dd.mkdir(parents=True, exist_ok=True)
        dirs.append(dd)
    manifests = [[] for _ in dirs]
    for k, e in enumerate(entries):
        pat = e.pop("_pat")
        b = k % len(dirs)
        manifests[b].append(e)
        torch.save(torch.tensor(pat, dtype=torch.float32), str(dirs[b].joinpath(e["id"] + ".pt")))
    for man, dd in zip(manifests, dirs):
        _save_manifest(man, dd)
    print(f"{fol}: 帶外收割 {len(oi)}+低側收割 {len(li)}+margin 樂透 {len(mi)}"
          f"+彩票 {len(wi)} → {len(dirs)} 夾 {[len(m) for m in manifests]}")


def select_r22mix(args):
    """R22 分布組合批（Ricky 拍板 2026-07-12「降王朝比例、探索其他分布;短期表現下滑可容忍」）:
      O 10 帶外過濾哨兵 —— 紅利耗盡（R21 O<M 連兩批、oob ρ 連三批死）,留一口監測復活
      M 50 王朝樂透 —— 兩池 70/30 照舊,付房租（三標穩定產出）
      C 40 冷支深耕 —— 冷支池**專屬配額**（batch4 三標 55% 非王朝標記=開採不足證據）
      Q 30 偏科生修復 —— 錨=深帶外帶內爛（w4_007 oob4.54 等）,pred_wm 降冪選=用 SM 活著的能力
      H 12 hslot 部分槽劑量 —— 低側構造法了斷（王×槽長 3/5/8/12）
      W 8 大跳彩票 —— d 26-60,死區持續複驗
    存活測試（發車前寫死）:C/Q 臂三標率 ≥6%（隨機基準）=礦脈活,連兩批 <6% 收臂;
    H 臂=lo_realized 劑量反應重現＋「兩批內三標且 lo_realized<+1.0 → 低側重啟,否則正式收案」;
    O 哨兵=前瞻 oob ρ ≥0.3 復活才回名額。紀錄候選門檻照 champions → 下批公證。id 前綴 {o,m,k,q,h,w}<5+batch>。"""
    from scipy.ndimage import label as _label
    from antenna.training import setup_responses
    from antenna.zoo import SURROGATES
    rng = np.random.default_rng(args.seed + args.batch)
    POISON = ("g2_029", "t14", "vg0795")
    DYN = ("c18", "vg0338", "vg0396", "g1_038", "g1_039", "r2_016", "r3_001")

    def loadp(fol, pid):
        f = _dir(fol).joinpath(pid + ".pt")
        if not f.exists():
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
            ("a024", "dedust_addmap_input", "a024_c25r9c11s3"),
            ("ccr9s2", "dedust_r17_input", "cc_c25_r6s2_r9s2"),
            ("ccx00", "dedust_r17_input", "cc_x00_r5s2_r8s3"),
            ("vg0765", "dedust_r19b_input", "vg0765_a024"),
            ("c25", "dedust_ref3_input", "c25_a15w10_2_22"),
            ("x00", "dedust_wide_input", "x00_c21k2"),
            ("g16", "dedust_r15ga_input", "g16_r15"),
            ("i02", "dedust_r15inf_input", "i02_r15")]
    #? 吸收=R21/R22 全史＋本輪前批贏家（兩池歸池照 DYN 標記;馬太由 70/30 抽樣稅制衡）
    rnd = getattr(args, "round", 22)
    prev = [f"dedust_r21b{b}{s}" for b in range(1, 6) for s in "abc"] \
        + [f"dedust_r21g{g}{s}" for g in (1, 2) for s in "abc"] \
        + [f"dedust_r22b{b}{s}" for b in range(1, 4) for s in "abcdef"] \
        + ["dedust_r22g1a", "dedust_r22g1b", "dedust_r22g1c"] \
        + [f"dedust_r{rnd}b{b}{s}" for b in range(1, args.batch) for s in "abcdefgh"]
    if rnd >= 25:                                        # R25 起:滾動吸收 R23..R(rnd−1) 全批+填空池+自產店（公證店 rNNn* 除外——重複測不進錨池）
        import re as _re
        pre = tuple([f"dedust_r{k}" for k in range(23, rnd)] + ["dedust_auto"])
        prev += [d for d in sorted(os.listdir(str(DATASET_PATH)))
                 if d.startswith(pre) and not d.endswith(("_input", "_src"))
                 and not _re.match(r"dedust_r\d+n", d)]
    for st in prev:
        rp = DATASET_PATH.joinpath(st, "results.json")
        if not rp.exists():
            continue
        res = json.load(open(str(rp), encoding="utf-8"))
        for i, r in res.items():
            if "wm" not in r or any(px in i for px in POISON):
                continue
            tri = _tri(r)
            if tri and (r["wm"][2] >= 0.15 or (r.get("oob_bad") or 99) < 9.5):
                ANCH.append((i, st + "_input", i))
    #? Q 臂偏科生錨（深帶外/帶內爛;單次值,修復目標=把帶內拉回來）
    SPEC = [("w4_007", "dedust_r21b4a_input", "w4_007_o1_062_vg033"),
            ("w4_003", "dedust_r21b4a_input", "w4_003_m1_050_c18"),
            ("o3_020", "dedust_r21b3c_input", "o3_020_o2_048_o1_03"),
            ("m2_046", "dedust_r21b2b_input", "m2_046_r2_016")]
    #? F 碎片/低側修復臂（R25;Ricky 2026-07-13「以資料多樣性再降王朝與根的比例」）:
    #  錨=歷史 oob_bad 極低但帶內/rad 爛的實測載體（D 臂產物＋R9 池頂族;analysis-03 ★複驗:
    #  低側可壓、碎片區=帶外乾淨載體）——目標=保帶外、修 wm/rad;鍵=pred_sel（同 D）。
    #  學費制 3 批,判準寫死於 round-25 §1。
    FRAG = [("d23b3_000", "dedust_r23b3c_input", "d23b3_000_denovo"),
            ("d23b3_001", "dedust_r23b3d_input", "d23b3_001_denovo"),
            ("d23b3_006", "dedust_r23b3c_input", "d23b3_006_denovo"),
            ("d24b1_010", "dedust_r24b1a_input", "d24b1_010_denovo"),
            ("d24b2_004", "dedust_r24b2a_input", "d24b2_004_denovo"),
            ("d24b3_013", "dedust_r24b3d_input", "d24b3_013_denovo"),
            ("t09", "dedust_r9_input", "t09_top"),
            ("t03", "dedust_r9_input", "t03_top"),
            ("t07", "dedust_r9_input", "t07_top"),
            ("n09", "dedust_r9_input", "n09_near"),
            ("p00", "dedust_r7_input", "p00_orig")]
    P, PS, PF = {}, {}, {}
    for name, fol, pid in ANCH:
        if name not in P and not any(px in name for px in POISON):
            P[name] = loadp(fol, pid)
    for name, fol, pid in SPEC:
        PS[name] = loadp(fol, pid)
    if getattr(args, "f", 0):
        for name, fol, pid in FRAG:
            PF[name] = loadp(fol, pid)
    #? 反馬太④誤差錨點外掛（2026-07-15）:analyze batch 每批寫 error_anchors.json
    #  （SM |pred−real| top 8）——錯哪補哪,自動進錨點池（無 DYN 標記→歸冷支）。
    eaf = DATASET_PATH.joinpath("error_anchors.json")
    if eaf.exists():
        try:
            _ea = json.load(open(str(eaf), encoding="utf-8")).get("anchors", [])
        except Exception:
            _ea = []
        n_ea = 0
        for e in _ea:
            nm = "err_" + e["id"][:14]
            if nm in P or any(px in e["id"] for px in POISON):
                continue
            fe = _dir(e["input"]).joinpath(e["id"] + ".pt")
            if fe.exists():
                P[nm] = np.asarray(torch.load(str(fe), weights_only=True)).reshape(25, 25) > 0.5
                n_ea += 1
        if n_ea:
            print(f"誤差錨點外掛 +{n_ea}（error_anchors.json;錯哪補哪）")
    dyn_names = [n for n in P if any(m in n for m in DYN)]
    cold_names = [n for n in P if n not in dyn_names]
    print(f"r{getattr(args, 'round', 22)} b{args.batch} 錨點 {len(P)}"
          f"（王朝 {len(dyn_names)}/冷支 {len(cold_names)}）＋偏科生 {len(PS)}")

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
    hist0 = list(hist)                                    # 生成前快照（新穎性距離只對「真歷史」量,不含本批自身）

    def _mutate(src_pool, pick, want_wild=False):
        aname = pick()
        q = src_pool[aname]
        chain = []
        n_ops = int(rng.choice((2, 3, 4))) if want_wild else int(rng.choice((1, 2), p=(.65, .35)))
        for _ in range(n_ops):
            fn = OPS[int(rng.choice(len(OPS), p=ws))][0]
            q, desc = fn(q)
            if q is None:
                return None
            chain.append(desc)
        if not chain:
            return None
        q = (np.asarray(q).reshape(25, 25) > 0.5).copy()
        q[FEED] = True
        q, _n = strip_small(q, 4)
        q = _ensure_feed_pad(q, 4)
        st = piece_stats(q)
        if st["n_1px"] > 0 or not (230 <= st["metal_px"] <= 520):
            return None
        d = int((q != src_pool[aname]).sum())
        k = q.tobytes()
        if k in hist:
            return None
        hist.add(k)
        return dict(pat=q, parent=aname, ops=chain, d=d, stats=st)

    #? 根多樣性稅（R24 降根計畫;--root-cap>0 生效）:錨點沿 source_id 走到池根,
    #  單一根的已接受占比 ≤ cap——治「同山不同坡面」打轉（credit 實測:五紀錄鏈三條同根 g1_038）。
    _ROOT_IDX, _ROOT_CACHE = {}, {}
    if getattr(args, "root_cap", 0):
        for fol in _all_input_folders():
            for m in json.load(open(str(DATASET_PATH.joinpath(fol, "manifest.json")), encoding="utf-8")):
                _ROOT_IDX.setdefault(m["id"], m.get("source_id"))

    def _root(name):
        if name in _ROOT_CACHE:
            return _ROOT_CACHE[name]
        seen, cur = set(), name
        while cur in _ROOT_IDX and _ROOT_IDX[cur] and cur not in seen:
            seen.add(cur)
            cur = _ROOT_IDX[cur]
        _ROOT_CACHE[name] = cur
        return cur

    def pick_two_pool():
        #? 王朝/冷支抽樣比:R21 起 70/30;2026-07-15 戰略換軸（decisions）降 40/60——王系鄰域
        #  資料 ikpi ±0.05（教不了 SM）,錨點權重讓給冷支/新血。fallback 0.7=舊 select 重現性。
        dfrac = getattr(args, "dyn_frac", 0.7)
        pool = dyn_names if (rng.random() < dfrac and dyn_names) else cold_names
        return pool[int(rng.integers(0, len(pool)))] if pool else list(P)[0]

    def pick_cold():
        return cold_names[int(rng.integers(0, len(cold_names)))]

    def pick_spec():
        return list(PS)[int(rng.integers(0, len(PS)))]

    def pick_frag():
        return list(PF)[int(rng.integers(0, len(PF)))]

    #? 王系相似度稅（Ricky 2026-07-14:「資料集跟模型自我正向強化——再降低和王系高度相同的」）:
    #  根稅盲區=掛別根但長得像王（冷支 g16/a024 系同屬 w17 實心語言）。d_dyn=對全部王朝系錨點
    #  的最小 Hamming;--dyn-simcap>0 時,「d_dyn<12 的近王樣本」批內佔比 ≤ cap（同 root-cap 語法）。
    SIM_T = 20                                            # Ricky 2026-07-14 再壓:12→20（家族語言多落 d13-25,原漏抓）
    _dyn_pack = None
    if getattr(args, "dyn_simcap", 0) and dyn_names:
        _dyn_pack = np.packbits(np.stack([P[n].reshape(-1) for n in dyn_names]).astype(np.uint8), axis=1)
    _POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)

    def _near_dyn(pat):
        if _dyn_pack is None:
            return False
        q = np.packbits(pat.reshape(-1).astype(np.uint8))
        return int(_POP[np.bitwise_xor(_dyn_pack, q)].sum(axis=1).min()) < SIM_T

    def _gen(src, pick, target, dlo=1, dhi=25, wild=False):
        out, tries, counts, near = [], 0, {}, 0
        cap = getattr(args, "root_cap", 0)
        scap = getattr(args, "dyn_simcap", 0)
        while len(out) < target and tries < target * 30:
            tries += 1
            c = _mutate(src, pick, want_wild=wild)
            if c is None or not (dlo <= c["d"] <= dhi):
                continue
            if cap:
                rt = _root(c["parent"])
                if len(out) >= 12 and (counts.get(rt, 0) + 1) / (len(out) + 1) > cap:
                    hist.discard(c["pat"].tobytes())      # 退回查重集,別浪費 pattern
                    continue
                counts[rt] = counts.get(rt, 0) + 1
            if scap:
                nd = _near_dyn(c["pat"])
                if nd and len(out) >= 12 and (near + 1) / (len(out) + 1) > scap:
                    hist.discard(c["pat"].tobytes())
                    continue
                near += int(nd)
            out.append(c)
        return out

    #? 漏斗放大（2026-07-16 Ricky 拍板①:SM 打分免費,候選池 ×2——9k 資料的 SM 該篩更大的池）
    #? 漏斗二次放大（2026-07-30 Ricky 拍板:「SM 推理遠快於 HFSS」——v87 儀器換代後遠區誤差
    #  1.7→0.9,可信域擴張,虛擬預篩劑量 ×3;效率評 over ≥5 輪〔擴散型介入,decisions〕）
    core = _gen(P, pick_two_pool, (args.o + args.m) * 72)          # O+M+I 共池（24→72 倍候選）
    coldp = _gen(P, pick_cold, args.c * 48)                         # C 冷支專屬（16→48 倍）
    specp = _gen(PS, pick_spec, args.q * 36)                        # Q 修復池
    wildp = _gen(P, pick_two_pool, args.wild * 90, dlo=26, dhi=60, wild=True)
    fragp = _gen(PF, pick_frag, getattr(args, "f", 0) * 36, dhi=60) if PF else []   # F 修復池（粉塵錨 strip 後 d 大,放寬到 60）

    def _skeleton(p0):
        """骨架萃取（與 analysis/mesh 同口徑:高斯 σ0.8×門檻 0.6,≥6px 質量塊）→ blk footprint 聯集。"""
        from scipy.ndimage import gaussian_filter as _gf
        dens = _gf(p0.astype(float), 0.8, mode="constant")
        lab_m, nm = _label(dens > 0.6, structure=np.ones((3, 3), dtype=bool))
        blk = np.zeros_like(p0)
        for g in range(1, nm + 1):
            reg = lab_m == g
            if p0[reg].sum() >= 6:
                blk |= reg
        return blk

    #? Y 塊內 rad 手術臂（R28;R27 結構定案的直接應用）：錨=R27 三+二顆 half 電性半成品
    #  （wm/低側達標、全卡 rad −2~−4）;手術=挖點/開槽/中帶清理,**全限 blk mask 內、網布凍結**
    #  （生成後驗證塊外像素零變動,違者棄）;鍵=maximin(pred_wm, pred_rad)（rad 頭復鍵中）。
    #  判準寫死於 round-28 §1;kind=surgery,id 前綴 y。
    surgp = []
    if getattr(args, "surgery", 0):
        SURG = [("p00h", "dedust_r27b1a_input", "n27b1_018_p00"),
                ("t03h", "dedust_r27b1b_input", "n27b1_019_t03"),
                ("t07h", "dedust_r27b1c_input", "n27b1_020_t07"),
                ("t09h", "dedust_r27b1d_input", "n27b1_021_t09"),
                ("n09h", "dedust_r27b1f_input", "n27b1_017_n09")]
        for name, fol, pid in SURG:
            p0 = loadp(fol, pid)
            blk = _skeleton(p0)
            in_idx = np.flatnonzero((p0 & blk).reshape(-1))
            band = np.zeros_like(p0)
            band[10:15] = True
            band_idx = np.flatnonzero((p0 & blk & band).reshape(-1))
            target = getattr(args, "surgery", 0) * 10
            tries, made = 0, 0
            while made < target and tries < target * 40:
                tries += 1
                q = p0.copy()
                mode = int(rng.integers(0, 3))
                if mode == 0:                                     # 挖點 1-6px
                    k = int(rng.integers(1, 7))
                    q.reshape(-1)[rng.choice(in_idx, size=min(k, len(in_idx)), replace=False)] = False
                    ops = ["surg_carve", k]
                elif mode == 1:                                   # 塊內水平槽 1×L（hslot=rad 大旋鈕知識）
                    rr_, cc_ = np.where(p0 & blk)
                    j = int(rng.integers(0, len(rr_)))
                    r0, c0 = int(rr_[j]), int(cc_[j])
                    L = int(rng.integers(3, 9))
                    seg_mask = blk[r0, c0:c0 + L]
                    q[r0, c0:c0 + L][seg_mask] = False
                    ops = ["surg_slot", r0, c0, L]
                else:                                             # 中帶清理（analysis-03:高 rad 中帶乾淨）
                    if not len(band_idx):
                        continue
                    k = int(rng.integers(2, 9))
                    q.reshape(-1)[rng.choice(band_idx, size=min(k, len(band_idx)), replace=False)] = False
                    ops = ["surg_midband", k]
                q[FEED] = True
                q = _ensure_feed_pad(q, 4)
                if ((q != p0) & ~blk).any():                      # 網布凍結驗證:塊外零變動
                    continue
                st = piece_stats(q)
                if not (180 <= st["metal_px"] <= 560):
                    continue
                kb = q.tobytes()
                if kb in hist:
                    continue
                hist.add(kb)
                made += 1
                surgp.append(dict(pat=q, parent=name, ops=[ops], d=int((q != p0).sum()), stats=st))

    #? B 塊級承重圖探針（R28b2;Ricky 2026-07-14「對 4-4 每個區塊調整的重要性分析」——R16 承重圖
    #  升到骨架尺度）：錨=t07h/p00h（電性最好的兩顆 half）;每個密度分塊兩探針=ablate（整塊挖除）
    #  /halve（塊內棋盤挖半）——決定性零 rng;量 Δwm/Δrad/Δ低側=塊 × 三軸重要性表。kind=blockmap,id 前綴 b。
    bmapp = []
    if getattr(args, "blockmap", 0):
        for name, fol, pid in (("t07h", "dedust_r27b1c_input", "n27b1_020_t07"),
                               ("p00h", "dedust_r27b1a_input", "n27b1_018_p00")):
            p0 = loadp(fol, pid)
            from scipy.ndimage import gaussian_filter as _gf2
            dens = _gf2(p0.astype(float), 0.8, mode="constant")
            lab_b, nb = _label(dens > 0.6, structure=np.ones((3, 3), dtype=bool))
            gidx = [g for g in range(1, nb + 1) if p0[lab_b == g].sum() >= 6]
            for g in gidx:
                reg = (lab_b == g) & p0
                for mode, opn in (("ablate", "bm_ablate"), ("halve", "bm_halve")):
                    q = p0.copy()
                    if mode == "ablate":
                        q[reg] = False
                    else:
                        ii2, jj2 = np.indices(p0.shape)
                        q[reg & (((ii2 + jj2) % 2) == 0)] = False
                    q[FEED] = True
                    q = _ensure_feed_pad(q, 4)
                    st = piece_stats(q)
                    if not (150 <= st["metal_px"] <= 560):
                        continue
                    kb = q.tobytes()
                    if kb in hist:
                        continue
                    hist.add(kb)
                    bmapp.append(dict(pat=q, parent=name, ops=[[opn, int(g), int(reg.sum())]],
                                      d=int((q != p0).sum()), stats=st))

    #? U 承重圖導引組合手術（R28b3;b2 承重圖首份真值的直接應用）:錨=t07h,凍結命脈塊
    #  {4,7}與中承重{9,10},只動低承重塊 {2,5,6,8}（b2 實測:塊2 halve=wm−0.39 換 rad+1.42 lo−2.30）——
    #  抽 2-3 塊組合 halve/ablate（單塊=B 臂已測,hist 自動擋）;判準=疊加性+低側組合手術,寫死 round-28 §1 b3。
    #  kind=bmix,id 前綴 u。
    bmixp = []
    if getattr(args, "bmix", 0):
        p0 = loadp("dedust_r27b1c_input", "n27b1_020_t07")
        from scipy.ndimage import gaussian_filter as _gf3
        dens3 = _gf3(p0.astype(float), 0.8, mode="constant")
        lab3, nb3 = _label(dens3 > 0.6, structure=np.ones((3, 3), dtype=bool))
        LOWLOAD = [g for g in (2, 5, 6, 8) if g <= nb3]
        target = getattr(args, "bmix", 0)
        tries = 0
        while len(bmixp) < target and tries < target * 60:
            tries += 1
            sel = rng.choice(LOWLOAD, size=int(rng.integers(2, 4)), replace=False)
            q = p0.copy()
            combo = []
            for g in sorted(int(x) for x in sel):
                reg = (lab3 == g) & p0
                op = ("halve", "ablate")[int(rng.integers(0, 2))]
                if op == "ablate":
                    q[reg] = False
                else:
                    ii3, jj3 = np.indices(p0.shape)
                    q[reg & (((ii3 + jj3) % 2) == 0)] = False
                combo += [g, op]
            q[FEED] = True
            q = _ensure_feed_pad(q, 4)
            if ((q != p0) & ~((lab3 > 0) & p0)).any():        # 塊外零變動（同 Y 臂網布凍結語義）
                continue
            st = piece_stats(q)
            if not (150 <= st["metal_px"] <= 560):
                continue
            kb = q.tobytes()
            if kb in hist:
                continue
            hist.add(kb)
            bmixp.append(dict(pat=q, parent="t07h", ops=[["surg_bmix"] + combo],
                              d=int((q != p0).sum()), stats=st))

    #? G 臂（R29 主力;Ricky 2026-07-14 拍板「G 臂多跑一點」）:sm_invert gen 的 staging 候選讀入——
    #  梯度反傳生成（四帶 mix:free/surg/champ/oobp）;band/anchor 記進 ops,HFSS 回來後
    #  pred vs realized=各帶 adversarial 率=SM 盲區地圖。kind=grad,id 前綴 g;查重=hist+check-dup 雙保險。
    gradp = []
    if getattr(args, "g", 0) and getattr(args, "gstage", ""):
        gs = os.path.abspath(getattr(args, "gstage"))
        smf = json.load(open(os.path.join(gs, "staging_manifest.json"), encoding="utf-8"))
        for m in smf:
            q = np.asarray(torch.load(os.path.join(gs, m["id"] + ".pt"),
                                      weights_only=True)).reshape(25, 25) > 0.5
            kb = q.tobytes()
            if kb in hist:
                continue
            hist.add(kb)
            gradp.append(dict(pat=q, parent=f"{m['band']}_{m['anchor']}",
                              ops=[["grad", m["band"], m["anchor"], m["dlim"]]],
                              d=int(m["d"]), stats=piece_stats(q)))

    #? L 低側據點臂（R30;R29b3 gap 區破冰的擴張）:錨=lo ≤−3 的 7 顆據點（全 wm 爛）——
    #  鄰域變異找「lo 深∧wm 近活」中繼點;選拔鍵=r_feed 高者優先（analysis-05:feed 主件佔比
    #  =帶外最強旋鈕 ρ−0.48——結構先驗首次進 select）。kind=lobeach,id 前綴 l。
    lbp = []
    if getattr(args, "lbeach", 0) and getattr(args, "round", 0) >= 39:
        #? R39 左側家族化錨組（決策=round-39 §1）:錨=首例+近親（oob 6.8-7.1 帶）;半 ref 半 rej 續帳
        #  （balance 常駐);diagb 方向性續;判準=合格變異體 ≥5=族成立。
        NEWLAND = [("fm_c8p03", "dedust_c8tri_p03_input", "c8trip03_01"),
                   ("fm_c10p02", "dedust_c10tri_p02_input", "c10trip02_07"),
                   ("fm_c6t5p06", "dedust_c6tri5_p06_input", "c6tri5p06_21"),
                   ("fm_s38s1", "dedust_r38s1_input", "s38s1_18")]
        target = getattr(args, "lbeach", 0)
        cand_pool = []
        for name, fol, pid in NEWLAND:
            p0 = loadp(fol, pid)
            db0 = diag_bridge(p0)
            for j in range(target * 6 // len(NEWLAND) + 2):
                q = p0.copy()
                d_ = int(rng.integers(1, 9)) if j % 5 < 3 else int(rng.integers(9, 26))
                q.ravel()[rng.choice(625, size=d_, replace=False)] ^= True
                q[FEED] = True
                st_ = piece_stats(q)
                if not (180 <= st_["metal_px"] <= 560) or q.tobytes() in hist                         or diag_bridge(q) > db0:
                    continue
                hist.add(q.tobytes())
                cand_pool.append(dict(pat=q, parent=name, ops=[["fm_flip", d_]],
                                      d=int((q != p0).sum()), stats=st_))
        _cfgL = load_config(args.config)
        _labL = PORT_SPECS[_cfgL.port]["labels"]
        _nptsL = sum(_cfgL.targets[_labL[0]]["width"])
        from antenna.zoo import SURROGATES as _SUR_L
        #? 打分改 two（R39 絕對值換裝同步）——sm_two<vn> 在用 two,否則退 mlp
        _vnL = "".join(ch for ch in str(args.sm) if ch.isdigit())
        _ftL = DATASET_PATH.joinpath(f"sm_two{_vnL}.pth")
        if _ftL.exists():
            _sml = _SUR_L["cnn2"](os.path.join(REPO, "tmp", "dedust"), 25 * 25, (len(_labL), _nptsL))
            _sml.pre_load_model(_ftL, strict=True)
        else:
            _sml = _SUR_L["mlp"](os.path.join(REPO, "tmp", "dedust"), 25 * 25, (len(_labL), _nptsL))
            _sml.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
        _sml.model.eval()
        with torch.no_grad():
            _rawl = _sml.model(torch.stack([torch.tensor(c["pat"], dtype=torch.float32).reshape(-1)
                                            for c in cand_pool])).reshape(len(cand_pool), len(_labL), _nptsL)
        for k2, c in enumerate(cand_pool):
            c["_pw"] = float(worst_margin(_rawl[k2], _labL, _cfgL.targets)[0])
        #? lo 判別器進鍵（R39 判準達成:批前瞻 ρ 0.756/0.717 連兩批 ≥0.5）——F 臂 gate:
        #  pred_lo>-1 者出池（軟門檻,錨群 lo −2~−3.6;守家族左側身分）
        _flh = DATASET_PATH.joinpath(f"sm_lohead{_vnL}.pth")
        if _flh.exists():
            import torch.nn as _nnF
            _lhF = _nnF.Sequential(_nnF.Conv2d(1, 32, 3, padding=1), _nnF.ReLU(), _nnF.MaxPool2d(2),
                                   _nnF.Conv2d(32, 64, 3, padding=1), _nnF.ReLU(), _nnF.MaxPool2d(2),
                                   _nnF.Flatten(), _nnF.Linear(64 * 6 * 6, 256), _nnF.ReLU(),
                                   _nnF.Linear(256, 2))
            _lhF.load_state_dict(torch.load(str(_flh), weights_only=True))
            _lhF.eval()
            with torch.no_grad():
                _ploF = torch.cat([_lhF(torch.stack([torch.tensor(c["pat"], dtype=torch.float32)
                                                     .reshape(1, 25, 25) for c in cand_pool[i:i + 256]]))
                                   for i in range(0, len(cand_pool), 256)])[:, 1].numpy()
            keepF = [i for i in range(len(cand_pool)) if _ploF[i] <= -1.0]
            print(f"F 臂 lo gate（進鍵 R39）: {len(cand_pool)}→{len(keepF)}（pred_lo≤−1）")
            if len(keepF) >= target:
                cand_pool = [cand_pool[i] for i in keepF]
        order_l = sorted(range(len(cand_pool)), key=lambda i: -cand_pool[i]["_pw"])
        n_ref = target // 2
        refs = [cand_pool[i] for i in order_l[:n_ref]]
        lower = order_l[len(order_l) // 2:]
        rejs = [cand_pool[i] for i in rng.choice(lower, size=min(target - n_ref, len(lower)),
                                                 replace=False)]
        for c in refs:
            c["sel_by"] = "ref"
        for c in rejs:
            c["sel_by"] = "rej"
        lbp = refs + rejs
        for c in lbp:
            c.pop("_pw", None)
    elif getattr(args, "lbeach", 0) and getattr(args, "round", 0) >= 37:
        #? R37 左側大陸錨組（換系統戰略,decisions 2026-07-23）:錨=tri 前緣（c2rad 系雙錨）+
        #  t07/l31b2 家族多樣;SM 過濾 balance（Ricky:半 ref=SM top/半 rej=SM 判死下半區均勻抽,
        #  sel_by 記帳→量測假陰性）;diagb 方向性過濾（變體不得比錨增對角橋——左側家族天生
        #  diagb 14-16,絕對否決會殺整個大陸,保守解=世代往下壓）。
        NEWLAND = [("nl_c2r10", "dedust_c2rad_p10_input", "c2radp10_21"),
                   ("nl_c2r09", "dedust_c2rad_p09_input", "c2radp09_16"),
                   ("nl_t07", "dedust_r9_input", "t07_top"),
                   ("nl_l31b2", "dedust_r31b2f_input", "l31b2_005_lb_n09")]
        target = getattr(args, "lbeach", 0)
        cand_pool = []
        for name, fol, pid in NEWLAND:
            p0 = loadp(fol, pid)
            db0 = diag_bridge(p0)
            for j in range(target * 6 // len(NEWLAND) + 2):
                q = p0.copy()
                d_ = int(rng.integers(1, 16)) if j % 5 < 3 else int(rng.integers(16, 41))
                q.ravel()[rng.choice(625, size=d_, replace=False)] ^= True
                q[FEED] = True
                st_ = piece_stats(q)
                if not (180 <= st_["metal_px"] <= 560) or q.tobytes() in hist \
                        or diag_bridge(q) > db0:
                    continue
                hist.add(q.tobytes())
                cand_pool.append(dict(pat=q, parent=name, ops=[["nl_flip", d_]],
                                      d=int((q != p0).sum()), stats=st_))
        #? 池內 SM 打分（ref/rej 需要;局部載入,與後段全域打分同權重檔;cfg/labels 尚未定義=自備）
        _cfgL = load_config(args.config)
        _labL = PORT_SPECS[_cfgL.port]["labels"]
        _nptsL = sum(_cfgL.targets[_labL[0]]["width"])
        from antenna.zoo import SURROGATES as _SUR_L
        _sml = _SUR_L["mlp"](os.path.join(REPO, "tmp", "dedust"), 25 * 25, (len(_labL), _nptsL))
        _sml.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
        _sml.model.eval()
        with torch.no_grad():
            _rawl = _sml.model(torch.stack([torch.tensor(c["pat"], dtype=torch.float32).reshape(-1)
                                            for c in cand_pool])).reshape(len(cand_pool), len(_labL), _nptsL)
        for k2, c in enumerate(cand_pool):
            c["_pw"] = float(worst_margin(_rawl[k2], _labL, _cfgL.targets)[0])
        order_l = sorted(range(len(cand_pool)), key=lambda i: -cand_pool[i]["_pw"])
        n_ref = target // 2
        refs = [cand_pool[i] for i in order_l[:n_ref]]
        lower = order_l[len(order_l) // 2:]                      # SM 判死下半區
        rejs = [cand_pool[i] for i in rng.choice(lower, size=min(target - n_ref, len(lower)),
                                                 replace=False)]
        for c in refs:
            c["sel_by"] = "ref"
        for c in rejs:
            c["sel_by"] = "rej"
        lbp = refs + rejs
        for c in lbp:
            c.pop("_pw", None)
    elif getattr(args, "lbeach", 0) and getattr(args, "round", 0) >= 34:
        #? R34 去王朝錨組（表型 40% 線的錨組解）:錨全換非王朝結構筆（爬山鏈+t03r 同框系）。
        RADGATE = [("dd_s119", "dedust_r33s1_input", "s1_19_g32b3_034_"),
                   ("dd_s218", "dedust_r33s2_input", "s2_18_s1_19_g32b"),
                   ("dd_l005", "dedust_r33b2f_input", "l33b2_005_lb_t03r"),
                   ("dd_l007", "dedust_r33b2b_input", "l33b2_007_lb_t03r")]
        from scipy.ndimage import label as _lab4

        def _rfeed4(q):
            lab_, n_ = _lab4(q, structure=_CROSS)
            g_ = lab_[FEED]
            return float((lab_ == g_).sum() / max(q.sum(), 1)) if g_ > 0 else 0.0
        target = getattr(args, "lbeach", 0)
        cand_pool = []
        for name, fol, pid in RADGATE:
            p0 = loadp(fol, pid)
            for j in range(target * 4 // len(RADGATE) + 2):
                q = p0.copy()
                d_ = int(rng.integers(1, 16)) if j % 5 < 3 else int(rng.integers(16, 41))
                q.ravel()[rng.choice(625, size=d_, replace=False)] ^= True
                q[FEED] = True
                st_ = piece_stats(q)
                if not (180 <= st_["metal_px"] <= 560) or q.tobytes() in hist:
                    continue
                hist.add(q.tobytes())
                cand_pool.append(dict(pat=q, parent=name, ops=[["lb_flip", d_]],
                                      d=int((q != p0).sum()), stats=st_, rfeed=_rfeed4(q)))
        cand_pool.sort(key=lambda c: -c["rfeed"])
        lbp = cand_pool[:target]
    elif getattr(args, "lbeach", 0) and getattr(args, "round", 0) >= 33:
        #? R33 rad 閘攻堅錨組:同框系 rad 全負(六批)→錨換「lo 壓∧rad 半好」交集帶
        #  （g29b1_031=全史唯一 rad≥−1∧lo≤−2 筆;判準=同框∧rad≥−1 ≥1/批,round-33 §1）。
        RADGATE = [("lb_p00h31", "dedust_r29b1d_input", "g29b1_031_surg_p00h"),
                   ("lb_p00h04", "dedust_r28b3c_input", "b28b3_004_p00h"),
                   ("lb_deta10", "dedust_r30diag_input", "x30d_10_detach_l30b2_009_"),
                   ("lb_t03r", "dedust_r31b3d_input", "l31b3_003_lb_t03"),
                   ("lb_f2t11", "dedust_r20g2a_input", "f2_015_t11"),
                   ("lb_f2t04", "dedust_r20g2c_input", "f2_029_t04")]
        from scipy.ndimage import label as _lab3

        def _rfeed3(q):
            lab_, n_ = _lab3(q, structure=_CROSS)
            g_ = lab_[FEED]
            return float((lab_ == g_).sum() / max(q.sum(), 1)) if g_ > 0 else 0.0
        target = getattr(args, "lbeach", 0)
        cand_pool = []
        for name, fol, pid in RADGATE:
            p0 = loadp(fol, pid)
            for j in range(target * 4 // len(RADGATE) + 2):
                q = p0.copy()
                d_ = int(rng.integers(1, 16)) if j % 5 < 3 else int(rng.integers(16, 41))
                q.ravel()[rng.choice(625, size=d_, replace=False)] ^= True
                q[FEED] = True
                st_ = piece_stats(q)
                if not (180 <= st_["metal_px"] <= 560) or q.tobytes() in hist:
                    continue
                hist.add(q.tobytes())
                cand_pool.append(dict(pat=q, parent=name, ops=[["lb_flip", d_]],
                                      d=int((q != p0).sum()), stats=st_, rfeed=_rfeed3(q)))
        cand_pool.sort(key=lambda c: -c["rfeed"])
        lbp = cand_pool[:target]
    elif getattr(args, "lbeach", 0):
        #? b2 起錨集換「中繼帶」（R30b1 判讀:深淵據點 lo −8~−10 但 wm −11~−19 修不回;
        #  全語料掃出 89 筆中繼帶——half/手術系 lo −4~−5∧wm≈0,oob_bad 6.7-8.6=天花板下 2dB,
        #  卡的是 rad——修訂註記見 round-30 §1）。b1 用的深淵七錨保留註解供重現。
        LBEACH = [("lb_y10n09", "dedust_r28b2c_input", "y28b2_010_n09h"),
                  ("lb_n09", "dedust_r27b1f_input", "n27b1_017_n09"),
                  ("lb_y35t03", "dedust_r28b1d_input", "y28b1_035_t03h"),
                  ("lb_f3t07", "dedust_r20g3c_input", "f3_011_t07"),
                  ("lb_t03", "dedust_r27b1b_input", "n27b1_019_t03"),
                  ("lb_y15n09", "dedust_r28b2b_input", "y28b2_015_n09h")] \
            if (args.batch >= 2 or getattr(args, "round", 0) >= 31) else \
                 [("lb_dnv6", "dedust_r29b3c_input", "d29b3_006_denovo"),
                  ("lb_exk43", "dedust_r29b3d_input", "g29b3_043_champ_exking"),
                  ("lb_err0", "dedust_r29b3a_input", "i29b3_000_err_g29b2_06"),
                  ("lb_p00h39", "dedust_r29b3f_input", "g29b3_039_surg_p00h"),
                  ("lb_free18", "dedust_r29b3c_input", "g29b3_018_free_rand"),
                  ("lb_frag14", "dedust_r29b3e_input", "g29b3_014_free_randf"),
                  ("lb_p00f69", "dedust_r29b3f_input", "g29b3_069_oobp_p00f")]
        from scipy.ndimage import label as _lab2

        def _rfeed(q):
            lab_, n_ = _lab2(q, structure=_CROSS)
            g_ = lab_[FEED]
            return float((lab_ == g_).sum() / max(q.sum(), 1)) if g_ > 0 else 0.0
        target = getattr(args, "lbeach", 0)
        cand_pool = []
        for name, fol, pid in LBEACH:
            p0 = loadp(fol, pid)
            for j in range(target * 4 // len(LBEACH) + 2):
                q = p0.copy()
                d_ = int(rng.integers(1, 16)) if j % 5 < 3 else int(rng.integers(16, 41))
                q.ravel()[rng.choice(625, size=d_, replace=False)] ^= True
                q[FEED] = True
                st_ = piece_stats(q)
                if not (180 <= st_["metal_px"] <= 560) or q.tobytes() in hist:
                    continue
                hist.add(q.tobytes())
                cand_pool.append(dict(pat=q, parent=name, ops=[["lb_flip", d_]],
                                      d=int((q != p0).sum()), stats=st_, rfeed=_rfeed(q)))
        cand_pool.sort(key=lambda c: -c["rfeed"])
        lbp = cand_pool[:target]

    #? V 臂 response 空洞反演（R40 首航;decisions 2026-07-25「資料擴展主軸=response 空間」,
    #  判準=round-40 §1）:同鍋 response PCA（2D）→格網找空洞（對既有雲 NN 距離最大)→K=4 質心
    #  當目標;候選池=文法隨機+碎片隨機,SM（two 優先）預測 response 投影,每質心收最近 2 席。
    #  不指望三標——資訊增益臂（實測投影 NN 距離>批中位=開新區;連兩批空=收案）。kind=voidhunt,前綴 v。
    vh = []
    if getattr(args, "v", 0) and getattr(args, "round", 0) >= 40:
        from scipy.spatial import cKDTree as _KDT
        from script.sm_reanchor import _load_clean as _lcv
        _trv, _hov = _lcv()
        _cfgV = load_config(args.config)
        _labV = PORT_SPECS[_cfgV.port]["labels"]
        _nptsV = sum(_cfgV.targets[_labV[0]]["width"])
        Yv = np.stack([np.asarray(y_).ravel() for _, y_ in (_trv + _hov)]).astype(np.float64)
        _mu = Yv.mean(axis=0, keepdims=True)
        _, _, _vt = np.linalg.svd(Yv - _mu, full_matrices=False)
        _pcd = (Yv - _mu) @ _vt[:2].T
        _tree = _KDT(_pcd)
        gx = np.linspace(_pcd[:, 0].min(), _pcd[:, 0].max(), 48)
        gy = np.linspace(_pcd[:, 1].min(), _pcd[:, 1].max(), 48)
        _grid = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        _gd = _tree.query(_grid)[0]
        _hot = _grid[np.argsort(-_gd)[:160]]                 # 最空的 160 格點
        _cent = _hot[rng.choice(len(_hot), size=4, replace=False)]
        for _ in range(12):                                   # 簡易 k-means（決定性:seeded rng 初始化）
            _asg = np.argmin(((_hot[:, None] - _cent[None]) ** 2).sum(-1), axis=1)
            for ci in range(4):
                if (_asg == ci).any():
                    _cent[ci] = _hot[_asg == ci].mean(axis=0)
        print(f"V 臂空洞質心（PCA 座標）: {[tuple(np.round(c_, 1)) for c_ in _cent]}")
        vpool, _tv = [], 0
        while len(vpool) < 2000 and _tv < 20000:   # 漏斗二次放大(2026-07-30):600→2000
            _tv += 1
            q = _rand_blocks(rng) if rng.random() < 0.5 else _rand_frag(rng)
            q = q.copy()
            q[FEED] = True
            st_ = piece_stats(q)
            if not (140 <= st_["metal_px"] <= 560) or q.tobytes() in hist:
                continue
            vpool.append(dict(pat=q, parent="void", ops=[["voidrand", 0]], d=0, stats=st_))
        from antenna.zoo import SURROGATES as _SUR_V
        _vnV = "".join(ch for ch in str(args.sm) if ch.isdigit())
        _ftV = DATASET_PATH.joinpath(f"sm_two{_vnV}.pth")
        if _ftV.exists():
            _smv = _SUR_V["cnn2"](os.path.join(REPO, "tmp", "dedust"), 25 * 25, (len(_labV), _nptsV))
            _smv.pre_load_model(_ftV, strict=True)
        else:
            _smv = _SUR_V["mlp"](os.path.join(REPO, "tmp", "dedust"), 25 * 25, (len(_labV), _nptsV))
            _smv.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
        _smv.model.eval()
        with torch.no_grad():
            _rawv = torch.cat([_smv.model(torch.stack([torch.tensor(c["pat"], dtype=torch.float32)
                                                       .reshape(-1) for c in vpool[i:i + 256]]))
                               for i in range(0, len(vpool), 256)]).reshape(len(vpool), -1).numpy()
        _pcv = (_rawv.astype(np.float64) - _mu) @ _vt[:2].T
        per_c = max(1, getattr(args, "v", 0) // 4)
        for ci in range(4):
            dists = ((_pcv - _cent[ci]) ** 2).sum(-1)
            for i in np.argsort(dists)[:per_c]:
                if vpool[i].get("sel_by") is None and len(vh) < getattr(args, "v", 0):
                    vpool[i]["sel_by"] = f"vc{ci}"
                    vpool[i]["ops"] = [["void", int(ci), float(np.sqrt(dists[i]))]]
                    hist.add(vpool[i]["pat"].tobytes())
                    vh.append(vpool[i])
        print(f"V 臂: 池 {len(vpool)} → 選 {len(vh)}（4 質心 × {per_c}）")

    #? X 海峽臂（R32;Ricky「海峽加進去,但要找 SM 有一定期望的,不是為了填而填」）:
    #  管線首個**雙親算子**——王朝簇（wm 好）×中繼/同框簇（lo 好）雜交（水平割線拼接/高斯遮罩混合）,
    #  oversample ×4 → SM 期望閘=LCB(pred_wm−std) top（線上學習精神:SM 在迴圈引導探索）;
    #  目標=填「王朝↔中繼海峽」（pattern_map 增量疊圖:bridge 梯度失敗的地理原因=海峽無教材）。
    #  kind=xover,id 前綴 x。
    xop = []
    if getattr(args, "xover", 0):
        XB_SRC = [("dedust_r28b2c_input", "y28b2_010_n09h"),
                  ("dedust_r27b1f_input", "n27b1_017_n09"),
                  ("dedust_r28b1d_input", "y28b1_035_t03h"),
                  ("dedust_r20g3c_input", "f3_011_t07"),
                  ("dedust_r30b2d_input", "l30b2_009_lb_t03"),
                  ("dedust_r31b1f_input", "l31b1_017_lb_y10n09"),
                  ("dedust_r31b2f_input", "l31b2_005_lb_n09")]
        xA = [P[n] for n in dyn_names] if dyn_names else list(P.values())[:20]
        xB = [loadp(f_, p_) for f_, p_ in XB_SRC]
        from scipy.ndimage import gaussian_filter as _gfx
        xt = getattr(args, "xover", 0) * 4
        tries = 0
        while len(xop) < xt and tries < xt * 40:
            tries += 1
            a_ = xA[int(rng.integers(0, len(xA)))]
            b_ = xB[int(rng.integers(0, len(xB)))]
            if rng.random() < 0.5:                        # 水平割線拼接
                r_ = int(rng.integers(8, 17))
                q = (np.vstack([a_[:r_], b_[r_:]]) if rng.random() < 0.5
                     else np.vstack([b_[:r_], a_[r_:]])).copy()
                op_ = ["xover_h", r_]
            else:                                         # 高斯遮罩混合
                mask = _gfx(rng.random((25, 25)), 3) > 0.5
                q = np.where(mask, a_, b_).copy()
                op_ = ["xover_m"]
            q[FEED] = True
            q = _fix_diag_bridges(q)
            st_ = piece_stats(q)
            if not (200 <= st_["metal_px"] <= 550) or q.tobytes() in hist:
                continue
            hist.add(q.tobytes())
            xop.append(dict(pat=q, parent="xover", ops=[op_], d=-1, stats=st_))

    #? N 網架臂（R27;Ricky 2026-07-14「圖4-4 仔細看是幾個分塊→可做 variation」＋「27 做厚一點,網架」）:
    #  實測基礎=池頂家族 8-10 密度分塊載 66% 金屬+網布,t03/t09/n09/p00 共享同一骨架（scratch 2026-07-14）。
    #  骨架萃取=高斯 σ0.8×門檻 0.6（≥6px 質量塊,與分析口徑一致）;每錨四式變體:
    #    H1 mesh_redust×3=塊內原樣+網布同質量重佈（骨架承載電性?）
    #    H2 solidify_full=塊 footprint 填實+刪網布 / solidify_half=塊填實+網布保留（低側住骨架還是網布?）
    #    mesh_uniform=塊內原樣+棋盤均勻網布（網布要密度還是要實現?）
    #  ⚠ 物理探測批:不過可製造閘（塵=實驗變數,同 D 臂慣例）;kind=mesh,id 前綴 n。
    meshp = []
    if getattr(args, "mesh", 0) and PF:
        for name in list(PF):
            p0 = PF[name]
            blk = _skeleton(p0)
            mesh_px = p0 & ~blk
            n_mesh = int(mesh_px.sum())
            out_idx = np.flatnonzero((~blk).reshape(-1))
            ii, jj = np.indices(p0.shape)
            cands = [(blk.copy(), ["solidify_full"]),
                     (blk | mesh_px, ["solidify_half"])]
            uni = np.flatnonzero((((ii + jj) % 2 == 0) & ~blk).reshape(-1))
            qv = (p0 & blk).reshape(-1).copy()
            if len(uni):
                qv[uni[np.linspace(0, len(uni) - 1, min(n_mesh, len(uni))).astype(int)]] = True
            cands.append((qv.reshape(25, 25), ["mesh_uniform"]))
            for t in range(3):
                qv = (p0 & blk).reshape(-1).copy()
                pick = rng.choice(out_idx, size=min(n_mesh, len(out_idx)), replace=False)
                qv[pick] = True
                cands.append((qv.reshape(25, 25), ["mesh_redust", t]))
            for q, ops in cands:
                q = (np.asarray(q).reshape(25, 25) > 0.5).copy()
                q[FEED] = True
                q = _ensure_feed_pad(q, 4)
                st = piece_stats(q)
                if not (180 <= st["metal_px"] <= 560):
                    continue
                k = q.tobytes()
                if k in hist:
                    continue
                hist.add(k)
                meshp.append(dict(pat=q, parent=name, ops=[ops], d=int((q != p0).sum()), stats=st))

    #? H 臂:部分槽劑量（王 × 槽長;主件最寬列開中央槽——構造法,決定性零 rng）
    def hslot_doses(anchor):
        p = P[anchor]
        rows = [(p[r].sum() if 6 <= r <= 18 else -1) for r in range(25)]
        r0 = int(np.argmax(rows))
        best, cur, start, bs = 0, 0, 0, 0
        for c in range(25):
            if p[r0, c]:
                cur = cur + 1 if cur else 1
                if cur == 1:
                    start = c
                if cur > best:
                    best, bs = cur, start
            else:
                cur = 0
        outs = []
        for L in (3, 5, 8, 12):
            if L > best - 2:
                continue
            q = p.copy()
            c0 = bs + (best - L) // 2
            q[r0, c0:c0 + L] = False
            q[FEED] = True
            q, _n = strip_small(q, 4)
            q = _ensure_feed_pad(q, 4)
            st = piece_stats(q)
            if st["n_1px"] > 0 or not (200 <= st["metal_px"] <= 520) or q.tobytes() in hist:
                continue
            hist.add(q.tobytes())
            outs.append(dict(pat=q, parent=anchor, ops=[["hslot_part", r0, L]],
                             d=int((q != p).sum()), stats=st))
        return outs
    #? H 錨點按批輪替（構造決定性,同錨同劑量=重複會被查重擋 → 每批換組;batch4 起回繞,屆時 H 判準已到期）
    H_POOL = ("r3_001", "a024", "c25", "r2_016", "x00", "g16", "ccr9s2", "i02", "vg0765")
    hs = [c for a in H_POOL[((args.batch - 1) * 3) % len(H_POOL):][:3] for c in hslot_doses(a)] \
        if getattr(args, "h", 12) else []

    #? S 槽鏈臂（b3 起;新臂協議先導 ≤15,Ricky 授權 2026-07-12）——證據:H 臂「部分槽 L5-8=帶內
    #  增益旋鈕但付 rad」跨 5/6 錨重現（b1 c25 +0.43/a024 +0.34/r3_001 +0.28;b2 g16 +0.46）。
    #  假設:槽的 wm 增益可保留、鏈上算子把 rad 修回。判準（發車前寫死,round-22 §1 修訂）:
    #  三標且 wm≥+0.30=成立;三標率連兩批 <6%=收臂。
    sc = []
    if getattr(args, "s", 0):
        S_ANCH = [a for a in ("c25", "a024", "r3_001", "g16") if a in P]
        tries = 0
        while len(sc) < args.s * 10 and tries < args.s * 120:
            tries += 1
            a = S_ANCH[tries % len(S_ANCH)]              # 輪替配錨（b3 教訓:隨機抽=g16 14/15 錨集中）
            p0 = P[a]
            r0 = int(np.argmax([(p0[r].sum() if 6 <= r <= 18 else -1) for r in range(25)]))
            best, cur, bs, start = 0, 0, 0, 0
            for c in range(25):
                if p0[r0, c]:
                    cur += 1
                    if cur == 1:
                        start = c
                    if cur > best:
                        best, bs = cur, start
                else:
                    cur = 0
            L = int(rng.integers(4, 10))
            if L > best - 2:
                continue
            off = int(rng.integers(0, best - L + 1))
            q = p0.copy()
            q[r0, bs + off: bs + off + L] = False
            chain = [["hslot_part", r0, L, off]]
            for _ in range(int(rng.integers(1, 3))):       # 鏈 1-2 算子修 rad
                fn = OPS[int(rng.choice(len(OPS), p=ws))][0]
                q2, desc = fn(q)
                if q2 is None:
                    break
                q, chain = q2, chain + [desc]
            q = (np.asarray(q).reshape(25, 25) > 0.5).copy()
            q[FEED] = True
            q, _n = strip_small(q, 4)
            q = _ensure_feed_pad(q, 4)
            st = piece_stats(q)
            if st["n_1px"] > 0 or not (200 <= st["metal_px"] <= 520):
                continue
            k = q.tobytes()
            if k in hist:
                continue
            hist.add(k)
            sc.append(dict(pat=q, parent=a, ops=chain, d=int((q != p0).sum()), stats=st))

    #? D de novo 臂（b3 起常設;Ricky 2026-07-12「非王朝系要有一定比例+針對訓 SM 去探索」）:
    #  池外隨機塊構造（不繼承任何血統）;篩選用 sm_harvest（34k 底座=現存分布最廣「非王朝 SM」）,
    #  資料 ≥150 後訓 sm_denovo 接棒。判準:任一三標=池外礦脈開張;連兩批零三標=收臂回觸發制。
    dn = []
    if getattr(args, "d", 0) and getattr(args, "round", 0) >= 43:
        #? R43 組文法 A/B（decisions「組文法生成系統」;判準=round-43 §1）:D 臂席位改文法槽——
        #  舊文法 2 對照＋GA/GB/GC 各 2＋GD/GDd 各 1（args.d=10 基準,少於 10 依序截斷）;
        #  每槽自建 15× 候選池,d_sm 只在**槽內**排序（配額固定=A/B 公平,SM 不跨槽挑食）;
        #  sel_by=dn_<文法> 逐批記帳。製造閘放寬 140-560（GC 輕結構刻意;n_1px 仍擋粉塵）。
        #? 槽版本:R43 首航→R44 GC汰/GA2進→R45 收斂（六批判定 round-44 §5:old 續任主力、
        #  GDd 留任〔對角機制載體〕,其餘收案）
        if getattr(args, "round", 0) >= 45:
            GRAM_SLOTS = [("old", 6), ("GDd", 4)]
        elif getattr(args, "round", 0) >= 44:
            GRAM_SLOTS = [("old", 2), ("GA", 2), ("GB", 2), ("GA2", 2), ("GD", 1), ("GDd", 1)]
        else:
            GRAM_SLOTS = [("old", 2), ("GA", 2), ("GB", 2), ("GC", 2), ("GD", 1), ("GDd", 1)]
        slots = []
        for g_, k_ in GRAM_SLOTS:
            slots += [g_] * k_
        slots = slots[:args.d]
        from collections import Counter as _Cnt
        for gname, k_ in sorted(_Cnt(slots).items(), key=lambda t: GRAM_SLOTS.index((t[0], [k for g, k in GRAM_SLOTS if g == t[0]][0]))):
            built, tries = 0, 0
            while built < k_ * 15 and tries < k_ * 400:
                tries += 1
                if gname == "old":
                    q = _rand_denovo_old(rng)
                else:
                    q = _rand_grammar(rng, gname)
                if q is None:
                    continue
                st = piece_stats(q)
                if st["n_1px"] > 0 or not (140 <= st["metal_px"] <= 560):
                    continue
                kb = q.tobytes()
                if kb in hist:
                    continue
                hist.add(kb)
                dn.append(dict(pat=q, parent="denovo", ops=[["gram", gname]], d=0, stats=st,
                               sel_by=f"dn_{gname}", _gram=gname, _gquota=k_))
                built += 1
    elif getattr(args, "d", 0):
        tries = 0
        while len(dn) < args.d * 40 and tries < args.d * 400:
            tries += 1
            q = _rand_denovo_old(rng)
            if q is None:
                continue
            st = piece_stats(q)
            if st["n_1px"] > 0 or not (230 <= st["metal_px"] <= 520):
                continue
            k = q.tobytes()
            if k in hist:
                continue
            hist.add(k)
            dn.append(dict(pat=q, parent="denovo", ops=[["blocks"]], d=0, stats=st))

    cfg = load_config(args.config)
    setup_responses(cfg)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    cache = os.path.join(REPO, "tmp", "r22_sm")
    os.makedirs(cache, exist_ok=True)
    sm = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
    sm.pre_load_model(DATASET_PATH.joinpath(args.sm), strict=True)
    allc = core + coldp + specp + wildp + hs + sc + fragp + meshp + surgp + bmapp + bmixp + gradp + lbp + xop + vh
    pats = torch.stack([torch.tensor(c["pat"], dtype=torch.float32).reshape(-1) for c in allc])
    with torch.no_grad():
        raw = sm.model(pats).reshape(len(allc), len(labels), n_pts)
    #? ensemble 不確定性（2026-07-16 Ricky 拍板②,記錄版）:sm_ens{N}_{1,2} 存在→三成員 pred_wm
    #  的 std 進 manifest（pred_std）;**第一版不進選批鍵**——判讀驗證「std 分桶 |pred−real| 校準」
    #  後再進鍵（高信心變現/低信心探索的原則性分流）。
    ens_raws = []
    _vn = "".join(ch for ch in os.path.basename(args.sm) if ch.isdigit())
    for _j in (1, 2):
        _fe = DATASET_PATH.joinpath(f"sm_ens{_vn}_{_j}.pth")
        if _fe.exists():
            #? ens 換代（v75 起成員=cnn2）:先試 cnn2,state_dict 不合退 mlp（舊版檔相容）
            _sme = None
            for _arch in ("cnn2", "mlp"):
                try:
                    _sme = SURROGATES[_arch](cache, 25 * 25, (len(labels), n_pts))
                    _sme.pre_load_model(_fe, strict=True)
                    break
                except Exception:
                    _sme = None
            if _sme is None:
                continue
            _sme.model.eval()
            with torch.no_grad():
                ens_raws.append(_sme.model(pats).reshape(len(allc), len(labels), n_pts))
    if ens_raws:
        print(f"ensemble 成員 ×{len(ens_raws)}（sm_ens{_vn}_*）→ pred_std 記錄")
    #? R33 混合鍵（記錄版,照 std 進鍵先例）:影子 CNN 前瞻 ρ 三連勝（0.59/0.56/0.64=「CNN=排序器」）
    #  → pred_wm_cnn 記 manifest;b1 只記錄（判讀審計「CNN 排序假設檢定」）,過了才進鍵。
    cnn_raw = None
    _lohead = None
    two_raw = None
    if getattr(args, "round", 0) >= 33:
        _fc = DATASET_PATH.joinpath(f"sm_shadow{_vn}.pth")
        if _fc.exists():
            _smc = SURROGATES["cnn"](cache, 25 * 25, (len(labels), n_pts))
            _smc.pre_load_model(_fc, strict=True)
            _smc.model.eval()
            with torch.no_grad():
                cnn_raw = _smc.model(pats).reshape(len(allc), len(labels), n_pts)
        #? R38b3 起:影子二號排序實權（轉正判定 §4b;pred_wm_two 記錄+O 臂 rank 移交）
        two_raw = None
        if getattr(args, "round", 0) >= 38:
            _ft = DATASET_PATH.joinpath(f"sm_two{_vn}.pth")
            if _ft.exists():
                _smt = SURROGATES["cnn2"](cache, 25 * 25, (len(labels), n_pts))
                _smt.pre_load_model(_ft, strict=True)
                _smt.model.eval()
                with torch.no_grad():
                    two_raw = _smt.model(pats).reshape(len(allc), len(labels), n_pts)
                print(f"影子二號（sm_two{_vn}）→ pred_wm_two 記錄+O 臂排序實權（轉正 R38b2）")
        if getattr(args, "round", 0) >= 38:
            #? lo 判別器記錄鍵（R38;analysis-06 臂B;R39 判進鍵）——sm_lohead<vn> 在才記
            _vn2 = "".join(ch for ch in str(args.sm) if ch.isdigit())
            _fl = DATASET_PATH.joinpath(f"sm_lohead{_vn2}.pth")
            if _fl.exists():
                import torch.nn as _nnl
                _lohead = _nnl.Sequential(_nnl.Conv2d(1, 32, 3, padding=1), _nnl.ReLU(), _nnl.MaxPool2d(2),
                                          _nnl.Conv2d(32, 64, 3, padding=1), _nnl.ReLU(), _nnl.MaxPool2d(2),
                                          _nnl.Flatten(), _nnl.Linear(64 * 6 * 6, 256), _nnl.ReLU(),
                                          _nnl.Linear(256, 2))
                _lohead.load_state_dict(torch.load(str(_fl), weights_only=True))
                _lohead.eval()
                print(f"lo 判別器（sm_lohead{_vn2}）→ pred_lo 記錄（R39 判進鍵）")
        if cnn_raw is not None:
            print(f"影子 CNN（sm_shadow{_vn}）→ pred_wm_cnn 記錄"
                  + ("（O 臂雙 rank 進鍵中）" if getattr(args, "batch", 1) >= 2 or getattr(args, "round", 0) >= 34 else "（混合鍵審計,b1 不進鍵）"))
    #? 配套解析一行化+manifest 記帳（audit 2026-07-29:配套按 --sm 版號字串配對,缺件=靜默停鍵——
    #  歷史 4 批輕量重錨 pred_std 缺件時 LCB 退化成裸預測從未入帳;成功/缺件都要顯性）
    _heads_str = ",".join([
        (f"ens{_vn}x{len(ens_raws)}" if ens_raws else "ens-"),
        (f"cnn{_vn}" if cnn_raw is not None else "cnn-"),
        (f"two{_vn}" if two_raw is not None else "two-"),
        (f"lo{_vn}" if _lohead is not None else "lo-")])
    _miss_h = [h for h in _heads_str.split(",") if h.endswith("-")]
    print(f"配套解析: sm=v{_vn} [{_heads_str}]"
          + (f" ⚠缺件停鍵:{'/'.join(_miss_h)}（ens 缺=LCB 退化裸預測）" if _miss_h else "（全載）"), flush=True)
    for k, c in enumerate(allc):
        w, _ = worst_margin(raw[k], labels, cfg.targets)
        c["pred_wm"] = _r(float(w))
        if ens_raws:
            _ws = [float(w)] + [float(worst_margin(er[k], labels, cfg.targets)[0]) for er in ens_raws]
            c["pred_std"] = _r(float(np.std(_ws)))
        if cnn_raw is not None:
            c["pred_wm_cnn"] = _r(float(worst_margin(cnn_raw[k], labels, cfg.targets)[0]))
        if getattr(args, "round", 0) >= 35:
            #? asym 記錄鍵（2026-07-22 幾何分析:wm ρ−0.63/lo ρ−0.51=rad↔lo 連續座標;記錄版,R36 判進鍵）
            _P = c["pat"].astype(float)
            _S = (_P + _P[:, ::-1]) / 2
            _A = (_P - _P[:, ::-1]) / 2
            c["asym"] = _r(float(np.linalg.norm(_A) / (np.linalg.norm(_S) + 1e-9)))
        if getattr(args, "round", 0) >= 36:
            c["diagb"] = diag_bridge(c["pat"])   # 對角橋記錄鍵（Ricky 2026-07-23;R37 進罰分）
        if two_raw is not None:
            c["pred_wm_two"] = _r(float(worst_margin(two_raw[k], labels, cfg.targets)[0]))
        if _lohead is not None:
            with torch.no_grad():
                c["pred_lo"] = _r(float(_lohead(torch.tensor(c["pat"], dtype=torch.float32)
                                                .reshape(1, 1, 25, 25))[0, 1]))
        r = raw[k].numpy()
        c["pred_oob"] = oob_metrics(r)["oob_bad"]
        ml = 10 * np.log10(np.clip(1 - 10 ** (r[0][:4] / 10), 1e-6, 1))
        c["pred_lor"] = _r(float((r[1][:4] + ml).max()))
        if two_raw is not None and getattr(args, "round", 0) >= 40:
            #? R40 換裝（round-40 §1;two 五批連勝誤差+ρ 判準）:主通道 pred_wm/oob/lor 全走 two,
            #  下游一切鍵（sel/LCB/臂 rank）自動吃 two;MLP 降審計鍵 pred_wm_mlp（前瞻對照用）。
            c["pred_wm_mlp"] = c["pred_wm"]
            c["pred_wm"] = c["pred_wm_two"]
            r2 = two_raw[k].numpy()
            c["pred_oob"] = oob_metrics(r2)["oob_bad"]
            ml2 = 10 * np.log10(np.clip(1 - 10 ** (r2[0][:4] / 10), 1e-6, 1))
            c["pred_lor"] = _r(float((r2[1][:4] + ml2).max()))
    #? rad 頭（2026-07-12,Ricky「補 K=16」）:通過 held-out ρ≥0.4 門檻才傳 --rad-head 進鍵;
    #  pred_rad 一律記 manifest 供前瞻驗證。
    radnet = None
    if getattr(args, "rad_head", None):
        import torch.nn as nn
        _rh = torch.load(str(DATASET_PATH.joinpath(args.rad_head)), weights_only=False)
        radnet = nn.Sequential(nn.Linear(625, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                               nn.Linear(256, 2 * _rh["K"]))
        radnet.load_state_dict(_rh["state"])
        _th = np.asarray(_rh["theta"], float)
        _phi = np.pi * (_th - _th.min()) / (_th.max() - _th.min())
        _B = torch.tensor(np.cos(np.arange(_rh["K"]).reshape(-1, 1) * _phi.reshape(1, -1)),
                          dtype=torch.float32)
        with torch.no_grad():
            _fits = (radnet(pats).reshape(len(allc), 2, _rh["K"]) @ _B).numpy()
        for k, c in enumerate(allc):
            c["pred_rad"] = _r(min(rad_window_margin(_th, _fits[k][0]),
                                   rad_window_margin(_th, _fits[k][1])))
    else:
        for c in allc:
            c["pred_rad"] = None
    #? D 臂預測:sm_harvest（34k 底座=分布最廣）篩池外樣本;rad 頭共用（訓於全史,偏差較小）
    if dn:
        smh = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
        #? --d-sm:D 臂專屬選拔器（R25b2 起=sm_denovo1,每批 train-denovo 重訓）;
        #  與 I 臂委員會搭檔（--denovo-sm）分離,一次只動一個旋鈕。
        smh.pre_load_model(DATASET_PATH.joinpath(getattr(args, "d_sm", None)
                                                 or getattr(args, "denovo_sm", "sm_harvest.pth")), strict=True)
        dpats = torch.stack([torch.tensor(c["pat"], dtype=torch.float32).reshape(-1) for c in dn])
        with torch.no_grad():
            draw = smh.model(dpats).reshape(len(dn), len(labels), n_pts)
        for k, c in enumerate(dn):
            w, _ = worst_margin(draw[k], labels, cfg.targets)
            c["pred_wm"] = _r(float(w))
            rr = draw[k].numpy()
            c["pred_oob"] = oob_metrics(rr)["oob_bad"]
            ml = 10 * np.log10(np.clip(1 - 10 ** (rr[0][:4] / 10), 1e-6, 1))
            c["pred_lor"] = _r(float((rr[1][:4] + ml).max()))
            c["pred_rad"] = None
        if radnet is not None:
            with torch.no_grad():
                dfits = (radnet(dpats).reshape(len(dn), 2, _rh["K"]) @ _B).numpy()
            for k, c in enumerate(dn):
                c["pred_rad"] = _r(min(rad_window_margin(_th, dfits[k][0]),
                                       rad_window_margin(_th, dfits[k][1])))

    #? B 新穎性紅利（R24 探索誘因包,Ricky 核准 2026-07-12;--novelty 才生效）:
    #  d_hist=與全史最近 Hamming;鍵折扣 λ·min(d,20),λ=0.02（上限 0.4 dB,蓋不過真實差距）
    if getattr(args, "novelty", False) and hist0:
        H = np.stack([np.frombuffer(k, dtype=bool) for k in hist0]).astype(np.float32)
        for pool in (core, coldp, specp, wildp, hs, sc, dn, fragp, meshp, surgp, bmapp):
            if not pool:
                continue
            A = np.stack([c["pat"].reshape(-1) for c in pool]).astype(np.float32)
            dmat = A @ (1 - H.T) + (1 - A) @ H.T          # (M,N) Hamming
            dmin = dmat.min(axis=1)
            for k, c in enumerate(pool):
                c["novelty"] = int(dmin[k])
    else:
        for pool in (core, coldp, specp, wildp, hs, sc, dn, fragp, meshp, surgp, bmapp):
            for c in pool:
                c["novelty"] = None

    def _nbonus(c):
        return 0.02 * min(c["novelty"], 20) if c.get("novelty") is not None else 0.0

    def _diverse(pool, idxs_sorted, n):
        out = []
        for i in idxs_sorted:
            if len(out) >= n:
                break
            if all(int((pool[i]["pat"] != pool[j]["pat"]).sum()) >= 6 for j in out):
                out.append(i)
        for i in idxs_sorted:
            if len(out) >= n:
                break
            if i not in out:
                out.append(i)
        return out

    #? R33 反王朝結構軟過濾（Ricky 2026-07-17「只擋底1大+上2中,其他都值得試」）:
    #  無實測佐證的候選池（core/coldp/dn/wildp）命中表型→score 罰 +2.0（黑名單制降權非硬擋）;
    #  錨定臂（L/G/B 泵/X）豁免。批內佔比統計印出=判準①的量測。
    if getattr(args, "round", 0) >= 33:
        n_dyn_pool = 0
        for pool in (core, coldp, dn, wildp):
            for c in pool:
                c["dynst"] = dyn_struct(c["pat"])
                n_dyn_pool += int(c["dynst"])
        n_all_pool = sum(len(p_) for p_ in (core, coldp, dn, wildp))
        print(f"結構判（候選池 core/cold/D/W）: 王朝表型 {n_dyn_pool}/{n_all_pool}"
              f"={100 * n_dyn_pool / max(n_all_pool, 1):.0f}%"
              f"（全史基線 81%;命中罰 +{float(getattr(args, 'struct_pen', 2.0)):.1f} 降權）")

    def _dynpen(c):
        pen_ = float(getattr(args, "struct_pen", 2.0)) if c.get("dynst") else 0.0
        #? R37 對角橋罰（Ricky 2026-07-23「不要對角線的那種」;analysis-05:有對角橋三標 14% vs 36%）
        if getattr(args, "diagb_pen", 0.0) and c.get("pat") is not None:
            if "diagb" not in c:
                c["diagb"] = diag_bridge(c["pat"])
            pen_ += float(args.diagb_pen) * min(c["diagb"], 5)
        return pen_

    if getattr(args, "key", "oob") == "sel":
        #? R23 起價值軸主鍵:pred_sel=pred_oob+κ·(wm 缺口＋rad 缺口);rad 項僅 --rad-head 過門檻時生效
        #  R31b2 起 std 進鍵（校準過:三分桶 0.94/4.00/10.05 完美單調——LCB 保守變現,低信心折價;
        #  探索臂 I/W/D 不動=天然吃高 std）。
        def _psel(i):
            c = core[i]
            wm_lcb = c["pred_wm"] - c.get("pred_std", 0.0)   # 信心調整後預測（LCB）
            pen = SEL_KAPPA * max(0.0, SEL_BUFFER - wm_lcb)
            if c.get("pred_rad") is not None and getattr(args, "rad_key", False):
                pen += SEL_KAPPA * max(0.0, -c["pred_rad"])   # rad 項:--rad-key 才進鍵（門檻 ρ≥0.4）
            return c["pred_oob"] + pen - _nbonus(c) + _dynpen(c)   # B 新穎性紅利＋R33 結構罰
        #? R33b2 起 CNN 排序進鍵（b1 檢定達成:CNN ρ+0.628 vs MLP +0.074,8.5 倍;判準見 round-33 §1）:
        #  「CNN=排序器/MLP=回歸器」——O 變現臂排序=雙 rank 平均（MLP _psel rank + CNN wm rank）,
        #  絕對值門檻/LCB 仍 MLP。CNN 分數不在（b1 前的批）自動退回純 _psel。
        if getattr(args, "round", 0) >= 33 and args.batch >= 2 \
                and any(c.get("pred_wm_cnn") is not None for c in core):
            r_mlp = {i: rk for rk, i in enumerate(sorted(range(len(core)), key=_psel))}
            r_cnn = {i: rk for rk, i in enumerate(sorted(range(len(core)),
                     key=lambda i: -(core[i].get("pred_wm_cnn") or -99)))}
            #? R36 起 CNN 單 rank（R35 收輪:連兩批三尺全贏=排序主鍵;--cnn-solo;判準=O 三標率
            #  掉過半回雙 rank）;絕對值門檻/LCB 仍 MLP。
            #? R38b3 起:two 在場=O 排序實權歸 two（轉正判定 §4b）;否則沿舊制。
            if any(c.get("pred_wm_two") is not None for c in core):
                r_two = {i: rk for rk, i in enumerate(sorted(range(len(core)),
                         key=lambda i: -(core[i].get("pred_wm_two") or -99)))}
                _okey = lambda i: r_two[i]
            else:
                _okey = (lambda i: r_cnn[i]) if getattr(args, "cnn_solo", False) \
                    else (lambda i: r_mlp[i] + r_cnn[i])
            oi = _diverse(core, sorted(range(len(core)), key=_okey), args.o)
        else:
            oi = _diverse(core, sorted(range(len(core)), key=_psel), args.o)
    else:
        trim = sorted(range(len(core)), key=lambda i: core[i]["pred_wm"], reverse=True)[:int(len(core) * .6)]
        oi = _diverse(core, sorted(trim, key=lambda i: core[i]["pred_oob"]), args.o)
    mi = list(rng.choice([i for i in range(len(core)) if i not in oi],
                         size=min(args.m, len(core) - len(oi)), replace=False))
    ki = _diverse(coldp, list(rng.permutation(len(coldp))), args.c)
    qi = _diverse(specp, sorted(range(len(specp)), key=lambda i: specp[i]["pred_wm"], reverse=True), args.q)
    wi = list(rng.choice(len(wildp), size=min(args.wild, len(wildp)), replace=False)) if wildp else []

    def _pselc(c):
        pen = SEL_KAPPA * max(0.0, SEL_BUFFER - c["pred_wm"])
        if c.get("pred_rad") is not None and getattr(args, "rad_key", False):
            pen += SEL_KAPPA * max(0.0, -c["pred_rad"])
        return c["pred_oob"] + pen - _nbonus(c) + _dynpen(c)
    if dn and dn[0].get("_gram"):
        #? R43 文法槽選拔:槽內 _pselc 排序取槽配額（SM 只在槽內挑,配額跨槽固定=A/B 公平）
        di = []
        for gname in dict.fromkeys(c["_gram"] for c in dn):
            gi = [i for i in range(len(dn)) if dn[i]["_gram"] == gname]
            gi.sort(key=lambda i: _pselc(dn[i]))
            di += gi[:dn[gi[0]]["_gquota"]]
    elif dn:
        di = _diverse(dn, sorted(range(len(dn)), key=lambda i: _pselc(dn[i])),
                      getattr(args, "d", 0))
    else:
        di = []
    fi = _diverse(fragp, sorted(range(len(fragp)), key=lambda i: _pselc(fragp[i])),
                  getattr(args, "f", 0)) if fragp else []
    #? N 臂選拔:按錨輪流、變體優先序 full→half→uniform→redust（H2 對照先進場,H1 隨配額擴大補齊）
    ni = []
    if meshp:
        _mprio = {"solidify_full": 0, "solidify_half": 1, "mesh_uniform": 2, "mesh_redust": 3}
        by_anchor = {}
        for idx, c in enumerate(meshp):
            by_anchor.setdefault(c["parent"], []).append(idx)
        for lst in by_anchor.values():
            lst.sort(key=lambda idx: (_mprio.get(meshp[idx]["ops"][0][0], 9), str(meshp[idx]["ops"][0][-1])))
        anchors_l, rr = sorted(by_anchor), 0
        while len(ni) < getattr(args, "mesh", 0) and any(by_anchor.values()):
            a = anchors_l[rr % len(anchors_l)]
            rr += 1
            if by_anchor[a]:
                ni.append(by_anchor[a].pop(0))
    #? Y 手術臂選拔:maximin(pred_wm, pred_rad)——目標=rad 修回而 wm 不塌;rad 頭沒載時退回 pred_wm
    if surgp:
        if radnet is not None:
            yi = _diverse(surgp, sorted(range(len(surgp)),
                          key=lambda i: min(surgp[i]["pred_wm"],
                                            surgp[i]["pred_rad"] if surgp[i].get("pred_rad") is not None else -9),
                          reverse=True), getattr(args, "surgery", 0))
        else:
            yi = _diverse(surgp, sorted(range(len(surgp)), key=lambda i: surgp[i]["pred_wm"], reverse=True),
                          getattr(args, "surgery", 0))
    else:
        yi = []
    bi = list(range(min(len(bmapp), getattr(args, "blockmap", 0))))   # 承重圖探針=決定性,全收到配額
    bxi = list(range(min(len(bmixp), getattr(args, "bmix", 0))))      # U 組合手術=生成即配額,全收
    gi2 = list(range(min(len(gradp), getattr(args, "g", 0))))         # G 梯度臂=staging 即配額,全收
    li2 = list(range(min(len(lbp), getattr(args, "lbeach", 0))))      # L 低側據點=r_feed 排序後全收
    vi2 = list(range(min(len(vh), getattr(args, "v", 0))))            # V 空洞反演=質心配額即全收
    xvi = sorted(range(len(xop)), key=lambda i: -(xop[i]["pred_wm"] - xop[i].get("pred_std", 0.0)))[:getattr(args, "xover", 0)]                                          # X 海峽=SM 期望閘 LCB top
    #? A 資訊臂（R24 探索誘因包）:兩 SM（本版 vs harvest 底座）預測分歧最大=資訊量最高的量測點
    #  （query-by-committee 主動學習）;KPI=模型更新量非三標率。
    ii = []
    if getattr(args, "i", 0):
        smh2 = SURROGATES["mlp"](cache, 25 * 25, (len(labels), n_pts))
        smh2.pre_load_model(DATASET_PATH.joinpath(getattr(args, "denovo_sm", "sm_harvest.pth")), strict=True)
        with torch.no_grad():
            hraw = smh2.model(pats).reshape(len(allc), len(labels), n_pts)
        for k, c in enumerate(allc):
            hw, _ = worst_margin(hraw[k], labels, cfg.targets)
            c["disagree"] = _r(abs(c["pred_wm"] - float(hw))
                               + 0.5 * abs(c["pred_oob"] - oob_metrics(hraw[k].numpy())["oob_bad"]))
        taken = set(oi) | set(mi)
        pool_rest = [i for i in range(len(core)) if i not in taken]
        ii = _diverse(core, sorted(pool_rest, key=lambda i: core[i].get("disagree", 0), reverse=True),
                      getattr(args, "i", 0))
    #? S 臂鍵:rad 頭過門檻（--rad-key）時用 maximin（min(pred_wm−0.30, pred_rad)＝綁束者優先）
    if sc and radnet is not None and getattr(args, "rad_key", False):
        si = _diverse(sc, sorted(range(len(sc)),
                                 key=lambda i: min(sc[i]["pred_wm"] - 0.30, sc[i]["pred_rad"]),
                                 reverse=True), getattr(args, "s", 0))
    elif sc:
        si = _diverse(sc, sorted(range(len(sc)), key=lambda i: sc[i]["pred_wm"], reverse=True),
                      getattr(args, "s", 0))
    else:
        si = []
    #? 命名:R23 起 round 號貫穿（id o23b1_*、夾 dedust_r23b1*;Ricky 規範）;R22 沿用舊 idn 不回改
    idt = f"{rnd}b{args.batch}" if rnd >= 23 else str(5 + args.batch)
    fam = f"{rnd}b{args.batch}"
    entries = []
    for arm, letter, idxs, src in (("oobharv", "o", oi, core), ("mlotto", "m", mi, core),
                                   ("coldmine", "k", ki, coldp), ("repair", "q", qi, specp),
                                   ("hslot", "h", list(range(len(hs))), hs),
                                   ("slotchain", "s", si, sc), ("denovo", "d", di, dn),
                                   ("fragfix", "f", fi, fragp), ("mesh", "n", ni, meshp),
                                   ("surgery", "y", yi, surgp), ("blockmap", "b", bi, bmapp),
                                   ("bmix", "u", bxi, bmixp), ("grad", "g", gi2, gradp),
                                   ("lobeach", "l", li2, lbp), ("xover", "x", xvi, xop),
                                   ("voidhunt", "v", vi2, vh),
                                   ("infogain", "i", ii, core), ("wild", "w", wi, wildp)):
        for j, i in enumerate(idxs):
            c = src[i]
            entries.append(dict(id=f"{letter}{idt}_{j:03d}_{c['parent'][:12]}", kind=arm,
                                family=f"{arm.upper()}_{fam}", removed_px=0, **c["stats"],
                                sm=os.path.basename(args.sm), heads=_heads_str,
                                source_id=c["parent"], ops=c["ops"], diff_px=c["d"],
                                pred_wm=c["pred_wm"], pred_oob=c["pred_oob"], pred_lor=c["pred_lor"],
                                pred_rad=c.get("pred_rad"), pred_std=c.get("pred_std"),
                                pred_wm_cnn=c.get("pred_wm_cnn"), dynst=c.get("dynst"),
                                asym=c.get("asym"), diagb=c.get("diagb"),
                                sel_by=c.get("sel_by"), pred_wm_two=c.get("pred_wm_two"),
                                pred_lo=c.get("pred_lo"), pred_wm_mlp=c.get("pred_wm_mlp"),
                                _pat=c["pat"]))
    dirs = []
    for suf in "abcdefgh"[:args.shards]:
        dd = _dir(f"dedust_r{rnd}b{args.batch}{suf}_input")
        dd.mkdir(parents=True, exist_ok=True)
        dirs.append(dd)
    manifests = [[] for _ in dirs]
    for k, e in enumerate(entries):
        pat = e.pop("_pat")
        b = k % len(dirs)
        manifests[b].append(e)
        torch.save(torch.tensor(pat, dtype=torch.float32), str(dirs[b].joinpath(e["id"] + ".pt")))
    for man, dd in zip(manifests, dirs):
        _save_manifest(man, dd)
    print(f"r{rnd} b{args.batch}: G{len(gi2)}+L{len(li2)}+O{len(oi)}+M{len(mi)}+C{len(ki)}+Q{len(qi)}"
          f"+H{len(hs)}+S{len(si)}+D{len(di)}+F{len(fi)}+N{len(ni)}+Y{len(yi)}+B{len(bi)}+U{len(bxi)}"
          f"+I{len(ii)}+W{len(wi)}+V{len(vi2)} → {len(dirs)} 夾 {[len(m) for m in manifests]}")


#! DEPRECATED（2026-07-10 起查重改 _all_input_folders() 自動掃描）——僅舊 select 內建去重仍引用,
#  新 code 一律不要用這個清單。
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


def _tri(r):
    """三標過（wm≥0 ∧ rad≥0）。rad_margin 走 _r() 兩位網格,0.00/−0.0 是合法合格值——
    **不可用 `or -9`**（falsy 會把貼線解判成缺件;audit 2026-07-29:全史 99 筆 rad==0.00,
    其中 −0.0 佔 12 筆〔round() 負零〕,錨池被稀釋 1.59%）。"""
    rm = r.get("rad_margin")
    return r["wm"][2] >= 0 and rm is not None and rm >= 0


def _pool_seed(seed, rnd, batch):
    """OOD 池有效 seed=base+round*1000+batch——round 必須參與推導。
    r51b2 撞 r50b2 事故(2026-08-02,20/20 全重複被 check-dup 攔):舊式 seed+batch 跨輪同批號
    同池,稽核 M6「每輪必換 seed」靠人記=必忘。round*1000 與舊有效 seed 空間(base+1..54)不相交。"""
    return seed + rnd * 1000 + batch


def select_neg(args):
    """R50 負片臂（型態體系軸;判準=round-50 §1/decisions「型態體系軸」條）:
    `script.neg_gen` 七臂池（決定性）→ farthest-point 覆蓋選席（**SM-blind**——影子 pred 收檔後
    以當版凍結 SM 離線補算,版本記 round 檔;冷啟動曲線起點=n=0）。夾=dedust_r<NN>b<批>b_input。"""
    from script.neg_gen import gen_pool, farthest_point, ARMS as _NEG_ARMS
    if args.pad < 1:
        raise SystemExit("--pad 最小 1(0=feed 無保護;稽核 L2)")
    arms = tuple(a.strip() for a in args.arms.split(",")) if args.arms else _NEG_ARMS
    pool = gen_pool(_pool_seed(args.seed, args.round, args.batch), args.pool, arms=arms, pad=args.pad)
    if args.stratify:
        #? 分層選席(稽核 M1,b2 起判準修訂:FPS 全域版餓死工程臂 eng1/sierp0 於 120 席)——
        #  每臂配額=均分+餘數給前臂;臂內仍用 FPS 保覆蓋
        by_arm = {}
        for j, (p_, m_) in enumerate(pool):
            by_arm.setdefault(m_["arm"], []).append(j)
        arms_sorted = sorted(by_arm)
        base, extra = divmod(args.n, len(arms_sorted))
        idx = []
        for k, a in enumerate(arms_sorted):
            q = base + (1 if k < extra else 0)
            sub = [pool[j] for j in by_arm[a]]
            picked = farthest_point(sub, min(q, len(sub)), seed=_pool_seed(args.seed, args.round, args.batch) + k)
            idx.extend(by_arm[a][t] for t in picked)
    else:
        idx = farthest_point(pool, args.n, seed=_pool_seed(args.seed, args.round, args.batch))
    input_dir = _dir(f"dedust_r{args.round}b{args.batch}b_input")
    if input_dir.is_dir() and any(input_dir.glob("*.pt")):
        raise SystemExit(f"{input_dir.name} 已存在且非空——拒寫防跨輪覆寫/正片 shards 撞夾(稽核 H1/H2)")
    input_dir.mkdir(parents=True, exist_ok=True)
    ARM_SHORT = {"eng": "eng", "grf_neg": "grfn", "grf_inv": "grfi", "grf_lab": "grfl",
                 "bool_cut": "bcut", "bool_keep": "bkee", "sierp": "sier"}
    manifest = []
    for k, j in enumerate(idx):
        pat, meta = pool[j]
        pid = f"z{args.round}b{args.batch}_{k:03d}_{ARM_SHORT[meta['arm']]}"
        torch.save(torch.tensor(pat.astype(np.float32)), str(input_dir.joinpath(f"{pid}.pt")))
        manifest.append(dict(id=pid, kind="negreg", family=f"NEG_{args.round}b{args.batch}",
                             arm=meta["arm"], gen_meta={k2: v for k2, v in meta.items() if k2 != "arm"},
                             pad=args.pad, seed=args.seed,
                             sm=None, heads="SM-blind(影子離線補)", **piece_stats(pat)))
    tmp = input_dir.joinpath("manifest.json.tmp")
    json.dump(manifest, open(str(tmp), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(str(tmp), str(input_dir.joinpath("manifest.json")))
    arms = {}
    for e in manifest:
        arms[e["arm"]] = arms.get(e["arm"], 0) + 1
    print(f"select-neg r{args.round}b{args.batch}: {len(manifest)} 筆 → {input_dir.name}"
          f"（池 {len(pool)},farthest-point;臂分布 {arms};有效 seed {_pool_seed(args.seed, args.round, args.batch)}）")


def select_bridge(args):
    """R50 橋接臂(Ricky 2026-08-01「正負片中間平滑過渡」):bri_dil/bri_ero/bri_mix 三式輪抽,
    母本=近期正片輸入夾(真實量測個體);kind=bridge、id 前綴 j;夾=dedust_r<NN>b<批>b_input(批號 30+
    保留橋接池)。SM-blind;店入 neg_stores(two 的課程學習教材)。"""
    from script.neg_gen import bridge_pool
    parents = []
    for fol in args.parent_inputs.split(","):
        pp = _dir(fol.strip())
        if not pp.is_dir():
            continue
        for f in sorted(pp.glob("*.pt")):
            try:
                parents.append(np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5)
            except Exception:
                pass
    if len(parents) < 20:
        raise SystemExit(f"母本不足({len(parents)})——檢查 --parent-inputs")
    pool = bridge_pool(_pool_seed(args.seed, args.round, args.batch), args.n, parents, pad=args.pad)
    input_dir = _dir(f"dedust_r{args.round}b{args.batch}b_input")
    if input_dir.is_dir() and any(input_dir.glob("*.pt")):
        raise SystemExit(f"{input_dir.name} 已存在且非空——拒寫(防覆寫)")
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k, (pat, meta) in enumerate(pool):
        pid = f"j{args.round}b{args.batch}_{k:03d}_{meta['arm']}"
        torch.save(torch.tensor(pat.astype(np.float32)), str(input_dir.joinpath(f"{pid}.pt")))
        manifest.append(dict(id=pid, kind="bridge", family=f"BRI_{args.round}b{args.batch}",
                             arm=meta["arm"], gen_meta={k2: v for k2, v in meta.items() if k2 != "arm"},
                             pad=args.pad, seed=args.seed, sm=None, heads="SM-blind", **piece_stats(pat)))
    tmp = input_dir.joinpath("manifest.json.tmp")
    json.dump(manifest, open(str(tmp), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(str(tmp), str(input_dir.joinpath("manifest.json")))
    arms = {}
    for e in manifest:
        arms[e["arm"]] = arms.get(e["arm"], 0) + 1
    print(f"select-bridge r{args.round}b{args.batch}: {len(manifest)} 筆 → {input_dir.name}"
          f"(母本 {len(parents)};臂分布 {arms})")


def select_meshconv(args):
    """網格收斂實驗(docs/discuss/proposal-mesh-convergence.md,diffsim session 提案 2026-08-03):
    同一批**已量測** pattern × 三組自適應網格設定重測,估 HFSS 真值自身抖動=rank ρ 天花板。
    ids 檔一行=`來源輸入夾:id 組別`(組別 A=高爭議/B=對照;S2 只收 A 組)。
    產出 dedust_r<NN>ms{0,1,2}_input 三夾:kind=meshconv(查重豁免)、各夾寫 hfss_setup.json。
    發車鐵則=**三夾 jobs-add 全帶同一個 --machine 釘選**(提案 §3.2 同機約束);
    且需三台 worker 都已更新(machine 欄位+hfss_setup.json 支援)才可入佇列。"""
    SETUPS = {0: dict(max_delta_s=0.02, max_passes=6, min_passes=5, min_converged=5),
              1: dict(max_delta_s=0.005, max_passes=20, min_passes=5, min_converged=3),
              2: dict(max_delta_s=0.002, max_passes=30, min_passes=5, min_converged=3)}
    rows = []
    for ln in open(args.ids_file, encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        src, grp = ln.split()
        fol, pid = src.split(":")
        rows.append((fol, pid, grp.upper()))
    if not rows:
        raise SystemExit("ids 檔空的")
    pats = {}
    for fol, pid, _ in rows:
        f = DATASET_PATH.joinpath(fol, f"{pid}.pt")
        if not f.exists():
            raise SystemExit(f"找不到 {fol}:{pid}")
        pats[pid] = torch.load(str(f), weights_only=True)
    for s_idx, setup in SETUPS.items():
        sel = [(fol, pid, g) for fol, pid, g in rows if s_idx < 2 or g == "A"]
        input_dir = _dir(f"dedust_r{args.round}ms{s_idx}_input")
        if input_dir.is_dir() and any(input_dir.glob("*.pt")):
            raise SystemExit(f"{input_dir.name} 已存在且非空——拒寫(防覆寫)")
        input_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for fol, pid, g in sel:
            torch.save(pats[pid], str(input_dir.joinpath(f"{pid}.pt")))
            manifest.append(dict(id=pid, kind="meshconv", family=f"MESH_S{s_idx}",
                                 group=g, src=fol, sm=None, heads="meshconv"))
        json.dump(setup, open(str(input_dir.joinpath("hfss_setup.json")), "w", encoding="utf-8"), indent=1)
        tmp = input_dir.joinpath("manifest.json.tmp")
        json.dump(manifest, open(str(tmp), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        os.replace(str(tmp), str(input_dir.joinpath("manifest.json")))
        print(f"select-meshconv S{s_idx}: {len(manifest)} 筆 → {input_dir.name}（setup={setup}）")
    print("⚠ 發車前確認:三台 worker 已更新(釘選+hfss_setup 支援);三夾 jobs-add 同一個 --machine。")


def select_senior(args):
    """R50 學長未殖民族驗證臂（b2 起 10 席;判準=round-50 §1②/decisions 雙外軸——出血統的外）。
    學長池 pool.npz(24,189)→top-300(池值)→greedy 家族(d≤20)領袖→池值降冪逐批驗;
    kind=senior、id 前綴 e;夾=dedust_r<NN>b<批>c_input。決定性（純 numpy,無隱藏隨機源）。"""
    pz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tmp", "pattern_anatomy", "pool.npz")
    z = np.load(pz, allow_pickle=True)
    wm = z["wm"][:, 2].astype(float)
    pats = np.unpackbits(z["packed"], axis=1)[:, :625].astype(bool)
    order = np.argsort(-wm)[:300]
    leaders = []                                          # (pool_idx, size)
    for i in order:
        for k, (li, _s) in enumerate(leaders):
            if int(np.sum(pats[i] != pats[li])) <= 20:
                leaders[k] = (li, leaders[k][1] + 1)
                break
        else:
            leaders.append((int(i), 1))
    #? 已量測領袖跳過（2026-08-01 b2 教訓:top10 有 6 個=R7/R9 早期已驗的池頂 p0x_orig 等——
    #  全史 hash 過濾,每批自動取「未測」前 n 名（消耗制,前批測過自動前進））
    hist = set()
    for _name in _all_input_folders():
        fol = _dir(_name) if not hasattr(_name, "glob") else _name
        for f in fol.glob("*.pt"):
            try:
                hist.add((np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes())
            except Exception:
                pass
    fresh = [(li, sz) for li, sz in leaders if pats[li].tobytes() not in hist]
    print(f"領袖 {len(leaders)} 名,已量測跳過 {len(leaders) - len(fresh)},未測 {len(fresh)}")
    sel = fresh[:args.n]
    if not sel:
        raise SystemExit(f"學長未測領袖耗盡（{len(leaders)} 名全數已驗）")
    input_dir = _dir(f"dedust_r{args.round}b{args.batch}c_input")
    if input_dir.is_dir() and any(input_dir.glob("*.pt")):
        raise SystemExit(f"{input_dir.name} 已存在且非空——拒寫（防覆寫）")
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k, (li, size) in enumerate(sel):
        pat = pats[li].reshape(25, 25).astype(np.float32)
        pid = f"e{args.round}b{args.batch}_{k:03d}_F{li}"
        torch.save(torch.tensor(pat), str(input_dir.joinpath(f"{pid}.pt")))
        manifest.append(dict(id=pid, kind="senior", family=f"SENIOR_{args.round}b{args.batch}",
                             pool_idx=int(li), pool_wm=round(float(wm[li]), 3), family_size=int(size),
                             sm=None, heads="SM-blind(帳面池值=學長模擬器口徑)", **piece_stats(pat > 0.5)))
    tmp = input_dir.joinpath("manifest.json.tmp")
    json.dump(manifest, open(str(tmp), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(str(tmp), str(input_dir.joinpath("manifest.json")))
    print(f"select-senior r{args.round}b{args.batch}: {len(manifest)} 領袖 → {input_dir.name}"
          f"（全名單 {len(leaders)} 名,本批池值範圍 {manifest[0]['pool_wm']:+.2f}~{manifest[-1]['pool_wm']:+.2f}）")


def select_graft(args):
    """R48 定向嫁接試點（判準寫死於 round-48 §1;Ricky 拍板方向②）:
    骨架=王朝合格解（x00/margin 王）× 引擎=左側深 lo 個體（合格首例/usable_lo 王/g・d 線終錨）。
    兩式:A 引擎替換（王朝主件 → 左側 XL 件,貼質心）/B 對角引擎加掛（左側對角密集件掛主件上緣 gap≥1）。
    n 筆=16 base（2 骨架×4 引擎×2 式）+抖動;決定性 seed;全測不經 SM 排序。"""
    from scipy.ndimage import label as _lab
    S8g = np.ones((3, 3), bool)
    SKELS = [("x00", "dedust_wide_input", "x00_c21k2"),
             ("m42", "dedust_r42b1a_input", "m42b1_003_o26b3_022_o2")]
    ENGS = [("c8t", "dedust_c8tri_p03_input", "c8trip03_01"),
            ("c41", "dedust_c41grp2_p02_input", "c41grp2p02_02"),
            ("g46", "dedust_c45g3_p02_input", "c45g3p02_06"),
            ("d47", "dedust_c47d3_p02_input", "c47d3p02_12")]

    def _ld(fol, pid):
        return np.asarray(torch.load(str(DATASET_PATH.joinpath(fol, pid + ".pt")),
                                     weights_only=True)).reshape(25, 25) > 0.5

    def _comps(p8):
        lab, n = _lab(p8, structure=S8g)
        return lab, [(i, int((lab == i).sum())) for i in range(1, n + 1)]

    def _diag_rich(p8):
        #? 引擎件=對角最密集的 8-conn 組（內部 4-conn 裂件數-1 = 內部對角橋數）,尺寸 10-120
        lab, cs = _comps(p8)
        best, best_sc = None, -1
        for i, s in cs:
            if not (10 <= s <= 120):
                continue
            m = lab == i
            _, n4 = _lab(m)
            sc = n4 - 1
            if sc > best_sc:
                best, best_sc = m, sc
        if best is None and cs:
            i, _s = max(cs, key=lambda t: t[1])
            best = lab == i
        return best

    def _shift(m, dr, dc):
        out = np.zeros_like(m)
        rr, cc = np.where(m)
        nr, nc = rr + dr, cc + dc
        ok = (nr >= 0) & (nr < 25) & (nc >= 0) & (nc < 25)
        out[nr[ok], nc[ok]] = True
        return out

    def _graft(skel, eng, mode, jr, jc):
        lab_s, cs_s = _comps(skel)
        mi, _ = max(cs_s, key=lambda t: t[1])
        main = lab_s == mi
        lab_e, cs_e = _comps(eng)
        if mode == "A":                                   # 引擎替換:主件 → 左側 XL 件
            ei, _ = max(cs_e, key=lambda t: t[1])
            piece = lab_e == ei
            base = skel & ~main
            tgt = np.argwhere(main).mean(0)
            src_c = np.argwhere(piece).mean(0)
            dr, dc = int(round(tgt[0] - src_c[0])) + jr, int(round(tgt[1] - src_c[1])) + jc
            q = base | _shift(piece, dr, dc)
        else:                                             # B 對角引擎換翼:翼槽=天然淨空位
            #? v3（前兩版教訓:王朝密度 ~55%,全域掃不到 gap≥1 空位——盆地太滿）:
            #  拆第二大件（翼）,對角引擎件放進翼槽（貼質心,掃 ±3 偏移求 gap≥1）=「共用配件換引擎」反向版
            lab_e2, cs_e2 = _comps(eng)
            pieces = sorted(((i, s) for i, s in cs_e2 if 8 <= s <= 90), key=lambda t: -t[1])
            if not pieces:
                return None
            if len(cs_s) < 2:
                return None
            wi, _ = sorted(cs_s, key=lambda t: -t[1])[1]
            wing = lab_s == wi
            base = skel & ~wing
            tgt = np.argwhere(wing).mean(0)
            from scipy.ndimage import binary_dilation as _bdg
            q = None
            for ei, _s in pieces[:3]:
                piece = lab_e2 == ei
                pc = np.argwhere(piece)
                for orr in (0, -1, 1, -2, 2, -3, 3):
                    for occ in (0, -1, 1, -2, 2, -3, 3):
                        dr = int(round(tgt[0] - pc.mean(0)[0])) + jr + orr
                        dc = int(round(tgt[1] - pc.mean(0)[1])) + jc + occ
                        cand = _shift(piece, dr, dc)
                        if cand.sum() < pc.shape[0] * 0.9:
                            continue
                        if not (_bdg(cand, structure=S8g) & base).any():
                            q = base | cand
                            break
                    if q is not None:
                        break
                if q is not None:
                    break
            if q is None:
                return None
        q = q.copy()
        q[FEED] = True
        if not (140 <= int(q.sum()) <= 560):
            return None
        return q

    rng = np.random.default_rng(args.seed)
    skels = {k: _ld(f, i) for k, f, i in SKELS}
    engs = {k: _ld(f, i) for k, f, i in ENGS}
    sk_id = {k: i for k, _f, i in SKELS}
    en_id = {k: i for k, _f, i in ENGS}
    ind = _dir(args.out)
    ind.mkdir(parents=True, exist_ok=True)
    hist = set()
    for _fh in _all_input_folders():
        _mp = DATASET_PATH.joinpath(_fh, "manifest.json")
        if _mp.exists():
            try:
                for _m in json.load(open(str(_mp), encoding="utf-8")):
                    _fp = DATASET_PATH.joinpath(_fh, f"{_m['id']}.pt")
                    if _fp.exists():
                        hist.add(np.asarray(torch.load(str(_fp), weights_only=True))
                                 .reshape(-1).__gt__(0.5).tobytes())
            except Exception:
                continue
    manifest, tries = [], 0
    combos = [(sk, en, md) for sk in skels for en in engs for md in ("A", "B")]
    ci = 0
    while len(manifest) < args.n and tries < args.n * 60:
        tries += 1
        if ci < len(combos):
            sk, en, md = combos[ci]
            jr = jc = 0
            ci += 1
        else:
            sk, en, md = combos[int(rng.integers(0, len(combos)))]
            jr, jc = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
        q = _graft(skels[sk], engs[en], md, jr, jc)
        if q is None or q.tobytes() in hist:
            continue
        hist.add(q.tobytes())
        pid = f"gr48_{len(manifest):03d}_{sk}{md}_{en}"
        torch.save(torch.tensor(q, dtype=torch.float32), str(ind.joinpath(pid + ".pt")))  # float32(store dtype 慣例)
        manifest.append(dict(id=pid, kind="graft", family=f"GRAFT_{sk}{md}",
                             removed_px=0, source_id=sk_id[sk],
                             ops=[["graft_" + md, en_id[en], jr, jc]],
                             diff_px=int((q != skels[sk]).sum()), sel_by=None,
                             **piece_stats(q)))
    _save_manifest(manifest, ind)
    from collections import Counter
    cnt = Counter(m["family"] for m in manifest)
    print(f"graft 試點 {len(manifest)} 筆 → {args.out}  {dict(cnt)}")
    print(f"（估 {len(manifest) * 3} 分 ≈ {len(manifest) * 3 / 60:.1f} hr;判準=round-48 §1:lo 保留率 ≥20%）")


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
           if man_kind.get(k) not in ("repeat", "notarize", "meshconv")}  # 蓄意重複(公證/網格收斂重測)不算違規
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


def select_scope(args):
    """顯微鏡包（decisions 2026-07-17「窮舉×防陷困折衷」）:單一高價值錨 d=1 鄰域**全枚舉**
    （624 變體=625 像素扣 FEED,小空間完備「不漏」）→ CNN 排序 top N 送測（SM 用強項「排序」,
    Ricky:SM 的工作=找有潛力的域）→ **錨每輪輪換**（防陷）。每輪限一夾 25 筆=預算封頂。
    用法: select-scope --anchor <id> --source-input <夾> --cnn sm_shadow46.pth --input dedust_r33s1_input"""
    from antenna.zoo import SURROGATES
    cfg = load_config(args.config)
    labels = PORT_SPECS[cfg.port]["labels"]
    n_pts = sum(cfg.targets[labels[0]]["width"])
    src_pt = _dir(args.source_input).joinpath(f"{args.anchor}.pt")
    if not src_pt.exists():
        raise SystemExit(f"找不到 {src_pt}——確認 --source-input / --anchor。")
    p0 = np.asarray(torch.load(str(src_pt), weights_only=True)).reshape(25, 25) > 0.5
    variants, idxs = [], []
    for px in range(625):
        if (px // 25, px % 25) == FEED:
            continue                                     # FEED 像素不可動
        q = p0.copy()
        q.ravel()[px] ^= True
        variants.append(q)
        idxs.append(px)
    cache = os.path.join(REPO, "tmp", "dedust")
    os.makedirs(cache, exist_ok=True)
    smc = SURROGATES["cnn"](cache, 25 * 25, (len(labels), n_pts))
    smc.pre_load_model(DATASET_PATH.joinpath(args.cnn), strict=True)
    smc.model.eval()
    pats = torch.stack([torch.tensor(q, dtype=torch.float32).reshape(-1) for q in variants])
    with torch.no_grad():
        raw = smc.model(pats).reshape(len(variants), len(labels), n_pts)
    scores = [float(worst_margin(raw[k], labels, cfg.targets)[0]) for k in range(len(variants))]
    rmix = int(getattr(args, "rand_mix", 0))
    top = sorted(range(len(variants)), key=lambda k: -scores[k])[:args.n - rmix]
    if rmix > 0:
        rng_ = np.random.default_rng(20260718)
        rest = [k for k in range(len(variants)) if k not in set(top)]
        order = top + list(rng_.choice(rest, size=rmix, replace=False))
    else:
        order = top
    input_dir = _dir(args.input)
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for j, k in enumerate(order):
        pid = f"s{args.tag}_{j:02d}_{args.anchor[:10]}"
        torch.save(torch.tensor(variants[k], dtype=torch.float32), str(input_dir.joinpath(f"{pid}.pt")))
        manifest.append(dict(id=pid, kind="scope", family=f"SCOPE_{args.anchor[:12]}", removed_px=0,
                             source_id=args.anchor, ops=[["scope_d1", idxs[k]]], diff_px=1,
                             pred_wm_cnn=_r(scores[k]), sel_by=("cnn" if j < args.n - rmix else "rand"),
                             **piece_stats(variants[k])))
    _save_manifest(manifest, input_dir)
    print(f"顯微鏡包 → {input_dir}: {args.anchor} d=1 全枚舉 {len(variants)} → CNN top {len(manifest)}"
          f"（分數 {scores[order[0]]:+.2f} ~ {scores[order[-1]]:+.2f};照常 check-dup → jobs-add）")


def _chain_score(v, goal):
    """爬山鏈目標鍵（判準寫死;decisions「tier 架構」2026-07-21）:
    wm=追高（rad 崩輕罰）;dual=雙線距離 min（wm−buffer 與 9.0−oob 縮放 0.3——都正=雙線破）;
    rad=同框內爬 rad（掉出同框=大罰）;lo/hi=合格門檻內壓單側帶外
    （2026-07-23 Ricky「左右側單獨看,持續往下壓」——左右側拆帳紀錄制,decisions）。"""
    w = v["wm"][2]
    r = v.get("rad_margin")
    r = -9.0 if r is None else r
    ob = v.get("oob_bad")
    ob = 99.0 if ob is None else ob
    lo = v.get("oob_gain_max_lo")
    if goal == "wm":
        return w - (1.0 if r < -1 else 0.0)
    if goal == "dual":
        return min(w - 0.15, (9.0 - ob) * 0.3)
    if goal in ("lo", "hi"):
        s = v.get(f"oob_gain_max_{goal}")
        return -float(s) if (s is not None and w >= 0.15 and r >= 0) else -99.0
    if goal == "tri":
        #? R37 左側大陸會師鍵（decisions 2026-07-23）:lo≤−2 門檻內爬 min(wm−buffer, rad)——
        #  兩軸同正=左側合格解（全史 0 筆的里程碑,公證+推播）。
        return min(w - 0.15, r) if (lo is not None and lo <= -2.0) else -99.0
    if goal == "rad":
        return r if (w >= -2 and lo is not None and lo <= -2) else -99.0
    raise SystemExit(f"未知 goal {goal}")


def chain(args):
    """爬山鏈 daemon（tier 0 插隊層;Ricky 2026-07-21「同時跑好幾個線上學習」）:
    線上學習的閉環結構、訓練搬到批間——迴圈=發包（錨 d=1 純隨機 --n 筆,--prio 1 插隊）→
    等收檔 → 目標鍵判讀 → 勝錨=新錨續爬 / 連 --dry 包無勝=收鏈。
    判準寫死於 CLI（--goal;發鏈前定,鏈中不改）;機時=一鏈一包在飛（~25 筆/70 分=≤1/3 機隊）,
    鏈數由人控制（≤2 條,decisions）。記帳=docs/chains/<name>.jsonl（append-only）。
    用法: python -m script.dedust chain --name c1wm --anchor <id> --source-input <夾> --goal wm"""
    import time
    import subprocess
    import hashlib
    #? md5 種子（2026-07-25 修:hash() 有進程鹽=跨進程不可重現,違反生成決定性鐵則）
    rng_master = np.random.default_rng(int(hashlib.md5(args.name.encode()).hexdigest()[:8], 16))
    src_pt = _dir(args.source_input).joinpath(f"{args.anchor}.pt")
    if not src_pt.exists():
        raise SystemExit(f"找不到 {src_pt}")
    anchor_pat = np.asarray(torch.load(str(src_pt), weights_only=True)).reshape(25, 25) > 0.5
    #? 發車驗錨（audit 2026-07-29:c47d1 教訓——錨在 goal 鍵盆地外→全包 −99 燒 50 筆;
    #  analysis-07:近王錨組級包實證低效〔200 HFSS 僅 +0.09〕→ wm>−2 拒發組級包）
    _ost = args.source_input[:-6] if args.source_input.endswith("_input") else args.source_input
    _orp = DATASET_PATH.joinpath(_ost, "results.json")
    if _orp.exists():
        _ores = json.load(open(str(_orp), encoding="utf-8"))
        if args.anchor in _ores and "wm" in _ores[args.anchor]:
            _av = _ores[args.anchor]
            _asc = _chain_score(_av, args.goal)
            if _asc <= -98:
                raise SystemExit(f"拒發車:錨 {args.anchor} 在 goal={args.goal} 鍵下 score={_asc:+.1f}"
                                 f"（盆地外——c47d1 教訓;lo 門檻用 oob_gain_max_lo）")
            if getattr(args, "mutator", "px") == "group" and _av["wm"][2] > -2:
                raise SystemExit(f"拒發車:錨 wm {_av['wm'][2]:+.2f} > −2——近王帶組級變異包實證低效"
                                 f"（analysis-07;確認局部極大請改 px 包）")
        else:
            print(f"⛰ {args.name} ⚠ 驗錨略過:{_ost}/results.json 查無 {args.anchor}", flush=True)
    else:
        print(f"⛰ {args.name} ⚠ 驗錨略過:{_ost} 無 results.json", flush=True)
    #? 全史 hash 本地防撞（2026-07-25 c41grp 教訓:組級算子常產 diff=1 單 px 變異=等價 d1,
    #  used 集只擋 px 半→20 包全數 check-dup 撞歷史零發車。啟動載一次,check-dup 仍是最後閘門）。
    hist_keys = set()
    for _fh in _all_input_folders():
        _dh = DATASET_PATH.joinpath(_fh)
        _mph = _dh.joinpath("manifest.json")
        if not _mph.exists():
            continue
        try:
            for _mh in json.load(open(str(_mph), encoding="utf-8")):
                _fp = _dh.joinpath(f"{_mh['id']}.pt")
                if _fp.exists():
                    hist_keys.add(np.asarray(torch.load(str(_fp), weights_only=True))
                                  .reshape(-1).__gt__(0.5).tobytes())
        except Exception:
            continue
    print(f"⛰ {args.name} 全史 hash 載入 {len(hist_keys)} 筆（本地防撞）", flush=True)
    anchor_id, anchor_score = args.anchor, args.anchor_score
    log_dir = os.path.join(REPO, "docs", "chains")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{args.name}.jsonl")
    dry, pack = 0, 0
    used = set()          # 同錨已抽 px——跨包記憶（2026-07-22 修:包內 used 導致 p02 起狂撞
                          #  p01 已測變體,19 連撞空轉收鏈）;換錨時重置。

    def _preload_used(aid):
        """接棒鏈預載:掃全史 manifest 找同錨已測的 d=1 px（chain_d1/scope_d1 ops）——
        新 daemon 不知前任鏈測過哪些,不預載=95% 撞歷史（c1d4 實測第三 bug）。"""
        out = set()
        for fol2 in os.listdir(str(DATASET_PATH)):
            if not fol2.endswith("_input"):
                continue
            mp2 = DATASET_PATH.joinpath(fol2, "manifest.json")
            if not mp2.exists():
                continue
            try:
                for m2 in json.load(open(str(mp2), encoding="utf-8")):
                    if m2.get("source_id") == aid:
                        for op2 in (m2.get("ops") or []):
                            if isinstance(op2, list) and len(op2) == 2                                     and op2[0] in ("chain_d1", "scope_d1"):
                                out.add(int(op2[1]))
            except Exception:
                continue
        return out

    used |= _preload_used(anchor_id)
    if used:
        print(f"⛰ {args.name} 預載同錨已測 px {len(used)} 個（接棒防撞）")
    #? 域專家模式（Ricky 2026-07-22「不同分布的 SM 做不同 tier 0」）:鏈資料微調全域 SM=域專家
    #  → 包生成改「d=1 全枚舉→專家排序 top n−k＋隨機 k 對照」（微尺度導航已敗兩次〔CNN ρ−0.39/
    #  無加值〕,第三試=密集鄰域特訓版;對照組鐵律,判準=連兩包專家半勝→標配,否則退純隨機）。
    expert = None
    if getattr(args, "expert", False):
        from antenna.zoo import SURROGATES as _SUR_E
        from antenna.training import load_config as _lc, PORT_SPECS as _PS
        _cfg_e = _lc(DEFAULT_CFG)
        _lab_e = _PS[_cfg_e.port]["labels"]
        _npts_e = sum(_cfg_e.targets[_lab_e[0]]["width"])
        import glob as _ge
        _base_sm = sorted(_ge.glob(str(DATASET_PATH.joinpath("sm_reanchor*.pth"))),
                          key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))[-1]
        print(f"⛰ {args.name} 專家模式:底座 {os.path.basename(_base_sm)},鏈資料微調,top {args.n - args.exp_rand}+隨機 {args.exp_rand}")
    while dry < args.dry and pack < args.max_packs:
        pack += 1
        store = f"dedust_{args.name}_p{pack:02d}"
        ind = _dir(store + "_input")
        ind.mkdir(parents=True, exist_ok=True)
        manifest = []
        avail = [px for px in range(625) if px not in used and (px // 25, px % 25) != FEED]
        if len(avail) < args.n:
            print(f"⛰ {args.name} 錨鄰域枯竭（{len(used)}/624 已抽）——收鏈")
            dry = args.dry
        elif expert is not None:
            #? 專家排序半＋隨機對照半（sel_by 記 manifest,判讀分半記帳）
            vs = []
            for px in avail:
                q = anchor_pat.copy()
                q.ravel()[px] ^= True
                vs.append((px, q))
            pats_e = torch.stack([torch.tensor(q, dtype=torch.float32).reshape(-1) for _, q in vs])
            expert.model.eval()
            with torch.no_grad():
                raw_e = expert.model(pats_e).reshape(len(vs), len(_lab_e), _npts_e)
            from antenna.losses import worst_margin as _wme
            sc_e = []
            for k2 in range(len(vs)):
                w_, _ = _wme(raw_e[k2], _lab_e, _cfg_e.targets)
                ob_ = oob_metrics(raw_e[k2].numpy())["oob_bad"]
                sc_e.append(min(float(w_) - 0.15, (9.0 - ob_) * 0.3) if args.goal == "dual" else float(w_))
            order_e = sorted(range(len(vs)), key=lambda k2: -sc_e[k2])
            n_top = args.n - args.exp_rand
            picks = order_e[:n_top]
            rest = [k2 for k2 in order_e[n_top:]]
            picks += list(rng_master.choice(rest, size=args.exp_rand, replace=False))
            for j2, k2 in enumerate(picks):
                px, q = vs[k2]
                used.add(px)
                pid = f"{args.name}p{pack:02d}_{j2:02d}"
                torch.save(torch.tensor(q, dtype=torch.float32), str(ind.joinpath(pid + ".pt")))
                manifest.append(dict(id=pid, kind="scope", family=f"CHAIN_{args.name}", removed_px=0,
                                     source_id=anchor_id, ops=[["chain_d1", px]], diff_px=1,
                                     sel_by=("exp" if j2 < n_top else "rand"),
                                     **piece_stats(q)))
        else:
            #? --mutator group（R41 C 臂）:包內 grp_frac 組級變異＋其餘 px 對照（sel_by=grp/px
            #  分半記帳,判準=round-41 §1:同包配對比較勝率/合格產出）;純 px 模式=原行為。
            grp_n = int(round(args.n * getattr(args, "grp_frac", 0.7))) \
                if getattr(args, "mutator", "px") == "group" else 0
            seen_q, gfails = {anchor_pat.tobytes()}, 0
            while len(manifest) < args.n:
                if len(manifest) < grp_n and gfails < 500:
                    res = _group_mutate(anchor_pat, rng_master)
                    if res is None or res[0].tobytes() in seen_q or res[0].tobytes() in hist_keys:
                        gfails += 1
                        continue
                    q, opdesc, dpx = res
                    seen_q.add(q.tobytes())
                    hist_keys.add(q.tobytes())
                    if dpx == 1:                      # 單 px 組級變異=等價 d1,同步記 used
                        used.add(int(np.argmax((q != anchor_pat).ravel())))
                    pid = f"{args.name}p{pack:02d}_{len(manifest):02d}"
                    torch.save(torch.tensor(q, dtype=torch.float32), str(ind.joinpath(pid + ".pt")))
                    manifest.append(dict(id=pid, kind="scope", family=f"CHAIN_{args.name}",
                                         removed_px=0, source_id=anchor_id, ops=[opdesc],
                                         diff_px=dpx, sel_by="grp", **piece_stats(q)))
                    continue
                if len(used) >= 590:
                    print(f"⛰ {args.name} 錨鄰域枯竭（{len(used)}/624 已抽）——收鏈")
                    dry = args.dry
                    break
                px = int(rng_master.integers(0, 625))
                if px in used or (px // 25, px % 25) == FEED:
                    continue
                used.add(px)
                q = anchor_pat.copy()
                q.ravel()[px] ^= True
                if q.tobytes() in hist_keys:          # 全史撞位（他鏈/scope 測過）——px 已記 used,跳
                    continue
                hist_keys.add(q.tobytes())
                pid = f"{args.name}p{pack:02d}_{len(manifest):02d}"
                torch.save(torch.tensor(q, dtype=torch.float32), str(ind.joinpath(pid + ".pt")))
                manifest.append(dict(id=pid, kind="scope", family=f"CHAIN_{args.name}", removed_px=0,
                                     source_id=anchor_id, ops=[["chain_d1", px]], diff_px=1,
                                     sel_by=("px" if grp_n else None),
                                     **piece_stats(q)))
        if len(manifest) < args.n:
            break
        _save_manifest(manifest, ind)
        cd = subprocess.run([sys.executable, "-m", "script.dedust", "check-dup",
                             "--input", store + "_input"], capture_output=True)
        if cd.returncode != 0:
            #? exit 1 語義判別（audit 2026-07-29:查重撞歷史與腳本崩潰共用 exit 1——check_dup 跑到底
            #  必印 "<input>:" 摘要行,沒有=崩潰〔如 NAS 讀檔失敗〕,不可誤當撞歷史無聲空轉燒 max_packs）
            _cdout = (cd.stdout or b"").decode("utf-8", "replace") + (cd.stderr or b"").decode("utf-8", "replace")
            if f"{store}_input:" not in _cdout:
                raise SystemExit(f"{store} check-dup 崩潰（非查重）:\n{_cdout[-800:]}")
            print(f"⚠ {store} 查重撞歷史——重抽下一包（不發車）")
            continue
        #? jobs.json 讀-改-寫無鎖——兩鏈同時 jobs-add 會互踩丟單（2026-07-22 實測:c2rad 首包
        #  被 c1dual 覆蓋）。修=入列後驗證,丟了重試（隨機退避錯開）。
        for _try in range(4):
            subprocess.run([sys.executable, "-m", "script.dedust", "jobs-add",
                            "--input", store + "_input", "--store", store,
                            "--prio", str(args.prio)], capture_output=True)
            time.sleep(2 + int(rng_master.integers(0, 6)))
            jl = json.load(open(str(DATASET_PATH.joinpath("jobs.json")), encoding="utf-8"))
            if any(j["store"] == store for j in jl):
                break
        else:
            raise SystemExit(f"{store} 入列失敗 ×4——查 jobs.json")
        print(f"⛰ {args.name} p{pack:02d} 發包（錨 {anchor_id[:24]},prio {args.prio}）,等收檔…", flush=True)
        sd = DATASET_PATH.joinpath("jobs_state")
        while not sd.joinpath(store + ".done").exists():
            if sd.joinpath(store + ".fail").exists():
                raise SystemExit(f"{store} .fail——人工介入")
            time.sleep(120)
        res = json.load(open(str(DATASET_PATH.joinpath(store, "results.json")), encoding="utf-8"))
        scored = [(k, _chain_score(v, args.goal), v) for k, v in res.items()
                  if "error" not in v and "wm" in v]
        best_id, best_s, best_v = max(scored, key=lambda t: t[1])
        #? 全出局誠實記帳（audit 2026-07-29:c47d1 全包 −99 仍記假 best 且首包無錨分時會「勝」）
        gated = sum(1 for _t in scored if _t[1] <= -98)
        all_out = gated == len(scored)
        win = (not all_out) and (anchor_score is None or best_s > anchor_score)
        rec = dict(pack=pack, n=len(scored), anchor=anchor_id, best=best_id,
                   best_score=_r(best_s), anchor_score=(None if anchor_score is None else _r(anchor_score)),
                   win=bool(win), wm=_r(best_v["wm"][2]), rad=_r(best_v.get("rad_margin")),
                   oob=_r(best_v.get("oob_bad")),
                   lo=_r(best_v.get("oob_gain_max_lo")), hi=_r(best_v.get("oob_gain_max_hi")),
                   gated=gated, all_out=bool(all_out))
        if getattr(args, "expert", False):
            #? 分半記帳（專家 vs 隨機對照;判準=連兩包 exp 半勝→標配）
            man_e = {m["id"]: m for m in _load_manifest(ind)}
            eh = [sc for k3, sc, _ in scored if man_e.get(k3, {}).get("sel_by") == "exp"]
            rh = [sc for k3, sc, _ in scored if man_e.get(k3, {}).get("sel_by") == "rand"]
            if eh and rh:
                rec["exp_best"] = _r(max(eh))
                rec["rand_best"] = _r(max(rh))
                rec["exp_med"] = _r(float(np.median(eh)))
                rec["rand_med"] = _r(float(np.median(rh)))
            #? 專家微調:鏈至今全部已收包的 (pattern,response) 低 lr 熱訓（域密集教材）
            try:
                #? SampleStore 內容物=AntennaPattern/Response——座標與 spec 要先裝（同 sm_reanchor 口徑;
                #  v1 漏裝→sizer 例外,expert 靜默退隨機跑了三包）
                from antenna.pattern import AntennaPattern as _AP_E
                _AP_E.setDefaultCoordinate((0, 25, 0, 25))
                from antenna.training import setup_responses as _sr_e
                _sr_e(_cfg_e)
                from antenna.utils.store import SampleStore as _SS
                from torch.utils.data import TensorDataset as _TD
                xs_, ys_ = [], []
                for pk2 in range(1, pack + 1):
                    st2 = DATASET_PATH.joinpath(f"dedust_{args.name}_p{pk2:02d}")
                    if not st2.is_dir():
                        continue
                    ss = _SS(st2, verbose=False)
                    for i3 in range(len(ss)):
                        x3, y3 = ss[i3]
                        xs_.append(torch.as_tensor(x3, dtype=torch.float32).reshape(-1))
                        ys_.append(torch.as_tensor(y3, dtype=torch.float32))
                if len(xs_) >= 20:
                    expert = _SUR_E["mlp"](os.path.join(REPO, "tmp", "chain_exp"), 25 * 25,
                                           (len(_lab_e), _npts_e), lr=3e-4)
                    expert.pre_load_model(_base_sm, strict=True)
                    expert.train_by_datas(_TD(torch.stack(xs_), torch.stack(ys_)),
                                          epochs=15, batch_size=32, verbose=False)
                    print(f"⛰ {args.name} 專家微調完成（域資料 {len(xs_)} 筆,底座 {os.path.basename(_base_sm)}）")
            except Exception as _ee:
                print(f"⚠ 專家微調失敗（退純隨機續跑）: {_ee}")
                expert = None
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"⛰ {args.name} p{pack:02d} 收檔: best {best_id} score {best_s:+.3f}"
              f"（wm{best_v['wm'][2]:+.2f}/oob{best_v.get('oob_bad', 99):.2f}）"
              + (f"⚠ 全包出局（gated {gated}/{len(scored)}——查錨/goal 口徑）" if all_out else "")
              + ("→ 換錨續爬" if win else f"→ 無勝錨（dry {dry + 1}/{args.dry}）"), flush=True)
        if win:
            dry = 0
            used = _preload_used(best_id)                 # 換錨=新鄰域,重置+預載（防撞）
            anchor_id, anchor_score = best_id, best_s
            anchor_pat = np.asarray(torch.load(str(ind.joinpath(best_id + ".pt")),
                                               weights_only=True)).reshape(25, 25) > 0.5
        else:
            dry += 1
    print(f"⛰ {args.name} 收鏈: {pack} 包;終錨 {anchor_id} score "
          f"{anchor_score if anchor_score is not None else float('nan'):+.3f};帳 {log_path}")


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

    #? 求解設定覆蓋(2026-08-03 網格收斂實驗,proposal-mesh-convergence):輸入夾放 hfss_setup.json
    #  (鍵=max_delta_s/max_passes/min_passes/min_converged)即整夾生效;無檔=歷來預設(0.02/6/5/5)。
    #  存證:複製一份進 store_dir——結果夾自帶「這批用什麼設定量的」,不靠人記。
    hfss_setup = {}
    _setup_f = input_dir.joinpath("hfss_setup.json")
    if _setup_f.exists():
        with open(str(_setup_f), encoding="utf-8") as f:
            hfss_setup = json.load(f)
        allowed = {"max_delta_s", "max_passes", "min_passes", "min_converged"}
        bad = set(hfss_setup) - allowed
        if bad:
            raise SystemExit(f"hfss_setup.json 不明鍵 {bad}（合法鍵={sorted(allowed)}）")
        with open(str(store_dir.joinpath("hfss_setup.json")), "w", encoding="utf-8") as f:
            json.dump(hfss_setup, f, ensure_ascii=False, indent=1)
        print(f"⚠ 非預設求解設定生效: {hfss_setup}（來源 {_setup_f.name};已存證進 store）")

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
                s = SinglePortRadSimulator(record_path=str(out), sweep_type=args.sweep, **hfss_setup)
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

    def _errstr(e):
        """COM 錯誤可讀化:HRESULT → 名稱標籤（原訊息常是 cp950 亂碼,碼比字可靠;2026-07-11 37 教訓）。"""
        known = {-2147417851: "RPC_E_SERVERFAULT(HFSS 內部例外)",
                 -2147023170: "RPC_S_CALL_FAILED(HFSS 行程死亡)",
                 -2147023174: "RPC_S_SERVER_UNAVAILABLE(HFSS 已關閉)",
                 -2147352567: "DISP_E_EXCEPTION(COM 例外)"}
        a = getattr(e, "args", ())
        codes = [c for c in a[:1] if isinstance(c, int)]
        if len(a) > 2 and isinstance(a[2], tuple):       # com_error excepinfo 的 scode=真正原因
            codes += [c for c in a[2] if isinstance(c, int) and c < 0]
        tags = " ".join(known.get(c, f"0x{c & 0xFFFFFFFF:08X}") for c in codes)
        return (tags + " | " if tags else "") + str(e)

    sim = _open_sim()
    fails = 0                                            # 連續失敗計數（HFSS 壞死保險絲）
    blowouts = 0                                         # 保險絲熔斷次數（冷卻重生循環,2026-07-15）
    try:
        #? 批尾自動補測（2026-07-11,37 HFSS bug）:主輪跑完殘留 error → 殺透重開 HFSS 再補,
        #  最多 --retry-pass 輪——COM 偶發錯多為 transient（b1a 手動補測 4 筆全清=依據）,
        #  免人工刪 .done+.claim 重派;同一筆 attempts≥3 判毒樣本嫌疑,不再試（人工判）。
        for rpass in range(1 + getattr(args, "retry_pass", 2)):
            if rpass:
                todo = [(n, m) for n, m in enumerate(manifest)
                        if "error" in results.get(m["id"], {})
                        and results[m["id"]].get("attempts", 1) < 3]
                if not todo:
                    break
                print(f"↻ 批尾補測第 {rpass} 輪：殘留 error {len(todo)} 筆,殺透重開 HFSS 重試")
                try:
                    _guard(sim.quit, 120, "補測前 HFSS 關閉")
                except Exception:
                    pass
                _kill_hfss()
                sim = _open_sim()
                fails = 0
            aborted = False
            for k, (num, m) in enumerate(todo):
                p = torch.load(str(input_dir.joinpath(f"{m['id']}.pt")), weights_only=True)
                print(f"[{m['id']}] 模擬中… (本次第 {k + 1}/{len(todo)} 筆;manifest #{num + 1}/{len(manifest)})")
                done_evt, fired = threading.Event(), threading.Event()
                threading.Thread(target=_watchdog, args=(done_evt, fired), daemon=True).start()
                try:
                    sim.start(num)
                    result = sim(p)
                    elapsed = sim.end()
                except Exception as e:                   #! 單筆失敗不炸整批：記 error、下一筆（比照線上 skip）
                    done_evt.set()
                    es = ("watchdog_timeout: " if fired.is_set() else "") + _errstr(e)
                    att = results.get(m["id"], {}).get("attempts", 0) + 1
                    results[m["id"]] = {"error": es, "attempts": att}
                    _flush()
                    print(f"  ✗ (第 {att} 次) {es}")
                    fails += 1
                    if not rpass and fails >= getattr(args, "max_fail", 5):
                        #? 冷卻重生（2026-07-15,216 間歇型故障）:連敗先冷卻再試,循環用盡才判死——
                        #  死亡判定從「連 5 敗」延長為「5 敗 × --max-blowout 循環,間隔 --cooldown 秒」。
                        blowouts += 1
                        mb = getattr(args, "max_blowout", 3)
                        if blowouts >= mb:
                            raise SystemExit(f"連敗保險絲熔斷 {blowouts} 循環（各連 {fails} 敗,冷卻無效）"
                                             "——HFSS 壞死,中止本批（已完成部分已落檔,修復後重跑即續）")
                        cd = getattr(args, "cooldown", 600)
                        print(f"  ⚡ 連 {fails} 敗——冷卻 {cd}s 後重開再試（熔斷 {blowouts}/{mb - 1}）")
                        _kill_hfss()
                        import time as _tc
                        _tc.sleep(cd)
                        fails = 0
                        sim = _open_sim()
                        continue
                    if rpass and fails >= 3:             # 補測輪不拉保險絲:連 3 敗=HFSS 又壞,放棄補測照 done 收
                        print("  補測連 3 敗——放棄補測,殘留 error 交 .done 記錄")
                        aborted = True
                        break
                    _kill_hfss()                         # 先殺透再重開（quit 對殭屍 COM 會卡,跳過）
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
                        return "yield"
                #? tier-2 讓位（2026-07-12 Ricky「空窗期跑 tier2,一接到 job 就暫停」）:
                #  填空批（prio ≥ tier2_prio）每筆完成後掃佇列,出現可認領的 tier-1 → 釋放 claim 退出;
                #  進度全在 results.json（冪等）,之後任一機續跑——暫停/恢復零額外儲存機制。
                if getattr(args, "job_prio", 0) >= getattr(args, "tier2_prio", 99):
                    _qp = DATASET_PATH.joinpath("jobs.json")
                    _sd = DATASET_PATH.joinpath("jobs_state")
                    try:
                        _jobs = json.load(open(str(_qp), encoding="utf-8")) if _qp.exists() else []
                    except Exception:
                        _jobs = []
                    for _j in _jobs:
                        if _j.get("prio", 9) >= getattr(args, "tier2_prio", 99):
                            continue
                        _st = _j["store"]
                        if _sd.joinpath(_st + ".done").exists() or _sd.joinpath(_st + ".fail").exists() \
                                or _sd.joinpath(_st + ".claim").exists():
                            continue
                        print(f"⏸ tier-1 job {_st} 出現——tier-2 讓位（進度已落檔,空窗時任一機續跑）")
                        if cp is not None:
                            try:
                                os.remove(cp)
                            except OSError:
                                pass
                        return "yield"

                resp = torch.stack([torch.as_tensor(result[l]).float().reshape(-1) for l in labels])
                w, per = worst_margin(resp, labels, cfg.targets)
                entry = {"wm": [_r(per[labels[0]]), _r(per[labels[1]]), _r(w)], "time_s": _r(elapsed, 1)}

                rad = sim.last_radiation                 # 方向圖順手收：±window 覆蓋餘裕 + 原始資料落檔
                if isinstance(rad, dict) and rad.get("theta") is not None:
                    torch.save(rad, str(rad_dir.joinpath(f"{m['id']}.pt")))
                    cuts = {f"phi{phi}": _r(rad_window_margin(rad["theta"], rad[f"phi{phi}"], window, floor))
                            for phi in (0, 90) if rad.get(f"phi{phi}") is not None}
                    if cuts:
                        entry["rad"] = cuts
                        entry["rad_margin"] = min(cuts.values())
                entry.update(oob_metrics(resp))           # 帶外選擇性 (2026-07-07 起隨批入檔)
                entry["sel"] = sel_score(entry["wm"][2], entry.get("rad_margin"), entry["oob_bad"])
                store.add(p, resp)                       # (pattern, 真響應) 入庫：可再餵 SM 重錨/Stage-3
                results[m["id"]] = entry
                _flush()
                print(f"  ✓ wm={entry['wm']}  rad_margin={entry.get('rad_margin', '—')}  {entry['time_s']}s")
            if aborted:
                break
    finally:
        try:
            _guard(sim.quit, 120, "HFSS 收尾關閉")
        except Exception:
            pass
    poison = [i for i, v in results.items() if "error" in v and v.get("attempts", 1) >= 3]
    if poison:
        print(f"⚠ 毒樣本嫌疑（3 連敗）{len(poison)} 筆: {','.join(poison[:10])}——人工判;重派=刪 .done+.claim")
    if args.out is None:
        #! 工作目錄=純暫存（結果全在 NAS）,跑完即刪——不清會吃滿系統碟:216 事件 2026-07-15,
        #  C 槽 0GB=78 個 job 的 HFSS 專案暫存,磁碟見底→COM 例外 0x80070223 爆發（重開機=假好轉）。
        #  只刪預設命名目錄;--out 自訂路徑視為使用者要保留。續跑不受影響（HFSS 專案會重建）。
        import shutil
        shutil.rmtree(str(out), ignore_errors=True)
        print(f"（工作目錄已清: {out}）")
    print(f"\n完成。結果：{results_path}；報表：python -m script.dedust report")


def _probe_check(me, sd):
    """機況探針（Ricky 2026-07-15「強化 worker,在 NAS 執行指令查各台狀況+整理」）:
    worker 每輪 poll 檢查 jobs_state/probe_<ip末段>.json,執行**白名單動作**後寫 _result 檔並刪指令。
    白名單=status（磁碟/殘留/git 版/HFSS 行程）/cleanup（刪 _dedust_* 殘留;空閒時執行=天然安全）
    ——不執行任意 shell（安全紅線）。發令端=`dedust probe --machine <末段> [--action ...]`。
    ⚠ 回應時機=worker 空閒的 poll 輪;跑 job 中不回應（最長延遲≈一個 job ~70 分）。"""
    import glob
    import shutil
    import subprocess
    import time
    tag = me.split(".")[-1]
    pf = sd.joinpath(f"probe_{tag}.json")
    if not pf.exists():
        return
    try:
        act = json.load(open(str(pf), encoding="utf-8")).get("action", "status")
    except Exception:
        act = "status"
    out = dict(machine=me, at=time.strftime("%Y-%m-%d %H:%M:%S"), action=act)
    try:
        junk = [d for d in glob.glob("_dedust_*") if os.path.isdir(d)]
        jgb = sum(os.path.getsize(os.path.join(r, f))
                  for d in junk for r, _, fs in os.walk(d) for f in fs) / 1e9
        if act == "status":
            du = shutil.disk_usage(os.path.abspath(os.sep))
            try:
                rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                              text=True, stderr=subprocess.DEVNULL).strip()
            except Exception:
                rev = "?"
            try:
                tl = subprocess.check_output(["tasklist", "/FI", "IMAGENAME eq ansysedt.exe"],
                                             text=True, stderr=subprocess.DEVNULL)
                n_hfss = tl.count("ansysedt.exe")
            except Exception:
                n_hfss = -1
            tmp = os.environ.get("TEMP", "")
            tgb = sum(os.path.getsize(os.path.join(r, f))
                      for r, _, fs in os.walk(tmp) for f in fs
                      if os.path.exists(os.path.join(r, f))) / 1e9 if tmp else -1
            out.update(disk_free_gb=round(du.free / 1e9, 1), disk_total_gb=round(du.total / 1e9, 1),
                       workdirs=len(junk), workdirs_gb=round(jgb, 2), temp_gb=round(tgb, 1),
                       git=rev, hfss_procs=n_hfss, cwd=os.getcwd())
        elif act == "cleanup":
            for d in junk:
                shutil.rmtree(d, ignore_errors=True)
            out.update(cleaned=len(junk), freed_gb=round(jgb, 2))
        else:
            out["error"] = f"未知動作 {act}（白名單: status / cleanup）"
    except Exception as e:
        out["error"] = str(e)
    tmpf = str(sd.joinpath(f"probe_{tag}_result.json")) + ".tmp"
    with open(tmpf, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmpf, str(sd.joinpath(f"probe_{tag}_result.json")))
    try:
        os.remove(str(pf))
    except OSError:
        pass
    print(f"📡 探針回應: {act} → probe_{tag}_result.json")


def probe(args):
    """發機況探針給指定正式機（開發機側;worker 空閒 poll 輪回應,見 _probe_check）。
    用法: python -m script.dedust probe --machine 216 [--action status|cleanup] [--wait 360]"""
    import time
    _, sd = _jobs_paths()
    pf = sd.joinpath(f"probe_{args.machine}.json")
    rf = sd.joinpath(f"probe_{args.machine}_result.json")
    if rf.exists():
        os.remove(str(rf))
    with open(str(pf), "w", encoding="utf-8") as f:
        json.dump(dict(action=args.action, from_at=time.strftime("%Y-%m-%d %H:%M:%S")), f)
    print(f"探針已發（{args.machine}/{args.action}）——worker 空閒 poll 輪回應,最長等 {args.wait}s…")
    t0 = time.time()
    while time.time() - t0 < args.wait:
        if rf.exists():
            print(json.dumps(json.load(open(str(rf), encoding="utf-8")), ensure_ascii=False, indent=1))
            return
        time.sleep(5)
    print("⚠ 逾時無回應——worker 沒跑/跑 job 中（佔線）/舊版沒有探針;指令檔留在佇列,worker 回來會補答")


# ---------------------------------------------------------------- 資料工廠（NAS 派工,2026-07-10）
def _jobs_paths():
    """佇列=DATASET_PATH/jobs.json（人/agent 編輯）;狀態檔=jobs_state/<store>.{claim,done,fail}。"""
    sd = DATASET_PATH.joinpath("jobs_state")
    sd.mkdir(parents=True, exist_ok=True)
    return DATASET_PATH.joinpath("jobs.json"), sd


def _jobs_lock_acquire(timeout=90.0, stale=180.0):
    """jobs.json 寫鎖（jobs_state/jobs.lock,O_EXCL;SMB 上近似原子）。
    兩起並發壞檔（2026-07-22/07-24,多 daemon 同時 jobs-add 互踩）的治本——
    讀-改-寫全程持鎖;陳鎖（mtime 超過 stale 秒=持鎖者已死）自動破鎖。回傳鎖路徑。"""
    import time
    _, sd = _jobs_paths()
    sd.mkdir(exist_ok=True)
    lk = str(sd.joinpath("jobs.lock"))
    t0 = time.time()
    while True:
        try:
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}".encode())
            os.close(fd)
            return lk
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lk) > stale:
                    os.remove(lk)
                    continue
            except OSError:
                continue                                   # 鎖剛被釋放——立即重試
            if time.time() - t0 > timeout:
                raise SystemExit(f"jobs.lock 佔用 >{timeout:.0f}s——查殭屍鎖 {lk}")
            time.sleep(0.4 + (os.getpid() % 7) * 0.15)     # 錯開重試（免 seed:非選樣路徑）


def jobs_add(args):
    """把一個批次加進 NAS 派工佇列（round 檔照常開、check-dup 照常跑——佇列只管「誰去燒」）。"""
    qp, _ = _jobs_paths()
    if not DATASET_PATH.joinpath(args.input, "manifest.json").exists():
        raise SystemExit(f"{args.input} 無 manifest——先跑 select 與 check-dup")
    lk = _jobs_lock_acquire()
    try:
        jobs = json.load(open(str(qp), encoding="utf-8")) if qp.exists() else []
        if any(j["store"] == args.store for j in jobs):
            raise SystemExit(f"{args.store} 已在佇列")
        job = dict(input=args.input, store=args.store, prio=args.prio)
        if getattr(args, "machine", None):
            job["machine"] = args.machine                # 釘選:只有這台認領(同機三組對照等)
        jobs.append(job)
        tmp = str(qp) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(jobs, f, ensure_ascii=False, indent=1)
        os.replace(tmp, qp)
    finally:
        try:
            os.remove(lk)
        except OSError:
            pass
    print(f"已入佇列: {args.store} (prio {args.prio});目前 {len(jobs)} 個 job")


def _rand_denovo_old(rng):
    """D 臂舊生成器（R43 起降格為文法 A/B 對照席;原 select 內嵌碼抽出）:
    饋點墊+隨機加塊到金屬預算+strip_small——回傳 q（呼叫端做製造閘/查重）。"""
    q = np.zeros((25, 25), dtype=bool)
    q[FEED[0] - 2:, FEED[1] - 2:FEED[1] + 3] = True
    target = int(rng.integers(250, 500))
    for _ in range(24):
        if int(q.sum()) >= target:
            break
        r0, c0 = int(rng.integers(0, 20)), int(rng.integers(0, 20))
        h, w0 = int(rng.integers(2, 8)), int(rng.integers(2, 8))
        q[r0:r0 + h, c0:c0 + w0] = True
    q[FEED] = True
    q, _n = strip_small(q, 4)
    return _ensure_feed_pad(q, 4)


def _rand_blocks(rng):
    """文法採樣 v2（Ricky 2026-07-16「隨機變高斯雜訊沒意義」——analysis-05 規律=生成文法）:
    主件住下半（含 feed 側,151+px 傾向）+翼 1-2 塊住上半（row0-8,三標過的翼緊湊在高處）+
    小件 1-3 顆擺槓桿區（中央柱上段/外緣）+稀網布;迴避中件散佈（16-63px=輸家語言）。
    非雜訊、非王系複製——「同語言的新句子」。"""
    q = np.zeros((25, 25), bool)
    h, w = int(rng.integers(7, 13)), int(rng.integers(13, 24))     # 主件（下半場,蓋 feed 側）
    r0 = int(rng.integers(12, 26 - h))
    c0 = int(rng.integers(0, 26 - w))
    q[r0:r0 + h, c0:c0 + w] = True
    for _ in range(int(rng.integers(1, 3))):                        # 翼級（上半,row0-8）
        h2, w2 = int(rng.integers(3, 8)), int(rng.integers(7, 17))
        r2 = int(rng.integers(0, max(9 - h2, 1)))
        c2 = int(rng.integers(0, 26 - w2))
        q[r2:r2 + h2, c2:c2 + w2] = True
    for _ in range(int(rng.integers(1, 4))):                        # 小件（中央柱上段/外緣槓桿區）
        h3, w3 = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        if rng.random() < 0.5:
            r3, c3 = int(rng.integers(5, 11)), int(rng.integers(10, 14))   # 中央柱上段
        else:
            r3, c3 = int(rng.integers(8, 12)), (int(rng.integers(0, 3)) if rng.random() < 0.5
                                                else int(rng.integers(21, 24)))  # 外緣
        q[r3:min(r3 + h3, 25), c3:min(c3 + w3, 25)] = True
    return q | (rng.random((25, 25)) < float(rng.uniform(0.03, 0.15)))


def _rand_frag(rng):
    """碎片語言隨機:15-30 個 1-3px 小塊+網布（學長碎片族的隨機版）。"""
    q = np.zeros((25, 25), bool)
    for _ in range(int(rng.integers(15, 31))):
        h, w = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        r0, c0 = int(rng.integers(0, 26 - h)), int(rng.integers(0, 26 - w))
        q[r0:r0 + h, c0:c0 + w] = True
    return q | (rng.random((25, 25)) < float(rng.uniform(0.15, 0.35)))


def _fix_diag_bridges(q):
    """橋接型對角修復（Ricky 2026-07-16「避免對角=可製造性,都加入,部分阻擋」）:
    只修跨島「尖角碰」;島內對角不動。修法=**剪尖角（detach）**——r30diag 探針因果
    （2026-07-16）:detach rad 中位 +0.3 系統性優於 attach（連好）−0.2——刪較小島側的
    接觸像素=電性正確的可製造性修復。決定性;一輪掃描（部分阻擋語義）。"""
    from scipy.ndimage import label as _lb
    q = q.copy()
    lab, _n = _lb(q, structure=_CROSS)
    sizes = np.bincount(lab.ravel())
    for r in range(24):
        for c in range(25):
            for dc in (-1, 1):
                c2 = c + dc
                if 0 <= c2 < 25 and q[r, c] and q[r + 1, c2] \
                        and not q[r + 1, c] and not q[r, c2]:
                    a, b = lab[r, c], lab[r + 1, c2]
                    if a != b:
                        if sizes[a] < sizes[b]:
                            q[r, c] = False
                        else:
                            q[r + 1, c2] = False
    return q


_SELFGEN_BASES = None


def _selfgen_chunk(me, args):
    """佇列全空時的自產 tier-2（Ricky 2026-07-12「HFSS 不准停;不需要你主動干涉的機制」）:
    從歷史 pattern 隨機翻 1-12 bit、查重全史,湊 --selfgen 筆進本機專屬夾 dedust_auto<ip>_input
    → 以 job_prio=999 跑:run() 的讓位檢查對「任何可認領 job」生效,正式資料一進佇列立即讓位。
    單機專屬 store=零互踩;決定性=seed(manifest 長度×1000+ip 末段);資料照常進重錨與查重掃描。"""
    global _SELFGEN_BASES
    tag = "auto" + me.split(".")[-1]
    ind = _dir(f"dedust_{tag}_input")
    ind.mkdir(parents=True, exist_ok=True)
    mp = ind.joinpath("manifest.json")
    manifest = json.load(open(str(mp), encoding="utf-8")) if mp.exists() else []
    if _SELFGEN_BASES is None:
        #? 王系親代過濾（Ricky 2026-07-16「還是很容易出現王系的」）:hist_flip 的親代池排除
        #  王朝家族近親（d_dyn<20）——王系殘影歸零;查重集 hist 仍收全歷史。
        DYN_ = ("c18", "vg0338", "vg0396", "g1_038", "g1_039", "r2_016", "r3_001")
        hist, bases, dynp = set(), [], []
        for fol in _all_input_folders():
            dd = DATASET_PATH.joinpath(fol)
            for m in json.load(open(str(dd.joinpath("manifest.json")), encoding="utf-8")):
                f = dd.joinpath(m["id"] + ".pt")
                if f.exists():
                    p = np.asarray(torch.load(str(f), weights_only=True)).reshape(25, 25) > 0.5
                    hist.add(p.tobytes())
                    bases.append((p, m["id"]))
                    if any(t in m["id"] for t in DYN_):
                        dynp.append(p.reshape(-1))
        _pk = np.packbits(np.stack(dynp).astype(np.uint8), axis=1)
        _pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
        keep = []
        for p, mid in bases:
            d_ = int(_pop[np.bitwise_xor(_pk, np.packbits(p.reshape(-1).astype(np.uint8)))]
                     .sum(axis=1).min())
            if d_ < 20:
                continue
            #? R37 種子換系統（Ricky 2026-07-23「增加其他系統比例」）:王朝表型種子只留 ~20%
            #  （決定性抽樣=md5 取模——python hash() 有進程鹽不可用,守生成決定性鐵則）;
            #  左側家族種子全保。
            import hashlib as _hl2
            if dyn_struct(p) and (_hl2.md5(p.tobytes()).digest()[0] % 5) != 0:
                continue
            keep.append(p)
        #? SM 粗篩員（漏斗化:生成 3× 挑 top;權重=NAS 最新 sm_reanchor*）
        import glob as _gl
        import re as _re
        vs = sorted((int(mo.group(1)), f) for f in _gl.glob(str(DATASET_PATH.joinpath("sm_reanchor*.pth")))
                    if (mo := _re.search(r"sm_reanchor(\d+)\.pth$", f)))
        sm_ = None
        if vs:
            try:
                from antenna.training import setup_responses
                cfg_ = load_config(args.config)
                setup_responses(cfg_)
                labels_ = PORT_SPECS[cfg_.port]["labels"]
                n_pts_ = sum(cfg_.targets[labels_[0]]["width"])
                from antenna.zoo import SURROGATES as _SUR
                sm_ = _SUR["mlp"](str(_dir("_selfgen_cache")), 25 * 25, (len(labels_), n_pts_))
                sm_.pre_load_model(vs[-1][1], strict=True)
                sm_.model.eval()
                print(f"⚙ 自產 SM 粗篩員: sm_reanchor{vs[-1][0]}")
            except Exception as e_:
                print(f"⚙ SM 粗篩員載入失敗（退回無篩選）: {e_}")
                sm_ = None
        _SELFGEN_BASES = (keep, hist, sm_, (cfg_, labels_, n_pts_) if sm_ else None)
        print(f"⚙ 自產基底載入:歷史 {len(bases)} 筆,非王系親代 {len(keep)} 筆")
    bases, hist, sm_scr, sm_ctx = _SELFGEN_BASES
    rng = np.random.default_rng(len(manifest) * 1000 + int(me.split(".")[-1]))
    #? 三分生成（歷史大翻 bit 50-150〔非王系親代〕/文法塊語言/碎片語言）＋ SM 垃圾過濾
    #  （Ricky 2026-07-16「隨機變高斯雜訊沒意義」:生成 3× 候選,pred_wm ≥−8 過濾雜訊級——
    #  是「過濾」不是「擇優」,通過者保持隨機=去雜訊不塌縮回 SM 自信區〔防馬太〕）。
    cands, tries = [], 0
    want = args.selfgen * (3 if sm_scr is not None else 1)
    while len(cands) < want and tries < want * 300:
        tries += 1
        mode = tries % 3
        if mode == 0:
            q = bases[int(rng.integers(0, len(bases)))].copy()
            k = int(rng.integers(50, 151))
            q.ravel()[rng.choice(625, size=k, replace=False)] ^= True
            src, ops, dpx = "hist_flip", [["flips", k]], k
        elif mode == 1:
            q = _rand_blocks(rng)
            q[FEED] = True
            src, ops, dpx = "rand_blocks", [["randb"]], -1
        else:
            q = _rand_frag(rng)
            q[FEED] = True
            src, ops, dpx = "rand_frag", [["randf"]], -1
        q = _fix_diag_bridges(q)                         # 可製造性:橋接型對角修復（2026-07-16）
        if not (200 <= int(q.sum()) <= 550) or q.tobytes() in hist:
            continue
        #? 反王朝結構過濾（2026-07-22 補實作——round-34 曾誤記「已生效」,實際靠飛輪自然降;
        #  Ricky「只擋底1大+上2中,其他都值得試」;文法帶命中率高=重抽由翻bit/碎片補）。
        if dyn_struct(q):
            continue
        hist.add(q.tobytes())
        cands.append((q, src, ops, dpx))
    if sm_scr is not None and len(cands) > args.selfgen:
        cfg_s, labels_s, npts_s = sm_ctx
        pats_ = torch.stack([torch.tensor(q, dtype=torch.float32).reshape(-1) for q, _, _, _ in cands])
        with torch.no_grad():
            raw_ = sm_scr.model(pats_).reshape(len(cands), len(labels_s), npts_s)
        pw = [float(worst_margin(raw_[i], labels_s, cfg_s.targets)[0]) for i in range(len(cands))]
        ok_i = [i for i in range(len(cands)) if pw[i] >= -8.0]
        if len(ok_i) < args.selfgen:                     # 通過太少→pred 最高者補足
            rest = sorted((i for i in range(len(cands)) if i not in set(ok_i)),
                          key=lambda i: -pw[i])
            ok_i += rest[:args.selfgen - len(ok_i)]
        print(f"⚙ SM 垃圾過濾: {len(cands)} 候選 → {len(ok_i[:args.selfgen])} 入批"
              f"（pred≥−8 通過 {sum(1 for i in ok_i if pw[i] >= -8)}）")
        cands = [cands[i] for i in ok_i[:args.selfgen]]
    made = []
    for q, src, ops, dpx in cands[:args.selfgen]:
        pid = f"a{tag[4:]}_{len(manifest) + len(made):05d}"
        torch.save(torch.tensor(q, dtype=torch.float32), str(ind.joinpath(pid + ".pt")))
        made.append(dict(id=pid, kind="selfgen", family=f"AUTO_{tag}", removed_px=0,
                         **piece_stats(q), source_id=src, ops=ops, diff_px=dpx))
    if not made:
        return False
    manifest += made
    _save_manifest(manifest, ind)
    print(f"⚙ 自產 tier-2 {len(made)} 筆 → dedust_{tag}（任何 job 入佇列即讓位）")
    ns = argparse.Namespace(config=args.config, input=f"dedust_{tag}_input", store=f"dedust_{tag}",
                            out=None, sweep=args.sweep, timeout=args.timeout, max_fail=args.max_fail,
                            retry_pass=0, job_prio=999, tier2_prio=999, claim_path=None, claim_me=me)
    run(ns)
    return True


def worker(args):
    """資料工廠 worker（Ricky 2026-07-10「三台都變資料收集系統,用 NAS 控制」）——正式機常駐:
    迴圈＝讀 jobs.json（prio 升冪）→ 跳過 done/被認領 → **原子認領**（jobs_state/<store>.claim,
    O_EXCL 建檔含機器 IP）→ run（斷點續跑＋單筆 watchdog＋連敗保險絲＋批尾自動補測）→ 標 done → 下一個;
    佇列空 → 睡 --poll 秒再掃。**同 store 兩機並跑由 claim 檔擋掉**（results.json 互踩防護）。
    stale 接管:claim 存在但 store 無進度超過 --stale 分鐘（機器死了）→ 別台可接手續跑。
    停止:建 jobs_state/STOP（跑完當前 job 收工）或 Ctrl-C。死亡判定三層（2026-07-15 升級,216 教訓）:
    ①連 --max-fail 敗=熔斷→冷卻 --cooldown 秒重開再試,--max-blowout 循環用盡才判死;
    ②判死→寫 <store>.fail（JSON 記 machines 名單）並停機;③別台 worker 見 .fail 名單無自己
    →自動接管重跑（毒批收斂:全機敗過=永久 fail 等人工;舊純文字 .fail 不自動接管）。"""
    import time
    from antenna.utils.web import get_local_ip
    me = get_local_ip()
    qp, sd = _jobs_paths()
    #? 啟動清掃（2026-07-15 磁碟滿事件）:「跑完即刪」只管正常結束,這裡掃中斷殘留（判死/讓位/
    #  當機留下的 _dedust_*）——worker 啟動當下本機無 active run,cwd 的 _dedust_* 全是垃圾。
    #  存量歷史（216 C 槽 0GB 的 78 個目錄）也靠這行:pull 新版重啟 worker 即清,免手動。
    import glob
    import shutil
    junk = [d for d in glob.glob("_dedust_*") if os.path.isdir(d)]
    if junk:
        tot = sum(os.path.getsize(os.path.join(r, f))
                  for d in junk for r, _, fs in os.walk(d) for f in fs)
        for d in junk:
            shutil.rmtree(d, ignore_errors=True)
        print(f"🧹 啟動清掃: {len(junk)} 個中斷殘留目錄（~{tot / 1e9:.1f} GB）已清"
              f"——正常兜底（跑完即刪管正常結束,這裡收 Ctrl-C/當機/讓位留下的）: {','.join(junk[:4])}")
    print(f"worker 上線 @ {me}（poll {args.poll}s / 單筆 timeout {args.timeout}s / stale {args.stale}m）")
    while True:
        if sd.joinpath("STOP").exists():
            print("STOP 檔存在,worker 收工")
            break
        _probe_check(me, sd)                             # 機況探針（status/cleanup;空閒輪回應）
        jobs = sorted(json.load(open(str(qp), encoding="utf-8")), key=lambda j: j.get("prio", 9)) \
            if qp.exists() else []
        picked = None
        for j in jobs:
            st = j["store"]
            if j.get("machine") and j["machine"] != me:
                continue                                  # 釘選批(網格收斂等):只給指定機器認領
            if sd.joinpath(st + ".done").exists():
                continue
            cp = sd.joinpath(st + ".claim")
            fp = sd.joinpath(st + ".fail")
            prior_fail = []
            if fp.exists():
                #? .fail 跨機接管（2026-07-15,216 反覆判死）:fail=「那台機判死」非「批判死」——
                #  名單裡沒有自己→接手重試（原人工「刪 .fail 重派」自動化）;每台失敗都記名,
                #  三台輪完=毒批,沒人再接、永久 fail 等人工。舊純文字 .fail 不接管（保守）。
                try:
                    fmach = json.load(open(str(fp), encoding="utf-8")).get("machines")
                except Exception:
                    fmach = None
                if not fmach or me in fmach:
                    continue
                print(f"接管 fail job {st}（{','.join(fmach)} 判死,本機未試——自動重派）")
                try:
                    os.remove(str(fp))
                except OSError:
                    continue                              # 別台剛搶先接管
                try:
                    os.remove(str(cp))                    # 判死機器的殘留 claim 一併清
                except OSError:
                    pass
                prior_fail = fmach
            elif cp.exists():
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
                json.dump(dict(machine=me, at=time.strftime("%Y-%m-%d %H:%M:%S"),
                               **({"prior_fail": prior_fail} if prior_fail else {})), f)
            picked = j
            break
        if picked is None:
            if args.once:
                print("佇列空/無可認領,--once 收工")
                break
            if getattr(args, "selfgen", 0):              # 佇列空 → 自產 tier-2,HFSS 不停(Ricky 2026-07-12)
                if _selfgen_chunk(me, args):
                    continue
            print(f"({time.strftime('%H:%M:%S')}) 佇列無可認領,{args.poll}s 後再掃")   # 心跳:睡眠不裝死
            time.sleep(args.poll)
            continue
        st = picked["store"]
        print(f"▶ 認領 {st}（input {picked['input']}）")
        ns = argparse.Namespace(config=args.config, input=picked["input"], store=st, out=None,
                                sweep=args.sweep, timeout=args.timeout, max_fail=args.max_fail,
                                cooldown=args.cooldown, max_blowout=args.max_blowout,
                                retry_pass=args.retry_pass, job_prio=picked.get("prio", 9),
                                tier2_prio=args.tier2_prio,
                                claim_path=str(sd.joinpath(st + ".claim")), claim_me=me)
        try:
            if run(ns) == "yield":                       # tier-2 讓位/被接管:不標 done,回佇列重掃
                continue
        except SystemExit as e:                          #? 判死:.fail 記機器名單（JSON,別台自動接管）、停機等人工
            prior = []
            try:
                prior = json.load(open(str(sd.joinpath(st + ".claim")), encoding="utf-8")) \
                    .get("prior_fail", [])
            except Exception:
                pass
            with open(str(sd.joinpath(st + ".fail")), "w", encoding="utf-8") as f:
                json.dump(dict(machines=prior + [me], last=str(e),
                               at=time.strftime("%Y-%m-%d %H:%M:%S")), f, ensure_ascii=False)
            print(f"✗ {st} 中止:{e}\nworker 停機（別台 worker 會自動接管此 store;本機修復後重啟 worker 即可）")
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


def watch(args):
    """收檔偵測（blocking;弱模型化 2026-07-12——Monitor 直接掛本命令,不再手寫 bash watcher）:
    每 --poll 秒掃 jobs_state,各 store 到終態印一行;全部終態結束（有 .fail → exit 1,全 done → exit 0）。
    用法: Monitor(command='python -m script.dedust watch --stores dedust_r23b4a,...')"""
    import time
    _, sd = _jobs_paths()
    stores = [s.strip() for s in args.stores.split(",") if s.strip()]
    seen, fail, warned = set(), False, set()
    while len(seen) < len(stores):
        for st in stores:
            if st in seen:
                continue
            fp, dp = sd.joinpath(st + ".fail"), sd.joinpath(st + ".done")
            if fp.exists():
                #? 接管寬限（2026-07-15）:.fail 可被別台 worker 自動接管（檔會消失回到等 .done）——
                #  先警告不判終態,超過 --fail-grace 分鐘沒人接才 FAIL。
                age_m = (time.time() - os.path.getmtime(str(fp))) / 60
                if age_m < args.fail_grace:
                    if st not in warned:
                        print(f"⚠ {st} 保險絲 .fail——等跨機接管（寬限 {args.fail_grace}m）", flush=True)
                        warned.add(st)
                    continue
                print(f"{st} FAIL(保險絲,{int(age_m)}m 無人接管): "
                      f"{open(str(fp), encoding='utf-8').read()[:150]}", flush=True)
                seen.add(st)
                fail = True
            elif dp.exists():
                print(f"{st} DONE: {open(str(dp), encoding='utf-8').read()[:200]}", flush=True)
                seen.add(st)
        if len(seen) < len(stores):
            time.sleep(args.poll)
    print("全部終態——可收檔判讀（analyze batch）", flush=True)
    raise SystemExit(1 if fail else 0)


def jobs_ls(args):
    """看佇列現況（人用;零 token）。預設隱藏已 done 的歷史 job（只列筆數）——接手降噪
    （2026-07-12,90 行→活躍區）;--all 列全部。"""
    import time
    qp, sd = _jobs_paths()
    if not qp.exists():
        print("（無 jobs.json）")
        return
    hidden = 0
    for j in sorted(json.load(open(str(qp), encoding="utf-8")), key=lambda j: j.get("prio", 9)):
        st = j["store"]
        state = "排隊中"
        for tag in ("fail", "done", "claim"):
            fp = sd.joinpath(f"{st}.{tag}")
            if fp.exists():
                info = open(str(fp), encoding="utf-8").read()[:100]
                age = (time.time() - __import__("os").path.getmtime(str(fp))) / 60
                state = f"{tag}（{age:.0f} 分前）{info}"
                break
        if state.startswith("done") and not getattr(args, "all", False):
            hidden += 1
            continue
        rp = DATASET_PATH.joinpath(st, "results.json")
        mp = DATASET_PATH.joinpath(j["input"], "manifest.json")
        total = len(json.load(open(str(mp), encoding="utf-8"))) if mp.exists() else "?"
        done_n = 0
        if rp.exists():
            res = json.load(open(str(rp), encoding="utf-8"))
            done_n = sum(1 for v in res.values() if "wm" in v)
        print(f"[prio {j.get('prio', 9)}] {st}: {done_n}/{total} | {state}")
    if hidden:
        print(f"（已隱藏 {hidden} 個 done 歷史 job;--all 列全部）")


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
    s.add_argument("--shards", type=int, default=3, help="輸出切幾夾（夾多於機器=慢機不拖全隊;batch5 起建議 6）")
    s.add_argument("--o", type=int, default=-1, help="O 臂筆數（-1=核心半數;0=純樂透填空批,配 --tag）")
    s.add_argument("--tag", default=None, help="填空批命名（如 g1 → 夾 dedust_r21g1*_input、id mg1_*;正批留空）")
    s.set_defaults(fn=select_r21harvest)

    s = sub.add_parser("select-r22mix", help="R22 分布組合批：O哨兵+M王朝+C冷支+Q偏科修復+H hslot+W彩票（六臂,判準見 docstring）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260712)
    s.add_argument("--sm", default="sm_reanchor12.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--o", type=int, default=10)
    s.add_argument("--m", type=int, default=50)
    s.add_argument("--c", type=int, default=40)
    s.add_argument("--q", type=int, default=30)
    s.add_argument("--h", type=int, default=12, help="H hslot 劑量臂（0=關;b3 起低側收案後關）")
    s.add_argument("--s", type=int, default=0, help="S 槽鏈臂（部分槽+修 rad 鏈;新臂先導 ≤15）")
    s.add_argument("--wild", type=int, default=8)
    s.add_argument("--shards", type=int, default=6)
    s.set_defaults(fn=select_r22mix, round=22, key="oob")

    s = sub.add_parser("select-r23", help="R23 價值軸批：O 主力 sel_score 鍵+M+C+S+D denovo+W（同 r22mix 機器,round 號貫穿命名）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260712)
    s.add_argument("--sm", default="sm_reanchor15.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--o", type=int, default=46, help="O 主力臂（pred_sel 升冪=價值軸主鍵）")
    s.add_argument("--m", type=int, default=33)
    s.add_argument("--c", type=int, default=40)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=15, help="S 槽鏈臂（判準沿用 R22 §1 修訂）")
    s.add_argument("--d", type=int, default=12, help="D de novo 常設臂（學費預算制,b3 起;篩選用 --denovo-sm）")
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm",
                   help="D 臂專屬篩選 SM（預設 34k 底座;資料夠後換 sm_denovo*）")
    s.add_argument("--i", type=int, default=0, help="I 資訊臂（兩 SM 分歧 top=主動學習;R24 起 12）")
    s.add_argument("--novelty", action="store_true", help="B 新穎性紅利進鍵（λ=0.02·min(d,20);R24 起開）")
    s.add_argument("--root-cap", type=float, default=0.0, dest="root_cap",
                   help="根多樣性稅:單一池根占比上限（0=關;R24 起 0.4——治同根打轉）")
    s.add_argument("--wild", type=int, default=4)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head1.pth", dest="rad_head",
                   help="rad 頭權重（一律算 pred_rad 記 manifest 供前瞻;None 字串=關）")
    s.add_argument("--rad-key", action="store_true", dest="rad_key",
                   help="pred_rad 進選批鍵（前提:held-out/前瞻 ρ≥0.4;預設只記不用）")
    s.set_defaults(fn=select_r22mix, round=23, key="sel")

    s = sub.add_parser("select-r24", help="R24 降根計畫：根多樣性稅+池外梯度(D20/I12)+誘因包(同 r22mix 機器)")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260713)
    s.add_argument("--sm", default="sm_reanchor19.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--o", type=int, default=40)
    s.add_argument("--m", type=int, default=25, help="對照+王朝保底(≥25 前瞻統計/近王產線)")
    s.add_argument("--c", type=int, default=30)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=15)
    s.add_argument("--d", type=int, default=20, help="D de novo 學費批(池外梯度加碼)")
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12, help="I 資訊臂(兩 SM 分歧=主動學習)")
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.4, dest="root_cap",
                   help="根多樣性稅:單根占比上限(R24=0.4;三標率跌>50% 回 0.5)")
    s.add_argument("--wild", type=int, default=8)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head2.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.set_defaults(fn=select_r22mix, round=24, key="sel")

    s = sub.add_parser("select-r25", help="R25 多樣性加碼：根稅 0.6+王朝 48%+F 碎片/低側修復臂(同 r22mix 機器)")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260713)
    s.add_argument("--sm", default="sm_reanchor22.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--o", type=int, default=32)
    s.add_argument("--m", type=int, default=20, help="對照+王朝保底(前瞻統計/近王產線)")
    s.add_argument("--c", type=int, default=20)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=20)
    s.add_argument("--d", type=int, default=8, help="D 復航 b2 起(Ricky 2026-07-13 裁決=看進步趨勢;b1=0)")
    s.add_argument("--d-sm", default="sm_denovo1.pth", dest="d_sm",
                   help="D 臂專屬選拔器(sm_reanchor train-denovo 產出,每批重訓)")
    s.add_argument("--f", type=int, default=24, help="F 碎片/低側修復臂(錨=帶外極乾淨載體;學費制 3 批)")
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=18, help="I 資訊臂(兩 SM 分歧=主動學習;b1=22)")
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap",
                   help="根多樣性稅 R25=0.6(軸相關枯竭觸發:margin 連三批無新高)")
    s.add_argument("--wild", type=int, default=8, help="b1=12;b2 起讓 4 席給 D 復航")
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head2.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.set_defaults(fn=select_r22mix, round=25, key="sel")

    s = sub.add_parser("select-r26", help="R26 帶外前瞻復活驗證：退 rad 鍵+I 26 加碼+F 12 二期+D 8 判決批(同 r22mix 機器)")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260714)
    s.add_argument("--sm", default="sm_reanchor25.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--o", type=int, default=32)
    s.add_argument("--m", type=int, default=20, help="對照+oob 前瞻統計母體(主判準)")
    s.add_argument("--c", type=int, default=20)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=20)
    s.add_argument("--d", type=int, default=8, help="D 第二期判決批(min sel 89.8/93.0→本輪判)")
    s.add_argument("--d-sm", default="sm_denovo3.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=12, help="F 第二期縮編(趨勢正續 1 期;round-26 §1)")
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=26, help="I 資訊臂加碼(R25b3 61% 爆發)")
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--wild", type=int, default=12)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head2.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key",
                   help="R26 預設退鍵(R25 連兩批<0.3);復鍵=續記前瞻連兩批>=0.3")
    s.set_defaults(fn=select_r22mix, round=26, key="sel")

    s = sub.add_parser("select-r27", help="R27 加厚雙主軸：N 網架臂(骨架+網布四式)×R26 延續(D 加碼/I 高配)(同 r22mix 機器)")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260714)
    s.add_argument("--sm", default="sm_reanchor27.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--o", type=int, default=26)
    s.add_argument("--m", type=int, default=20, help="前瞻統計母體(維持 n=20)")
    s.add_argument("--c", type=int, default=14)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=14)
    s.add_argument("--d", type=int, default=14, help="D 加碼(第二期判決過:min sel 連降 65.0→51.7)")
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=8, help="F 席位:依 R26b3 收官判定,可 --f 0 關")
    s.add_argument("--mesh", type=int, default=24, help="N 網架臂(骨架+網布;H2 對照優先,H1 隨批擴)")
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=22)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--wild", type=int, default=8)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head2.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.set_defaults(fn=select_r22mix, round=27, key="sel")

    s = sub.add_parser("select-r28", help="R28 塊內 rad 手術：Y 36（half 半成品錨,網布凍結,maximin 鍵）+梯子延續(同 r22mix 機器)")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260715)
    s.add_argument("--sm", default="sm_reanchor30.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--o", type=int, default=18)
    s.add_argument("--m", type=int, default=20, help="前瞻統計母體(不動)")
    s.add_argument("--c", type=int, default=8)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=10)
    s.add_argument("--d", type=int, default=18, help="D 加碼(Ricky 2026-07-14 再降王類)")
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=8, help="F 最小席位(待 Ricky 裁決)")
    s.add_argument("--mesh", type=int, default=0, help="N 臂功成休兵(H1/H2 答畢)")
    s.add_argument("--surgery", type=int, default=36, help="Y 塊內 rad 手術(half 五錨,網布凍結)")
    s.add_argument("--blockmap", type=int, default=0, help="B 塊級承重圖探針(t07h/p00h 逐塊 ablate/halve,決定性;b2 搭載)")
    s.add_argument("--bmix", type=int, default=0, help="U 承重圖導引組合手術(t07h 低承重塊 2-3 塊組合;b3 搭載)")
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=18, help="I 續高配(ikpi 首讀 I−M +0.20 成立)")
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.12, dest="dyn_simcap",
                   help="王系相似度稅:d_dyn<20 的近王/近似樣本批內佔比上限（Ricky 2026-07-14 二次加壓 0.12;0=關）")
    s.add_argument("--wild", type=int, default=14)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head2.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.set_defaults(fn=select_r22mix, round=28, key="sel")

    s = sub.add_parser("select-r29", help="R29 G 臂主力批：梯度反傳 staging 76+常規臂 74（判準寫死於 round-29 檔）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260716)
    s.add_argument("--sm", default="sm_reanchor33.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--g", type=int, default=76, help="G 梯度臂（sm_invert gen staging 讀入,絕對主力）")
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"), help="staging 夾路徑（sm_invert gen 產物）")
    s.add_argument("--o", type=int, default=12, help="O 梯子房租")
    s.add_argument("--m", type=int, default=14, help="前瞻統計母體")
    s.add_argument("--c", type=int, default=6)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=6)
    s.add_argument("--d", type=int, default=12, help="D 降額（b3 倒退+對決四連敗;資訊帳仍榜首故不砍臂）")
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0, help="F 退役（修復精神由 G/U 繼承）")
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=14, help="I 降額（ikpi 帳 +0.20/+0.09/−0.00 紅利消退）")
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.12, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.4, dest="dyn_frac",
                   help="錨點抽樣王朝池機率（2026-07-15 戰略換軸 0.7→0.4;冷支/新血讓位）")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head2.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.set_defaults(fn=select_r22mix, round=29, key="sel")

    s = sub.add_parser("select-r30", help="R30 SM 準度輪 2：G64+L20 低側據點（r_feed 鍵首航）+恆溫加碼配額（判準寫死於 round-30 檔）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260717)
    s.add_argument("--sm", default="sm_reanchor36.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--g", type=int, default=64, help="G 梯度臂（free28 含碎片 init/champ24/surg8/oobp4）")
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=20, help="L 低側據點臂（gap 7 錨鄰域,r_feed 鍵）")
    s.add_argument("--o", type=int, default=8)
    s.add_argument("--m", type=int, default=14, help="前瞻統計母體(不動)")
    s.add_argument("--c", type=int, default=4)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=2)
    s.add_argument("--d", type=int, default=16, help="D 恆溫加碼（12→16）")
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.12, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.4, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head2.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.set_defaults(fn=select_r22mix, round=30, key="sel")

    s = sub.add_parser("select-r31", help="R31 王系凍結輪：champ 換錨中繼帶+L20 續攻+M 樂透王錨降權（判準寫死於 round-31 檔）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260718)
    s.add_argument("--sm", default="sm_reanchor39.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--g", type=int, default=64, help="G（free28/champ-bridge24〔中繼錨〕/oobp12;surg 併 bridge）")
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=20, help="L 中繼帶鄰域續攻（r_feed 鍵）")
    s.add_argument("--o", type=int, default=8)
    s.add_argument("--m", type=int, default=14, help="前瞻統計母體(不動)")
    s.add_argument("--c", type=int, default=4)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=2)
    s.add_argument("--d", type=int, default=16)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.12, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac",
                   help="王朝抽樣再壓 0.4→0.2（王系凍結,Ricky 2026-07-16 叮嚀）")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head39.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.set_defaults(fn=select_r22mix, round=31, key="sel")

    s = sub.add_parser("select-r32", help="R32 海峽輪：X 雜交臂（雙親+SM 期望閘 LCB）+L24 續攻+影子 CNN 對決（判準寫死於 round-32 檔）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260719)
    s.add_argument("--sm", default="sm_reanchor42.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=24, help="X 海峽雜交臂（雙親,oversample×4→LCB top）")
    s.add_argument("--g", type=int, default=44, help="G（free28/oobp16）")
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=24, help="L 中繼帶續攻（r_feed 鍵）")
    s.add_argument("--o", type=int, default=8)
    s.add_argument("--m", type=int, default=14, help="前瞻統計母體(不動)")
    s.add_argument("--c", type=int, default=4)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=12)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.12, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=8)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head42.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.set_defaults(fn=select_r22mix, round=32, key="sel")

    s = sub.add_parser("select-r33", help="R33 反王朝結構輪：表型軟過濾（黑名單=底1大+上2中）+RADGATE 錨（rad 閘攻堅）+CNN 混合鍵記錄+B 泵續投（判準寫死於 round-33 檔）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260720)
    s.add_argument("--sm", default="sm_reanchor45.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=60, help="G（free24/oobp12/B泵24=selfgen 錨帶集中）")
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=24, help="L rad 閘攻堅（RADGATE 六錨:lo 壓∧rad 半好交集帶）")
    s.add_argument("--o", type=int, default=8)
    s.add_argument("--m", type=int, default=14, help="前瞻統計母體(不動)")
    s.add_argument("--c", type=int, default=4)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=16, help="D +4（反王朝結構自由帶）")
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=16, help="I 資訊臂 +4（連三批 3-4 三標穩定產線）")
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.12, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=8)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head45.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--struct-pen", type=float, default=2.0, dest="struct_pen",
                   help="王朝表型罰分（b3 判準上調 4.0）")
    s.set_defaults(fn=select_r22mix, round=33, key="sel")

    s = sub.add_parser("select-r34", help="R34 第二血脈輪：champ-I（I 系近王錨）+L 去王朝錨組+I 24 加碼（判準寫死於 round-34 檔）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260721)
    s.add_argument("--sm", default="sm_reanchor48.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=60, help="G（free24/oobp12/champ-I 24）")
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=12, help="L 去王朝錨組（爬山鏈+t03r 四錨）")
    s.add_argument("--o", type=int, default=8)
    s.add_argument("--m", type=int, default=14, help="M 凍結對照臂(不動)")
    s.add_argument("--c", type=int, default=4)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=16)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=24, help="I 加碼（連五批爬升+平王筆出自 I 系）")
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.12, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=12)
    s.add_argument("--shards", type=int, default=6)
    s.add_argument("--rad-head", default="rad_head48.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.set_defaults(fn=select_r22mix, round=34, key="sel")

    s = sub.add_parser("chain", help="爬山鏈 daemon（tier 0 插隊;錨 d=1 純隨機包→判讀→換錨迴圈;≤2 條並行=decisions）")
    s.add_argument("--name", required=True, help="鏈名（夾名 dedust_<name>_pNN;帳 docs/chains/<name>.jsonl）")
    s.add_argument("--anchor", required=True)
    s.add_argument("--source-input", required=True, dest="source_input")
    s.add_argument("--goal", required=True, choices=["wm", "dual", "rad", "lo", "hi", "tri"],
                   help="目標鍵（發鏈前寫死;lo/hi=合格門檻內壓單側;tri=lo≤−2 內爬 min(wm−0.15,rad)=左側合格解會師鍵）")
    s.add_argument("--anchor-score", type=float, default=None, dest="anchor_score",
                   help="錨的已知 score（首包 baseline;不給=首包必換錨）")
    s.add_argument("--n", type=int, default=25)
    s.add_argument("--prio", type=int, default=1, help="tier 0=1（插隊）")
    s.add_argument("--dry", type=int, default=2, help="連 N 包無勝錨=收鏈")
    s.add_argument("--max-packs", type=int, default=20, dest="max_packs")
    s.add_argument("--expert", action="store_true", help="域專家模式:鏈資料微調 SM→枚舉排序 top+隨機對照（dual/wm 鏈適用）")
    s.add_argument("--exp-rand", type=int, default=13, dest="exp_rand", help="對照隨機席（其餘=專家 top）")
    s.add_argument("--mutator", choices=["px", "group"], default="px",
                   help="包生成算子:px=d1 翻轉（原行為）;group=組級變異 70/30 混 px 對照（R41 C 臂,判準=round-41 §1）")
    s.add_argument("--grp-frac", type=float, default=0.7, dest="grp_frac", help="group 模式組級佔比")
    s.set_defaults(fn=chain)

    s = sub.add_parser("select-r35", help="R35 新節奏：批 75（3 夾）高頻迭代;asym 記錄鍵（判準寫死於 round-35 檔）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260723)
    s.add_argument("--sm", default="sm_reanchor51.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=30, help="G（free24/oobp6）")
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=12, help="L 爬山錨組")
    s.add_argument("--o", type=int, default=4)
    s.add_argument("--m", type=int, default=7, help="M 凍結對照（跨批合併讀）")
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=6)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=8)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.12, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=6)
    s.add_argument("--shards", type=int, default=3)
    s.add_argument("--rad-head", default="rad_head51.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.set_defaults(fn=select_r22mix, round=35, key="sel")

    s = sub.add_parser("select-r36", help="R36 抗線輪：批 50（2 夾;tier 再平衡降格）;CNN 單 rank;free 減半;rad-key 退鍵（判準寫死於 round-36 檔）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260724)
    s.add_argument("--sm", default="sm_reanchor54.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=18, help="G（free12/oobp6;free 減半=外推區止損）")
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=8)
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5, help="M 凍結對照（跨批合併讀）")
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=3)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=8, help="I 甜蜜點續持")
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.12, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=3)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head54.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key", help="R36 預設退鍵（連兩批 <0.3）;復鍵帳跨批記錄")
    s.add_argument("--cnn-solo", action="store_true", default=True, dest="cnn_solo",
                   help="O 臂 CNN 單 rank（R35 收輪轉正;--no-cnn-solo 回雙 rank）")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.set_defaults(fn=select_r22mix, round=36, key="sel")

    s = sub.add_parser("select-r37", help="R37 左側大陸殖民輪：批 50;L 臂新大陸錨組+ref/rej balance;diagb 罰;rad-key/cnn-solo 皆退（判準寫死於 round-37 檔）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260725)
    s.add_argument("--sm", default="sm_reanchor57.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12, help="G（free6/oobp6;SM 盲區探測+誤差錨工廠定位）")
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=12, help="L 新大陸錨組（tri 前緣+t07/l31b2;半ref半rej）")
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=4)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=8)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=4)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head57.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo",
                   help="R36 判定回退雙 rank——預設關（排序ρ≠top-k 選拔）")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen",
                   help="對角橋罰/橋（上限 5 橋;Ricky 2026-07-23）")
    s.set_defaults(fn=select_r22mix, round=37, key="sel")

    s = sub.add_parser("select-r38", help="R38 影子二號輪：批 54;lo 判別器記錄鍵;L 半ref半rej 常駐（判準寫死於 round-38 檔）")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260726)
    s.add_argument("--sm", default="sm_reanchor60.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=12)
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=6)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=8)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=6)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head60.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=38, key="sel")

    s = sub.add_parser("select-r39", help="R39 左側家族化輪：F 臂家族錨組（首例+近親）;two 絕對值換裝;判準寫死於 round-39 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260727)
    s.add_argument("--sm", default="sm_reanchor63.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=16, help="F 臂家族錨組（半ref半rej）")
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=6)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=8)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=6)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head63.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=39, key="sel")

    s = sub.add_parser("select-r40", help="R40 換裝與空洞輪：two 主通道換裝（MLP 降審計鍵）;V 臂 response 空洞反演首航（G12/I12/V8/M5/O3/K2/D10/W10=62）;F 臂撤;判準寫死於 round-40 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260728)
    s.add_argument("--sm", default="sm_reanchor66.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0, help="F 臂撤（R39 孤點結論）")
    s.add_argument("--v", type=int, default=8, help="V 臂 response 空洞反演（4 質心×2）")
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=10)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head66.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=40, key="sel")

    s = sub.add_parser("select-r41", help="R41 常態線（同 r40 配置:two 主通道+V 臂;組測試組另走鏈線 C 臂）;判準寫死於 round-41 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260729)
    s.add_argument("--sm", default="sm_reanchor69.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0)
    s.add_argument("--v", type=int, default=8)
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=10)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head69.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=41, key="sel")

    s = sub.add_parser("select-r42", help="R42 常態輪（同 r41 配置:two 主通道+V 臂常駐）;判準寫死於 round-42 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260730)
    s.add_argument("--sm", default="sm_reanchor71.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0)
    s.add_argument("--v", type=int, default=8)
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=10)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head71.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=42, key="sel")

    s = sub.add_parser("select-r43", help="R43 組文法首航：D 臂 10 席=文法槽（舊2/GA2/GB2/GC2/GD1/GDd1,槽內排序配額固定）;判準寫死於 round-43 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260731)
    s.add_argument("--sm", default="sm_reanchor74.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0)
    s.add_argument("--v", type=int, default=8)
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=10)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head74.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=43, key="sel")

    s = sub.add_parser("select-r44", help="R44 文法二輪：D 臂槽=舊2/GA2/GB2/GA2(組義槽)2/GD1/GDd1（GC 汰）;判準寫死於 round-44 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260801)
    s.add_argument("--sm", default="sm_reanchor77.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0)
    s.add_argument("--v", type=int, default=8)
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=10)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head77.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=44, key="sel")

    s = sub.add_parser("select-r45", help="R45 接力輪：D 臂槽收斂 old6/GDd4（round-44 §5 判定）;判準寫死於 round-45 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260802)
    s.add_argument("--sm", default="sm_reanchor80.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0)
    s.add_argument("--v", type=int, default=8)
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=10)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head80.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=45, key="sel")

    s = sub.add_parser("select-r46", help="R46 接力二輪：批線常態（=r45 配置 old6/GDd4）;判準寫死於 round-46 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260803)
    s.add_argument("--sm", default="sm_reanchor83.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0)
    s.add_argument("--v", type=int, default=8)
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=10)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head83.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=46, key="sel")

    s = sub.add_parser("select-r47", help="R47 接力三輪：批線常態（=r46 配置）;判準寫死於 round-47 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260804)
    s.add_argument("--sm", default="sm_reanchor85.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0)
    s.add_argument("--v", type=int, default=8)
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=10)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head85.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=47, key="sel")

    s = sub.add_parser("select-r48", help="R48 定向嫁接輪：批線常態（=r47 配置+漏斗二次放大）;判準寫死於 round-48 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260805)
    s.add_argument("--sm", default="sm_reanchor89.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0)
    s.add_argument("--v", type=int, default=8)
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=10)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head89.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=48, key="sel")

    s = sub.add_parser("select-r49", help="R49 兩段式制度化輪：批線常態（=r48 配置,v91 配套）;判準寫死於 round-49 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260806)
    s.add_argument("--sm", default="sm_reanchor91.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=12)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0)
    s.add_argument("--v", type=int, default=8)
    s.add_argument("--o", type=int, default=3)
    s.add_argument("--m", type=int, default=5)
    s.add_argument("--c", type=int, default=2)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=10)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=12)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=10)
    s.add_argument("--shards", type=int, default=2)
    s.add_argument("--rad-head", default="rad_head91.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=49, key="sel")

    s = sub.add_parser("select-r50", help="R50 型態體系軸:正片保底 30(=r49 配置縮編半);負片臂另走 select-neg;判準=round-50 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260807)
    s.add_argument("--sm", default="sm_reanchor95.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=6)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0)
    s.add_argument("--v", type=int, default=2)
    s.add_argument("--o", type=int, default=2)
    s.add_argument("--m", type=int, default=3)
    s.add_argument("--c", type=int, default=1)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=5)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=6)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=5)
    s.add_argument("--shards", type=int, default=1)
    s.add_argument("--rad-head", default="rad_head95.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=50, key="sel")

    s = sub.add_parser("select-r51", help="R51 橋接與進鍵輪:正片 30(=r50 配置);判準=round-51 檔")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--seed", type=int, default=20260809)
    s.add_argument("--sm", default="sm_reanchor99.pth")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--xover", type=int, default=0)
    s.add_argument("--g", type=int, default=6)
    s.add_argument("--gstage", default=os.path.join("tmp", "invert_stage"))
    s.add_argument("--lbeach", type=int, default=0)
    s.add_argument("--v", type=int, default=2)
    s.add_argument("--o", type=int, default=2)
    s.add_argument("--m", type=int, default=3)
    s.add_argument("--c", type=int, default=1)
    s.add_argument("--q", type=int, default=0)
    s.add_argument("--h", type=int, default=0)
    s.add_argument("--s", type=int, default=0)
    s.add_argument("--d", type=int, default=5)
    s.add_argument("--d-sm", default="sm_denovo2.pth", dest="d_sm")
    s.add_argument("--f", type=int, default=0)
    s.add_argument("--mesh", type=int, default=0)
    s.add_argument("--surgery", type=int, default=0)
    s.add_argument("--blockmap", type=int, default=0)
    s.add_argument("--bmix", type=int, default=0)
    s.add_argument("--denovo-sm", default="sm_harvest.pth", dest="denovo_sm")
    s.add_argument("--i", type=int, default=6)
    s.add_argument("--novelty", action="store_true")
    s.add_argument("--root-cap", type=float, default=0.6, dest="root_cap")
    s.add_argument("--dyn-simcap", type=float, default=0.08, dest="dyn_simcap")
    s.add_argument("--dyn-frac", type=float, default=0.2, dest="dyn_frac")
    s.add_argument("--wild", type=int, default=5)
    s.add_argument("--shards", type=int, default=1)
    s.add_argument("--rad-head", default="rad_head99.pth", dest="rad_head")
    s.add_argument("--rad-key", action="store_true", dest="rad_key")
    s.add_argument("--cnn-solo", action="store_true", default=False, dest="cnn_solo")
    s.add_argument("--no-cnn-solo", action="store_false", dest="cnn_solo")
    s.add_argument("--struct-pen", type=float, default=4.0, dest="struct_pen")
    s.add_argument("--diagb-pen", type=float, default=2.0, dest="diagb_pen")
    s.set_defaults(fn=select_r22mix, round=51, key="sel")

    s = sub.add_parser("select-neg", help="R50 負片臂:neg_gen 七臂池→farthest-point 覆蓋選席(SM-blind;判準=round-50/decisions 型態體系軸;每輪必換 --seed)")
    s.add_argument("--round", type=int, required=True, help="必填防跨輪覆寫(稽核 H1)")
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--pool", type=int, default=600)
    s.add_argument("--pad", type=int, default=5)
    s.add_argument("--stratify", action="store_true", help="每臂配額分層(b2 起;稽核 M1 判準修訂)")
    s.add_argument("--arms", default=None, help="逗號臂單(如排除 sierp——全臂僅 2 圖已於 b9 覆蓋;稽核 M2)")
    s.add_argument("--seed", type=int, default=20260808)
    s.set_defaults(fn=select_neg)

    s = sub.add_parser("select-senior", help="R50 學長未殖民族臂:pool top-300 家族領袖池值降冪逐批驗(kind=senior;判準=round-50 §1②)")
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--n", type=int, default=10)
    s.set_defaults(fn=select_senior)

    s = sub.add_parser("select-meshconv", help="網格收斂實驗:已量測 pattern×三組求解設定重測(kind=meshconv;判準=proposal-mesh-convergence §3;發車=同機釘選)")
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--ids-file", required=True, help="一行=`來源輸入夾:id 組別(A/B)`;S2 只收 A")
    s.set_defaults(fn=select_meshconv)

    s = sub.add_parser("select-bridge", help="R50 橋接臂:正負片中間過渡(dil/ero/mix;kind=bridge;批號 30+)")
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--n", type=int, default=280)
    s.add_argument("--pad", type=int, default=5)
    s.add_argument("--seed", type=int, default=20260811)
    s.add_argument("--parent-inputs", dest="parent_inputs",
                   default="dedust_r49b1a_input,dedust_r49b2a_input,dedust_r49b3a_input,dedust_r48b1a_input")
    s.set_defaults(fn=select_bridge)

    s = sub.add_parser("select-graft", help="R48 嫁接試點:王朝骨架×左側引擎(A 替換/B 對角加掛);判準=round-48 §1")
    s.add_argument("--out", default="dedust_g48graft1_input")
    s.add_argument("--n", type=int, default=25)
    s.add_argument("--seed", type=int, default=4810)
    s.set_defaults(fn=select_graft)

    s = sub.add_parser("select-scope", help="顯微鏡包:錨 d=1 全枚舉→CNN 排序 top N（25 筆/輪封頂,錨輪換防陷;decisions 2026-07-17）")
    s.add_argument("--anchor", required=True)
    s.add_argument("--source-input", required=True, dest="source_input")
    s.add_argument("--cnn", default="sm_shadow45.pth")
    s.add_argument("--n", type=int, default=25)
    s.add_argument("--tag", default="1", help="包序號（入 id 前綴 s<tag>_）")
    s.add_argument("--rand-mix", type=int, default=0, dest="rand_mix",
                   help="混入隨機 k 筆當對照（CNN top n-k+隨機 k;驗證 CNN 微尺度選擇加值）")
    s.add_argument("--input", required=True)
    s.add_argument("--config", default=DEFAULT_CFG)
    s.set_defaults(fn=select_scope)

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
    s.add_argument("--max-fail", type=int, default=5, dest="max_fail", help="連續失敗幾筆＝保險絲熔斷一次")
    s.add_argument("--cooldown", type=int, default=600, help="熔斷後冷卻秒數（殺透 HFSS 睡完重開再試）")
    s.add_argument("--max-blowout", type=int, default=3, dest="max_blowout",
                   help="熔斷幾循環判 HFSS 壞死中止（延長死亡判定,2026-07-15）")
    s.add_argument("--retry-pass", type=int, default=2, dest="retry_pass",
                   help="批尾自動補測輪數（殘留 error 殺重開 HFSS 重試;0=關）")
    s.set_defaults(fn=run)

    s = sub.add_parser("worker", help="資料工廠 worker：常駐認領 NAS 佇列 job（jobs.json）自動跑批")
    s.add_argument("--config", default=DEFAULT_CFG)
    s.add_argument("--sweep", default="Interpolating", choices=["Interpolating", "Discrete", "Fast"])
    s.add_argument("--poll", type=int, default=300, help="佇列空時幾秒掃一次")
    s.add_argument("--timeout", type=int, default=900)
    s.add_argument("--max-fail", type=int, default=5, dest="max_fail")
    s.add_argument("--cooldown", type=int, default=600)
    s.add_argument("--max-blowout", type=int, default=3, dest="max_blowout")
    s.add_argument("--retry-pass", type=int, default=2, dest="retry_pass")
    s.add_argument("--stale", type=int, default=45, help="claim 無進度幾分鐘可被接管")
    s.add_argument("--once", action="store_true", help="只跑一個 job 就收工（測試用）")
    s.add_argument("--tier2-prio", type=int, default=8, dest="tier2_prio",
                   help="prio ≥ 此值=tier-2 填空批（每筆完成後掃佇列,tier-1 出現即讓位）")
    s.add_argument("--selfgen", type=int, default=12,
                   help="佇列全空時自產 tier-2 段長（歷史翻bit+查重;0=關;任何 job 入佇列即讓位）")
    s.set_defaults(fn=worker)

    s = sub.add_parser("jobs-add", help="把批次加進派工佇列（select+check-dup 先跑完）")
    s.add_argument("--input", required=True)
    s.add_argument("--store", required=True)
    s.add_argument("--prio", type=int, default=5, help="小=先跑")
    s.add_argument("--machine", default=None, help="釘選 IP 末段(216/218/37)——只有這台認領;同機對照實驗用")
    s.set_defaults(fn=jobs_add)

    s = sub.add_parser("jobs-ls", help="看派工佇列現況（預設隱藏 done 歷史;--all 列全部）")
    s.add_argument("--all", action="store_true", help="連 done 歷史 job 一起列")
    s.set_defaults(fn=jobs_ls)

    s = sub.add_parser("probe", help="機況探針：經 NAS 向正式機發白名單指令（worker 空閒輪回應）")
    s.add_argument("--machine", required=True, help="IP 末段（216/218/37）")
    s.add_argument("--action", default="status", choices=["status", "cleanup"],
                   help="status=磁碟/殘留/git/HFSS 行程;cleanup=刪 _dedust_* 殘留")
    s.add_argument("--wait", type=int, default=360, help="等回應秒數（worker 跑 job 中會佔線）")
    s.set_defaults(fn=probe)

    s = sub.add_parser("watch", help="收檔偵測（blocking;Monitor 直接掛;全終態 exit0/含 fail exit1）")
    s.add_argument("--stores", required=True, help="逗號分隔 store 名（dedust_r23b4a,...）")
    s.add_argument("--poll", type=int, default=180)
    s.add_argument("--fail-grace", type=int, default=90, dest="fail_grace",
                   help=".fail 出現後等跨機接管幾分鐘,超過才判終態 FAIL")
    s.set_defaults(fn=watch)

    s = sub.add_parser("report", help="匯總表（貼 round 檔 §4）")
    s.add_argument("--input", default=DEFAULT_INPUT)
    s.add_argument("--store", default=DEFAULT_STORE)
    s.set_defaults(fn=report)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
