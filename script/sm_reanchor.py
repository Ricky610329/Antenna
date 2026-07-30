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
#? 2026-07-12 檔案化（弱模型化:重錨不再改原始碼）——清單在 configs/clean_stores.txt,
#  追加走 `train --add "storeA,storeB"`（自動 append+訓練;git diff 可審計）。
_CS_PATH = os.path.join(REPO, "configs", "clean_stores.txt")


def _move_opt_state(sm, device):
    for st in sm.optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)


from contextlib import contextmanager


@contextmanager
def _train_device(sm):
    """訓練段上 GPU（有卡才）、訓完搬回 CPU——存檔/eval/下游全維持 CPU 語義（2026-07-30 GPU 開關）。
    load_torch 已 map 到 config.device,跨裝置載入天生安全;shell 的 device setter 只搬 model,
    optimizer state 這裡自己搬。⚠ CUDA 訓練非 bit 級決定性（生成決定性鐵則只約束 select,不含 SM 訓練）。"""
    if not torch.cuda.is_available():
        yield
        return
    prev = _config.device
    _config.device = "cuda:0"
    sm.device = "cuda:0"
    _move_opt_state(sm, "cuda:0")
    try:
        yield
    finally:
        sm.device = "cpu"
        _move_opt_state(sm, "cpu")
        _config.device = prev


def _cs_sort_key(name):
    """公證店排序鍵:dedust_rNNn*（notarize）排最前,讓 3/3 重測樣本在「首見即贏」去重中勝出。"""
    import re
    return 0 if re.match(r"dedust_r\d+n", name) else 1


def _load_clean_stores():
    if not os.path.exists(_CS_PATH):
        raise SystemExit(f"{_CS_PATH} 不存在——CLEAN_STORES 已檔案化(2026-07-12),請自 git 還原該檔")
    out = []
    with open(_CS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line and line not in out:
                out.append(line)
    #? 自產 tier-2（selfgen dedust_auto*）與**鏈店（dedust_c*）**永遠是乾淨 HFSS 真值 → 自動納入
    #  （2026-07-13 selfgen 修;2026-07-24 鏈店修:R34 後鏈資料 ~700 筆最密前緣教材兩頭落空——
    #  重錨只 --add 批線店、auto 自動含、鏈店漏接=sm_two 在 tri 牆鄰域 wm 預測偏 −3.6 的真因）。
    import glob
    for pat in ("dedust_auto*", "dedust_c*"):
        for p in sorted(glob.glob(str(DATASET_PATH.joinpath(pat)))):
            name = os.path.basename(p)
            if name.endswith("_input"):
                continue
            if os.path.isdir(p) and DATASET_PATH.joinpath(name, "results.json").exists() and name not in out:
                out.append(name)
    #? 公證店（dedust_rNNn*）前移——「certified 先見先贏」去重不變式從註解變事實
    #  （audit 2026-07-29:原順序公證店 0/23 勝出,3/3 均值標籤被首見單測值蓋掉;list.sort 穩定,其餘順序不動）
    out.sort(key=_cs_sort_key)
    return tuple(out)


CLEAN_STORES = _load_clean_stores()
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
    #? 凍結成員強制留 held-out（2026-07-16 修:「每第 5 筆」隨資料插入整體位移——v40 實測凍結
    #  交集剩 416/1772,凍結尺失效;修法=凍結清單成員一律進 ho,其餘照 hash 每 5 筆）。
    import hashlib as _h
    import json as _j
    fz_path = os.path.join(REPO, "configs", "heldout_frozen.json")
    fzk = set(_j.load(open(fz_path, encoding="utf-8"))["keys"]) if os.path.exists(fz_path) else set()
    tr, ho = [], []
    j2 = 0
    for k in sorted(seen):
        #! 凍結清單的 md5=bool bytes（評估端口徑）——切分端 key 是 float32 bytes,必須轉 bool 再湊
        #  （v41 實測踩坑:float32 直接 md5 → 0 匹配,凍結尺假縮水 353）。
        kb = _h.md5((np.frombuffer(k, dtype=np.float32) > 0.5).tobytes()).hexdigest()
        if kb in fzk:
            ho.append(seen[k])
        else:
            (ho if j2 % 5 == 0 else tr).append(seen[k])
            j2 += 1
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


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)
DYN = ("c18", "vg0338", "vg0396", "g1_038", "g1_039", "r2_016", "r3_001")


def _dyn_pack():
    """王朝家族參照集（分層 held-out 用;同 analyze/tax 口徑）。"""
    import json
    pats = []
    for fol in os.listdir(str(DATASET_PATH)):
        mp = DATASET_PATH.joinpath(fol, "manifest.json")
        if not fol.endswith("_input") or not mp.exists():
            continue
        for m in json.load(open(str(mp), encoding="utf-8")):
            if any(t in m["id"] for t in DYN):
                f = DATASET_PATH.joinpath(fol, m["id"] + ".pt")
                if f.exists():
                    pats.append(np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5)
    return np.packbits(np.stack(pats).astype(np.uint8), axis=1)


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


def _build_ds(tr, replay, over, mode="pattern"):
    """密度反權重訓練集（反馬太②,2026-07-15）:權重 ∝ 1/局部密度,
    複製式實作（integer 重複,不動核心 train_by_datas）。train 與影子 CNN 共用**同一鍋**
    ——對決公平性的前提。回 (ds, reps)。
    mode="response"（R37 A/B,Ricky 2026-07-23「從 response 多樣性下手」）:密度改在 response
    特徵空間算（wm/lo/hi/oob 四維 z-score,歐氏 <1.0 為鄰）——合格聚落自動降權、稀有 response
    區（左側壓低族）自動加權=「SM 配比」自動版。"""
    if mode == "response":
        from script.dedust import oob_metrics
        feats = []
        for x, y in tr:
            yy = y.reshape(len(LABELS), -1)
            w_, _ = worst_margin(yy, LABELS, _cfg.targets)
            m = oob_metrics(yy.numpy())
            feats.append((float(w_), m.get("oob_gain_max_lo", 0.0),
                          m.get("oob_gain_max_hi", 0.0), m.get("oob_bad", 0.0)))
        F = np.asarray(feats, dtype=np.float32)
        F = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-9)
        nbrs = np.array([int((np.linalg.norm(F - F[i], axis=1) < 1.0).sum()) - 1
                         for i in range(len(F))])
    else:
        trpk = np.packbits(np.stack([(x.numpy().reshape(-1) > 0.5) for x, _ in tr]).astype(np.uint8), axis=1)
        nbrs = np.array([int((_POP8[np.bitwise_xor(trpk, trpk[i])].sum(axis=1) < 30).sum()) - 1
                         for i in range(len(tr))])
    w = 1.0 / (1.0 + nbrs)
    reps = np.clip(np.round(over * w / w.mean()), 1, over * 3).astype(int)
    dense_parts = []
    for i, r_ in enumerate(reps):
        dense_parts += [tr[i]] * int(r_)
    parts = [_tds(dense_parts)] + ([_tds(replay)] if replay else [])
    return ConcatDataset(parts), reps


def train(args):
    global CLEAN_STORES
    if getattr(args, "add", None):                       # 重錨一鍵化:append clean_stores.txt 再訓
        import time as _t
        new = [s.strip() for s in args.add.split(",") if s.strip() and s.strip() not in CLEAN_STORES]
        if new:
            with open(_CS_PATH, "a", encoding="utf-8") as f:
                f.write(f"# {args.out} 追加（{_t.strftime('%Y-%m-%d')}）\n")
                for n in new:
                    f.write(n + "\n")
            CLEAN_STORES = CLEAN_STORES + tuple(new)
            print(f"clean_stores.txt +{len(new)}: {','.join(new)}")
    tr, ho = _load_clean()
    replay, _ = _load_harvest(args.replay, args.val)
    print(f"乾淨真值 {len(tr) + len(ho)} 筆（train {len(tr)} / held-out {len(ho)}）＋ harvest 重放 {len(replay)}")
    sm = _make_sm()
    sm.pre_load_model(DATASET_PATH.joinpath("sm_harvest.pth"), strict=True)
    ds, reps = _build_ds(tr, replay, args.over, mode=getattr(args, "ds_mode", "pattern"))
    print(f"訓練集 {len(ds)} 筆（密度反權重:重複 中位×{int(np.median(reps))} 範圍"
          f" [{int(reps.min())},{int(reps.max())}],孤樣本重學/王朝密集降權 + 重放）,"
          f"epochs={args.epochs}, batch={args.batch}")
    losses = sm.train_by_datas(ds, epochs=args.epochs, batch_size=args.batch, verbose=True)  # MLP 留 CPU(GPU 0.8x 實測)
    out = DATASET_PATH.joinpath(args.out)
    sm.save_as(out)
    print(f"loss: 首 {losses[0]:.3f} → 末 {losses[-1]:.3f}；權重 → {out}")
    #? KPI 主軸①:SM 準度曲線（decisions 2026-07-15 戰略換軸）——每版重錨自動評 held-out,
    #  append docs/kpi.csv。反馬太③分層:held-out 從自家偏分布切=溫度計歪——按 d_dyn 分
    #  近(<20)/中(20-60)/遠(>60)三帶各報,「遠域誤差沒降=馬太在贏」直接可讀。
    import time as _t2
    errs = _wm_errs(sm, ho)
    med, p90 = float(np.median(errs)), float(np.percentile(errs, 90))
    dpk = _dyn_pack()
    hd = np.array([int(_POP8[np.bitwise_xor(dpk, np.packbits((x.numpy().reshape(-1) > 0.5)
                   .astype(np.uint8)))].sum(axis=1).min()) for x, _ in ho])

    def _band(mask):
        return float(np.median(errs[mask])) if mask.any() else float("nan")
    e_near, e_mid, e_far = _band(hd < 20), _band((hd >= 20) & (hd <= 60)), _band(hd > 60)
    #? 凍結基準集（2026-07-16 Ricky 拍板①:held-out 隨批擴張=尺在漂,v38 的 1.30 可能只是考卷變難
    #  ——固定基準跨版可比;首跑自動凍結當下 held-out）
    import hashlib as _hl
    import json as _js
    fz_path = os.path.join(REPO, "configs", "heldout_frozen.json")
    keys_ho = [_hl.md5((x.numpy().reshape(-1) > 0.5).tobytes()).hexdigest() for x, _ in ho]
    if not os.path.exists(fz_path):
        with open(fz_path, "w", encoding="utf-8") as f:
            _js.dump(dict(frozen_at=args.out, n=len(keys_ho), keys=keys_ho), f)
        print(f"凍結基準集建立: {len(keys_ho)} 筆（{args.out} 時刻）→ configs/heldout_frozen.json")
    fzk = set(_js.load(open(fz_path, encoding="utf-8"))["keys"])
    fmask = np.array([k in fzk for k in keys_ho])
    fz_med = _band(fmask)
    fz_far = _band(fmask & (hd > 60))
    #? 制度合訓（2026-07-16 Ricky 拍板③b:rad_head2 化石=九版沒吃新方向圖）:每版重錨配訓同期 rad 頭
    rad_rho, rad_mae = float("nan"), float("nan")
    if not getattr(args, "no_rad", False):
        vnum = "".join(ch for ch in os.path.basename(args.out) if ch.isdigit())
        rad_out = f"rad_head{vnum}.pth" if vnum else "rad_head_auto.pth"
        print(f"—— 制度合訓: rad 頭全量重訓 → {rad_out}")
        rad_rho, rad_mae = _train_rad_core(30, rad_out)
    #? ensemble 不確定性成員（2026-07-16 Ricky 拍板②）:2 顆異 seed 半程成員（+主 SM=3 成員）
    #  → select 打分記 pred_std;第一版不進選批鍵,判讀驗證 std 校準後再分流（變現/探索）。
    if not getattr(args, "no_ens", False):
        #? ens 換代（Ricky 2026-07-26 核准三升級①,v75 起）:成員 MLP→cnn2=與主通道同口徑
        #  （R40 換裝時的混口徑註記至此清償）;暖啟動=最新 sm_two*;無 two 檔退回舊 MLP 路。
        #  判準=std 校準單調性維持（analyze 判讀）。
        vnum2 = "".join(ch for ch in os.path.basename(args.out) if ch.isdigit())
        import glob as _g2
        _twos = sorted(_g2.glob(str(DATASET_PATH.joinpath("sm_two*.pth"))),
                       key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
        for j, sd_ in enumerate((17, 42), 1):
            torch.manual_seed(sd_)
            if _twos:
                cache_e = os.path.join(REPO, "tmp", "sm_reanchor")
                n_pts_e = sum(_cfg.targets[LABELS[0]]["width"])
                sm_e = SURROGATES["cnn2"](cache_e, 25 * 25, (len(LABELS), n_pts_e))
                sm_e.pre_load_model(_twos[-1], strict=True)
                arch_e = f"cnn2←{os.path.basename(_twos[-1])}"
            else:
                sm_e = _make_sm()
                sm_e.pre_load_model(DATASET_PATH.joinpath("sm_harvest.pth"), strict=True)
                arch_e = "mlp←sm_harvest"
            with _train_device(sm_e):
                sm_e.train_by_datas(ds, epochs=max(args.epochs // 2, 10), batch_size=args.batch,
                                verbose=False)
            eo = f"sm_ens{vnum2}_{j}.pth" if vnum2 else f"sm_ens_auto_{j}.pth"
            sm_e.save_as(DATASET_PATH.joinpath(eo))
            e_med = float(np.median(_wm_errs(sm_e, ho)))
            print(f"ensemble 成員 {j}（{arch_e},seed {sd_},{max(args.epochs // 2, 10)}ep）→ {eo}"
                  f"（held-out 中位 {e_med:.3f}）")
    kp = os.path.join(REPO, "docs", "kpi.csv")
    hdr = "date,sm,heldout_n,wm_err_med,wm_err_p90,err_near,err_mid,err_far,frozen_med,frozen_far,rad_rho,rad_mae\n"
    if os.path.exists(kp):                                # 舊 8 欄檔升級（一次性,舊行補空欄）
        lines = open(kp, encoding="utf-8").read().splitlines()
        if lines and lines[0].count(",") == 7:
            with open(kp, "w", encoding="utf-8") as f:
                f.write(hdr)
                for ln in lines[1:]:
                    f.write(ln + ",,,\n")
    newfile = not os.path.exists(kp)
    with open(kp, "a", encoding="utf-8") as f:
        if newfile:
            f.write(hdr)
        f.write(f"{_t2.strftime('%Y-%m-%d %H:%M')},{args.out},{len(errs)},{med:.3f},{p90:.3f},"
                f"{e_near:.3f},{e_mid:.3f},{e_far:.3f},{fz_med:.3f},{fz_far:.3f},"
                f"{rad_rho:.3f},{rad_mae:.3f}\n")
    print(f"held-out 準度: |wm err| 中位 {med:.3f} / P90 {p90:.3f}（n={len(errs)}）"
          f" | 分層 近{e_near:.2f}/中{e_mid:.2f}/遠{e_far:.2f}"
          f" | 凍結基準 {fz_med:.2f}/遠 {fz_far:.2f}（n={int(fmask.sum())}）"
          f" | rad ρ{rad_rho:+.2f} → docs/kpi.csv")
    #? 影子 CNN 對決（R32 判準;2026-07-17）:每版重錨平行訓一顆 CNN 挑戰者,吃**同一鍋** ds。
    #  尺1（凍結基準分層）即訓即評;尺2 前瞻/尺3 adv 率=下批 analyze batch 雙模盲測。
    #  連兩批三尺全贏→轉正（zoo "cnn" 成預設+全鏈換錨）;輸→降級異架構 ensemble 成員。
    if not getattr(args, "no_shadow", False):
        vnum3 = "".join(ch for ch in os.path.basename(args.out) if ch.isdigit())
        ep_sh = args.shadow_epochs or args.epochs * 2
        _train_shadow_core(ds, ho, hd, fmask, vnum3, ep_sh, args.batch,
                           mlp_ref=(med, e_near, e_far, fz_med, fz_far))


def _train_shadow_core(ds, ho, hd, fmask, vnum, epochs, batch, mlp_ref=None):
    """影子 CNN 訓練＋尺1 評估（train 制度段與 train-shadow 補訓共用）。
    ⚠ 誠實註記:MLP 自 sm_harvest（學長池 24k 預訓）熱啟,CNN 從零——epochs 預設 2× 補償,
    不對稱記錄在案（挑戰者要贏的就是「含預訓優勢的現役」）。"""
    from antenna.zoo import SURROGATES
    import time as _t
    cache = os.path.join(REPO, "tmp", "sm_shadow")
    os.makedirs(cache, exist_ok=True)
    n_pts = sum(_cfg.targets[LABELS[0]]["width"])
    torch.manual_seed(7)
    smc = SURROGATES["cnn"](cache, 25 * 25, (len(LABELS), n_pts))
    print(f"—— 影子 CNN 對決: 從零訓 {epochs}ep（同鍋 ds;尺2/尺3=下批 analyze batch 雙模盲測）")
    with _train_device(smc):
        losses = smc.train_by_datas(ds, epochs=epochs, batch_size=batch, verbose=False)
    out = f"sm_shadow{vnum}.pth" if vnum else "sm_shadow_auto.pth"
    smc.save_as(DATASET_PATH.joinpath(out))
    errs = _wm_errs(smc, ho)

    def _b(m):
        return float(np.median(errs[m])) if m.any() else float("nan")
    med, p90 = float(np.median(errs)), float(np.percentile(errs, 90))
    e_near, e_far = _b(hd < 20), _b(hd > 60)
    fz_med, fz_far = _b(fmask), _b(fmask & (hd > 60))
    print(f"影子 loss 首 {losses[0]:.3f} → 末 {losses[-1]:.3f} → {out}")
    print(f"影子尺1: |wm err| 中位 {med:.3f}/P90 {p90:.3f} | 近{e_near:.2f}/遠{e_far:.2f}"
          f" | 凍結 {fz_med:.2f}/遠 {fz_far:.2f}")
    if mlp_ref:
        m_med, m_near, m_far, m_fz, m_fzf = mlp_ref
        win = fz_med < m_fz and fz_far < m_fzf
        print(f"  vs MLP（中位 {m_med:.2f}/凍結 {m_fz:.2f}/凍結遠 {m_fzf:.2f}）→ 尺1 "
              + ("**CNN 贏**" if win else "MLP 守住") + "（判定記 round 檔;連兩批三尺全贏才轉正）")
    kp = os.path.join(REPO, "docs", "kpi_shadow.csv")
    newf = not os.path.exists(kp)
    with open(kp, "a", encoding="utf-8") as f:
        if newf:
            f.write("date,shadow,heldout_n,med,p90,near,far,frozen_med,frozen_far\n")
        f.write(f"{_t.strftime('%Y-%m-%d %H:%M')},{out},{len(errs)},{med:.3f},{p90:.3f},"
                f"{e_near:.3f},{e_far:.3f},{fz_med:.3f},{fz_far:.3f}\n")
    return out


def train_shadow_cmd(args):
    """獨立補訓影子 CNN（給既有版本補影子,如 v42;制度段=每版 train 自動帶）。
    資料管線與 train 完全同鍋（_load_clean+_load_harvest+密度反權重）。"""
    import hashlib as _hl
    import json as _js
    tr, ho = _load_clean()
    replay, _ = _load_harvest(args.replay, args.val)
    ds, _ = _build_ds(tr, replay, args.over)
    print(f"乾淨真值 {len(tr) + len(ho)} 筆（train {len(tr)} / held-out {len(ho)}）＋ 重放 {len(replay)}")
    dpk = _dyn_pack()
    hd = np.array([int(_POP8[np.bitwise_xor(dpk, np.packbits((x.numpy().reshape(-1) > 0.5)
                   .astype(np.uint8)))].sum(axis=1).min()) for x, _ in ho])
    fz_path = os.path.join(REPO, "configs", "heldout_frozen.json")
    fzk = set(_js.load(open(fz_path, encoding="utf-8"))["keys"]) if os.path.exists(fz_path) else set()
    fmask = np.array([_hl.md5((x.numpy().reshape(-1) > 0.5).tobytes()).hexdigest() in fzk
                      for x, _ in ho])
    vnum = "".join(ch for ch in os.path.basename(args.out) if ch.isdigit())
    _train_shadow_core(ds, ho, hd, fmask, vnum, args.epochs, args.batch)


def _augmirror(items):
    """鏡射增強（analysis-06:響應=頻域曲線,左右鏡射不變;訓練資料 ×2）。"""
    out = list(items)
    for x, y in items:
        xm = torch.flip(x.reshape(25, 25), dims=[1]).reshape(x.shape)
        out.append((xm, y))
    return out


def _train_two_core(tr, replay, over, ho, fmask, vnum, epochs, batch):
    """影子二號＋lo 判別器（R38 制度段;analysis-06 臂A/臂B 投產版）。
    二號=cnn2（ResBlock+BN）鏡射增強同鍋;lo 頭=小 CNN 直接回歸 (wm,lo) 標量。
    尺1 落 docs/kpi_two.csv;盲測（尺2/尺3）=analyze batch 三模段。"""
    from antenna.zoo import SURROGATES
    from script.dedust import oob_metrics as _om
    import time as _t2
    cache = os.path.join(REPO, "tmp", "sm_two")
    os.makedirs(cache, exist_ok=True)
    n_pts = sum(_cfg.targets[LABELS[0]]["width"])
    tr_aug = _augmirror(tr) + replay
    #? 配方=bake-off 實證版（原始池+鏡射,不疊密度過採樣——首跑教訓:8×reps 疊 2×aug=16× 有效
    #  訓練量,CPU 3hr 未收;analysis-06 贏的就是 flat pot+40ep）
    ds_aug = _tds(tr_aug)
    torch.manual_seed(7)
    sm2 = SURROGATES["cnn2"](cache, 25 * 25, (len(LABELS), n_pts))
    print(f"—— 影子二號: cnn2 從零訓 {epochs}ep（鏡射增強 ×2,同鍋）")
    with _train_device(sm2):
        losses = sm2.train_by_datas(ds_aug, epochs=epochs, batch_size=batch, verbose=False)
    out2 = f"sm_two{vnum}.pth" if vnum else "sm_two_auto.pth"
    sm2.save_as(DATASET_PATH.joinpath(out2))
    errs = _wm_errs(sm2, ho)
    med, fz = float(np.median(errs)), float(np.median(errs[fmask])) if fmask.any() else float("nan")
    print(f"二號 loss 首 {losses[0]:.3f} → 末 {losses[-1]:.3f} → {out2}")
    print(f"二號尺1: |wm err| 中位 {med:.3f} | 凍結 {fz:.3f}（盲測=下批 analyze 三模段）")
    #? lo 判別器（臂B）:標量 (wm,lo) 回歸——lo 軸=左側戰役導航;select 記 pred_lo（R39 判進鍵）
    import torch.nn as _nn
    scal = _nn.Sequential(_nn.Conv2d(1, 32, 3, padding=1), _nn.ReLU(), _nn.MaxPool2d(2),
                          _nn.Conv2d(32, 64, 3, padding=1), _nn.ReLU(), _nn.MaxPool2d(2),
                          _nn.Flatten(), _nn.Linear(64 * 6 * 6, 256), _nn.ReLU(), _nn.Linear(256, 2))
    X, S = [], []
    for x, y in tr_aug:
        yy = y.reshape(len(LABELS), -1)
        w_, _ = worst_margin(yy, LABELS, _cfg.targets)
        X.append(x.reshape(1, 25, 25))
        S.append((float(w_), float(_om(yy.numpy()).get("oob_gain_max_lo", 0.0))))
    X = torch.stack(X); S = torch.tensor(S)
    opt = torch.optim.Adam(scal.parameters(), lr=1e-3)
    idx = np.arange(len(X))
    for ep in range(max(epochs // 2, 30)):
        np.random.shuffle(idx)
        for i in range(0, len(idx), batch):
            ii = idx[i:i + batch]
            opt.zero_grad()
            loss = _nn.functional.mse_loss(scal(X[ii]), S[ii])
            loss.backward(); opt.step()
    scal.eval()
    outl = f"sm_lohead{vnum}.pth" if vnum else "sm_lohead_auto.pth"
    torch.save(scal.state_dict(), str(DATASET_PATH.joinpath(outl)))
    with torch.no_grad():
        pl = torch.cat([scal(torch.stack([x.reshape(1, 25, 25) for x, _ in ho[i:i + 256]]))
                        for i in range(0, len(ho), 256)])[:, 1].numpy()
    real_lo = np.array([float(_om(y.reshape(len(LABELS), -1).numpy()).get("oob_gain_max_lo", 0.0))
                        for _, y in ho])
    from scipy.stats import spearmanr as _sp
    rho_lo, _ = _sp(pl, real_lo)
    print(f"lo 判別器: ho lo ρ {rho_lo:+.3f} / |err|med {np.median(np.abs(pl - real_lo)):.3f} → {outl}")
    kp = os.path.join(REPO, "docs", "kpi_two.csv")
    newf = not os.path.exists(kp)
    with open(kp, "a", encoding="utf-8") as f:
        if newf:
            f.write("date,sm,ho_n,two_med,two_frozen,lohead_rho,lohead_mae\n")
        f.write(f"{_t2.strftime('%Y-%m-%d %H:%M')},{out2},{len(ho)},{med:.3f},{fz:.3f},"
                f"{rho_lo:.3f},{float(np.median(np.abs(pl - real_lo))):.3f}\n")


def train_two_cmd(args):
    """獨立補訓影子二號＋lo 判別器（同鍋;R38 起 train 制度段自動帶=TODO,先手動）。"""
    import hashlib as _hl
    import json as _js
    tr, ho = _load_clean()
    replay, _ = _load_harvest(args.replay, args.val)
    fz_path = os.path.join(REPO, "configs", "heldout_frozen.json")
    fzk = set(_js.load(open(fz_path, encoding="utf-8"))["keys"]) if os.path.exists(fz_path) else set()
    fmask = np.array([_hl.md5((x.numpy().reshape(-1) > 0.5).tobytes()).hexdigest() in fzk
                      for x, _ in ho])
    vnum = "".join(ch for ch in os.path.basename(args.out) if ch.isdigit())
    print(f"乾淨真值 train {len(tr)} / held-out {len(ho)} ＋ 重放 {len(replay)}（鏡射增強前）")
    _train_two_core(tr, replay, args.over, ho, fmask, vnum, args.epochs, args.batch)


def _load_denovo():
    """跨批萃取 kind=denovo 樣本（R24 §5 缺口落地,2026-07-13）：掃全部 *_input manifest,
    kind==denovo → input pattern 以 bool bytes 在同名 store 的 response 配對（sm_reanchor 只讀
    整店讀不到「散在批次店的 D 樣本」,這條路補上）。去重＋bytes 排序決定性切分（每 5 筆 held-out）。"""
    import json
    seen = {}
    for fol in sorted(os.listdir(str(DATASET_PATH))):
        if not fol.endswith("_input"):
            continue
        stname = fol[:-len("_input")]
        mp = DATASET_PATH.joinpath(fol, "manifest.json")
        if not mp.exists() or not DATASET_PATH.joinpath(stname).is_dir():
            continue
        ids = [m["id"] for m in json.load(open(str(mp), encoding="utf-8"))
               if m.get("kind") == "denovo"]
        if not ids:
            continue
        store = SampleStore(DATASET_PATH.joinpath(stname), verbose=False)
        smap = {}
        for i in range(len(store)):
            x, y = store[i]
            smap[(np.asarray(x).reshape(-1) > 0.5).tobytes()] = (
                torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
        for pid in ids:
            f = DATASET_PATH.joinpath(fol, pid + ".pt")
            if not f.exists():
                continue
            key = (np.asarray(torch.load(str(f), weights_only=True)).reshape(-1) > 0.5).tobytes()
            if key in smap and key not in seen:
                seen[key] = smap[key]
    items = [seen[k] for k in sorted(seen)]
    tr = [it for j, it in enumerate(items) if j % 5 != 0]
    ho = [it for j, it in enumerate(items) if j % 5 == 0]
    return tr, ho


def train_denovo(args):
    """sm_denovo：全史 D 臂樣本＋harvest 重放,自 sm_harvest 起訓;訓完就地對決
    （判準=R24 §1 寫死:D held-out 上贏 sm_harvest → D 臂換導引復航,輸了 D 續停）。"""
    from scipy.stats import spearmanr
    from script.dedust import oob_metrics
    tr, ho = _load_denovo()
    replay, _ = _load_harvest(args.replay, args.val)
    print(f"denovo 真值 {len(tr) + len(ho)} 筆（train {len(tr)} / held-out {len(ho)}）＋ harvest 重放 {len(replay)}")
    torch.manual_seed(0)
    sm = _make_sm()
    sm.pre_load_model(DATASET_PATH.joinpath("sm_harvest.pth"), strict=True)
    ds = ConcatDataset([_tds(tr)] * args.over + [_tds(replay)])
    losses = sm.train_by_datas(ds, epochs=args.epochs, batch_size=args.batch, verbose=True)  # MLP 留 CPU(GPU 0.8x 實測)
    out = DATASET_PATH.joinpath(args.out)
    sm.save_as(out)
    print(f"loss: 首 {losses[0]:.3f} → 末 {losses[-1]:.3f}；權重 → {out}")
    base = _make_sm()
    base.pre_load_model(DATASET_PATH.joinpath("sm_harvest.pth"), strict=True)
    print(f"\n== 對決（D held-out {len(ho)} 筆;⚠ n 小,ρ 波動大——方向參考,別當精確值）==")
    print("| 模型 | |wm err| 中位 | wm 排序 ρ | oob 排序 ρ |")
    print("|---|---|---|---|")
    for tag, mdl in (("sm_harvest(現行導引)", base), (args.out, sm)):
        pw, tw, po, to = [], [], [], []
        mdl.model.eval()
        with torch.no_grad():
            for x, y in ho:
                pred = mdl.model(x.flatten())
                w_p, _ = worst_margin(pred, LABELS, _cfg.targets)
                w_t, _ = worst_margin(y, LABELS, _cfg.targets)
                pw.append(float(w_p)); tw.append(float(w_t))
                po.append(oob_metrics(pred.numpy())["oob_bad"])
                to.append(oob_metrics(np.asarray(y))["oob_bad"])
        errs = np.abs(np.asarray(pw) - np.asarray(tw))
        print(f"| {tag} | {np.median(errs):.2f} | {spearmanr(pw, tw)[0]:+.3f} | {spearmanr(po, to)[0]:+.3f} |")
    print("→ 兩欄排序 ρ 都贏＝換導引（select 的 --denovo-sm 換本檔）;輸/平＝D 續停,回報使用者。")


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
        sm.train_by_datas(ds, epochs=epochs, batch_size=batch, verbose=False)  # MLP 留 CPU
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


def train_radhead2_cmd(args):
    """標量 rad 判別器（健檢反思④,2026-07-24）:pattern→rad_margin 直接回歸——
    rad=全系統綁定瓶頸∧曲線版 rad 頭 ρ 0.09↔0.46 震盪最弱儀器∧rad_margin 本來就是標量規格。
    同 held-out 切分（md5%7）;判準=M 臂前瞻對決舊 rad 頭,輸者退役。"""
    import hashlib
    import torch.nn as nn
    from scipy.stats import spearmanr
    X, C, M, keys, th = _rad_dataset()
    print(f"rad 資料 {len(X)} 筆;標量目標 rad_margin 中位 {np.median(M):+.2f}")
    side = np.array([int(hashlib.md5(k).hexdigest(), 16) % 7 == 0 for k in keys])
    Xt = torch.tensor(X[~side]).reshape(-1, 1, 25, 25)
    Mt = torch.tensor(M[~side], dtype=torch.float32).reshape(-1, 1)
    Xh = torch.tensor(X[side]).reshape(-1, 1, 25, 25)
    Mh = M[side]
    #? 鏡射增強:rad_margin=雙切面 min,左右鏡射下 phi0/phi90 各自鏡像,margin 不變 → ×2
    Xt = torch.cat([Xt, torch.flip(Xt, dims=[3])]); Mt = torch.cat([Mt, Mt])
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Flatten(), nn.Linear(64 * 6 * 6, 256), nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(Xt)
    for ep in range(args.epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            loss = nn.functional.mse_loss(net(Xt[idx]), Mt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        if ep % 10 == 9:
            print(f"  ep {ep + 1}/{args.epochs} loss {tot / n:.4f}", flush=True)
    net.eval()
    with torch.no_grad():
        pm = torch.cat([net(Xh[i:i + 256]) for i in range(0, len(Xh), 256)])[:, 0].numpy()
    rho, pv = spearmanr(pm, Mh)
    mae = float(np.abs(pm - Mh).mean())
    print(f"held-out {int(side.sum())} 筆: ρ={rho:+.3f} (p={pv:.1e}) / MAE {mae:.3f} dB"
          f"（舊 rad 頭震盪帶 0.09~0.46——首讀對照）")
    torch.save(net.state_dict(), str(DATASET_PATH.joinpath(args.out)))
    print(f"→ {args.out}（select 記 pred_rad2;判準=M 臂前瞻對決舊頭,輸者退役）")


def _train_rad_core(epochs, out):
    """rad 頭訓練核心（train_rad 與 train 制度合訓共用;2026-07-16 Ricky 拍板 b 案——
    rad_head2 化石問題:訓後九版沒吃過新方向圖=前瞻蹺蹺板頭號嫌疑）。回 (ρ, MAE)。"""
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
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 64):
            idx = perm[i:i + 64]
            fit = net(Xt[idx]).reshape(-1, 2, RAD_K) @ B
            loss = (((fit - Ct[idx]) ** 2) * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        if ep == 0 or ep % 5 == 4:
            print(f"  ep {ep + 1}/{epochs} loss {tot / n:.4f}", flush=True)
    with torch.no_grad():
        fit = (net(Xh).reshape(-1, 2, RAD_K) @ B).numpy()
    pm = np.array([min(rwm(th, fit[i][0]), rwm(th, fit[i][1])) for i in range(len(fit))])
    rho, p = spearmanr(pm, Mh)
    mae = float(np.abs(pm - Mh).mean())
    print(f"held-out {int(side.sum())} 筆:rad_margin 排序 ρ={rho:+.3f} (p={p:.1e}) / MAE {mae:.3f} dB")
    torch.save(dict(state=net.state_dict(), K=RAD_K, theta=np.asarray(th)),
               str(DATASET_PATH.joinpath(out)))
    print(f"→ {out}（判準:ρ≥0.4 才進 pred_sel 選批鍵）")
    return float(rho), mae


def train_rad(args):
    """rad 頭:625 → 2×K cosine 係數（|θ|≤90 半圓）→ 雙切面曲線;±45° 窗 2× 加權。
    K=16 依 rad-repr 表達力分析（Ricky 2026-07-12「感覺可以補 K=16」);資料=R7 起順收的全史方向圖。
    **判準（發車前寫死）**:held-out（pattern-hash 切分,防公證重複洩漏）rad_margin 排序 ρ≥0.4
    → pred_rad 進 pred_sel 罰項;否則只隨 manifest 記 pred_rad 供前瞻,不進選批鍵。"""
    _train_rad_core(args.epochs, args.out)


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
        s.add_argument("--add", default=None, help='逗號分隔新 store,先 append configs/clean_stores.txt 再訓（重錨一鍵化）')
        s.add_argument("--no-rad", action="store_true", dest="no_rad",
                       help="跳過制度合訓 rad 頭（預設每版重錨配訓同期 rad_headNN.pth）")
        s.add_argument("--no-ens", action="store_true", dest="no_ens",
                       help="跳過 ensemble 成員訓練（預設 2 顆異 seed 半程成員 → pred_std）")
        s.add_argument("--no-shadow", action="store_true", dest="no_shadow",
                       help="跳過影子 CNN 對決訓練（預設每版重錨平行訓 sm_shadowNN.pth）")
        s.add_argument("--shadow-epochs", type=int, default=None, dest="shadow_epochs",
                       help="影子 CNN epochs（預設 --epochs×2:從零訓補償,MLP 有 harvest 預訓）")
        s.add_argument("--ds-mode", default="pattern", choices=["pattern", "response"], dest="ds_mode",
                       help="密度反權重空間（R37 A/B:response=四維特徵鄰居,Ricky 2026-07-23）")
        s.add_argument("--grid-epochs", type=int, nargs="+", default=[40, 80])
        s.add_argument("--grid-over", type=int, nargs="+", default=[4, 8, 16])
        s.add_argument("--grid-replay", type=int, nargs="+", default=[1000, 2000])
        s.add_argument("--grid-batch", type=int, nargs="+", default=[64])
        s.set_defaults(fn=fn)
    s = sub.add_parser("train-denovo", help="sm_denovo:全史 kind=denovo 萃取+harvest 重放起訓,訓完就地對決 sm_harvest(D 臂復航判準)")
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--over", type=int, default=24, help="denovo 過採樣倍數(~84 筆×24≈與重放等量)")
    s.add_argument("--replay", type=int, default=2000)
    s.add_argument("--val", type=int, default=500)
    s.add_argument("--out", default="sm_denovo1.pth")
    s.set_defaults(fn=train_denovo)
    s = sub.add_parser("train-rad", help="rad 頭:pattern→K=16 cosine 雙切面(±45 窗加權);held-out ρ≥0.4 才進 pred_sel")
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--out", default="rad_head1.pth")
    s.set_defaults(fn=train_rad)
    s = sub.add_parser("train-two", help="獨立補訓影子二號+lo 判別器（R38;cnn2 鏡射增強+標量 lo 頭;bake-off 實證配方=flat pot+40ep）")
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--over", type=int, default=8)
    s.add_argument("--replay", type=int, default=2000)
    s.add_argument("--val", type=int, default=500)
    s.add_argument("--out", default="sm_reanchor60.pth", help="版本號來源（sm_twoNN/sm_loheadNN 取此檔數字）")
    s.set_defaults(fn=train_two_cmd)
    s = sub.add_parser("train-radhead2", help="標量 rad 判別器（健檢④;pattern→rad_margin 直回歸+鏡射;對決舊 rad 頭）")
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--out", default="sm_radhead2_60.pth")
    s.set_defaults(fn=train_radhead2_cmd)
    s = sub.add_parser("train-shadow", help="獨立補訓影子 CNN（給既有版補影子;制度段=train 自動帶;同鍋資料）")
    s.add_argument("--epochs", type=int, default=80)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--over", type=int, default=8)
    s.add_argument("--replay", type=int, default=2000)
    s.add_argument("--val", type=int, default=500)
    s.add_argument("--out", default="sm_shadow_auto.pth", help="用 sm_shadow<NN>.pth 對齊主 SM 版號")
    s.set_defaults(fn=train_shadow_cmd)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
