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


def _load_clean_stores():
    if not os.path.exists(_CS_PATH):
        raise SystemExit(f"{_CS_PATH} 不存在——CLEAN_STORES 已檔案化(2026-07-12),請自 git 還原該檔")
    out = []
    with open(_CS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line and line not in out:
                out.append(line)
    #? 自產 tier-2（selfgen dedust_auto*）永遠是乾淨 HFSS 真值 → 自動納入,免手維護
    #  （2026-07-13 修:之前 95 筆自產資料含 8 三標卻沒餵 SM=浪費,Ricky 指出）。
    import glob
    for p in sorted(glob.glob(str(DATASET_PATH.joinpath("dedust_auto*")))):
        name = os.path.basename(p)
        if os.path.isdir(p) and DATASET_PATH.joinpath(name, "results.json").exists() and name not in out:
            out.append(name)
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
        if _h.md5(k).hexdigest() in fzk:
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
    #? 反馬太②密度反權重（2026-07-15 Ricky 核准四機制全套）:訓練集分布=歷史選樣累積=偏——
    #  王朝密集區樣本被重複學習。權重 ∝ 1/局部密度（Hamming<30 鄰居數）,複製式實作
    #  （integer 重複,不動核心 train_by_datas）;均勻 ×over 過採樣退役。
    trpk = np.packbits(np.stack([(x.numpy().reshape(-1) > 0.5) for x, _ in tr]).astype(np.uint8), axis=1)
    nbrs = np.array([int((_POP8[np.bitwise_xor(trpk, trpk[i])].sum(axis=1) < 30).sum()) - 1
                     for i in range(len(tr))])
    w = 1.0 / (1.0 + nbrs)
    reps = np.clip(np.round(args.over * w / w.mean()), 1, args.over * 3).astype(int)
    dense_parts = []
    for i, r_ in enumerate(reps):
        dense_parts += [tr[i]] * int(r_)
    ds = ConcatDataset([_tds(dense_parts), _tds(replay)])
    print(f"訓練集 {len(ds)} 筆（密度反權重:重複 中位×{int(np.median(reps))} 範圍"
          f" [{int(reps.min())},{int(reps.max())}],孤樣本重學/王朝密集降權 + 重放）,"
          f"epochs={args.epochs}, batch={args.batch}")
    losses = sm.train_by_datas(ds, epochs=args.epochs, batch_size=args.batch, verbose=True)
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
        vnum2 = "".join(ch for ch in os.path.basename(args.out) if ch.isdigit())
        for j, sd_ in enumerate((17, 42), 1):
            torch.manual_seed(sd_)
            sm_e = _make_sm()
            sm_e.pre_load_model(DATASET_PATH.joinpath("sm_harvest.pth"), strict=True)
            sm_e.train_by_datas(ds, epochs=max(args.epochs // 2, 10), batch_size=args.batch,
                                verbose=False)
            eo = f"sm_ens{vnum2}_{j}.pth" if vnum2 else f"sm_ens_auto_{j}.pth"
            sm_e.save_as(DATASET_PATH.joinpath(eo))
            e_med = float(np.median(_wm_errs(sm_e, ho)))
            print(f"ensemble 成員 {j}（seed {sd_},{max(args.epochs // 2, 10)}ep）→ {eo}"
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
    losses = sm.train_by_datas(ds, epochs=args.epochs, batch_size=args.batch, verbose=True)
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
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
