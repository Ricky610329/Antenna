# -*- coding: utf-8 -*-
"""
script/sm_dual.py — dual-port SM 排序器（R58 施工包）。

用途只有一個：**批次線的排序器**——從幾千張候選 pattern 裡挑 top-N 送 HFSS，
省掉「每張都要 ~100 秒模擬」的錢。**不是**線上學習的 SM（不做單筆過擬合、不接 GEN 反傳）。

尺＝`antenna.losses.worst_margin_dual(response, ['S11','S21','S22'], targets)`，
targets 讀 `configs/dual_r1_eval.yaml`（與批次線判讀同一把尺，不另立標準）。

子命令地圖::

    train   建鍋（harvest_dual + dedust_r57* 全鍋去重）→ 分位分層 split（seed 寫死）
            → 訓 3 個 seed → 權重存 ROOTDIR/sm_dual_v1_s{0,1,2}.pth
            + split 索引存 ROOTDIR/sm_dual_v1_split.json
    eval    載權重，held-out 上印：逐通道 MAE / spearman ρ(wm_dual 口徑，單模型 vs ensemble)
            / top30 ∩ true top10% 命中數 + 超幾何 P → 「→ 品質閘: 過/不過」
    rank    --pool <npz 或資料夾> --top N → 以 3-seed ensemble 平均預測的 wm_dual 排序，
            輸出 id + pred_wm 表（--out 寫 CSV，否則 stdout）

品質閘（round-58 §1 寫死，eval 直接判）：
    (a) held-out 上 spearman ρ(pred wm_dual, true wm_dual) ≥ +0.40
    (b) pred top30 中落在 true top10% 的數量，超幾何檢定 P < 0.05（隨機期望 ≈ 3）

節流（開發機鐵則）：單例鎖（tmp/sm_dual.lock）＋ `torch.set_num_threads(4)` ＋ 進程降優先權；
GPU 有卡才用（CUDA 訓練非 bit 級決定性——決定性保證只到「鍋與 split」，不含權重）。
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch import nn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from antenna.losses import worst_margin_dual
from antenna.training import PORT_SPECS
from antenna.utils import DATASET_PATH, ROOTDIR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(REPO, "configs", "dual_r1_eval.yaml")
CACHE_PATH = os.path.join(REPO, "tmp", "sm_dual_pool.npz")
LOCK_PATH = os.path.join(REPO, "tmp", "sm_dual.lock")

LABELS = PORT_SPECS["dual"]["labels"]          # ['S11', 'S21', 'S22'] — dual response 列序
N_POINTS = 17                                  # 24-32GHz 17 點
N_PIX = 625                                    # 25×25
MARGINS = ("m1", "m2", "m3", "m4")             # wm_dual = min(m1..m4)（m5/m6 只記帳，不進 min）

#? 鍋的店清單與**順序＝去重優先權**（首見先贏，同 sm_reanchor 的「certified 先見先贏」慣例）：
#  公證重測店（n1/n2，同 pattern 重測）排最前 → 其重測值蓋過批次店的單測值；
#  smoke 與 harvest 重複、批次店與 n1/n2 重複，都靠這個順序解決。
#! v2 起（2026-08-11）鍋擴充到 R58/R59 與 autod 自產池；**dedust_r60*（kind=slotw 幾何變體）
#! 永不入鍋**——同 bits 不同幾何=毒資料（script/CLAUDE.md 鐵則 7）。autod 店動態發現。
_FIXED_STORES = (
    "dedust_r57n1", "dedust_r57n2", "dedust_r58n1", "dedust_r58n2",     # 公證重測（最高優先）
    "dedust_r57b1", "dedust_r57b2a", "dedust_r57b2b", "dedust_r57b2c",  # R57 批次量測
    "dedust_r57b3a", "dedust_r57b3b", "dedust_r57b3c",
    "dedust_r58b1a", "dedust_r58b1b", "dedust_r58b1c",                  # R58
    "dedust_r58b2a", "dedust_r58b2b", "dedust_r58b2c",
    "dedust_r58b3a", "dedust_r58b3b", "dedust_r58b3c",
    "dedust_r59b1a", "dedust_r59b1b", "dedust_r59b1c",                  # R59（通濾族＝新物種）
    "dedust_r59b2a", "dedust_r59b2b", "dedust_r59b2c",
    "dedust_r59b3a", "dedust_r59b3b", "dedust_r59b3c",
    "dedust_r59dx",                                                     # 對角探針（bits 忠實）
    "dedust_r57s216", "dedust_r57s37", "dedust_r57smoke",               # smoke（與 harvest 重複）
    "harvest_dual",                                                     # 學長池 ~10k（最低優先）
)


def _discover_stores():
    """固定清單 + 動態 autod 自產店（插在 harvest 之前=自量測優先於學長池）。"""
    import glob as _g
    autod = sorted(os.path.basename(p) for p in _g.glob(os.path.join(str(DATASET_PATH), "dedust_autod*"))
                   if not p.endswith("_input"))
    fixed = list(_FIXED_STORES)
    return tuple(fixed[:-1] + autod + fixed[-1:])


STORES = _discover_stores()

SPLIT_SEED = 58            #! held-out 抽樣 seed，寫死（改它＝換一份 held-out，等於換考卷）
HELDOUT_FRAC = 0.10
SPLIT_BINS = 10            # wm_dual 分位分層的箱數
GATE_RHO = 0.40            # 品質閘 (a)
GATE_TOPK = 30             # 品質閘 (b)：pred 取前 30
GATE_TRUE_FRAC = 0.10      # 品質閘 (b)：true 前 10%
GATE_P = 0.05

SM_VER = os.environ.get("SM_DUAL_VER", "v2")   #! v2=2026-08-11 鍋擴充(R58/R59+autod);v1 權重保留不覆蓋
WEIGHT_FMT = "sm_dual_" + SM_VER + "_s{}.pth"
SPLIT_JSON = "sm_dual_" + SM_VER + "_split.json"


# ────────────────────────────────────────────────────────────────────────────
# 節流：單例鎖 + 執行緒上限 + 降優先權（開發機鐵則，2026-08-06 卡機事件）
# ────────────────────────────────────────────────────────────────────────────
def throttle():
    torch.set_num_threads(4)
    try:
        import psutil
        psutil.Process(os.getpid()).nice(getattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS", 10))
    except Exception as e:      # psutil 沒裝/平台不支援 → 只是沒降權，不該擋工作
        print(f"[warn] 降優先權失敗（略過）：{e}")


class SingleInstance:
    """單例鎖：同時只准一個 sm_dual 在跑（鎖檔存 pid，死 pid 自動接管）。"""

    def __enter__(self):
        os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
        if os.path.exists(LOCK_PATH):
            try:
                old = int(open(LOCK_PATH, encoding="utf-8").read().strip() or 0)
            except ValueError:
                old = 0
            alive = False
            try:
                import psutil
                alive = psutil.pid_exists(old) and old != os.getpid()
            except Exception:
                alive = False
            if alive:
                raise SystemExit(f"已有 sm_dual 在跑（pid={old}）；單例鐵則 → 本次不啟動。"
                                 f"（確認沒在跑就刪 {LOCK_PATH}）")
        with open(LOCK_PATH, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        try:
            os.remove(LOCK_PATH)
        except OSError:
            pass
        return False


# ────────────────────────────────────────────────────────────────────────────
# 鍋：掃 store → 去重 → wm_dual/margins → 本機快取
# ────────────────────────────────────────────────────────────────────────────
def load_targets(cfg_path=CFG_PATH):
    """只讀 yaml 的 targets 段（不建 spec、不碰全域裝置狀態）。"""
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["targets"]


def pattern_key(x):
    """去重鍵：pattern bits 的 packbits bytes（float/bool/0-1 皆正規化到同一把尺）。"""
    bits = (np.asarray(x, dtype=np.float32).reshape(-1) > 0.5).astype(np.uint8)
    return np.packbits(bits).tobytes()


def margins_of(y, targets):
    """(3,17) 響應 → (wm, [m1,m2,m3,m4])。wm = min(m1..m4)（`worst_margin_dual` 定義）。"""
    wm, per = worst_margin_dual(np.asarray(y, dtype=np.float32).reshape(len(LABELS), -1), LABELS, targets)
    return float(wm), np.array([per[k] for k in MARGINS], dtype=np.float32)


def wm_of(y, targets):
    return margins_of(y, targets)[0]


def wm_array(Y, targets):
    return np.array([wm_of(y, targets) for y in Y], dtype=np.float64)


def _scan_stores(stores, workers=8):
    """依 STORES 順序掃 NAS，首見先贏去重。回 (X uint8 (n,625), Y float32 (n,3,17), src, n_raw)。"""
    from concurrent.futures import ThreadPoolExecutor
    seen, order, n_raw = {}, [], 0
    for name in stores:
        d = DATASET_PATH.joinpath(name)
        if not d.is_dir():
            print(f"[warn] store 不存在，略過：{name}")
            continue
        files = sorted(f for f in os.listdir(str(d)) if f.endswith(".pt"))
        t0 = time.time()
        #? NAS 逐檔 torch.load 是網路延遲綁定（單執行緒 ~100 筆/秒）→ 小 I/O 執行緒池；
        #  計算仍受 set_num_threads(4) 上限約束。
        with ThreadPoolExecutor(max_workers=workers) as ex:
            samples = list(ex.map(lambda f: torch.load(os.path.join(str(d), f), weights_only=True), files))
        kept = 0
        for x, y in samples:
            n_raw += 1
            k = pattern_key(x)
            if k in seen:
                continue
            seen[k] = (np.asarray(x, dtype=np.float32).reshape(-1) > 0.5).astype(np.uint8)
            order.append((k, np.asarray(y, dtype=np.float32).reshape(len(LABELS), N_POINTS), name))
            kept += 1
        print(f"  {name:<18} 讀 {len(files):>5} 筆，新增 {kept:>5}（{time.time() - t0:.1f}s）")
    #! 排序鍵＝pattern bytes → 鍋的順序與檔案系統列舉順序無關（split 決定性的前提）。
    order.sort(key=lambda t: t[0])
    X = np.stack([seen[k] for k, _, _ in order]).astype(np.uint8)
    Y = np.stack([y for _, y, _ in order]).astype(np.float32)
    return X, Y, [s for _, _, s in order], n_raw


def build_pool(refresh=False, cache=CACHE_PATH, stores=STORES):
    """建鍋（含本機快取）。回 dict(X, Y, wm, M, src, n_raw)；M=(n,4) 的 m1..m4。"""
    if os.path.exists(cache) and not refresh:
        z = np.load(cache, allow_pickle=False)
        if "M" in z:
            return {k: (z[k] if k != "src" else [str(s) for s in z["src"]])
                    for k in ("X", "Y", "wm", "M", "src")} | {"n_raw": int(z["n_raw"])}
        X, Y, src, n_raw = z["X"], z["Y"], [str(s) for s in z["src"]], int(z["n_raw"])
    else:
        print("建鍋（掃 NAS，只做一次；之後走本機快取，--refresh 可重建）：")
        X, Y, src, n_raw = _scan_stores(stores)
    targets = load_targets()
    t0 = time.time()
    wm, M = np.zeros(len(Y), np.float32), np.zeros((len(Y), 4), np.float32)
    for i, y in enumerate(Y):
        wm[i], M[i] = margins_of(y, targets)
    print(f"  wm_dual/margins 算完 {len(wm)} 筆（{time.time() - t0:.1f}s）")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, X=X, Y=Y, wm=wm, M=M, src=np.array(src), n_raw=np.array(n_raw))
    print(f"  快取 → {cache}")
    return {"X": X, "Y": Y, "wm": wm, "M": M, "src": src, "n_raw": n_raw}


def make_split(wm, seed=SPLIT_SEED, frac=HELDOUT_FRAC, n_bins=SPLIT_BINS):
    """wm_dual 分位分層抽 held-out。決定性：同一鍋 + 同 seed → 同一份索引。

    :return: (train_idx, heldout_idx)，皆為升序 np.int64 陣列。
    """
    wm = np.asarray(wm, dtype=np.float64)
    order = np.argsort(wm, kind="stable")          # stable → 同值時保持鍋序（鍋序本身決定性）
    rng = np.random.default_rng(seed)
    ho = []
    for b in np.array_split(order, n_bins):
        if len(b) == 0:
            continue
        k = max(1, int(round(len(b) * frac)))
        ho.extend(rng.choice(b, size=min(k, len(b)), replace=False).tolist())
    ho = np.array(sorted(set(ho)), dtype=np.int64)
    mask = np.ones(len(wm), dtype=bool)
    mask[ho] = False
    return np.flatnonzero(mask).astype(np.int64), ho


# ────────────────────────────────────────────────────────────────────────────
# 模型：MLP + BatchNorm，兩顆頭（響應 51 + margin 4）
# ────────────────────────────────────────────────────────────────────────────
class DualNet(nn.Module):
    """pattern (25×25 0/1) → dual 響應 (3,17) dB **＋** wm_dual 的四項 margin (m1..m4)。

    一個 MLP-BN 幹（預設 1024-1024-512）＋ 一顆 55 維輸出層，語意上兩顆頭：

    - **響應頭 (51)**：診斷用（逐通道 MAE、要看曲線時）。
    - **margin 頭 (4)**：**排序真正用的那個**。理由（R58 A/B 實測，同鍋同 inner val）：
      wm_dual = min over「band 上的 max/min」，先回歸響應再取極值會把逐點 ~3dB 的誤差
      放大成排序噪音（ρ≈0.39）；直接回歸四個 margin 純量 ρ≈0.51。兩者一起訓（多任務）
      比只訓 margin 略好，且免費附送響應曲線。

    架構刻意從簡（先求過閘）：A/B 過的對手裡，CNN（含 pool 與 no-pool 版）明顯輸給 MLP
    ——pattern→S 參數不是平移不變問題，pooling 反而丟掉「像素在哪」的資訊。
    標準化統計以 buffer 存進 state_dict → 載權重即帶正規化，不需外部 meta。
    """

    N_OUT = len(LABELS) * N_POINTS + len(MARGINS)      # 51 + 4

    def __init__(self, hidden=(1024, 1024, 512), dropout=0.1):
        super().__init__()
        layers, prev = [], N_PIX
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, self.N_OUT)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("t_mean", torch.zeros(self.N_OUT))
        self.register_buffer("t_std", torch.ones(self.N_OUT))

    def forward(self, x):
        """:param x: (B,625)/(B,25,25)/(B,1,25,25) → :return: (響應 (B,3,17) dB, margin (B,4) dB)。"""
        t = self.raw(x)
        return t[:, :-len(MARGINS)].reshape(-1, len(LABELS), N_POINTS), t[:, -len(MARGINS):]

    def raw(self, x):
        """55 維（51 響應 + 4 margin）反正規化後的原始輸出——訓練/存檔內部用。"""
        return self.net(x.reshape(x.shape[0], -1)) * self.t_std + self.t_mean


#? 上下翻轉時 55 維目標的重排：響應 [S11,S21,S22] → [S22,S21,S11]；margin m1(S11)↔m2(S22)，m3/m4 不動。
_UD_PERM = torch.as_tensor(list(range(34, 51)) + list(range(17, 34)) + list(range(0, 17)) + [52, 51, 53, 54])


def _augment(xb, tb, aug, device):
    """幾何對稱增強（免費資料，靠 dual 佈局的兩個對稱性）。

    - **lr（左右鏡射）**：兩個饋電點都在中央行（底邊中央 + 頂邊中央，見 `FeedReachability.dual_feed`），
      沿垂直中線鏡射把兩個埠都映到自己 → 三條 S 參數不變。
    - **ud（上下翻轉）**：上下翻把 port1 ↔ port2 互換 → S11 ↔ S22 對調、S21 因互易不變。

    #! 兩者都是「理應」——真 HFSS setup 若不對稱就會反過來害，所以走 A/B 決定，不當公理。
    #  R58 實測（inner val，normalized MSE 越低越好）：none 0.873 → lr 0.864 → lr+ud 0.773，
    #  ud 這一刀最有效 → 反過來也算是「setup 對 port 對稱」的實證。
    """
    if "lr" in aug:
        m = (torch.rand(len(xb), device=device) < 0.5).reshape(-1, 1, 1, 1)
        xb = torch.where(m, xb.flip(-1), xb)
    if "ud" in aug:
        m = (torch.rand(len(xb), device=device) < 0.5).reshape(-1, 1, 1, 1)
        xb = torch.where(m, xb.flip(-2), xb)
        tb = torch.where(m.reshape(-1, 1), tb[:, _UD_PERM.to(device)], tb)
    return xb, tb


def _targets_matrix(Y, M):
    """(n,3,17) 響應 + (n,4) margin → (n,55) 訓練目標。"""
    return np.concatenate([Y.reshape(len(Y), -1), M], axis=1).astype(np.float32)


def train_one(X, T, seed, *, epochs=300, bs=128, lr=2e-3, wd=1e-4, aug="lr+ud", w_margin=12.0,
              val_frac=0.05, device="cpu", model_fn=None, verbose=True):
    """訓一個 seed。內部再切 val（選最佳 checkpoint），**held-out 全程不碰**。

    :param T: (n,55) 目標（見 `_targets_matrix`）。
    :param w_margin: margin 那 4 維在 loss 裡的權重（4 維 vs 51 維，不加權會被響應淹掉）。
    :return: (model, best_val_loss)。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(round(len(X) * val_frac)))
    vi, ti = perm[:n_val], perm[n_val:]

    xt = torch.as_tensor(X, dtype=torch.float32)
    tt = torch.as_tensor(T, dtype=torch.float32)
    model = (model_fn or DualNet)().to(device)
    with torch.no_grad():                       # 正規化統計只用訓練子集（不含 val/held-out）
        model.t_mean.copy_(tt[ti].mean(dim=0).to(device))
        model.t_std.copy_(tt[ti].std(dim=0).clamp_min(1e-3).to(device))

    w = torch.ones(DualNet.N_OUT, device=device)
    w[-len(MARGINS):] = w_margin
    xtr, ttr = xt[ti].to(device), tt[ti].to(device)
    xva, tva = xt[vi].to(device), tt[vi].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state = float("inf"), None
    brng = np.random.default_rng(2000 + seed)

    for ep in range(epochs):
        model.train()
        idx = brng.permutation(len(ti))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb, tb = _augment(xtr[b].reshape(-1, 1, 25, 25), ttr[b], aug, device)
            #? loss 算在**標準化空間**（除以逐維 std）：raw dB 上 S21 的變異數是 S11/S22 的兩倍多
            #  （44 vs 20），直接對 raw 做 MSE 等於偷偷把權重壓在 S21 上；wm_dual 是 min over 四項、
            #  三個通道等重要，所以要等權。
            loss = ((((model.raw(xb) - tb) / model.t_std) ** 2) * w).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(((((model.raw(xva) - tva) / model.t_std) ** 2) * w).mean())
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"    seed{seed} ep{ep:>3} train={float(loss):.4f} val={vl:.4f} best={best:.4f}")
    model.load_state_dict(best_state)
    return model.to("cpu").eval(), best


# ────────────────────────────────────────────────────────────────────────────
# 預測 / 排序
# ────────────────────────────────────────────────────────────────────────────
def predict(models, X, bs=1024, device="cpu"):
    """回 (響應 (n_models,n,3,17), margin (n_models,n,4)) numpy。"""
    xt = torch.as_tensor(np.asarray(X), dtype=torch.float32).reshape(-1, N_PIX)
    R, G = [], []
    for m in models:
        m = m.to(device).eval()
        rr, gg = [], []
        with torch.no_grad():
            for i in range(0, len(xt), bs):
                r, g = m(xt[i:i + bs].to(device))
                rr.append(r.cpu().numpy())
                gg.append(g.cpu().numpy())
        R.append(np.concatenate(rr) if rr else np.zeros((0, len(LABELS), N_POINTS), np.float32))
        G.append(np.concatenate(gg) if gg else np.zeros((0, len(MARGINS)), np.float32))
    return np.stack(R), np.stack(G)


def rank_pool(models, X, ids, top=None, device="cpu"):
    """ensemble 平均 margin → wm=min(m1..m4)，由高到低排。回 list[(id, pred_wm, {m1..m4})]。"""
    G = predict(models, X, device=device)[1].mean(axis=0)
    rows = [(ids[i], float(g.min()), dict(zip(MARGINS, (float(v) for v in g)))) for i, g in enumerate(G)]
    rows.sort(key=lambda r: -r[1])
    return rows[:top] if top else rows


# ────────────────────────────────────────────────────────────────────────────
# 權重 I/O
# ────────────────────────────────────────────────────────────────────────────
def weight_path(seed, outdir=None):
    return os.path.join(str(outdir) if outdir else str(ROOTDIR), WEIGHT_FMT.format(seed))


def load_models(seeds=(0, 1, 2), outdir=None):
    models = []
    for s in seeds:
        p = weight_path(s, outdir)
        if not os.path.exists(p):
            raise SystemExit(f"權重不存在：{p}（先跑 `python -m script.sm_dual train`）")
        ck = torch.load(p, map_location="cpu", weights_only=True)
        m = DualNet(tuple(ck.get("hidden", (1024, 1024, 512))))
        m.load_state_dict(ck["model_state_dict"])
        models.append(m.eval())
    return models


# ────────────────────────────────────────────────────────────────────────────
# 子命令
# ────────────────────────────────────────────────────────────────────────────
def cmd_train(a):
    device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    pool = build_pool(refresh=a.refresh)
    X, Y, wm, M, src = pool["X"], pool["Y"], pool["wm"], pool["M"], pool["src"]
    tr, ho = make_split(wm)
    n_r57 = sum(1 for s in src if s.startswith("dedust_"))
    print(f"\n鍋：去重前 {pool['n_raw']} → 去重後 {len(X)}（harvest_dual {len(X) - n_r57} / dedust_r57* {n_r57}）")
    print(f"split（seed={SPLIT_SEED}，wm_dual {SPLIT_BINS} 分位分層 {HELDOUT_FRAC:.0%}）："
          f"train {len(tr)} / held-out {len(ho)}")
    print(f"wm_dual 分布：min {wm.min():.2f} / 中位 {np.median(wm):.2f} / max {wm.max():.2f}\n")

    T = _targets_matrix(Y, M)
    outdir = a.outdir or str(ROOTDIR)
    os.makedirs(outdir, exist_ok=True)
    metas = []
    for s in a.seeds:
        t0 = time.time()
        print(f"  [seed {s}] 訓練中（device={device}, epochs={a.epochs}, aug={a.aug}, w_margin={a.w_margin}）…")
        model, best = train_one(X[tr], T[tr], s, epochs=a.epochs, bs=a.bs, lr=a.lr,
                                aug=a.aug, w_margin=a.w_margin, device=device)
        p = weight_path(s, outdir)
        torch.save({"model_state_dict": model.state_dict(), "seed": s, "arch": "DualNet",
                    "hidden": (1024, 1024, 512), "epochs": a.epochs, "aug": a.aug,
                    "w_margin": a.w_margin, "val_loss": best, "n_train": int(len(tr)),
                    "split_seed": SPLIT_SEED, "stores": list(STORES)}, p)
        metas.append({"seed": s, "val_loss": best, "sec": round(time.time() - t0, 1), "path": p})
        print(f"  [seed {s}] 完成 val={best:.4f}，{time.time() - t0:.0f}s → {p}")

    import hashlib
    keys = [hashlib.sha1(np.packbits(X[i]).tobytes()).hexdigest() for i in ho]
    sp = os.path.join(outdir, SPLIT_JSON)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"split_seed": SPLIT_SEED, "frac": HELDOUT_FRAC, "bins": SPLIT_BINS,
                   "stores": list(STORES), "n_total": int(len(X)), "n_train": int(len(tr)),
                   "n_heldout": int(len(ho)), "heldout_idx": ho.tolist(),
                   "heldout_sha1": keys, "models": metas}, f, ensure_ascii=False, indent=1)
    print(f"\nsplit → {sp}")
    print("下一步：python -m script.sm_dual eval")


def _gate_line(name, rho, hit, n_ho, k_true, topk):
    from scipy.stats import hypergeom
    p = float(hypergeom.sf(hit - 1, n_ho, k_true, topk))
    ok = (rho >= GATE_RHO) and (p < GATE_P)
    print(f"  {name:<14} ρ={rho:+.4f}  top{topk}∩true-top10%={hit}/{topk}"
          f"（隨機期望 {topk * k_true / n_ho:.1f}，超幾何 P={p:.3g}）→ {'過' if ok else '不過'}")
    return ok, p


def cmd_eval(a):
    device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    from scipy.stats import spearmanr
    targets = load_targets()
    pool = build_pool(refresh=False)
    X, Y, wm = pool["X"], pool["Y"], pool["wm"]
    tr, ho = make_split(wm)
    models = load_models(tuple(a.seeds), a.outdir)

    #! split 對帳：權重旁的 json 若與現在算出來的 held-out 不一致（鍋變了/seed 改了）→ 明講，別靜默。
    sp = os.path.join(str(a.outdir) if a.outdir else str(ROOTDIR), SPLIT_JSON)
    if os.path.exists(sp):
        rec = json.load(open(sp, encoding="utf-8"))
        same = rec.get("heldout_idx") == ho.tolist()
        print(f"split 對帳：{'一致' if same else '⚠ 不一致（鍋或 seed 變過，數字不可與舊版比）'}"
              f"（json n_heldout={rec.get('n_heldout')}，現算 {len(ho)}）")

    Yh, wm_true = Y[ho], wm[ho].astype(np.float64)
    R, G = predict(models, X[ho], device=device)
    Re, Ge = R.mean(axis=0), G.mean(axis=0)

    print(f"\nheld-out n={len(ho)}（train {len(tr)}）")
    print("逐通道 MAE (dB, ensemble)：" + "  ".join(
        f"{L}={v:.3f}" for L, v in zip(LABELS, np.abs(Re - Yh).mean(axis=(0, 2)))))
    for j, s in enumerate(a.seeds):
        print(f"  seed{s} MAE：" + "  ".join(
            f"{L}={v:.3f}" for L, v in zip(LABELS, np.abs(R[j] - Yh).mean(axis=(0, 2))))
            + f"   全體 {np.abs(R[j] - Yh).mean():.3f}")
    print(f"  ensemble 全體 MAE：{np.abs(Re - Yh).mean():.3f}")

    k_true = max(1, int(round(len(ho) * GATE_TRUE_FRAC)))
    true_top = set(np.argsort(-wm_true, kind="stable")[:k_true].tolist())
    topk = min(GATE_TOPK, len(ho))

    print(f"\n品質閘（ρ ≥ {GATE_RHO:+.2f} 且 超幾何 P < {GATE_P}；true top10% = 前 {k_true} 名；"
          f"排序口徑＝margin 頭的 min(m1..m4)）")
    for j, s in enumerate(a.seeds):
        wp = G[j].min(axis=1)
        rho = float(spearmanr(wp, wm_true).statistic)
        hit = len(true_top & set(np.argsort(-wp, kind="stable")[:topk].tolist()))
        _gate_line(f"seed{s}", rho, hit, len(ho), k_true, topk)
    wp = Ge.min(axis=1)
    rho_e = float(spearmanr(wp, wm_true).statistic)
    hit_e = len(true_top & set(np.argsort(-wp, kind="stable")[:topk].tolist()))
    ok_e, p_e = _gate_line("ensemble", rho_e, hit_e, len(ho), k_true, topk)

    #? 對照組（不進閘）：改用「響應頭 → worst_margin_dual」排序——證明 margin 頭不是白加的。
    wp_r = wm_array(Re, targets)
    print(f"  [對照] 響應頭→worst_margin_dual 排序：ρ={float(spearmanr(wp_r, wm_true).statistic):+.4f}"
          f"（命中 {len(true_top & set(np.argsort(-wp_r, kind='stable')[:topk].tolist()))}/{topk}）")
    print("  [記帳] margin MAE (dB)：" + "  ".join(
        f"{k}={v:.3f}" for k, v in zip(MARGINS, np.abs(Ge - np.array(
            [margins_of(y, targets)[1] for y in Yh])).mean(axis=0))))

    print(f"\n→ 品質閘: {'過' if ok_e else '不過'}"
          f"（ensemble ρ={rho_e:+.4f}｜命中 {hit_e}/{topk}，P={p_e:.3g}）")


def _load_pool_arg(path):
    """rank 的 --pool：.npz（鍵 X/patterns + 選用 ids）或資料夾（manifest.json 或 *.pt）。"""
    if os.path.isfile(path) and path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        key = "X" if "X" in z else ("patterns" if "patterns" in z else None)
        if key is None:
            raise SystemExit(f"{path} 沒有 'X' 或 'patterns' 鍵（有 {list(z.keys())}）")
        X = z[key].reshape(len(z[key]), -1)
        ids = [str(i) for i in z["ids"]] if "ids" in z else [f"{i:05d}" for i in range(len(X))]
        return X.astype(np.float32), ids
    if not os.path.isdir(path):
        raise SystemExit(f"--pool 既不是 .npz 也不是資料夾：{path}")
    mf = os.path.join(path, "manifest.json")
    if os.path.exists(mf):                                    # dedust *_input 夾：<id>.pt = 純 pattern
        ids = [m["id"] for m in json.load(open(mf, encoding="utf-8"))]
        ids = [i for i in ids if os.path.exists(os.path.join(path, i + ".pt"))]
        xs = [torch.load(os.path.join(path, i + ".pt"), weights_only=True) for i in ids]
    else:                                                     # SampleStore 夾：<hash>.pt = (x, y)
        files = sorted(f for f in os.listdir(path) if f.endswith(".pt"))
        ids = [os.path.splitext(f)[0] for f in files]
        xs = [torch.load(os.path.join(path, f), weights_only=True) for f in files]
        xs = [t[0] if isinstance(t, (tuple, list)) else t for t in xs]
    if not ids:
        raise SystemExit(f"--pool 夾裡沒有 .pt：{path}")
    X = np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in xs])
    return X, ids


def cmd_rank(a):
    device = a.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    X, ids = _load_pool_arg(a.pool)
    if X.shape[1] != N_PIX:
        raise SystemExit(f"pattern 維度 {X.shape[1]} ≠ {N_PIX}")
    rows = rank_pool(load_models(tuple(a.seeds), a.outdir), X, ids, top=a.top, device=device)
    head = ["rank", "id", "pred_wm", "m1_S11", "m2_S22", "m3_S21pass", "m4_S21stop"]
    out = [[i + 1, r[0], round(r[1], 4)] + [round(r[2][k], 4) for k in MARGINS]
           for i, r in enumerate(rows)]
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerows(out)
        print(f"{len(out)} 筆（池 {len(ids)}）→ {a.out}")
    else:
        print("\t".join(head))
        for r in out:
            print("\t".join(str(c) for c in r))


def main(argv=None):
    ap = argparse.ArgumentParser(description="dual-port SM 排序器（批次線候選挑選）")
    #? 共用旗標放 parent（不放主 parser）：--seeds 是 nargs="+"，擺在子命令前會把子命令名吃掉。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default=None, help="cpu / cuda:0（預設有卡就用卡）")
    common.add_argument("--outdir", default=None, help=f"權重/split 目錄（預設 {ROOTDIR}）")
    common.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", parents=[common], help="建鍋 → 分層 split → 訓 3 seed → 存權重")
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--bs", type=int, default=128)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--w-margin", type=float, default=12.0, dest="w_margin")
    t.add_argument("--aug", default="lr+ud", choices=["none", "lr", "ud", "lr+ud"],
                   help="對稱增強（見 _augment）")
    t.add_argument("--refresh", action="store_true", help="重建鍋快取（重掃 NAS）")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", parents=[common], help="held-out 品質閘")
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("rank", parents=[common], help="候選池排序")
    r.add_argument("--pool", required=True, help=".npz（X/patterns[+ids]）或資料夾（manifest 或 *.pt）")
    r.add_argument("--top", type=int, default=50)
    r.add_argument("--out", default=None, help="輸出 CSV（省略則印 stdout）")
    r.set_defaults(func=cmd_rank)

    a = ap.parse_args(argv)
    throttle()
    with SingleInstance():
        a.func(a)


if __name__ == "__main__":
    main()
