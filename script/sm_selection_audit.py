# -*- coding: utf-8 -*-
"""SM 的**選批能力**稽核：把「準度」翻譯成「這一批能挑到多好的候選」。

## 為什麼需要這支（analysis-10 §7/§8 的產物）

批次線的五軸 KPI ①（SM 準度）＝ held-out 誤差 ＋ 前瞻 ρ。實測（§8）它是**有效的代理**：
在 `sm_reanchor` v50–v100 十三個版本上，`ρ vs top10% 命中率` 的 Spearman ＝ **+0.775**。

**但那個結論有前提**：SM 的改善是**全域均勻**的（多看資料 ⇒ 整條曲線都更貼）。
diffsim 那邊示範了反例——修一個**結構性**的物理 bug 讓 ρ 改善 **+124%**，
而選批的 P(勝隨機) 從 18% → 17%，**完全沒動**（§7.2）。

⇒ **落地規則（`docs/discuss/decisions.md`）**：
   改動只碰模型的**一部分機制**時（換架構／加某個特徵／換錨組／改 loss 的一項），
   **ρ 不夠，要直接量 top-K**。這支就是量它的。

## 量什麼

在一組**模型沒見過**的樣本上，模擬「用這個模型挑 K 筆去燒 HFSS」：
  - `P(勝隨機)`：這 K 筆的 best 勝過隨機挑 K 筆的機率（50% ＝ 與隨機無異）
  - `top10% 命中率`：選中的 K 筆有幾成落在池的前 10%（隨機期望 10%）
  - 兩者都用**重抽候選池**的 bootstrap 給 CI（重抽的是**樣本**不是 seed）

## 用法

    python -m script.sm_selection_audit --versions 88,94,100
    python -m script.sm_selection_audit --versions 100 --k 30 --stratum neg

⚠ 只讀：`torch.load` 推論而已，不寫 NAS、不碰 jobs/records/kpi、強制 CPU。
⚠ 樣本自動挑「**不在 `CLEAN_STORES` 內**」的（含自動納入的 `dedust_auto*`/`dedust_c*`
   ——那是 587 店不是 `clean_stores.txt` 的 513 行，漏扣會**系統性高估**）。
"""
import argparse
import os
import sys

import numpy as np
import torch


def _load_pool(stratum: str, n_max: int, seed: int):
    """回傳 (X, wm_true)：`fit` 分割中該層、且**不在任何 SM 訓練店**內的樣本。"""
    from script.sm_reanchor import CLEAN_STORES
    from script.diffsim import data as D
    from script.diffsim.eval import margins
    idx = D.load()
    split, _ = D.assign_split(idx)
    seen = np.isin(idx["store"], list(set(CLEAN_STORES)))
    m = (split == "fit") & (idx["stratum"] == stratum) & (~seen)
    if not m.any():
        raise SystemExit(f"stratum={stratum} 沒有 out-of-sample 樣本可用")
    sel = np.random.default_rng(seed).permutation(np.where(m)[0])[:n_max]
    y, _, _ = margins(idx["y"][sel])
    return idx["x"][sel].astype(np.float64), y


def _predict_wm(path, X):
    """載入一個 SM checkpoint，回傳它預測的 worst_margin。純推論、不改任何狀態。"""
    from script.sm_reanchor import _make_sm, LABELS, _cfg
    from antenna.losses import worst_margin
    m = _make_sm()
    m.pre_load_model(path, strict=True)
    xf = torch.as_tensor(X.reshape(len(X), -1), dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(xf), 256):
            o = m.model(xf[i:i + 256])
            out.extend(float(worst_margin(o[j], LABELS, _cfg.targets)[0]) for j in range(o.shape[0]))
    return np.asarray(out)


def audit(pred, y, k: int, nboot: int, seed: int = 0):
    """重抽**候選池**，回傳 (P(勝隨機), 命中率中位, 命中率 CI, ρ, |err| 中位)。

    #! `P(勝隨機)` 用**嚴格大於**（平手算輸）⇒ **完美預測器不會是 100%**：
    #  它必定選到池中最大值，但隨機挑 K 筆也有 ~K/n 的機率選到同一筆，那次算平手＝輸。
    #  n=800/K=60 時完美預測器實測 **~86%**（`test_selection_audit_calibration` 釘住這個刻度）。
    #  保留這個定義是因為它對所有方法一視同仁、且偏保守；讀數時記得
    #  **上限是 ~86% 不是 100%，50% 才是「與隨機無異」**。
    """
    from scipy.stats import spearmanr
    n = len(y)
    thr = np.percentile(y, 90)
    g = np.random.default_rng(seed)
    win = np.empty(nboot)
    hit = np.empty(nboot)
    for i in range(nboot):
        idx = g.integers(0, n, n)
        yy = y[idx]
        top = np.argsort(-pred[idx])[:k]
        win[i] = yy[top].max() > yy[g.choice(n, k, replace=False)].max()
        hit[i] = (yy[top] >= thr).mean()
    return (100 * win.mean(), 100 * np.median(hit),
            (100 * np.percentile(hit, 2.5), 100 * np.percentile(hit, 97.5)),
            spearmanr(pred, y).statistic, float(np.median(np.abs(pred - y))))


def main(argv=None):
    ap = argparse.ArgumentParser(description="SM 選批能力稽核（唯讀）")
    ap.add_argument("--versions", required=True,
                    help="逗號分隔的 sm_reanchor 版本號，如 88,94,100")
    ap.add_argument("--k", type=int, default=60, help="一批席次（預設 60）")
    ap.add_argument("--stratum", default="clean", choices=("clean", "neg"))
    ap.add_argument("--n", type=int, default=817, help="候選池上限")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args(argv)

    from antenna.utils import DATASET_PATH
    X, y = _load_pool(a.stratum, a.n, a.seed)
    print(f"候選池：stratum={a.stratum}  n={len(y)}  "
          f"（`fit` 分割中不在 CLEAN_STORES 的樣本）")
    print(f"真值 wm：中位 {np.median(y):+.2f}  p90 {np.percentile(y, 90):+.2f}  max {y.max():+.2f}")
    print(f"\n選 K={a.k}（P(勝隨機) 的 50% ＝ 與隨機無異；命中率的隨機期望 ＝ 10%）")
    print(f"{'版本':>6} {'層內ρ':>9} {'|err|中位':>11} {'P(勝隨機)':>10} {'top10%命中':>18}")
    for v in a.versions.split(","):
        v = v.strip()
        p = DATASET_PATH.joinpath(f"sm_reanchor{v}.pth")
        if not os.path.exists(str(p)):
            print(f"{'v' + v:>6}   找不到 {p}")
            continue
        try:
            pred = _predict_wm(p, X)
        except Exception as e:                                   # 舊版架構可能不相容
            print(f"{'v' + v:>6}   載入失敗：{str(e)[:60]}")
            continue
        win, hit, ci, rho, err = audit(pred, y, a.k, a.boot, a.seed)
        print(f"{'v' + v:>6} {rho:+9.4f} {err:9.3f} dB {win:9.0f}% "
              f"{hit:9.0f}% [{ci[0]:.0f}%, {ci[1]:.0f}%]")
    print("\n⚠ 這是單一 OOS 樣本集上的重量，不是批次線的官方 KPI 口徑；用來看**趨勢與量級**。")
    print("⚠ 何時該看這支而不是只看 ρ：改動只碰模型的一部分機制時（換架構／加特徵／換錨組／"
          "改 loss 的一項）——見 docs/discuss/decisions.md「驗收指標要用使用方式的指標」。")


if __name__ == "__main__":
    sys.exit(main())
