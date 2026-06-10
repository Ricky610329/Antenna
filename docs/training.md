# 訓練使用說明（config 驅動）

> 入口是 **`train.py`**，由外部 YAML config 驅動。已**取代** `train_single.py` / `train_dual.py`。
> 一頁速查見 [`quickstart.md`](quickstart.md)；系統架構見 [`architecture.html`](architecture.html)。

## 1. 執行

```bash
conda activate patch
python train.py configs/single_base.yaml      # 單埠
python train.py configs/dual_base.yaml         # 雙埠
python train.py configs/dual_sc.yaml           # 雙埠 + 論文 SC 連通性損失
```

一個 YAML = 一組實驗的完整設定。要跑新實驗就複製一個 config 改參數。

## 2. config 結構

```yaml
name: pixel_single_base       # 實驗名稱 (結果資料夾用)
port: single                  # single | dual  ← 決定模擬器/損失/饋電/響應「結構」

epochs: 1000
lr: 0.005
patience: 10                  # early-stop → rollback 的耐心值

loss:                         # 正則化權重 (0 = 不啟用)
  total_variation: 0.0
  island_suppression: 0.0
  spectral_connectivity: 0.0  # 論文主方法 (圖譜連通度)
  gap_closing: 0.0

hfss:                         # 代理模型 (SM) 線上訓練
  lr: 0.001
  min_loss: 0.1
  max_epoch: 20000

scheduler:
  on_plateau: linear          # linear | peak

surrogate:                    # 模型載入策略 (見第 4 節)
  pretrained: old_sm.pth
  offline_dataset: patch_single_mirror

targets:                      # 目標響應 (side=兩端, center=中央, width=梯形寬度)
  S11:  { side: 0,   center: -10, width: [5, 0, 7, 0, 5], method: low }
  Gain: { side: -19, center: 4,   width: [5, 0, 7, 0, 5], method: high }
```

## 3. port 決定的「結構性元件」（不放 YAML，由 code 解析）

| port | 模擬器 | 損失 hook | 饋電塊 | 響應標籤 |
| --- | --- | --- | --- | --- |
| `single` | SinglePortSimulator | custom_loss_minmax (`method: low/high`) | lower | S11, Gain |
| `dual` | DualPortSimulator | interval_loss (`interval: [-1, 1]`) | lower + upper | S11, S21, S22 |

## 4. 模型載入（`surrogate` 區段，可在 config 指定）

`prepare_surrogate()` 依序判斷：

1. **斷點續跑**：結果夾已存在且 `temp` 有 epoch → 載回 GEN/SM，從上次續跑。
2. **預訓練檔**：`surrogate.pretrained` 指向的權重檔存在 → 直接載入。
3. **離線預訓練**：否則用 `surrogate.offline_dataset` 從頭預訓練 SM。
4. **全新**：皆無 → SM 從隨機權重起步（純靠線上學習）。

`surrogate.pretrained` / `offline_dataset` 相對於 `DATASET_PATH`；設 `null` 則略過該步。

## 5. 現成 configs（對照舊 `MultiConfig` 編號）

| config | port | 重點 | 舊編號 |
| --- | --- | --- | --- |
| `single_base.yaml` | single | 基準 | 1 / 2 / 5 |
| `single_tv.yaml` | single | TV 0.01 | 3 / 4 ※ |
| `single_tv50.yaml` | single | TV 50 | 7 |
| `single_island.yaml` | single | 孤島抑制 100 | 8 / 9 |
| `single_island1.yaml` | single | 孤島抑制 1 | 10 |
| `single_peak.yaml` | single | on_plateau peak | 6 |
| `dual_base.yaml` | dual | 基準 | 6 / 7 / 8 |
| `dual_sc.yaml` | dual | SC 連通性 0.0005 | 9 |
| `dual_tv100.yaml` | dual | TV 100 | 1 / 4 |
| `dual_tv1.yaml` | dual | TV 1 | 2 |
| `dual_island.yaml` | dual | 孤島抑制 100 | 3 / 5 ※※ |

- ※ 舊 3/4 還含 KuoHung SM 暖身，目前 config 驅動尚未支援（規劃中）。
- ※※ 舊 dual 3/5 的 `island_suppression` 鍵名打錯而失效；轉成 config 後鍵名正確 → **孤島抑制實際生效**。舊 5 的 `relu` 從未被讀取，已捨棄。

## 6. 前置需求（正式機）

- Windows + ANSYS HFSS（COM）；連得到實驗室內網（自動掛 NAS `T:`）。
- 結果寫到 `ROOTDIR\result\[Patch-<port>-...]<name>\`（含 log / config 快照 / checkpoint / online / pic）。

## 7. 斷點續跑

用**相同 config**（→ 結果夾名相同）再跑，且 `temp` 有 epoch，會自動載回 GEN/SM 從上次續跑。想重頭跑就改 config 的 `name` 或清掉對應 result 資料夾。
