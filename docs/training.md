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
**鍵名有白名單把關**：任何區段內打錯鍵會直接報錯（不會默默用預設值跑完整個實驗）。

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

sm_train:                     # 代理模型 (SM) 線上訓練
  lr: 0.001
  min_loss: 0.1
  max_epoch: 20000

# seed: 0                     # 選填：固定隨機種子 (可重現性，論文實驗建議設)

scheduler:                    # ACP (論文核心機制) 超參數，全部可調；預設值=原腳本設定
  on_plateau: linear          # linear | peak
  # T_0: 100  T_mult: 1  lr_min: 1.0e-6  temp_max: 4.0  temp_min: 0.1
  # warmup_ratio: 0.2  patience: 25  factor: 0.7

generator: sigmoid            # 生成器：antenna/zoo.py 的名字 (可省略 → sigmoid)；見第 4 節
                              # 微調/暖啟動: {name: sigmoid, hidden: [...], pretrained: xxx.pth}

surrogate:                    # 代理 SM：載入策略 (name 可省略 → mlp)；見第 4 節
  pretrained: old_sm.pth
  offline_dataset: patch_single_mirror
  # warmup: "1"               # 選填：KuoHung 參考圖樣編號，對 SM 做單筆暖身

targets:                      # 目標響應 (side=兩端, center=中央, width=梯形寬度)
  S11:  { side: 0,   center: -10, width: [5, 0, 7, 0, 5], method: low }
  Gain: { side: -19, center: 4,   width: [5, 0, 7, 0, 5], method: high }
```

## 3. port 決定的「結構性元件」（不放 YAML，由 code 解析）

| port | 模擬器 | 損失 hook | 饋電塊 | 響應標籤 |
| --- | --- | --- | --- | --- |
| `single` | SinglePortSimulator | custom_loss_minmax (`method: low/high`) | lower | S11, Gain |
| `dual` | DualPortSimulator | interval_loss (`interval: [-1, 1]`) | lower + upper | S11, S21, S22 |

## 4. 模型架構 + 載入（`generator` / `surrogate` 區段）

### 架構：模型動物園（`antenna/zoo.py`）

所有可用的 GEN / SM 架構都登記在 **`antenna/zoo.py`**（唯一的註冊文件），config 用**名字**指定：

| 區段 | 預設名字 | 對應實作 | 預設 `hidden` |
| --- | --- | --- | --- |
| `generator` | `sigmoid` | `SigmoidGenerator` | `[1024, 1024]` |
| `surrogate` | `mlp` | `HFSSNet`（`MLPSurrogate`） | `[2048, 1024, 512, 128, 64]` |

- 兩區段都可省略 → 用預設（與舊 `train_single/dual` **完全相同**）。
- `hidden` 為選填微調（一次性實驗用）；常用的變體請在 zoo 登記成新名字。
- **新增模型**：寫一個純 `nn.Module`（GEN 約定：spec 向量 → logits，**不做**二值化、不碰 tau——STE 二值化由訓練管線統一套用、tau 由 ACP 控制），在 zoo 加一行即可。

### 載入策略（`prepare_models()` 依序判斷）

1. **斷點續跑**：結果夾已存在且 `metrics.csv` 有 epoch → 載回 GEN/SM，從上次續跑（其餘略過）。
2. **GEN 預載入**：`generator.pretrained` 權重檔存在 → 載入 GEN（暖啟動；架構需相容）。
3. **SM 載入**：`surrogate.pretrained` 存在 → 直接載入；否則用 `surrogate.offline_dataset` 從頭預訓練。
4. **KuoHung 暖身**：`surrogate.warmup`（參考圖樣編號，如 `"1"`）→ 取 `KuoHung.load()` 對 SM 做單筆暖身微調。

皆無 → GEN/SM 從隨機權重起步（純靠線上學習）。

`*.pretrained` / `offline_dataset` 相對於 `DATASET_PATH`；設 `null` 或省略則略過該步。
> 目前僅支援**完整權重載入**（L1/L2，架構需相同）。架構不同時的部分載入（transfer learning，L3）尚未實作。

## 5. 現成 configs（對照舊 `MultiConfig` 編號）

| config | port | 重點 | 舊編號 |
| --- | --- | --- | --- |
| `single_base.yaml` | single | 基準 | 1 / 2 / 5 |
| `single_tv.yaml` | single | TV 0.01 + KuoHung 暖身 | 3 / 4 ※ |
| `single_tv50.yaml` | single | TV 50 | 7 |
| `single_island.yaml` | single | 孤島抑制 100 | 8 / 9 |
| `single_island1.yaml` | single | 孤島抑制 1 | 10 |
| `single_peak.yaml` | single | on_plateau peak | 6 |
| `dual_base.yaml` | dual | 基準 | 6 / 7 / 8 |
| `dual_sc.yaml` | dual | SC 連通性 0.0005 | 9 |
| `dual_tv100.yaml` | dual | TV 100 | 1 / 4 |
| `dual_tv1.yaml` | dual | TV 1 | 2 |
| `dual_island.yaml` | dual | 孤島抑制 100 | 3 / 5 ※※ |

- ※ 舊 3/4 含 KuoHung SM 暖身，已由 `surrogate.warmup` 支援（`single_tv.yaml` 設 `"1"`＝舊 3 的 KuoHung-1；舊 4 用 KuoHung-2，改成 `"2"` 即可）。
- ※※ 舊 dual 3/5 的 `island_suppression` 鍵名打錯而失效；轉成 config 後鍵名正確 → **孤島抑制實際生效**。舊 5 的 `relu` 從未被讀取，已捨棄。

## 6. 前置需求（正式機）

- Windows + ANSYS HFSS（COM）；連得到實驗室內網（自動掛 NAS `T:`）。
- 結果夾（`ROOTDIR\result\[Patch-<port>-...]<name>\`）內容＝**自我說明的檔案制資料庫**：

```
config.yaml      # 設定快照 (原文)
status.json      # 運行心跳：state(running/finished/crashed)/機器/epoch
metrics.csv      # 純量時序 (一 epoch 一行；pd.read_csv 即可自訂畫圖)
patterns/        # 每筆模擬過的 (pattern, response, loss)，hash 即檔名
checkpoint/      # GEN 逐 epoch 權重 + sm.pth
online/          # 好樣本庫 (SampleStore)
tb/              # TensorBoard 事件檔 (監看方式見 quickstart 3.5)
summary.png      # 訓練結束的總覽圖
*.log            # 文字 log
```

## 7. 斷點續跑

用**相同 config**（→ 結果夾名相同）再跑，且 `metrics.csv` 有 epoch，會自動載回 GEN/SM 從上次續跑。想重頭跑就改 config 的 `name` 或清掉對應 result 資料夾。
