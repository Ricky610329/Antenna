# configs/ — 實驗對照表

> **一個 `*.yaml` = 一組實驗的完整設定。** 這份檔案是所有訓練實驗的索引：每個 config 在「對標哪個 baseline、改了什麼、想看什麼」一目了然。
>
> ⚠ **硬規則（CLAUDE.md 慣例）**：新增/修改任何 `configs/*.yaml` 或訓練腳本，**必須同步更新這份表**。產生新實驗前先掃這份表，避免重複造輪子。

## 怎麼跑

```bash
conda activate patch          # 正式機才有 HFSS
python train.py configs/single_base.yaml
```

- 入口固定是根目錄的 **`train.py`**（已取代舊 `train_single.py` / `train_dual.py`）。
- config 鍵名有白名單把關：打錯鍵直接報錯，不會默默吃預設值。
- 斷點續跑：用**相同 config**（→ 結果夾名相同）再跑即自動續；要重頭跑就改 `name` 或清掉 result 夾。
- 設定/結構對照見 [`../docs/training.md`](../docs/training.md)；系統架構見 [`../docs/architecture.html`](../docs/architecture.html)。

## baseline 是誰

| port | baseline config | 說明 |
| --- | --- | --- |
| single | `single_base.yaml` | **學長單埠基準**（所有 loss 正則化＝0）。新單埠實驗都對標它。 |
| dual | `dual_base.yaml` | 學長雙埠基準。新雙埠實驗都對標它。 |

## 單埠 single（對標 `single_base.yaml`）

| config | 測試重點 | 與 base 的差異 | 舊編號 |
| --- | --- | --- | --- |
| `single_base.yaml` | 基準 | — | 1 / 2 / 5 |
| `single_sc.yaml` | **論文主方法：圖譜連通度（單埠版）** | `loss.spectral_connectivity: 0.0005` | （新增，對標 base 看 SC 是否有幫助） |
| `single_sc_rad.yaml` | **SC + 方向圖 loss** | 多 `radiation:` 區段（`enable: true`，beam_coverage_loss）；SM 多 rad head，`pretrained: old_sm.pth` 以 strict=False 部分載入暖啟動（與 base 同速起步） | （新增，對標 `single_sc` 看方向圖是否有幫助；**需正式機 HFSS**） |
| `single_sc_rad_mirror.yaml` | **SC + 方向圖 + 鏡像對稱** | 在 `single_sc_rad` 上 `generator: mirror`（唯一變因） | （新增，對標 `single_sc_rad` 看對稱約束在 rad 前線上是否更快/更好；**需正式機 HFSS**） |
| `single_sc_rad_replay.yaml` | **SC + 方向圖 + SM 經驗回放** | 在 `single_sc_rad` 上 `sm_train.mode: replay`（唯一變因；最新少數步＋回放整個緩衝取代「擬到收斂」反模式） | （新增，對標 `single_sc_rad` 看防遺忘/削暴衝是否更好；**需正式機 HFSS**） |
| `single_sc_rad_dlf.yaml` | **SC + 方向圖 + 動態損失過濾 DLF** | 在 `single_sc_rad` 上 `sm_train.mode: dlf`（學長論文 §3.5 驗證法：全收＋每輪用累計門檻 λ_t 篩菁英子集訓） | （新增，對標 `single_sc_rad`／`_replay`；論文消融 >50% 改善；**需正式機 HFSS**） |
| `single_sc_rad_boundary.yaml` | **DLF + Boundary loss（trust-region）** | 在 `single_sc_rad_dlf` 上 `loss.boundary`（拉 G 回 SM 已見分布；權重待 A/B 調） | （新增，對標 `single_sc_rad_dlf` 看「防 G 鑽 SM 盲區、少白燒 HFSS」是否更快/更好；**需正式機 HFSS**） |
| `single_tv.yaml` | TV 正則化 + KuoHung SM 暖身 | `loss.total_variation: 0.01`、`surrogate.warmup: "1"` | 3 / 4 |
| `single_tv50.yaml` | 強 TV | `loss.total_variation: 50` | 7 |
| `single_island.yaml` | 孤島抑制（強） | `loss.island_suppression: 100` | 8 / 9 |
| `single_island1.yaml` | 孤島抑制（弱） | `loss.island_suppression: 1` | 10 |
| `single_peak.yaml` | ACP plateau 策略 | `scheduler.on_plateau: peak` | 6 |
| `single_mirror.yaml` | **左右鏡像對稱生成器** | `generator: mirror`（MLP 出半邊 25×13→flip 成 25×25；對稱+搜尋空間砍半） | （新增，對標 base 看對稱約束是否更快/更好；學長舊法是資料增強、不保證對稱） |
| `single_sc_mirror.yaml` | **SC + 鏡像對稱** | 在 `single_sc` 上 `generator: mirror`（唯一變因） | （新增，對標 `single_sc` 看對稱約束在 SC 上是否更快/更好） |
| `single_sc_mirror_boundary.yaml` | **SC + 鏡像 + DLF + Boundary（無方向圖）** | 在 `single_sc_mirror` 上 `sm_train.mode: dlf` + `loss.boundary`（進階 SM 訓練疊在對稱生成器上） | （新增，mirror 進階版；對標 `single_sc_mirror`；無 rad） |

## 雙埠 dual（對標 `dual_base.yaml`）

| config | 測試重點 | 與 base 的差異 | 舊編號 |
| --- | --- | --- | --- |
| `dual_base.yaml` | 基準 | — | 6 / 7 / 8 |
| `dual_sc.yaml` | 論文主方法：圖譜連通度 | `loss.spectral_connectivity: 0.0005` | 9 |
| `dual_tv100.yaml` | 強 TV | `loss.total_variation: 100` | 1 / 4 |
| `dual_tv1.yaml` | 弱 TV | `loss.total_variation: 1` | 2 |
| `dual_island.yaml` | 孤島抑制 | `loss.island_suppression: 100` | 3 / 5 |

## 已知缺口 / 可補的實驗

- ~~單埠沒有 `spectral_connectivity` config~~ → 已補 `single_sc.yaml`（2026-06-19）。
- ~~方向圖 loss 尚未有 config~~ → 已補 `single_sc_rad.yaml`（2026-06-19，Stage 2 整合完成：`radiation:` 區段 + SM rad head + `beam_coverage_loss`）。**僅正式機可跑**（需 HFSS 取方向圖）。
- **方向圖 rad head 冷啟動**：`harvest_single` 沒有方向圖標籤 → rad head 線上從零學。優化（Stage 3，未做）：收 `harvest_single_rad`（好 pattern 補方向圖標籤）預訓練 rad head，再用 `radiation.offline_dataset` 載入。
- **方向圖訓練預設凍 trunk**（`radiation.freeze_trunk: true`）：隨機 rad 頭 + 不凍 trunk + 極端 dB target 曾把 S11/Gain backbone 帶歪、爆 NaN。現在 rad 頭只更新自己、rad target 會 clamp（±60dB）、SM 訓練有 NaN 防護網。要放梯度回 backbone 才設 `false`。
- **SM 線上更新重設計（profiling 找到瓶頸後）**：`train_one_data` 把「最新一筆擬到收斂（≤20000 步）」是 catastrophic-forgetting 反模式、也是速度大頭。`sm_train.mode`：`single`(原樣/反模式) → `replay`(全收＋回放整個緩衝) → `dlf`(全收＋每輪用累計門檻 λ_t 篩菁英子集訓，＝學長論文 §3.5 驗證法、消融 >50%)。opt-in，預設 single 保 golden。`radiation.sm_max_epoch`(rad 擬合上限,1000)。⚠ 舊 `online` store「<平均才收」＝論文點名的 baseline（寫入端篩→偽收斂），replay/dlf 改走「全收」緩衝。`loss.boundary`（已做）＝trust-region 正則（拉 G 回 SM 已見分布、防鑽盲區、少白燒 HFSS；需 replay/dlf 緩衝；權重待 A/B 調）。未來：prioritized 抽樣、SM 不確定性導引探索/聰明重啟（取代 ACP 盲目加熱）、rad head replay。

## 資料集標記（before rad / *_rad）

NAS（`DATASET_PATH = T:\碩一_鄒穎麒's\antenna\dataset`）上的離線資料集分兩代，**勿混疊**：

| 資料集 | 標籤 | 說明 |
| --- | --- | --- |
| `harvest_single`（24189）/ `harvest_dual`（10023） | **before rad** | response 只有 S11/Gain(/S21)，**無方向圖標籤**。各夾內有唯讀 `_BEFORE_RAD.md` 標記 |
| `harvest_single_rad` 等 `*_rad`（未來，Stage 3） | with rad | 好 pattern 補方向圖標籤的子集，**獨立新名**，不疊進舊集 |

- **約定**：rad 標籤資料一律走新名 `*_rad`，**永不**寫回 before-rad 舊集。
- marker 用 `.md`（`SampleStore` 只認 `*.pt`，且 init 會清 `*.tmp` → marker 不能用這兩種副檔名）。

## 新增實驗 SOP

1. 複製最接近的 baseline（`single_base.yaml` / `dual_base.yaml`）。
2. 改 `name`（決定 result 夾名，避免撞到別的實驗）。
3. **只改一個變因**（A/B 才乾淨）：通常是 `loss:` 區段開一個權重，或 `scheduler:` / `generator:` / `surrogate:` 換一項。
4. **回來這份表加一行**（測什麼、與 base 差在哪）。← 這步是硬規則。
5. 跑：`python train.py configs/<新檔>.yaml`（正式機）。結果夾＝自我說明的檔案制資料庫（`metrics.csv` / `patterns/` / `tb/`…，見 training.md §6）。

## 相關訓練腳本（`script/`）

| 腳本 | 是否訓練 | 功能 |
| --- | --- | --- |
| `../train.py` | ✅ 主入口 | config 驅動的單/雙埠訓練閉迴路 |
| `script/verify_radiation.py` | ❌ 驗證 | 正式機驗證 `SinglePortRadSimulator` 能否把方向圖資料抓出來（不訓練、不碰核心） |
| `script/kuohung.py` | ⚙ 資料 | KuoHung 參考圖樣載入（SM 單筆暖身用，對應 `surrogate.warmup`） |
| `script/harvest_legacy.py` | ⚙ 資料 | 從學長舊資料收割成自有 NAS 資料集（`harvest_single` / `harvest_dual`） |
| `script/convert_dataset.py` | ⚙ 資料 | 舊 `.dataset` 格式轉換 |
| `script/img2video.py`、`check_gpu.py`、`get_local_ip.py`、`kill.py`、`process_files.py` | ❌ 雜項 | 視覺化／環境／程序管理工具，與訓練無關 |
