# `train_single.py` 與 `train_dual.py` 使用說明（GAN branch）

> 適用範圍：`GAN` 分支根目錄下的 `train_single.py`（單埠貼片天線）與 `train_dual.py`（雙埠貼片天線）。
> 本文件只描述「如何執行這兩支腳本、各參數的意義、以及執行時的注意事項」，不涵蓋 `antenna/` 套件的內部演算法細節。

---

## 1. 這兩支腳本在做什麼

兩支都是**「生成模型 + 代理模型 + HFSS 模擬」三者交替的線上學習（online learning）訓練主程式**，流程概念相同：

1. 生成模型 `SigmoidGEN`（GEN）吃「目標頻率響應」→ 產生一張 25×25 的像素天線 pattern。
2. 把 pattern 丟進 **ANSYS HFSS** 做電磁模擬，得到真實的 S 參數（`real_loss`）。
3. 用模擬結果即時訓練**代理模型**（Surrogate Model, `OldSM`），讓它逼近 HFSS。
4. 用代理模型對 GEN 反傳遞，更新 GEN（`fake_loss`）。
5. 重複，並用 early-stop / rollback 機制在卡住時退回最佳 epoch 重練。

兩者唯一的本質差異是**單埠 vs 雙埠**（埠數、目標響應、饋入點、固定金屬塊位置不同），程式骨架幾乎一模一樣。

> ⚠️ 注意 scope：這是研究/實驗用的訓練腳本，不是 production 工具。執行高度依賴實驗室環境（HFSS 軟體 + 網路磁碟 + 預訓練檔），無法在一般機器上直接重現。

---

## 2. 執行前置需求

| 需求 | 說明 |
| --- | --- |
| **作業系統** | Windows（simulator 透過 `win32com` 呼叫 HFSS，僅限 Windows）。 |
| **ANSYS HFSS** | 必須安裝並可被 COM 介面 `AnsoftHFSS.HfssScriptInterface` 啟動。沒有 HFSS 跑到 `simulator.open()` 會直接失敗。 |
| **Python 環境** | conda env `ant`（先 `conda activate ant`），需 `pywin32`、`torch`、`loguru` 等（見 `requirements.txt`）。 |
| **網路磁碟 `T:`** | 腳本開頭會 `connect_network_drive("T:", r"\\140.123.106.219\temp", "user", "ailab120")` 自動掛載實驗室共享磁碟。`ROOTDIR = T:\碩二_吳維文's\Patch Antenna\Experiment`，所有結果與資料集都在這。**需連得到該實驗室內網**。 |
| **資料集 / 預訓練代理模型** | 放在 `DATASET_PATH = ROOTDIR\dataset`：<br>· single → `old_sm.pth`、DataManager `patch_single_mirror`<br>· dual → `patch_dual.pth`、DataManager `patch_dual` |

> 兩支腳本開頭都會 `config.device = "cpu"`，預設用 CPU 跑（GEN/SM 都很小，瓶頸在 HFSS 模擬）。

---

## 3. 如何執行

兩支腳本都用 **`MultiConfig` 機制**：第一個命令列參數 `sys.argv[1]` 決定要跑哪一組設定。

```bash
conda activate ant

# 單埠：跑設定編號 1
python train_single.py 1

# 雙埠：跑設定編號 4
python train_dual.py 4
```

- **必須帶編號**。`MultiConfig` 內部讀 `sys.argv[1]`，不帶參數會 `IndexError`。
- 編號就是下面設定表中的 key（single 為 `1`–`10`，dual 為 `1`–`9`）。
- 編號決定實驗名稱與該組要啟用的 loss 權重等超參數。

執行後結果會寫到 `ROOTDIR\result\<實驗名稱>\`（名稱由設定的 `name` 模板展開，含裝置 IP 末碼與 hash id）。

---

## 4. 單埠 vs 雙埠 差異對照

| 項目 | `train_single.py` | `train_dual.py` |
| --- | --- | --- |
| 埠數 | 單埠 | 雙埠 |
| Simulator | `SinglePortSimulator`（`single_port.sab`） | `DualPortSimulator`（`dual_port.sab`） |
| 目標響應標籤 | `S11`, `Gain` | `S11`, `S21`, `S22` |
| Loss hook | `custom_loss_minmax`（`method='low'/'high'`） | `interval_loss`（區間容許值 `[target-1, target+1]`） |
| 固定金屬塊 | 只加 `lower`（座標 `(10,15,20,25)`） | 加 `lower` + `upper`（`upper` 在 `(10,15,0,5)`） |
| 饋入點 | `FeedReachability.single_feed()`（1 點） | `FeedReachability.dual_feed()`（2 點） |
| 預訓練代理模型 | `old_sm.pth` | `patch_dual.pth` |
| 訓練資料集 | `patch_single_mirror` | `patch_dual` |
| 代理模型線上再訓練（rollback 時） | `smodel.train_by_datas(online_dataset)`（全部） | `smodel.train_by_datas(online_dataset.filter(upper=平均 real_loss))`（只用較好的） |
| 線上資料寫入條件 | 只有 `real_loss < 平均` 才存入 online dataset | 每次模擬都存入 online dataset |
| 代理模型尚未預訓練時的行為 | 先用 dataset 預訓練 SM，**接著繼續往下訓練** | 先用 dataset 預訓練 SM 後 **`exit()` 直接結束**（需再跑一次） |

設定相同的部分：`config.epochs = 1000`、`config.lr = 0.005`、`patience = 10`、scheduler 都是 `AdaptiveCyclicalScheduler`（`T_0=100`、`lr_max=0.005`、`temp_max=4.0`、`patience=25`）。

---

## 5. `MultiConfig` 設定編號一覽

### `train_single.py`

| 編號 | 名稱重點 | 啟用的超參數 |
| --- | --- | --- |
| `1` / `2` | pixel_base_1 / _2 | 預設（無額外 loss） |
| `3` | base_1 + TV | `total_variation_loss=0.01`、`KuoHung='KuoHung-1'` |
| `4` | base_2 + TV | `total_variation_loss=0.01`、`KuoHung='KuoHung-2'` |
| `5` | on_plateau linear | `on_plateau='linear'` |
| `6` | on_plateau peak | `on_plateau='peak'` |
| `7` | linear + tv50 | `total_variation_loss=50`、`on_plateau='linear'` |
| `8` | linear + is100 | `island_suppression_loss=100`、`on_plateau='linear'` |
| `9` | is100 | `island_suppression_loss=100` |
| `10` | is1 | `island_suppression_loss=1` |

### `train_dual.py`

| 編號 | 名稱重點 | 啟用的超參數 |
| --- | --- | --- |
| `1` | oldloss_tv100 | `total_variation_loss=100` |
| `2` | tv1 | `total_variation_loss=1` |
| `3` | oldloss_is100 | `island_suppression=100` ⚠️（見第 8 節，實際不會生效） |
| `4` | intervalloss_tv100 | `total_variation_loss=100` |
| `5` | intervalloss_isrelu9 | `island_suppression=100`、`relu=0.9` ⚠️（兩者皆不生效） |
| `6` / `7` / `8` | intervalloss / dlfavg 系列 | 預設（無額外 loss 權重） |
| `9` | sc0.0005 | `spectral_connectivity_loss=0.0005` |

### 設定 key 意義

| key | 作用 |
| --- | --- |
| `name` | 結果資料夾與實驗名稱模板，可用 `{device}`、`{hash_id}`。 |
| `total_variation_loss` | Total Variation 正則化權重（讓 pattern 平滑、減少破碎）。 |
| `island_suppression_loss` | 孤島抑制 loss 權重（懲罰孤立金屬塊）。**注意 dual 寫成 `island_suppression` 會失效。** |
| `spectral_connectivity_loss` | 頻譜連通性 loss 權重。 |
| `gap_closing_loss` | 縫隙閉合 loss 權重（兩支腳本 loss 都有讀，但設定表內未列）。 |
| `on_plateau` | scheduler 遇到 plateau 的行為，`'linear'` 或 `'peak'`。 |
| `KuoHung` | （single 限定）預訓練時要載入的 KuoHung 參考 pattern。 |

---

## 6. 訓練主迴圈（每個 epoch 做的事）

兩支腳本的 `while epoch < config.epochs + 1` 迴圈邏輯一致：

1. `simulator.start(epoch)`：在 HFSS 開一個新專案。
2. **early-stop 判斷**：若 `real_loss` 連續 `patience(=10)` 次沒進步 → **rollback** 回最佳 epoch 的 GEN 權重，並用 online dataset 再訓練代理模型。
3. GEN 由目標響應生成 pattern，加上固定金屬塊（single：`+lower`；dual：`+lower+upper`）。
4. **去重**：若該 pattern 之前模擬過（在 `patch_pattern_buf`）就直接取舊結果（`jump += 1`），否則丟 HFSS 模擬，得 `real_loss`，並即時訓練 + 存代理模型。
5. 更新最小 loss / 最佳 epoch，存 `config`。
6. 用代理模型對 pattern 預測響應，組 `loss = 響應誤差 + 各正則化項`，`loss.backward()` 更新 GEN（`fake_loss`）。
7. 存 GEN、`simulator.end()` 關專案、`simulator.clean()`（預設只保留最近 5 個 HFSS 專案）。
8. 畫圖：每個 epoch 輸出一張 2×3 的結果圖到 `pic/`（pattern、S 參數對目標、scheduler、loss 曲線、饋入可達率/耗時）。

跑完 1000 epoch 後呼叫 `Complete(...)`，會**寄送 email 通知**（`send_email=True`）。

---

## 7. 輸出結果結構

結果在 `ROOTDIR\result\<實驗名稱>\`：

```
<實驗名稱>/
├─ <實驗名稱>.log      # loguru 日誌
├─ config.*           # 本次所有設定快照
├─ temp*              # Record("temp")，逐 epoch 的 loss / pattern buffer / 斷點資訊
├─ pic/               # 每個 epoch 的結果圖
├─ checkpoint/        # GEN / 代理模型 權重
├─ HFSS/              # HFSS 專案檔
└─ online/            # 線上累積的 (pattern, 模擬結果) 資料集
```

---

## 8. 斷點續跑

- `get_result_path(...)` 回傳的 `CONTINUE_RUN` 來自「結果資料夾是否已存在」。
- 若用**相同設定編號**再次執行（hash_id 相同 → 資料夾名相同），且 `temp` 內有 `epoch`，會自動：
  - `generator.change(TEMP('epoch'), load=True)` 載回 GEN 權重，
  - `smodel.load()` 載回代理模型，
  - 從上次的 `epoch` 繼續。
- 想重新從頭跑，就改設定的 `name` 或清掉對應 result 資料夾。

---

## 9. 注意事項 / 已知陷阱

1. **必帶設定編號**：`python train_xxx.py <編號>`，否則 `MultiConfig` 讀 `sys.argv[1]` 會 `IndexError`。
2. **dual 第一次無預訓練模型會 `exit()`**：`train_dual.py` 在沒有 `patch_dual.pth` 時，預訓練完代理模型就 `exit()` 結束。需**再執行第二次**才會進入正式訓練；`train_single.py` 則會直接接著跑。
3. **dual 的 `island_suppression` / `relu` 設定不會生效**：loss 計算讀的是 `MULTICONFIG("island_suppression_loss", 0)`，但 dual 設定 `3`、`5` 寫的 key 是 `island_suppression`（少了 `_loss`），`relu` 也沒有任何地方讀取 → 這兩個值實際被忽略、退回預設 0。若要啟用孤島抑制，需把 key 改成 `island_suppression_loss`。
4. **硬編碼的網路磁碟與帳密**：`connect_network_drive` 內含實驗室 IP 與帳密，且所有路徑指向 `T:`。換環境需自行調整 `ROOTDIR` / `DATASET_PATH` 與掛載設定。
5. **強依賴 HFSS**：沒有 ANSYS HFSS（或 COM 介面被佔用/當機）時 `simulator.open()` / `start()` 會失敗；`clean()` 會自動關閉舊專案、保留最近 5 個。
6. **設定名稱與內容偶有不一致**：例如 single 編號 `8` 名稱寫 `tv100` 但實際設的是 `island_suppression_loss=100`。以「啟用的超參數」欄為準，不要只看名稱。
7. **完成會寄信**：訓練結束 `Complete(..., send_email=True)` 會發送 email，跑前確認 email 設定正確或可接受。

---

## 10. 快速範例

```bash
# 啟用環境
conda activate ant

# 單埠，基礎設定
python train_single.py 1

# 單埠，加 TV 正則化（base_1）
python train_single.py 3

# 雙埠，第一次跑（會先預訓練代理模型後 exit，需再跑一次）
python train_dual.py 1
# 第二次：正式進入訓練主迴圈
python train_dual.py 1

# 雙埠，加頻譜連通性 loss
python train_dual.py 9
```
