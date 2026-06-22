# 開發者工作流（development.md）

> 給**要改程式碼的人**看的：測試怎麼跑、golden 怎麼維護、怎麼加新東西、branch 策略。
> 「怎麼跑訓練」見 [`quickstart.md`](quickstart.md)；「config 怎麼寫」見 [`training.md`](training.md)；系統原理見 [`architecture.html`](architecture.html)。

## 1. 環境

| 機器 | conda env | 用途 |
| --- | --- | --- |
| 開發機 | `ant`（miniforge） | 改 code、跑測試 |
| 正式機 | `patch` | 跑真實訓練（HFSS + NAS） |

需 Python 3.11+。測試不需要 HFSS、不掛 NAS、不需要 GPU。

```bash
conda activate ant
python -m pytest tests/ -q   # 在 repo 根目錄執行
```

> 注意：**一定要 `python -m pytest`、且在 repo 根目錄跑**。`python -m` 會把 cwd 加進
> `sys.path`（這是找到 `antenna` 套件的唯一機制，沒有 pytest.ini）；裸 `pytest` 會直接
> `ModuleNotFoundError`。在別處（含 git worktree）跑則會 import 到錯的 `antenna/`。

## 2. 測試結構

| 檔案 | 測什麼 | 防什麼 |
| --- | --- | --- |
| `tests/conftest.py` | session 共用：安裝單埠響應規格 + `golden` fixture | — |
| `tests/test_characterization.py` | 純函式行為（STE 二值化、正則化、loss hook、R_feed） | 改公式時的數值漂移 |
| `tests/test_baseline_loop.py` | `run_training` 整圈（Mock 模擬器 + 固定種子，跑 6 epochs） | 訓練迴圈重構造成行為改變 |
| `tests/test_config.py` | YAML 解析、port 解析、區段鍵白名單、seed、ACP 參數流 | config 欄位破壞 / 鍵名錯字 |
| `tests/test_model_loading.py` | zoo 建模、`prepare_models` 載入分支（MagicMock） | 模型建構/載入邏輯破壞 |
| `tests/test_models_shell.py` | `Models` 外殼行為合約（換檔/存讀/title 把關/NaN 防護） | 外殼重構造成行為改變 |
| `tests/test_response_spec.py` | 響應規格實例（自包含/原子安裝/dual 順序雙軌） | 全域狀態回歸 |
| `tests/test_sample_store.py` | SampleStore（一筆一檔/去重/持久化） | 資料層破壞 |
| `tests/test_runstate.py` | RunState（csv 續跑/去重快取/best_epoch） | 訓練狀態破壞 |
| `tests/test_monitor.py` | TB 監控（降級/記錄內容/summary.png） | 監控破壞訓練 |
| `tests/test_radiation_simulator.py` | `SinglePortRadSimulator` 結構（子類/不開 COM/配方）— 無 HFSS | 方向圖萃取器誤改、誤碰既有單埠流程 |
| `tests/test_beam_coverage_loss.py` | `beam_coverage_loss`（相對 boresight 平頂+中央峰；floor/boresight/窗/單邊/可微）— 無 HFSS | 方向圖 loss 行為退化 |
| `tests/test_radiation_integration.py` | 方向圖接閉迴路（rad head 形狀/freq 參數零擾動/config 白名單/mock 端到端）— 無 HFSS | 方向圖整合誤碰 S11/Gain 或 golden |

CI（GitHub Actions，`.github/workflows/tests.yml`）在 push / PR 時自動跑：**pyflakes 守門**（擋 undefined name）→ 全部測試（windows runner，因 import 鏈含 pywin32）。結果看 repo 的 **Actions** 頁籤。

## 3. Golden 測試怎麼維護（最重要）

採 **approval testing**：第一次跑會把數值寫進基準檔，之後每次跑都比對（容差 `1e-4`）。

| 基準檔 | 來源 |
| --- | --- |
| `tests/golden.json` | 純函式數值（characterization） |
| `tests/golden_loop.json` | 單埠 mock 迴圈逐 epoch 數列 |
| `tests/golden_loop_dual.json` | 雙埠 mock 迴圈逐 epoch 數列 |

### 測試紅了（golden drift）怎麼辦

1. **先假設是 bug**。看 drift 訊息裡的鍵名與 Δ，找出哪個改動影響了數值。
2. 判斷：
   - **意外改變** → 修 code，不動 golden。
   - **刻意改變**（例如換公式、修 bug 後數值本來就該動）→ 更新基準：
     ```bash
     # 刪掉對應的 golden 檔，重跑一次（會自動以「目前行為」重建基準）
     rm tests/golden_loop.json
     pytest tests/test_baseline_loop.py -q
     ```
     並在 **commit message 裡寫明為什麼更新 golden**（這是行為改變的紀錄點）。
3. 經驗法則：**重構（不改行為）絕不該動 golden**；golden 動了 = 行為變了，要嘛是 bug、要嘛要說清楚。

### Golden 的參考環境：torch 2.7.1 + 開發機硬體（重要）

golden 數值**綁定 torch 版本與 CPU**：
- 換 torch 版本會漂（SC loss 走特徵值分解最敏感，2.7→2.12 漂 Δ≈1.0）→ CI 釘住 `torch==2.7.1`（= 開發機 `ant` env 版本）。
- 換 CPU 也會漂（SIMD/MKL kernel 路徑不同；GitHub runner 實測相對漂移 ~0.4%）→ 無法跨硬體精確比對。

因此採**雙軌容差**（`conftest.golden_tol`）：

| 環境 | 容差 | 角色 |
| --- | --- | --- |
| 本機（開發機） | 絕對 `1e-4` | **精檢**，數值真相的來源；0.1% 級的細微 bug 在這裡抓 |
| CI（`CI=true` 自動判斷） | 相對 `1%`；**`r_feed` 只檢查值域** | **粗檢**：import / config / 單元測試全部嚴格，golden 擋大方向 |

> `r_feed` 為何 CI 不比數值：它是離散指標（可達金屬比例，由連通分量決定）。跨硬體浮點
> 漂移可能讓 STE 二值化翻轉個別像素 → 整塊連通性改變 → r_feed **跳階** >1%，任何容差
> 都會偶爾爆（flaky）。本機 golden 仍以 1e-4 精檢 r_feed。

> 誠實的代價：0.2% 級的細微行為差異（如當年 dual register_order bug）CI 抓不到，
> **commit 前本機跑 `python -m pytest tests/ -q` 仍是必要紀律**。

**要升級 torch 時**：升開發機 → 刪三個 golden 檔重生基準 → 同步改 `.github/workflows/tests.yml` 的釘版 → 一個 commit 一起進，message 說明。

### 已知的非直覺行為（別「修」它們）

- `island_suppression_loss` 對全金屬 pattern **不是 0**（≈0.0937）：`avg_pool2d` 零填充所致，是既有行為。
- dual 的 target 註冊順序**必須是 S11→S22→S21**（`PORT_SPECS.register_order`）：決定 GEN 輸入向量排列，動了 golden 就會漂。
- `DataManager`/`Data`/`Record` 的 pickle **payload 格式不可破壞**：舊 `.dataset`/`.record`/checkpoint 仍要能讀（這些類別都只 pickle 純資料、不含自身，故把 Data/DataManager 搬到 `antenna/legacy/` 不影響舊檔）。

## 4. 怎麼加新東西

### 新實驗（最常見）
複製一個 `configs/*.yaml`、改 `name` 和參數即可，**不用改 code**。欄位說明見 [`training.md`](training.md)。

### 新的 GEN / SM 架構（模型動物園）
1. 寫一個純 `nn.Module`。**GEN 約定**：建構簽名 `(in_dim, out_dim, **參數)`、forward 是
   spec 向量 → logits，**不做** STE 二值化、不碰 tau（那是訓練管線的固定一步，tau 由 ACP 控制）。
2. 在 **`antenna/zoo.py`** 的 `GENERATORS` / `SURROGATES` 加一行。
3. config 寫名字：`generator: <名字>`。
4. 在 `tests/test_model_loading.py` 加架構測試（斷層數/寬度即可）。

只調層數/寬度不用走上面流程：config 寫 `{name: sigmoid, hidden: [...]}` 即可。

### 新的 port 模式（罕見）
在 `antenna/training.py` 的 `PORT_SPECS` 加一組（labels、register_order、feeds、make_r_feed），並準備對應模擬器。

### 型別註解慣例
**型別註解＝輕量文件**（不做靜態型別檢查）：簡單註解（`x: Tensor`、`-> dict`）歡迎；
**TypeVar / Generic / ParamSpec / @overload 一律不用**（2026-06 已全數清除，types.py 258→47 行）。
CI 只用 pyflakes 擋 undefined name。

## 4.5 資料層（新舊雙軌）

| | 新格式 `SampleStore`（標準） | 舊 `DataManager`（→ `antenna/legacy/`，只讀舊 `.dataset`） |
| --- | --- | --- |
| 形式 | **一筆一檔**：`<資料夾>/<內容hash>.pt` | 整個資料集一個 pickle（`<name>.data`） |
| append | 寫一個 ~3KB 小檔，O(1) | **全量重寫**整個 pkl（NAS 上很慢） |
| 去重 | 檔名 = 內容 hash，存在即重複 | 自維護指紋集 |
| 損壞 | 壞一檔損一筆 | 壞一檔全滅（有 backup 緩解） |

- **online**（訓練收集）與 **offline**（SM 預訓練）都已用新格式：configs 的 `offline_dataset` 指向自有 NAS 收割的 `harvest_single`/`harvest_dual`（SampleStore）。
- `train.py` 的 `load_dataset()` 自動偵測：資料夾→SampleStore；否則 lazy 從 `antenna.legacy` 載 DataManager 讀舊 pkl。
- 舊 `Data`/`DataManager` 已隔離到 **`antenna/legacy/`**，`antenna/` 核心零 legacy 依賴。`script/harvest_legacy.py` 已把學長 result/ 的真實模擬樣本收割成 harvest_single/dual（故 `convert_dataset` 功成身退）。app.py 的 `/dataset` 瀏覽頁已改讀 SampleStore。

### 結果夾即資料庫（訓練狀態，新格式）

訓練狀態也去 pickle 化（`RunState`，`antenna/utils/runstate.py`），取代舊 `temp.record`
（那是每 epoch 全量重寫的單一 pickle——最後一個 O(n²) NAS 寫入者）：

```
result/[實驗]/
  config.yaml      # 設定快照 (原文)
  status.json      # 運行管理心跳：state(running/finished/crashed)/機器/epoch/updated_at
  metrics.csv      # 純量時序，一 epoch 一行 → pd.read_csv 三行就能 pyplot 自訂畫圖
  patterns/        # 每筆模擬過的 (pattern, response, loss)，hash 即檔名 = 去重快取
  checkpoint/  online/  tb/  summary.png
```

- **監控 vs 運行管理是兩件事**：TB 看曲線/圖（監控）；`status.json` 回答「誰還活著、跑到哪、在哪台機器」（管理）。
- 斷點續跑改讀 `metrics.csv`：升級前還在半路的舊 run（只有 temp.record）續跑不相容——跑完再升級或重來。
- `RunState` 在訓練路徑取代 `temp.record`；但 `Record` 類別本身是**核心**（ACP 排程器記 lr/tau、每個 checkpoint 存 `record.state_dict()`），留在 `antenna/utils/record.py`。

## 4.6 方向圖（radiation pattern）→ loss（Stage 0 萃取已驗證、Stage 1 loss 已實作）

S11 / 正向 Gain 都是「vs **頻率**」；閉迴路要再納入**方向圖**＝gain vs **角度**（固定頻率）。
分階段、刻意與既有 S11/Gain 路徑解耦、零侵入：
**Stage 0** 把資料抓出來（已在正式機驗證 OK，2026-06-19）、
**Stage 1** 寫 `beam_coverage_loss`（已實作 + 單元測試）、
**Stage 2** SM 多 rad head 並掛進訓練（**已實作**，`radiation:` 區段 + `single_sc_rad.yaml`；golden 零漂移已驗）、
**Stage 3** 補方向圖離線資料解冷啟動（待做）。

### 現況：`SinglePortRadSimulator`（`antenna/patch/patch_simulator/single_port_rad.py`）

繼承 `SinglePortSimulator`，沿用其全部建模 / 求解 / S11&Gain 流程；只在求解**後**多匯出方向圖。

- **配方**（照使用者 HFSS 操作截圖）：`dB(GainTotal)` vs **Theta**、固定 **Freq=28GHz**、
  **Phi∈{0°,90°}**（E-plane / H-plane 兩主平面切面）、解 `Setup1 : LastAdaptive`、
  沿用父類別已建的 3D 無限球面（theta/phi step 2° → 181 點）。
- **三道安全鎖**（對應「不准搞壞」）：
  1. `__call__` 回傳值與父類別**完全相同**（只有 S11/Gain）——方向圖**不**塞進回傳 dict
     （否則 criterion 依 `spec.labels` 用 `zip` 對齊會**靜默錯位**，見 `antenna/response.py`）。
  2. 父類別程式碼**一行未改**，方向圖只在求解後「多做」報表。
  3. 萃取整段包 `try/except`：報表失敗也只記 warning、**絕不**拖垮已取得的 S11/Gain。
- 資料落點：`<record>/HFSS/result/NN_patch_RadGain_{num}_phi{0,90}.csv` ＋ 暫存 `self.last_radiation`。
- **尚未接進 `build_simulator`**：正常 `train.py` 仍用舊單埠模擬器（不收方向圖）；目前只能靠下方腳本單獨跑。

### 怎麼在正式機驗證「有這個資料」

```bash
python -m script.verify_radiation                                       # configs/single_base.yaml + 置中方塊圖樣
python -m script.verify_radiation --config configs/single_peak.yaml --pattern xxx.pt
```

跑完看 `✅ 通過`，並到 `<out>/HFSS/result/` 看那兩個 CSV。本機（無 HFSS）只能跑結構測試
`tests/test_radiation_simulator.py`（不開 COM）；**實際萃取只能正式機驗**。

### 設計決定與路線（2026-06-19 定案；取代先前「統一 response 向量」舊構想）

- **學長規格 ≠ 我們規格**：學長要求「phi=0°/90° 在 theta∈±55° 內、gain 不得比 0°（boresight）低 >3dB」。
  這是**「他們的」要求**，我們當 v1 起步、之後可調，不寫死。
- **loss（Stage 1 已實作）**：`beam_coverage_loss`（`antenna/losses.py`，測試 `tests/test_beam_coverage_loss.py`）。
  **相對** boresight（錨在預測 `G0 = rad_pred[θ≈0]`），兩個單邊 relu 項：① 逼窗內 ≥ `G0−floor_db`、② 逼 0° 最高；
  窗內超過 floor 後不再罰＝「越高越好」。**分工**：boresight 絕對增益由既有 `Gain` target（method='high'）負責，
  本函式只塑形 → 故方向圖**不需**在 `targets:` 寫梯形曲線（相對、自我歸一化）。
- **SM 走 multi-head（Stage 2，已實作）**：`HFSSNet` 的 `fc_patch`（freq）**原封不動**，rad head 從 `fc_patch[:-1]`（＝末層 Linear 之前的 64 維共用 backbone）分叉 → `forward_rad`。
  - golden 安全的關鍵：`head_rad` 在 `fc_patch` 之後才建立，從零建構時 freq 參數的 RNG 抽取序列完全不變（測試 `test_rad_head_does_not_perturb_freq_params` 直接驗證 fc_patch 參數逐一相同）；`fc_patch` 名稱不動 → 舊 `sm.pth` 零 remap。
  - **rad head ＝ 平滑 cosine 基底頭（解鋸齒）**：`head_rad` 不直接吐 `n_theta` 個獨立值（裸 `Linear` 無平滑先驗 + 凍 trunk 下擬不到收斂 → 預測鋸齒），改吐 K=`n_basis`（預設 16）個 cosine 係數，乘固定基底 `B(K,n_theta)` 展開：`pred = coeffs @ B`。輸出 **band-limited → 結構上必平滑**，且只擬 K 個數收斂快。`B` 是不可訓 buffer、用 `torch.cos/arange` 建（不吃 RNG → golden 零漂移）；`set_rad_theta` 依實際 HFSS θ 網格（整 run 固定）逐欄重建 → 對位正確、HFSS 匯出序未排序也 OK。測試 `test_rad_head_basis_is_band_limited`（K=1→沿 θ 恆為常數）。⚠ head 形狀變了 → 舊 rad-run 的 `sm.pth` 不能續跑（freq-only checkpoint 不受影響）。
  - **trunk 預設凍（`freeze_trunk: true`，已實作）**：`train_one_data_rad` 凍住 `head_rad` 以外的參數 → 方向圖頭只更新自己、不污染 S11/Gain backbone（隨機 rad 頭曾把 trunk 帶歪→爆 NaN）。設 `false` 才放梯度回 backbone（兩者互相牽動）。測試 `test_rad_freeze_trunk_protects_backbone` / `test_rad_unfrozen_updates_trunk`。
- **訓練接法（`run_training`，全程 `rad_on` 閘住）**：① `build_simulator` 開啟時換 `SinglePortRadSimulator`；② 每筆真實模擬後讀 `simulator.last_radiation`、`train_one_data_rad` 訓方向圖頭；③ 過 `warmup_epochs` 後把 `beam_coverage_loss` 加進 GEN loss。`rad_on=False` → 全部不執行（golden 零漂移）。
- **監控（TensorBoard，比照 pattern）**：on_epoch 多帶 `rad_loss` 純量 + `radiation` 快照（theta/pred/real/window/floor）。`monitor.py` 記 `loss/rad_loss`（Scalars）＋ `radiation/gain_vs_angle`（Images，有 epoch 滑桿：phi0/phi90 實線=HFSS 真實、虛線=SM 預測，疊 ±window 與 G(0°)−floor_db 線）；`summary.png` 也多一格方向圖。無方向圖實驗這些鍵不存在 → 監控行為不變。
- **冷啟動解法（Stage 3，待做）**：`harvest_single/dual` **只有 S11/Gain、沒有方向圖** → rad head 線上從零學。做法：按 loss 挑好 pattern
  （per-run `online` SampleStore 已是）→ **只把這批**丟 `SinglePortRadSimulator` 補 phi0/phi90 標籤
  → 存成 `DATASET_PATH/harvest_single_rad`（SampleStore）→ pretrain rad head。比重抽便宜，配 warmup 閘門。
- **config（Stage 2，已實作）**：`radiation:{enable,weight,window_deg,floor_db,boresight_weight,warmup_epochs,n_theta,n_basis,freeze_trunk,sm_max_epoch,sm_min_loss}` 區段，預設 off → 既有 config/golden 零影響。範例見 `configs/single_sc_rad.yaml`。
  - rad 版 SM 多了 head → 舊 `old_sm.pth` 以 `pre_load_model(strict=False)` 部分載入暖啟動（fc_patch+optimizer 對位灌入、head_rad 維持隨機；測試 `test_pre_load_partial_warm_starts_rad_sm`）。
- **theta 解析度**：目前 2°（父類別 3D 球面）；要 1°（`Elevation` 球）是小改動。
- **`LastAdaptive`**：現行 `Setup1:LastAdaptive` 已驗證取得到遠場（Stage 0 OK）；若哪天取不到改 `"Setup1 : Sweep"`。
- **dual 未做**：先把 single 跑順、驗證對了再複製到 `dual_port.py`。

> 設計全圖見 [`pipeline_loss_acp.html`](pipeline_loss_acp.html)（pipeline + loss 結構 + ACP + 方向圖接法）。

## 5. Branch / commit 慣例

- 開發都在 **`GAN`** 分支；測試全綠後 `main` 以 fast-forward 同步：
  ```bash
  git push origin GAN
  git checkout main && git merge --ff-only GAN && git push origin main
  git checkout GAN
  ```
- Commit message 用繁體中文，首行 `type: 摘要`（feat / refactor / docs / fix / test）。
- 行為有變（golden 更新）時，在 message 裡寫明原因與驗證方式。

## 6. 開發環境陷阱（踩過的坑）

- `conda run -n ant python -c "<多行>"` 會掛掉 → 直接用完整路徑 `~/miniforge3/envs/ant/python.exe`。
- 終端機印中文亂碼 → 設 `PYTHONIOENCODING=utf-8`。
- `.gitignore` **不支援行內註解**（`pattern  # 註解` 會把整行當 pattern），註解要獨立成行。
- `archive/`、`tmp/`、`docs/Paper.pdf` 是 gitignored 的本機資料，不要 commit。
