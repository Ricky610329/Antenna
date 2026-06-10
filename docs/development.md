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
| `tests/conftest.py` | session 共用：單埠全域註冊 + `golden` fixture | — |
| `tests/test_characterization.py` | 純函式行為（STE 二值化、正則化、loss hook、R_feed） | 改公式時的數值漂移 |
| `tests/test_baseline_loop.py` | `run_training` 整圈（Mock 模擬器 + 固定種子，跑 6 epochs） | 訓練迴圈重構造成行為改變 |
| `tests/test_config.py` | YAML 解析、port 結構解析、錯誤拒絕 | config 欄位破壞 |
| `tests/test_model_loading.py` | registry 建模、`prepare_models` 載入分支（MagicMock） | 模型建構/載入邏輯破壞 |

CI（GitHub Actions，`.github/workflows/tests.yml`）在 push / PR 時自動跑全部測試（windows runner，因 import 鏈含 pywin32）。結果看 repo 的 **Actions** 頁籤。

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
- `Record`/`DataManager`/`Data` 的**序列化格式不可破壞**：`application/app.py` 還在讀舊結果。

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

## 4.5 資料層（新舊雙軌）

| | 新格式 `SampleStore`（未來標準） | 舊 `DataManager`（保留給學長 code） |
| --- | --- | --- |
| 形式 | **一筆一檔**：`<資料夾>/<內容hash>.pt` | 整個資料集一個 pickle（`<name>.data`） |
| append | 寫一個 ~3KB 小檔，O(1) | **全量重寫**整個 pkl（NAS 上很慢） |
| 去重 | 檔名 = 內容 hash，存在即重複 | 自維護指紋集 |
| 損壞 | 壞一檔損一筆 | 壞一檔全滅（有 backup 緩解） |

- **online**（訓練中收集）已用新格式；**offline**（NAS 舊資料集）過渡期仍走 DataManager。
- `train.py` 的 `load_dataset()` 自動偵測：資料夾→新格式、否則→舊 pkl。
- 正式轉換：`python -m script.convert_dataset patch_single_mirror`（**先與維護者確認再跑**；
  不刪舊檔，轉完自動切新格式）。app.py 的 dataset 瀏覽頁只認舊格式。

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
- `Record` 與舊 `temp.record` 完全保留（app.py 歷史檔案館用）。

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
