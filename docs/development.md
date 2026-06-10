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
pytest tests/ -q        # 在 repo 根目錄執行（沒有 pytest.ini，靠 cwd 解析 antenna 套件）
```

> 注意：**一定要在 repo 根目錄跑**。在別處（含 git worktree）跑會 import 到錯的 `antenna/`。

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

### Golden 的參考環境：torch 2.7.1（重要）

golden 數值**綁定 torch 版本**：換版本會造成浮點漂移（SC loss 走特徵值分解最敏感，2.7→2.12 漂 Δ≈1.0；迴圈 golden 會逐 epoch 放大）。因此：

- CI 釘住 `torch==2.7.1`（= 開發機 `ant` env 的版本），**不要**為了漂移放寬 `tol`（會讓真 bug 躲過去）。
- **要升級 torch 時**的正確順序：升開發機 → 刪三個 golden 檔重生基準 → 同步改 `.github/workflows/tests.yml` 的釘版 → 一個 commit 一起進，message 說明。

### 已知的非直覺行為（別「修」它們）

- `island_suppression_loss` 對全金屬 pattern **不是 0**（≈0.0937）：`avg_pool2d` 零填充所致，是既有行為。
- dual 的 target 註冊順序**必須是 S11→S22→S21**（`PORT_SPECS.register_order`）：決定 GEN 輸入向量排列，動了 golden 就會漂。
- `Record`/`DataManager`/`Data` 的**序列化格式不可破壞**：`application/app.py` 還在讀舊結果。

## 4. 怎麼加新東西

### 新實驗（最常見）
複製一個 `configs/*.yaml`、改 `name` 和參數即可，**不用改 code**。欄位說明見 [`training.md`](training.md)。

### 新的 GEN / SM 架構
1. 在 `antenna/models.py`（GEN）或 `antenna/smodels.py`（SM）寫新的 `nn.Module`。
2. 在 `antenna/training.py` 的 `GENERATOR_REGISTRY` / `SURROGATE_REGISTRY` 註冊一個 `type` 名稱。
3. config 裡 `generator.type` / `surrogate.type` 指定它。
4. 在 `tests/test_model_loading.py` 加架構測試（照現有測試的樣子，斷層數/寬度即可）。

### 新的 port 模式（罕見）
在 `antenna/training.py` 的 `PORT_SPECS` 加一組（labels、register_order、feeds、make_r_feed），並準備對應模擬器。

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
