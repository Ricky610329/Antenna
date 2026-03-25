# CLAUDE.md

本檔案提供 Claude Code (claude.ai/code) 在此專案中的操作指引。

## 專案概述

Antenna 是一個基於 PyTorch 的研究專案，用於透過深度學習優化**微帶貼片天線 (microstrip patch antenna)** 與**可重構智慧表面 (Reconfigurable Intelligent Surface, RIS)**。整合 HFSS 電磁模擬器（透過 Windows COM 自動化）進行模擬迴圈訓練，並提供代理模型 (surrogate model) 來取代高成本的 HFSS 模擬。

程式碼中的註解與文件皆使用繁體中文。

## 分支

主要開發在 **`GAN`** 分支上進行，`main` 為穩定分支。

## 環境建置

```bash
conda create --name antenna python=3.11
conda activate antenna
pip install -r requirements.txt
```

需要 Windows 環境（pywin32 用於 HFSS COM 介面）。可能需要連接網路磁碟 `T:` 以存取資料集（`\\140.123.106.219\temp`）。

## 執行方式

訓練腳本為頂層的 `train_*.py` 檔案，每個對應不同的訓練變體：
```bash
python train_single.py    # 單埠天線
python train_dual.py      # 雙埠天線
python train_ris.py       # RIS
```

網頁應用程式（Flask 結果檢視器）：
```bash
python application/run_waitress.py          # 正式環境
python application/app.py -dev              # 開發環境
```

## 架構

### 核心類別 (`antenna/__init__.py`)

- **`AntennaPattern`** — 表示天線設計的 2D 二值像素圖。支援合併、模擬、二值化、突變。具有類別層級狀態：使用前需呼叫 `setDefaultCoordinate()` 與 `register_simulator()`。
- **`AntennaResponse`** — 表示頻域響應曲線（S11、Gain、S21 等）。透過靜態方法（`registerLabels()`、`registerLossHook()`、`registerTargetResponse()`）註冊標籤、損失函數鉤子與目標響應。
- **`TargetResponse`** / **`MultiResponses`** — 目標/期望的響應曲線，支援自訂損失函數註冊。

### 訓練基礎架構 (`antenna/models.py`, `antenna/smodels.py`)

- **`Models`** — 高度泛型的訓練封裝器（`Models[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams]`）。管理模型、優化器、排程器、損失函數，以及用於檢查點的 `Record`。
- **`SurrogateModel`** — 繼承 `Models`，用於 HFSS 代理模型建模，可在不執行完整 HFSS 模擬的情況下訓練。

### 模擬器整合 (`antenna/patch/`, `antenna/ris/`)

- **`PatchSimulator`**（基底類別）、**`SinglePortSimulator`**、**`DualPortSimulator`** — HFSS COM 介面，僅限 Windows。
- **`RISSimulator`** — RIS 圖案模擬。

### 工具模組 (`antenna/utils/`)

- **`Config` / `config`** — 全域單例設定（device、NAME、RESULT_PATH、epochs、lr 等）。支援動態屬性設定。
- **`Record`** — 模型與訓練歷史的持久化檢查點。
- **`Figure`** — Matplotlib 封裝器，支援多子圖與自動儲存。
- **`Path`** — 擴充 `pathlib.Path`，新增 `not_exist_create()`、`del_from_glob()`、`manage_file_count()`。
- **`DataManager`** / **`size_converter`**（`antenna/utils/data.py`）— 資料集管理與彈性張量重塑。
- **`connect_network_drive`**、**`Email`**（`antenna/utils/web.py`）— 網路磁碟與通知工具。

### 損失函數

分散於多個模組中：
- `antenna/__init__.py` — `total_variation_loss()`、`island_suppression_loss()`
- `antenna/patch/__init__.py` — `custom_loss_r()`、`custom_loss_g()`、`custom_loss_minmax()`、`interval_loss()`
- `antenna/ris/__init__.py` — `custom_loss()`
- `antenna/functions.py` — `custom_loss_interval()`、`GapClosingLoss`、`SpectralConnectivityLoss`

### 典型訓練腳本模式

```python
from antenna import *
from antenna.utils import *
from antenna.models import Models
from antenna.patch import SinglePortSimulator

connect_default_drive()
RESULT_PATH, CONTINUE_RUN = get_result_path('[...][{device}] ...', rootdir=ROOTDIR)

AntennaPattern.setDefaultCoordinate((0, n, 0, n))
AntennaPattern.register_simulator(simulator)
AntennaResponse.registerLabels('response', ..., x='...')
```

## 重要路徑

- `ROOTDIR`：`T:\碩二_吳維文's\Patch Antenna\Experiment`（網路磁碟）
- `DATASET_PATH`：`ROOTDIR/dataset`
- 結果儲存於：`result/<run-name>/`
- 實驗追蹤：Weights & Biases (`wandb`)

## 測試與檢查

目前沒有正式的測試框架、Linter 設定或 CI/CD 流程。
