# CLAUDE.md

本檔案提供 Claude Code (claude.ai/code) 在此專案中的操作指引。

## 專案概述

Antenna 是一個基於 PyTorch 的研究專案，用於透過深度學習優化**微帶貼片天線 (microstrip patch antenna)** 與**可重構智慧表面 (Reconfigurable Intelligent Surface, RIS)**。整合 HFSS 電磁模擬器（透過 Windows COM 自動化）進行模擬迴圈訓練，並提供代理模型 (surrogate model) 來取代高成本的 HFSS 模擬。

程式碼中的註解與文件皆使用繁體中文。

## 分支

主要開發在 **`GAN`** 分支上進行，`main` 為穩定分支。`modernize` 為現代化重構分支。

## 環境建置

```bash
conda create --name antenna python=3.11
conda activate antenna
pip install -e ".[dev]"
```

需要 Windows 環境（pywin32 用於 HFSS COM 介面）。網路磁碟帳密透過環境變數設定（參考 `.env.example`）。

## 執行方式

### Hydra CLI（新）
```bash
python -m antenna train +experiment=train_single
python -m antenna train +experiment=train_dual epochs=2000
python -m antenna train +experiment=train_ris environment.device=cuda:0
```

### 原始訓練腳本（仍可用）
```bash
python train_single.py
python train_dual.py
python train_ris.py
```

### 網頁應用程式
```bash
python application/run_waitress.py          # 正式環境
python application/app.py -dev              # 開發環境
```

## 開發工具

```bash
pre-commit install        # 安裝 pre-commit hooks
ruff check .              # Lint
ruff format .             # 格式化
pytest                    # 測試
```

## 架構

### 套件結構
```
antenna/
├── core/           # 核心類別（AntennaPattern, AntennaResponse）
├── models/         # 生成器與訓練封裝
│   ├── base.py         # Models 泛型訓練封裝器
│   ├── generators/     # SigmoidGEN, GumbelSigmoidGEN, SPGEN, CVAE, MirrorCVAE
│   ├── surrogates/     # 代理模型（SurrogateModel）
│   └── autograd/       # 自訂 autograd 函數
├── simulators/     # PatchSimulator, SinglePort, DualPort, RIS（re-export hub）
├── losses/         # 損失函數
│   ├── patch_losses.py     # custom_loss_r/g/minmax, interval_loss
│   ├── regularization.py   # TV loss, island suppression, SC/GC loss, FeedReachability
│   └── mirror.py           # mirror, gumbel_sinkhorn
├── schedulers/     # AdaptiveCyclicalScheduler（已解耦 tau callback）
├── training/       # Trainer class（開發中）
├── configs/        # Hydra structured config dataclasses
├── conf/           # Hydra YAML 設定檔
│   ├── config.yaml
│   ├── response/       # single_port, dual_port, ris
│   └── experiment/     # train_single, train_dual, train_ris
└── utils/          # Config, Record, Figure, Path, DataManager
```

### 核心類別

- **`AntennaPattern`** (`core/pattern.py`) — 2D 二值像素圖，支援合併、模擬、二值化。具有 class-level 狀態（`setDefaultCoordinate()`, `register_simulator()`）。
- **`AntennaResponse`** (`core/response.py`) — 頻域響應曲線。透過 `registerLabels()`, `registerTargetResponse()`, `registerLossHook()` 設定。
- **`Models`** (`models/base.py`) — 泛型訓練封裝器，管理模型、優化器、排程器、檢查點。

### 設定管理

使用 **Hydra** 進行設定管理：
- YAML 設定檔位於 `antenna/conf/`
- Structured configs 位於 `antenna/configs/schema.py`
- 支援 CLI override：`python -m antenna train epochs=2000`
- 支援實驗組合：`+experiment=train_single`

### 重要路徑

- 結果儲存於：`result/<run-name>/`
- 實驗追蹤：Weights & Biases (`wandb`)
- 網路磁碟帳密：環境變數（`ANTENNA_NETWORK_DRIVE_*`, `ANTENNA_EMAIL_*`）
