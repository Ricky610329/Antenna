# Antenna — 深度學習輔助的天線最佳化工具

> 以 PyTorch 為核心、整合 HFSS 電磁模擬器的天線最佳化研究專案。
> 涵蓋**微帶貼片天線 (microstrip patch antenna)** 與**可重構智慧表面 (Reconfigurable Intelligent Surface, RIS)**。

---

## 目錄

1. [這份專案是什麼？](#1-這份專案是什麼)
2. [名詞速查](#2-名詞速查)
3. [五分鐘快速上手](#3-五分鐘快速上手)
4. [安裝與環境](#4-安裝與環境)
5. [執行方式](#5-執行方式)
6. [專案結構](#6-專案結構)
7. [核心類別](#7-核心類別)
8. [生成器 (Generators)](#8-生成器-generators)
9. [代理模型 (Surrogates)](#9-代理模型-surrogates)
10. [損失函數 (Losses)](#10-損失函數-losses)
11. [排程器 (Schedulers)](#11-排程器-schedulers)
12. [Trainer 與訓練流程](#12-trainer-與訓練流程)
13. [Hydra 設定系統](#13-hydra-設定系統)
14. [HFSS 整合](#14-hfss-整合)
15. [RIS 模擬](#15-ris-模擬)
16. [資料管理 (DataManager)](#16-資料管理-datamanager)
17. [實驗追蹤 (Record / wandb)](#17-實驗追蹤-record--wandb)
18. [Web 應用程式](#18-web-應用程式)
19. [測試與開發工具](#19-測試與開發工具)
20. [擴充指南](#20-擴充指南)
21. [疑難排解](#21-疑難排解)

---

## 1. 這份專案是什麼？

天線設計長期以來依賴工程師手動調整幾何參數、再送進 HFSS 這類電磁模擬器驗證。這個專案要做的事情很直接：**用深度學習產生天線幾何，並透過可微的代理模型讓最佳化自動進行**。

核心想法：

- 把天線貼片視為一張 **2D 二值像素圖 (pattern)** — 每個像素代表銅箔有無。
- 訓練一個**生成器 (generator)**，輸出這張 pattern。
- 透過**代理模型 (surrogate model)** 把 pattern 映射到頻域響應 (如 S11、Gain)，提供可微的梯度路徑。
- 定期呼叫真正的 HFSS 模擬來校正代理模型、蒐集新資料。
- 使用自訂的**損失函數**把「響應貼近目標」、「pattern 幾何合理（無孤島、可饋電）」等條件轉化成可優化的目標。

實務上支援三種任務：

| 任務類型 | 模擬器 | 典型應用 |
|----------|--------|----------|
| `single_port` | HFSS 單埠貼片 | S11 / Gain 最佳化 |
| `dual_port`   | HFSS 雙埠貼片 | 雙頻 / MIMO |
| `ris`         | 解析式陣列因子 | 反射陣列、波束導向 |

---

## 2. 名詞速查

| 術語 | 意義 |
|------|------|
| **Pattern** | 貼片的 2D 銅箔分佈（例：25×25 的二值圖） |
| **Response** | 頻域響應，如 S11、S21、Gain，通常是數十個頻點的向量 |
| **Generator (GEN)** | 輸出 pattern 的神經網路（SigmoidGEN、CVAE 等） |
| **Surrogate Model (SM)** | 近似 HFSS 的可微神經網路：pattern → response |
| **HFSS** | Ansys 的 3D 電磁模擬器；本專案透過 Windows COM 自動化驅動 |
| **RIS** | Reconfigurable Intelligent Surface，可調相位反射陣列 |
| **Tau (τ)** | Sigmoid/Gumbel-Sigmoid 的溫度參數，控制輸出二值化程度 |

---

## 3. 五分鐘快速上手

以下假設已經完成[安裝](#4-安裝與環境)。

### 最快啟動 — 跑單埠貼片訓練

```bash
conda activate ant
python -m antenna train +experiment=train_single
```

此指令會：

1. 讀取 `antenna/conf/config.yaml` + `antenna/conf/experiment/train_single.yaml`
2. 建立 25×25 的 AntennaPattern 空間
3. 使用 SigmoidGEN 生成器 + OldSM 代理模型
4. 目標響應：S11 在中心頻段壓低至 -10 dB、Gain 達 4 dB
5. 執行 1000 個 epoch（可用 `epochs=100` 覆寫）

執行結果會寫到 `result/<experiment_name>/`，包含 log、checkpoint、pattern 圖、響應曲線。

### 改一下超參數

```bash
# 換裝置
python -m antenna train +experiment=train_single environment.device=cuda:0

# 覆寫 epoch 與學習率
python -m antenna train +experiment=train_single epochs=2000 optimizer.lr=0.001

# 打開特定的正則化
python -m antenna train +experiment=train_single total_variation_loss_weight=0.1

# 只 dump 合成後的設定（不訓練），確認參數對不對
python -m antenna train +experiment=train_single --cfg job
```

### 在 Python 腳本裡用這個套件

```python
import torch
from antenna import AntennaPattern, AntennaResponse, config
from antenna.models.generators.sigmoid_gen import SigmoidGEN

# 設定 pattern 空間
AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
pattern_size = AntennaPattern.size(flatten=True)  # 625

# 目標響應軸
AntennaResponse.registerLabels("S11", "Gain", x="n257")
response_size = AntennaResponse.size(flatten=True)

# 建生成器
model = SigmoidGEN(response_size, pattern_size).to(config.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 一次前向
target_response = torch.randn(response_size, device=config.device)
pattern_logits = model(target_response)
```

---

## 4. 安裝與環境

### 系統需求

- **作業系統**：建議 **Windows**（HFSS COM 介面只能在 Windows 上跑；Linux / macOS 可訓練代理模型或 RIS，但不能跑真實 HFSS）
- **Python**：3.11
- **GPU**：可選，CUDA 12.1+；CPU 也能跑
- **HFSS**：Ansys Electronics Desktop（想跑 patch 模擬時需要）

### 建立環境

```bash
conda create --name ant python=3.11
conda activate ant
pip install -e ".[dev]"
```

`-e` 是 editable install，改 source 不用重裝；`[dev]` 會順便裝 `ruff`、`pre-commit`、`pytest`。

### 設定 Git hooks

```bash
pre-commit install
```

pre-commit 會在每次 commit 前跑 `ruff check --fix` + `ruff-format` + trailing whitespace 等檢查。

### 環境變數（可選）

網路磁碟帳密 / 例外通知 email 等敏感資訊透過環境變數載入，請複製 `.env.example`：

```bash
cp .env.example .env   # 然後填入實際帳密
```

對應變數：

| 變數 | 用途 |
|------|------|
| `ANTENNA_NETWORK_DRIVE_USER` | 網路磁碟帳號 |
| `ANTENNA_NETWORK_DRIVE_PASSWORD` | 網路磁碟密碼 |
| `ANTENNA_EMAIL_USER` | 例外通知寄件 email |
| `ANTENNA_EMAIL_PASSWORD` | 例外通知寄件密碼 |

只有呼叫 `connect_default_drive()` 或 `global_exception_handler()` 時才會用到；開發或測試可忽略。

---

## 5. 執行方式

### 5.1 Hydra CLI（推薦）

```bash
python -m antenna train [override...]
python -m antenna train +experiment=train_single
python -m antenna train +experiment=train_dual
python -m antenna train +experiment=train_ris
python -m antenna train +experiment=train_ris_v7best   # RIS 已知最佳設定
```

> **`train_ris_v7best` preset**：把 V4–V7 多輪實驗找到的最佳超參封裝成單一 yaml。
> 含 AdaptiveCyclicalScheduler + scheduler decoupling rollback + 適當 tau 範圍避免梯度 vanishing。
> 參數選擇理由詳見 [`docs/architecture.md`](docs/architecture.md) §8.4c。

**常用 override 範例**：

```bash
# 改訓練長度
python -m antenna train +experiment=train_single epochs=2000

# 改裝置
python -m antenna train +experiment=train_ris environment.device=cuda:0

# 改座標範圍（RIS：25×25 → 30×30）
python -m antenna train +experiment=train_ris pattern.coordinate=[0,30,0,30]

# 換排程器
python -m antenna train +experiment=train_single scheduler._target_=torch.optim.lr_scheduler.ReduceLROnPlateau scheduler.factor=0.5

# 結果輸出根目錄（推薦：寫到本地 ./result/，避免污染共用網路磁碟）
python -m antenna train +experiment=train_ris environment.rootdir=.

# multirun（同時跑多組）
python -m antenna -m train +experiment=train_single optimizer.lr=0.001,0.005,0.01
```

### 5.2 Python API

想把 Trainer 嵌入自己的 orchestration 時：

```python
from omegaconf import OmegaConf
from antenna.training.trainer import Trainer

cfg = OmegaConf.load("antenna/conf/experiment/train_single.yaml")
# ... 疊加 overrides ...
trainer = Trainer(cfg)
trainer.run()
```

### 5.3 舊訓練腳本（legacy）

在 modernize 分支重構之前，所有訓練由根目錄的 `train_*.py` 獨立腳本驅動：

| 腳本 | 用途 |
|------|------|
| `train_single.py` | 單埠貼片 |
| `train_dual.py` | 雙埠貼片 |
| `train_ris.py` | RIS |
| `train_selection_single.py` | 單埠「選擇式」訓練（pattern 從候選池挑） |
| `train_selection_dual.py` | 雙埠選擇式 |
| `train_single_mirror.py` | 加入 mirror 對稱約束的單埠 |

> **已刪除（依賴的 symbol 在 modernize 中被移除）**：`train_ris_old.py`、`train_ris_ssm.py`、`train_selection_dual_ssm.py`。

**這些腳本仍可執行**，內容和 Hydra CLI 等價，但不享受 structured config 的型別保護；推薦新實驗走 `python -m antenna train ...`。

### 5.4 Web 結果檢視器

```bash
# 開發環境
python application/app.py -dev

# 正式環境
python application/run_waitress.py
```

開啟瀏覽器到 `http://<本機 IP>:<port>` 可以：

- 瀏覽 `result/` 下每個實驗的 checkpoint、pattern、響應曲線
- 互動式 pattern 生成器
- 下載產生的資料集

### 5.5 結果視覺化工具（命令列）

訓練完成後可以用 `script/` 下的工具產出檢視圖：

```bash
# 單一 run 的完整 inspection（loss/tau/pattern/response/best-hard pattern + 10 sample 三聯圖）
python script/inspect_ris_run.py result/RIS-v7-v4like-decoupled

# 多 run cross-overlay 比較（loss / tau / response 疊圖 + summary 表）
python script/compare_ris_runs.py result/RIS-v4-* result/RIS-v7-* result/RIS-v9-*
```

`inspect_ris_run.py` 產出在 `result/<run>/pic/`：
- `loss_curves.png`、`pattern_evolution.png`、`response_vs_target.png`、`best_pattern_hard.png`
- `samples/sample_NN.png` × 10 張：每張顯示「輸入 target / 輸出 pattern / 實際響應」三聯圖。**用於驗證 generator 是否真為 conditional**——若 10 個不同 target 卻得到同樣 pattern，就表示 generator collapse（見 [`docs/architecture.md`](docs/architecture.md) §8.4d）。

`compare_ris_runs.py` 產出在 `result/_compare/<timestamp>/`：
- `loss_compare.png`、`tau_compare.png`、`response_compare.png`、`summary.md`
- 支援 mixed-target 比較（每個 run 從自己的 `config.yaml` 讀回訓練時的 target）

---

## 6. 專案結構

```
Antenna/
├─ antenna/               # 主套件
│  ├─ core/
│  │  ├─ pattern.py           # AntennaPattern（2D 像素圖）
│  │  └─ response.py          # AntennaResponse, TargetResponse, MultiResponses
│  ├─ models/
│  │  ├─ base.py              # Models 泛型訓練封裝器（save/load/step）
│  │  ├─ components.py        # 共用 nn.Module 零件
│  │  ├─ generators/          # SigmoidGEN, GumbelSigmoidGEN, SPGEN, CVAE, MirrorCVAE, ...
│  │  ├─ surrogates/          # OldSM, UNetSM, HFSSNet, EnhancedHFSSUNet
│  │  └─ autograd/            # 自訂 autograd.Function（sign_f, GumbelSigmoid 等）
│  ├─ losses/
│  │  ├─ patch_losses.py      # custom_loss_r/g/minmax, custom_loss_boundary
│  │  ├─ regularization.py    # TVLoss, IslandSuppression, SC/GC Loss, FeedReachability
│  │  ├─ mirror.py            # 鏡像對稱約束、gumbel_sinkhorn
│  │  └─ interval.py          # interval_loss
│  ├─ schedulers/
│  │  └─ adaptive_cyclical.py # AdaptiveCyclicalScheduler（warmup + cosine + plateau）
│  ├─ training/
│  │  └─ trainer.py           # 統一 Trainer class（Hydra 驅動）
│  ├─ configs/
│  │  └─ schema.py            # Hydra structured config dataclasses
│  ├─ conf/                   # Hydra YAML 設定
│  │  ├─ config.yaml              # 根設定
│  │  ├─ environment/default.yaml # 執行環境
│  │  ├─ experiment/              # 完整實驗預設組合
│  │  │  ├─ train_single.yaml
│  │  │  ├─ train_dual.yaml
│  │  │  └─ train_ris.yaml
│  │  └─ response/                # 目標響應
│  │     ├─ single_port.yaml
│  │     ├─ dual_port.yaml
│  │     └─ ris.yaml
│  ├─ patch/patch_simulator/  # HFSS 貼片模擬（COM 自動化）
│  │  ├─ single_port.py
│  │  ├─ dual_port.py
│  │  ├─ _common.py               # 共用 HFSS 腳本
│  │  └─ sab/                     # HFSS 幾何模板（.sab 二進位檔）
│  ├─ ris/
│  │  └─ simulate_ris.py          # 解析式 RIS 陣列因子模擬（純 numpy/torch）
│  ├─ utils/
│  │  ├─ config.py                # Config / MultiConfig（legacy runtime config）
│  │  ├─ data.py                  # DataManager（pattern/response 資料集）
│  │  ├─ record.py                # Record（epoch log + early-stopping）
│  │  ├─ figure.py                # Figure（pattern/response 畫圖）
│  │  ├─ path.py                  # Path（Pathlib 的 antenna 特化版）
│  │  ├─ json_utils.py            # JsonFile（巢狀 key 的 JSON 存取）
│  │  ├─ hashing.py               # TID, shake_128
│  │  ├─ torch_utils.py           # device 檢測 / checkpoint 載入
│  │  ├─ web.py                   # 網路磁碟、email 通知、本機 IP
│  │  └─ utils.py                 # re-export hub
│  ├─ __init__.py             # 套件頂層 re-export
│  ├─ __main__.py             # Hydra CLI entry（python -m antenna train）
│  ├─ types.py                # 型別 hub
│  ├─ functions.py            # 歷史 re-export（losses / schedulers）
│  ├─ smodels.py              # surrogates 的 backward-compat shim
│  └─ ranger.py               # Ranger optimizer（第三方）
├─ application/           # Flask 結果檢視器
├─ script/                # 輔助腳本（kill, img2video, check_gpu...）
├─ tests/                 # 477 個 pytest 單元測試
├─ result/                # 訓練結果（git ignored）
├─ train_*.py             # legacy 訓練腳本（6 個）
├─ pyproject.toml
├─ requirements.txt
├─ .pre-commit-config.yaml
└─ .github/workflows/lint.yml   # CI：ruff check + ruff format --check
```

---

## 7. 核心類別

這三個 class 是整個 library 的心臟。

### 7.1 `AntennaPattern`（`antenna/core/pattern.py`）

**職責**：管理 2D 像素圖 (pattern)，提供合併、模擬、二值化等操作。

**class-level 狀態**（整個 runtime 共享）：

```python
AntennaPattern.setDefaultCoordinate((x_min, x_max, y_min, y_max))
AntennaPattern.register_simulator(my_simulator)   # pattern.simulate() 會呼叫它
AntennaPattern.size(flatten=True)                 # 攤平後的尺寸（像素數）
```

**instance 生命週期**：

```python
# 從 tensor logit 建立
p = AntennaPattern(pattern_logits)

# 二值化
p_bin = p.binarization(tau=0.1, threshold=0.5)

# 繪圖
p.plot()                # 單張
p.plot_individual()     # 含座標軸的完整版

# 呼叫已註冊的 simulator
response = p.simulate()

# 合併 / 反轉
p3 = p1.merge(p2)
p_inv = p.invert()
```

**常見注意事項**：

- `binarize()` 的實作在 `modernize` 之前對**非方形 pattern** 有 bug（用了 `(0, len(bi), 0, len(bi))` 而非 `(0, w, 0, h)`）；現已修復。
- `binarization()` 的 `tau` / `threshold` 若傳 `0` 會被正確處理（之前用 `tau or default` 會誤判 0）。
- `tau` 有下限 `1e-4` 避免除零。

### 7.2 `AntennaResponse`（`antenna/core/response.py`）

**職責**：管理頻域響應與目標響應的比對機制。

**三步驟設定**：

```python
# 1. 註冊 labels 與 x 軸
AntennaResponse.registerLabels("S11", "Gain", x="n257")

# 2. 註冊目標響應（可在訓練前決定）
AntennaResponse.registerTargetResponse({
    "S11": TargetResponse(side=0, center=-10, width=[5,0,7,0,5]),
    "Gain": TargetResponse(side=-19, center=4, width=[5,0,7,0,5]),
})

# 3. 註冊每個 label 的 loss function
AntennaResponse.registerLossHook(
    S11=(custom_loss_minmax, {"method": "low"}),
    Gain=(custom_loss_minmax, {"method": "high"}),
)
```

**TargetResponse**：以「寬度區塊」定義容許的響應曲線形狀；`side` 是外側值、`center` 是中心值、`width=[5,0,7,0,5]` 表達「左側 5 點 side → 0 點過渡 → 7 點 center → 0 點過渡 → 5 點 side」。

**MultiResponses**：處理「多個 label 一次算 loss」的容器，`trainer.run()` 內部會走這條路。

### 7.3 `Models`（`antenna/models/base.py`）

**職責**：把「神經網路 + optimizer + scheduler + criterion + checkpoint 管理」打包。**所有 Generator / Surrogate 最終都被包進 Models**。

```python
m = Models(
    name="single_port_generator",
    model=SigmoidGEN(response_size, pattern_size),
    optimizer=torch.optim.Adam,
    optimizer_params={"lr": 0.005, "betas": (0.5, 0.999)},
    scheduler=AdaptiveCyclicalScheduler,
    scheduler_params={...},
    criterion=custom_loss_minmax,
)

# 訓練一步
loss = m.step(predicted_response, target_response)

# 儲存 / 載入
m.save(path)
m.load(path)
m.save_as(other_path)  # 另存
m.change(pattern_dict=...)  # 換資料後重新綁定
```

**關鍵方法**：

- `step(y_pred, y_true)`：計算 loss、backward、step、zero_grad，一次到位。
- `pre_load_model(path)`：從 checkpoint 載入權重、並驗證沒有 NaN/Inf。
- `requires_grad(bool)`：凍結 / 解凍所有參數。
- `FloatTensor` 屬性：裝置相容的 `torch.FloatTensor` 別名（舊腳本用）。

---

## 8. 生成器 (Generators)

所有生成器都吃**目標響應**，產出 **pattern logits**（未二值化的連續值）。位於 `antenna/models/generators/`。

| 類別 | 檔案 | 特色 |
|------|------|------|
| `SigmoidGEN` | `sigmoid_gen.py` | 最基本：`FC → Sigmoid`，輸出 [0,1] pattern |
| `GumbelSigmoidGEN` | `gumbel_sigmoid_gen.py` | 加 Gumbel 雜訊 + 退火 tau，鼓勵近二值 |
| `SPGEN` | `sp_gen.py` | Selection Pattern：從預先給定的 pattern_table 挑 |
| `CVAE` | `cvae.py` | Conditional VAE，用於資料增強 |
| `MirrorCVAE` | `mirror_cvae.py` | 加入鏡像對稱約束的 CVAE |
| `OldGEN` | `old_gen.py` | 歷史版本，舊 train_*.py 仍在用 |
| `GradientEstimator` | `gradient_estimator.py` | 代理模型意義上的「梯度估計器」（不是 autograd Function） |

**共用 helper `_build_fc_patch()`** 把「stack of Linear + activation」的樣板抽出來，所有生成器共用。

**退火機制**：GumbelSigmoidGEN 的 `anneal_tau()` 讓 tau 隨 epoch 遞減，初期軟、後期近硬二值。搭配 `AdaptiveCyclicalScheduler` 的 tau callback 能在 plateau 時調整。

---

## 9. 代理模型 (Surrogates)

位於 `antenna/models/surrogates/`。所有代理模型輸入 pattern、輸出 response，**必須可微**。

| 類別 | 檔案 | 架構 |
|------|------|------|
| `OldSM` | `surrogate_model.py` | 歷史版本，FC stack，所有 train_*.py 預設用它 |
| `UNetSM` | `surrogate_model.py` | UNet 編解碼 |
| `HFSSNet` | `hfss_net.py` | 純 FC MLP，輕量代理 |
| `EnhancedHFSSUNet` | `unet.py` | CNN + self-attention UNet，有 `DoubleConvWithDropout` 共用塊 |

**訓練資料**：來自真實 HFSS 模擬的 `(pattern, response)` 對；代理模型學到 HFSS 的近似行為。

**在 trainer 中**：代理模型與生成器**交替訓練**：

1. 生成器從目標響應產出 pattern
2. 代理模型預測響應，計算 loss（相對目標）
3. backward 到生成器
4. 定期呼叫真 HFSS 產生新訓練對，fine-tune 代理模型

---

## 10. 損失函數 (Losses)

### 10.1 Patch losses（`losses/patch_losses.py`）

針對「預測響應 vs 目標響應」的損失。

- `custom_loss_r` / `custom_loss_g`：對 side/center 用 SmoothL1，對過渡區容忍
- `custom_loss_minmax`：支援 `method="low"`（低於閾值才計入）、`method="high"`（高於）、`method="interval"`（區間外才計入）
- `custom_loss_boundary`：`custom_loss_r/g` 的底層 helper
- `_one_sided_penalty`、`_make_criterion`：共用內部 helpers

### 10.2 Regularization（`losses/regularization.py`）

針對 pattern 本身的幾何約束：

- **TV loss** (`TotalVariationLoss`)：懲罰相鄰像素差異大，鼓勵連續區域
- **Island Suppression**：在 `AntennaPattern` 的 method，懲罰孤立 pixel
- **SpectralConnectivityLoss**：以圖 Laplacian 的特徵值差距衡量連通性
- **GapClosingLoss**：鼓勵關閉縫隙
- **FeedReachability**：確保所有銅箔與饋電點連通（用 scipy.ndimage.label）

這些 loss 的權重在 `config.yaml` 直接暴露：

```yaml
total_variation_loss_weight: 0.0
island_suppression_loss_weight: 0.0
spectral_connectivity_loss_weight: 0.0
gap_closing_loss_weight: 0.0
```

### 10.3 Mirror / Interval（`losses/mirror.py`、`losses/interval.py`）

- `mirror(pattern)`：計算左右 / 上下對稱性
- `gumbel_sinkhorn_rectangular`：可微的排列矩陣（透過 Sinkhorn 在 log space 做）
- `interval_loss`：頻段上下限約束；低於下限或高於上限才累積 loss

### 10.4 RIS-specific losses（`antenna/ris/__init__.py`）

兩個 RIS 專用 loss 都在 `LOSS_FN_REGISTRY` 註冊好可從 YAML 直接呼叫：

- **`custom_loss`**：tolerance-style — sidelobe 區超過 threshold 才罰、main beam 跌到 sidelobe 以下才罰；不超出範圍時 fallback 到小 MSE。**缺點**：main beam 沒「越高越好」的梯度動機，generator 把主峰擺到位後就停止 push。
- **`custom_loss_directivity`**：tolerance + reward — 沿用 sidelobe 平方 penalty，但**main beam 直接用 `-mean(prediction)` 當 loss 項**，響應越高 loss 越低。期望突破 §8.4d 的 generator collapse。
  - 參數：`sidelobe_threshold`（dB）、`main_beam_weight`（reward 項權重，預設 0.1）

> **⚠️ 已知調參陷阱**：兩項單位不同（side_loss 是 dB²、main_reward 是 dB）。
> V13 實測 main_beam_weight=0.1 時 loss 飆到 ~40（從 main_reward 的線性 dB 來），
> generator 反而往 sidelobe 平壓的方向跑（壓低 main beam → side_loss=0 → 但 main_reward 變大），
> 訓練退化。建議：
> - 把 main_reward 也用平方/clip 控制範圍，例如 `(target_high - main).clamp(min=0).pow(2)`
> - 或大幅降低 main_beam_weight（如 0.001）讓 sidelobe penalty 主導
> - 或先固定 sidelobe_threshold 為實際達得到的值（例如 -25 而非 -20）

YAML 切換：
```yaml
response:
  label_configs:
    response:
      loss_fn: custom_loss_directivity
      loss_params:
        sidelobe_threshold: -20.0
        main_beam_weight: 0.1
```

---

## 11. 排程器 (Schedulers)

`antenna/schedulers/adaptive_cyclical.py` 提供一個自訂 scheduler：

**`AdaptiveCyclicalScheduler`** — warmup → cosine 下降 → plateau 偵測 → 必要時 restart。參數：

| 參數 | 意義 |
|------|------|
| `T_0` | 一個週期的長度（epoch） |
| `T_mult` | 每次 restart 後週期長度的倍增 |
| `lr_max` / `lr_min` | cosine 範圍 |
| `temp_max` / `temp_min` | 同步更新 Gumbel/sigmoid 的 tau |
| `warmup_ratio` | warmup 佔週期比例（0.2 表示前 20% 線性升 lr） |
| `patience` / `factor` | plateau 偵測的 torch ReduceLROnPlateau 參數 |
| `mode` | `"min"` 或 `"max"` |
| `on_plateau` | plateau 的 tau 行為：`"linear"` / `"cosine"` |

也能透過 `scheduler._target_` 切成 `torch.optim.lr_scheduler.ReduceLROnPlateau` 或其他標準 scheduler。

---

## 12. Trainer 與訓練流程

`antenna/training/trainer.py` 的 `Trainer` class 把所有東西綁起來。

> **詳細架構圖、online learning 細節、參數表、現況觀察**：見 [`docs/architecture.md`](docs/architecture.md)

**初始化順序**（重要，修過 bug）：

```
環境 → 路徑 → 追蹤(Record) → 模擬器 → 天線 → 模型
                ↑
        必須在模型之前，resume 才能正確讀到歷史 checkpoint
```

**Registry 機制**（從 YAML 字串對應到實作）：

```python
LOSS_FN_REGISTRY = {
    "custom_loss_minmax": custom_loss_minmax,
    "custom_loss_r": custom_loss_r,
    "custom_loss_g": custom_loss_g,
    "custom_loss": ris_custom_loss,       # RIS 專用
}

MODEL_REGISTRY = {
    "sigmoid_gen": SigmoidGEN,
    "gumbel_sigmoid_gen": GumbelSigmoidGEN,
}

SIMULATOR_REGISTRY = {
    "single_port": _single_port_factory,  # lazy import HFSS
    "dual_port": _dual_port_factory,
    "ris": _ris_factory,
}
```

**擴充新 loss / model / simulator**：只要把條目加進對應 registry，YAML 就能用（見 [§20](#20-擴充指南)）。

---

## 13. Hydra 設定系統

設定檔採「structured configs + YAML」雙層架構：

**上層 — Python dataclass**（`antenna/configs/schema.py`）定義欄位型別與預設值。
**下層 — YAML**（`antenna/conf/*.yaml`）提供具體設定。

### 組合機制

```yaml
# antenna/conf/config.yaml
defaults:
  - environment: default           # 套用 environment/default.yaml
  - response: single_port          # 套用 response/single_port.yaml
  - _self_                         # 然後套用此檔其他欄位
```

```yaml
# antenna/conf/experiment/train_single.yaml
# @package _global_                # 表示這個 experiment 直接覆蓋 global
defaults:
  - override /response: single_port   # 強制用 single_port response

model: sigmoid_gen
simulator: single_port
scheduler:
  _target_: antenna.schedulers.adaptive_cyclical.AdaptiveCyclicalScheduler
  T_0: 100
  ...
```

### CLI override 語法

```bash
# 等號覆寫單欄
python -m antenna train epochs=2000

# . 巢狀
python -m antenna train optimizer.lr=0.001

# = 加陣列
python -m antenna train pattern.coordinate=[0,30,0,30]

# + 代表「追加」
python -m antenna train +experiment=train_single +new_field=hello

# ~ 代表「刪除」
python -m antenna train +experiment=train_single ~scheduler
```

### 預留的空目錄

`antenna/conf/{model,scheduler,simulator,surrogate}/` 目前是空的（只有 `.gitkeep`），為了未來可以加 `defaults:` 對應的 Hydra group 檔案預留。

---

## 14. HFSS 整合

### 14.1 PatchSimulator 介面

`antenna/patch/patch_simulator/` 下：

- `PatchSimulator`（抽象基底）
- `SinglePortSimulator` / `DualPortSimulator`（實作）
- `_common.py`（共用 HFSS 腳本 helper）
- `sab/*.sab`（HFSS 幾何模板二進位檔）

使用方式：

```python
from antenna.patch.patch_simulator import SinglePortSimulator

sim = SinglePortSimulator(record_path="result/my_run/")
AntennaPattern.register_simulator(sim)

response = pattern.simulate()   # 會開 HFSS、跑模擬、讀回 S-parameter
```

### 14.2 COM 自動化細節

- 啟動時呼叫 `gencache.EnsureDispatch("AnsoftHfss.HfssScriptInterface")`
- `_common.py` 提供 `assign_pixel_variables`, `import_substrate`, `create_patch_pixels`, `insert_analysis_setup`, `configure_3d_rad_field` 等共用函式
- 針對像素尺寸有 dict lookup 表，不是 if-cascade
- 沒有安裝 HFSS 時會 raise 可辨識的錯誤訊息

### 14.3 測試策略

`tests/test_patch_simulator.py` 用 `_FakeEditor` / `_FakeDesign` mock 驗證呼叫路徑，不需要真的 HFSS 即可通過 CI。

---

## 15. RIS 模擬

`antenna/ris/simulate_ris.py` 是**純 numpy/torch 實作**，不依賴 HFSS。

**原理**：以解析式陣列因子（Array Factor）計算 RIS 遠場響應。

```python
from antenna.ris import RISSimulator, custom_loss

sim = RISSimulator(
    element_num=25,           # 單邊元素數（總元素數 = 25² = 625）
    freq_hz=28e9,             # 28 GHz（毫米波）
    feed_distance_m=500e-3,
    inc_theta_deg=-40.0,
    inc_phi_deg=90.0,
)

# pattern: (25, 25) 的 0/1 tensor
response = sim(pattern)
loss = custom_loss(response, target)
```

全部可微，可以直接 backward。

---

## 16. 資料管理 (DataManager)

`antenna/utils/data.py` 的 `DataManager` 管理 `(pattern, response, label)` 資料集。

```python
from antenna.utils import DataManager

dm = DataManager(save_path="result/my_run/datasets.pkl")
dm.append(pattern, response, label="S11")
dm.save()
# 之後可以：
dm.load("result/my_run/datasets.pkl")
dataset = dm.to_torch_dataset()
```

支援：

- **去重**（透過 `make_hashable` 算 data_id）
- **篩選**（`dm.filter(label="S11")`）
- **原子性 pickle save**（先寫暫存再 rename）
- **timestamped backup**

---

## 17. 實驗追蹤 (Record / wandb)

### Record（`antenna/utils/record.py`）

輕量級 epoch log：

```python
from antenna.utils import Record

rec = Record(save_path="result/my_run/record.pkl")
for epoch in range(epochs):
    ...
    rec.append(loss=loss.item(), lr=current_lr, pattern=pattern_np)
    if rec.early_stop(patience=10):
        break
rec.end()      # 存檔
```

### wandb

未來會整合（trainer.py 有預留空間）。CLAUDE.md 的 `實驗追蹤: Weights & Biases (wandb)` 是規劃中。

---

## 18. Web 應用程式

`application/app.py` 是 Flask 結果檢視器。

**路由**：

| 路徑 | 功能 |
|------|------|
| `/` | 首頁：列出所有實驗 |
| `/record/<id>` | 看某實驗的訓練過程 |
| `/dataset/<name>` | 檢視資料集統計 |
| `/generator` | 互動式 pattern 生成器 |
| `/result/<path:filename>` | 下載結果檔（用 `send_from_directory` 防路徑穿越） |

**安全**：路由參數過 `_validate_name()` 檢查（只允許 `[\w\-.]+` 且排除 `..`）。

---

## 19. 測試與開發工具

```bash
# 跑全部測試（目前 477 個）
pytest

# 跑單一檔
pytest tests/test_trainer.py -v

# Lint
ruff check .

# 格式化
ruff format .

# 一次跑完 pre-commit 所有 hooks
pre-commit run --all-files
```

**測試覆蓋**：tests/ 下 18 個檔，涵蓋 core / losses / schedulers / trainer / utils / surrogates / autograd / patch_simulator（smoke） / ris（smoke）。

**CI**：`.github/workflows/lint.yml` 在 push 至 `main` / `GAN` / `modernize` 時跑 ruff check + format --check。

---

## 20. 擴充指南

### 20.1 新增一個 loss function

1. 在 `antenna/losses/` 內寫函式：
   ```python
   def my_loss(pred, target, **params):
       ...
   ```
2. 在 `trainer.py` 的 `LOSS_FN_REGISTRY` 加入 `"my_loss": my_loss`
3. YAML 就能用 `loss_fn: my_loss`

### 20.2 新增一個 generator

1. 繼承 `torch.nn.Module`，實作 `forward(response)` 回傳 pattern logits
2. 放到 `antenna/models/generators/`，`__init__.py` 記得 re-export
3. 在 `trainer.py` 的 `MODEL_REGISTRY` 加入對應
4. `tests/test_generators.py` 補上 forward shape / generate shape 測試

### 20.3 新增一個 simulator

1. 寫 factory function（lazy import，避免測試環境崩）：
   ```python
   def _my_factory(cfg, result_path):
       from antenna.my.module import MySimulator
       return MySimulator(...)
   ```
2. 在 `SIMULATOR_REGISTRY` 加 `"my_type": _my_factory`
3. 新增對應 `antenna/conf/response/my_type.yaml`
4. 可選：新增 `antenna/conf/experiment/train_my.yaml` 作為 defaults 組合

### 20.4 新增一個 Hydra experiment

複製 `antenna/conf/experiment/train_single.yaml`，改 `# @package _global_` 之下的欄位，跑 `python -m antenna train +experiment=<新檔名>`。

---

## 21. 疑難排解

| 症狀 | 可能原因 / 解法 |
|------|-----------------|
| `ImportError: DLL load failed` 在 pywin32 | Windows 環境但 pywin32 沒正確安裝；重裝 `pip install --force-reinstall pywin32` |
| `pytest -x` 找不到 `antenna` | `pip install -e ".[dev]"` 忘了；或 conda env 沒啟用 |
| Hydra override 不生效 | 加 `--cfg job` 看合成後的設定，確認該欄位路徑正確 |
| Hydra 報 `Could not override 'X'` / `Key 'X' is not in struct` | 此欄位不在 structured config schema；改用 `++` 前綴強制加（如 `++scheduler.T_0=100`） |
| CUDA OOM | 降 batch、縮小 pattern coordinate、或 `environment.device=cpu` |
| `Expected all tensors to be on the same device` 在 generator forward | 已修（commit 358628a）— 若再遇到請確認 `Models.__init__` 沒在傳 `device=config.device` 早綁定 default |
| HFSS 開不起來 | 確認 Ansys AEDT 已安裝；`gencache.EnsureDispatch` 失敗時會 raise |
| `test_configs_schema.py` 被跳過 | 缺 `hydra-core` / `omegaconf`；`pip install -e ".[dev]"` 會補 |
| 網路磁碟連不上 | `.env` 的 `ANTENNA_NETWORK_DRIVE_*` 帳密錯誤 |
| 訓練結果寫到別人的網路磁碟資料夾 | `ROOTDIR` 預設行為；用 `environment.rootdir=.` 或 `ANTENNA_ROOTDIR` 環境變數覆寫，將結果落地本地 `result/` |
| 在 git worktree 內跑 pytest，匯入到主 repo 的 antenna/ | `pyproject.toml` 的 `pythonpath=["."]` 會被 pytest 解析到主 repo；改用 `pytest -o "pythonpath=<worktree-abs-path>"` 或 cd 到 worktree 內驗證 `python -c "import antenna; print(antenna.__file__)"` |
| ruff format 在 CI 失敗但本地過 | 整 repo 跑 `ruff format .`，注意 Unit 20 之後有幾個檔案曾經漏做 |
| Pre-commit 第一次太慢 | ruff / pre-commit 需下載 hook repo，之後會被快取 |
| `AdaptiveCyclicalScheduler` tau 沒退火（一直停在 4.0 附近） | 已修（commit 497d694）— scheduler 預設 callback 不影響 GumbelSigmoidGEN.tau；trainer 已自動接 callback，但若用 `scheduler._target_=` 覆寫到自訂類別需確認 callback 仍有效 |
| `Can't pickle local object '_cb'` 在 Models.save | 已修（commit 40ff5b4）— scheduler.state_dict 之前會把 closure 連帶序列化 |

---

## 授權

依循上游 repo 的授權條款。

## 聯絡

Issue tracking 走 GitHub Issues。
