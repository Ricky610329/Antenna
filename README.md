天線 (Antenna)
===========
微帶貼片天線 (microstrip patch antenna) 的**反向設計**：用生成器＋可微分代理模型＋HFSS 模擬器做 online learning，
由目標頻率響應反推 25×25 像素化天線圖樣。（早期的 RIS 變體已移除，見 git 歷史。）

> **快速開始**：訓練入口是 `python train.py configs/<實驗>.yaml`（由外部 YAML config 驅動）。
> 環境啟用 / 切 branch / 執行指令一頁速查見 [`docs/quickstart.md`](docs/quickstart.md)。

文件
------
完整說明都在 [`docs/`](docs/)，導覽見 [`docs/README.md`](docs/README.md)：

| 文件 | 用途 |
| --- | --- |
| [`quickstart.md`](docs/quickstart.md) | 環境 / 切 branch / 執行指令（一頁） |
| [`training.md`](docs/training.md) | config 結構、port、模型架構/載入、變體對照 |
| [`architecture.html`](docs/architecture.html) | 系統架構與論文機制對照 |

安裝
------

### 建立虛擬環境
若已經有了就不需再建立
```bash
conda create --name antenna python=3.11

# 檢查是否安裝成功
conda env list
```

### 啟動虛擬環境
每次使用cmd都要執行
```bash
conda activate antenna

# 查看該環境目前套件
conda list
```

> 註：實際執行環境的 conda env 名稱依機器而定（正式機是 `patch`），以 [`docs/quickstart.md`](docs/quickstart.md) 為準。


### 安裝依賴完竟
依照自己的需求備註

```bash
pip install -r requirements.txt

# Update requirements.txt
pip freeze > requirements.txt
```

檔案介紹
-----------

```python
Antenna
├─ train.py            # 訓練入口：python train.py configs/xxx.yaml
├─ configs             # 一檔一實驗的 YAML 設定
├─ antenna             # 主套件
|  ├─ pattern.py       # 圖樣抽象 (AntennaPattern：座標/merge/二值化/simulate)
|  ├─ response.py      # 響應抽象 (AntennaResponse 家族、響應規格)
|  ├─ training.py      # 單/雙埠共用訓練核心 (run_training)
|  ├─ zoo.py           # 模型動物園：可用的 GEN/SM 架構都登記在這
|  ├─ monitor.py       # TensorBoard 監控 + 結尾總覽圖
|  ├─ losses.py        # 可製造性損失 (TV/SC/GapClosing) + R_feed 指標
|  ├─ models           # 模型層：shell (外殼) / generators (GEN) / surrogates (SM)
|  ├─ optim            # 優化層：ranger (優化器) / scheduler (ACP)
|  ├─ patch            # microstrip patch antenna
|  |  └─ patch_simulator   # HFSS 模擬器 (single_port / dual_port)
|  └─ utils            # config / RunState / SampleStore (+ legacy Record/DataManager)
├─ tests               # pytest (golden + 單元測試)
├─ docs                # 文件 (見 docs/README.md)
├─ script              # 輔助腳本
└─ result              # 執行後自動生成
```
