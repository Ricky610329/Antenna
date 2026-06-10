天線 (Antenna)
===========
裡面包含 微帶貼片天線(microstrip patch antenna) 與 可重構智慧表面(Reconfigurable Intelligent Surface, RIS)

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
|  ├─ training.py      # 單/雙埠共用訓練核心 (run_training)
|  ├─ models.py        # 生成器 G (SigmoidGEN)
|  ├─ smodels.py       # 代理模型 SM (HFSSNet / OldSM)
|  ├─ functions.py     # 損失 + ACP 排程器
|  ├─ patch            # microstrip patch antenna
|  |  └─ patch_simulator   # HFSS 模擬器 (single_port / dual_port)
|  └─ utils            # config / Record / DataManager 等工具
├─ tests               # pytest (golden + 單元測試)
├─ docs                # 文件 (見 docs/README.md)
├─ script              # 輔助腳本
└─ result              # 執行後自動生成
```
