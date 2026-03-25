天線 (Antenna)
===========
裡面包含 微帶貼片天線(microstrip patch antenna) 與 可重構智慧表面(Reconfigurable Intelligent Surface, RIS)

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


### 安裝依賴
```bash
# 建議方式（可編輯安裝）
pip install -e ".[dev]"

# 或者使用 requirements.txt
pip install -r requirements.txt
```

### 開發設定
```bash
pre-commit install
```

檔案介紹
-----------

```
Antenna
├─ antenna/       # 主套件
│  ├─ core/       # 核心類別（AntennaPattern, AntennaResponse）
│  ├─ models/     # 生成器與代理模型
│  ├─ simulators/ # HFSS / RIS 模擬器
│  ├─ losses/     # 損失函數
│  ├─ training/   # 訓練迴圈
│  ├─ schedulers/ # 學習率排程器
│  ├─ conf/       # Hydra YAML 設定檔
│  └─ utils/      # 工具模組
├─ application/   # Flask 結果檢視器
├─ tests/         # 測試
├─ script/        # 輔助腳本
└─ result/        # 執行後自動生成
```
