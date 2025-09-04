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
├─ antenna    # 主套件
|  ├─ patch   # microstrip patch antenna
|  |  └─ patch_simulator
|  |      ├─ sab # HFSS用
|  |      └─ ...
|  ├─ ris    # Reconfigurable Intelligent Surface (RIS)
|  |  └─ ...
|  └─ ...
├─ script     # 腳本
|  └─ ...
└─ result     # 執行後自動生成
```
