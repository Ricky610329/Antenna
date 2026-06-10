# 速查：環境 + 切 branch + 執行

> 一頁搞定。完整說明見 [`training.md`](training.md)。

## 1. 啟用環境（每次開新終端都要）

```bash
conda activate patch
```

> 一定要先 activate（需 Python 3.11+）。沒開對環境會出現
> `ImportError: cannot import name 'Self' from 'typing'`。

## 2. 拿最新程式碼 / 切 branch

```bash
git fetch origin        # 抓遠端最新（只下載，不動檔案）
git checkout GAN        # 切到 GAN 分支（要 main 就 git checkout main）
git pull                # 更新目前分支到最新
git log --oneline -3    # 確認最新 commit
```

常用查詢：
```bash
git status              # 看目前在哪個分支 + 有無改動
git branch -a           # 列出所有分支
```

## 3. 執行（config 驅動）

```bash
python train.py configs/single_base.yaml      # 單埠
python train.py configs/dual_base.yaml         # 雙埠
python train.py configs/dual_sc.yaml           # 雙埠 + SC 連通性損失
```

- **一個 YAML = 一組實驗**。要跑別的就指定別的 config（`configs/` 下有現成的）。
- 跑新實驗：複製一個 config 改參數即可。各 config 與舊編號對照見 `training.md` 第 5 節。
- 用**相同 config** 再跑會自動斷點續跑。
- 舊指令 `python train_single.py <編號>` 已移除，改用上面的方式。

## 4. 前置需求（正式機）

- Windows + 已安裝 ANSYS HFSS（透過 COM 啟動）。
- 連得到實驗室內網（腳本會自動掛 NAS `T:`，結果/資料集都在那）。

## 5. 常見錯誤

| 訊息 | 原因 / 解法 |
| --- | --- |
| `ImportError: ... 'Self' from 'typing'` | 沒 `conda activate patch`（用到舊 Python）。 |
| `用法: python train.py configs/...` | 忘了帶 config 路徑。 |
| `FileNotFoundError: configs/...` | config 路徑打錯。 |
| `simulator.open()` 失敗 | 沒有 HFSS / COM 被佔用。 |
