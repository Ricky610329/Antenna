# 速查：環境 + 切 branch + 執行

> 一頁搞定。詳細參數請看 [`train_single_dual_usage.md`](train_single_dual_usage.md)。

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

## 3. 執行（必帶設定編號）

```bash
python train_single.py 1     # 單埠，編號 1–10
python train_dual.py 1       # 雙埠，編號 1–9
```

- **一定要帶編號**，否則會 `IndexError`。編號對照表見 usage 文件第 5 節。
- **雙埠第一次**沒有 `patch_dual.pth` 時，會先預訓練代理模型後 `exit()` → **要再跑第二次**才進主訓練。
- 用**相同編號**再跑會自動斷點續跑（從上次 epoch 接著跑）。

## 4. 前置需求（正式機）

- Windows + 已安裝 ANSYS HFSS（透過 COM 啟動）。
- 連得到實驗室內網（腳本會自動掛 NAS `T:`，結果/資料集都在那）。

## 5. 常見錯誤

| 訊息 | 原因 / 解法 |
| --- | --- |
| `ImportError: ... 'Self' from 'typing'` | 沒 `conda activate patch`（用到舊 Python）。 |
| `IndexError`（在 MultiConfig） | 忘了帶設定編號，例 `python train_single.py 1`。 |
| `simulator.open()` 失敗 | 沒有 HFSS / COM 被佔用。 |
