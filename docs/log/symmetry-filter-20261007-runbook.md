# R80 操作：金屬對稱量測、有限批次、獨立 worker

本次只執行 R80 金屬左右對稱量測。R81 的 smoke、Discrete、mesh 與所有其他 spec-change 實驗全部延後，不得加入目前佇列。worker 由使用者在正式機啟動；目前尚未啟動 HFSS，也沒有本輪量測結果。

scope 固定為 `symmetry_filter_20261007`，worker 固定使用 `--selfgen 0 --poll 60 --stale 120`。啟動器會拒絕 scope 內任何 dual/filter job。

## 本次唯一佇列

部署目標為私人 NAS：

```text
T:\碩二_鄒穎麒's\antenna\experiments\r80_symmetry_20261007
```

只允許以下三個 jobs：

| store | 次數 | 用途 |
| --- | ---: | --- |
| `dedust_r80b1` | 48 | R80 首批 |
| `dedust_r80repeat1` | 2 | 兩個代表圖形的第一次重測 |
| `dedust_r80repeat2` | 2 | 同兩個代表圖形的第二次重測 |

合計 52 次量測。conductor 已將158個檔案部署到上述私人NAS路徑並逐檔比對；預設 `-CheckOnly` 成功列出 `0/48`、`0/2`、`0/2`，combined queue 也已確認會在worker啟動前遭拒。HFSS仍未啟動。部署證據見 [R80私人部署紀錄](assets/r80_private_deployment_20261007.json)。舊可攜 ZIP 與62次量測收據僅是歷史紀錄，不是本次交付或啟動方式。

## 正式機啟動

開發與每台正式機 worker 統一使用 Git `GAN`（使用者 2026-10-07 確認）；遠端已包含 worker 實作 `29a6834`，不需要 ZIP。在每台 Antenna repo 根目錄執行：

```powershell
git checkout GAN
git pull --ff-only origin GAN
conda activate patch
powershell -NoProfile -ExecutionPolicy Bypass -File .\script\start_scoped_worker.ps1 -CheckOnly
powershell -NoProfile -ExecutionPolicy Bypass -File .\script\start_scoped_worker.ps1
```

`-ExecutionPolicy Bypass` 只套用在這個 PowerShell 程序，不會永久修改系統原則。`-CheckOnly` 只解析私人 `ROOTDIR` 下的 `experiments/r80_symmetry_20261007/dataset`、列出 scope 佇列並檢查job的single-port身分，不啟動HFSS；完整input驗證已在備料與部署階段另外完成。確認只有上述三個single-port jobs後，再執行不帶 `-CheckOnly` 的啟動命令。通常不必手填 `-DatasetRoot`；腳本會由私人設定自動解析完整路徑。

一台正式機只開一個本輪 worker，也不要同時執行其他 HFSS 任務。無工作時 worker 會等待；中斷後保留 results、tensor、markers 與 claim 狀態，供原機恢復或稽核。

## 本機備料與 controller

本次預設備料只做 single phase，只讀歷史 `dedust_r55sym_input` 的 34 個幾何 seeds：

```powershell
$env:OMP_NUM_THREADS='4'
python -m script.prepare_symmetry_filter prepare `
  --history-root C:/Users/ricky/antenna_nas_backup/dataset `
  --out tmp/r80_symmetry_20261007_single
```

預設 `--phase single`；目前不使用 `--phase combined`，它保留給後續 R81 階段。single controller 位於 `tmp/r80_symmetry_20261007_single/single`；NAS初始備份位於 `controller_initial/single`，已納入逐檔部署比對。

備料輸出不得覆蓋；若需重建，使用新的輸出名稱並保留舊證據。歷史 response 不匯入為新真值，橋寬維持 0.1 mm。

## 回填

首批完成後，保留完整共享 dataset，不可只帶 `results.json`。在開發機以 local controller 回填：

```powershell
python -m script.exploration feedback --work-dir tmp/r80_symmetry_20261007_single/single --batch 1 --dataset-root <私人 dataset 完整路徑> --result-manifest <私人 dataset 完整路徑>/dedust_r80b1/results.json
python -m script.exploration train --work-dir tmp/r80_symmetry_20261007_single/single --through-batch 1
```

目前只備料並派送上述52次。R80第二、三批仍依原研究設計：先完成首批收檔、預測稽核與模型回填，再產生下一批；不能預先混入目前佇列。R81及其他規格變更實驗維持延後。

## 可證明範圍

完整回歸 **544 passed / 361.29 秒**（`OMP_NUM_THREADS=4`），golden 原始 bytes 不變；pyflakes 通過。worker 實作 `29a6834` 已包含於遠端 `GAN`。部署收據中的 `main` 為首次交付的歷史紀錄；現行更新與啟動一律使用 `GAN`。

程式測試、`-CheckOnly` 與部署稽核只能證明備料、佇列及啟動邊界，不能證明HFSS已執行或R80假設成立。目前私人NAS部署與52次single佇列檢查已完成；worker待使用者啟動，HFSS尚未開始。
