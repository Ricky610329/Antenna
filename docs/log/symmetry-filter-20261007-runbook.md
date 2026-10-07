# R80 操作：金屬對稱資料工廠與獨立 worker

目前先執行 R80 金屬左右對稱量測；使用者已啟動worker，並於2026-10-07授權每30分鐘監看，完成對稱任務後接續spec調整與R81驗證。R81工程檢查及正式批次待R80完成才派送。最新量測狀態以[操作板](../../configs/ONGOING.md)為準。

scope 固定為 `symmetry_filter_20261007`，worker 固定使用 `--selfgen 0 --poll 60 --stale 120`。啟動器會拒絕 scope 內任何 dual/filter job。

## 私人佇列與初始 pilot

部署目標為私人 NAS：

```text
T:\碩二_鄒穎麒's\antenna\experiments\r80_symmetry_20261007
```

初始 pilot 保留以下三個 jobs，後續依使用者授權增加16筆的對稱工廠 jobs：

| store | 次數 | 用途 |
| --- | ---: | --- |
| `dedust_r80b1` | 48 | R80 首批 |
| `dedust_r80repeat1` | 2 | 兩個代表圖形的第一次重測 |
| `dedust_r80repeat2` | 2 | 同兩個代表圖形的第二次重測 |

初始合計52次量測；部署時158個檔案均逐檔比對，combined queue在worker啟動前遭拒。這是部署時的證據，三台worker現已啟動。後续16筆factory jobs同樣限定single/profile/scope；最新數量與實測狀態以[操作板](../../configs/ONGOING.md)為準。[初始部署紀錄](assets/r80_private_deployment_20261007.json)、舊ZIP與62次收據均保留為歷史，不是目前完整佇列。

## 正式機啟動

開發與每台正式機 worker 統一使用 Git `GAN`（使用者 2026-10-07 確認）；遠端已包含 worker 實作 `29a6834`，不需要 ZIP。在每台 Antenna repo 根目錄執行：

```powershell
git checkout GAN
git pull --ff-only origin GAN
conda activate patch
powershell -NoProfile -ExecutionPolicy Bypass -File .\script\start_scoped_worker.ps1 -CheckOnly
powershell -NoProfile -ExecutionPolicy Bypass -File .\script\start_scoped_worker.ps1
```

`-ExecutionPolicy Bypass` 只套用在這個 PowerShell 程序，不會永久修改系統原則。`-CheckOnly` 只解析私人 `ROOTDIR` 下的 `experiments/r80_symmetry_20261007/dataset`、列出 scope 佇列並檢查job的single-port身分，不啟動HFSS；完整input驗證已在備料與部署階段另外完成。確認都是本輪single-port jobs後，再執行不帶 `-CheckOnly` 的啟動命令。通常不必手填 `-DatasetRoot`；腳本會由私人設定自動解析完整路徑。已運作的worker不需因本機SM/controller更新而重啟。

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

## 原 pilot 回填與目前工廠更新

首批完成後，保留完整共享 dataset，不可只帶 `results.json`。在開發機以 local controller 回填：

```powershell
python -m script.exploration feedback --work-dir tmp/r80_symmetry_20261007_single/single --batch 1 --dataset-root <私人 dataset 完整路徑> --result-manifest <私人 dataset 完整路徑>/dedust_r80b1/results.json
python -m script.exploration train --work-dir tmp/r80_symmetry_20261007_single/single --through-batch 1
```

以上指令僅回填原pilot controller。目前10,240筆工廠使用獨立`tmp/r80_factory_20261007/training_v2`及`symmetry_training.py`，每48筆有效唯一新結果觸發更新（單次最多96）；`snapshot_successes`可凍結長批已成功的子集合，不必等整個長批跑完。原store與送測前預測均保留，低性能有效資料也入訓練。新policy見`configs/single_r80_symmetry_factory.yaml`及R80最新追加紀錄。R81接續對稱階段，先通過工程檢查再擴大量測。

## 可證明範圍

完整回歸 **544 passed / 361.29 秒**（`OMP_NUM_THREADS=4`），golden 原始 bytes 不變；pyflakes 通過。worker 實作 `29a6834` 已包含於遠端 `GAN`。部署收據中的 `main` 為首次交付的歷史紀錄；現行更新與啟動一律使用 `GAN`。

程式測試、`-CheckOnly` 與部署稽核只證明工程邊界；實測數量由後續NAS樣本、場型和hash重播稽核確認，見操作板及R80日誌。三機已持續產出，但目前不宣稱金屬對稱已改善場型，或新SM已優於歷史模型。
