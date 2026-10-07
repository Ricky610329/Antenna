# R80/R81 操作：具名量測、有限批次、獨立 worker

所有工作在 Antenna。emforge 未修改。這份指令供具 HFSS 的正式機使用；開發機只做 CPU 與備料。
本輪 scope 固定 `symmetry_filter_20261007`，worker 必須 `--selfgen 0`，避免啟動舊自產線。

## 開發機備料

```powershell
$env:OMP_NUM_THREADS='4'
python -m script.prepare_symmetry_filter prepare --history-root C:/Users/ricky/antenna_nas_backup/dataset --out tmp/symmetry_filter_20261007
```

輸出包含 `single/dual` 的本機探索狀態、seed snapshot、`dataset/` 輸入及 `preparation.json`。
只讀具名的 R55、R79 與 smp073 input bundles；原始歷史 response 不改、不匯入新真值。
同一輸出不覆蓋，重建須選新名稱。橋寬單位為 mm（0.1 / 0.075），不是檔名中的100/75。

## Worker 啟動

先把交付包的 `dataset` 放到所有正式機可見的同一個共享路徑；每台使用包中同版本 code 的**本機副本**，
在該副本根目錄啟動。不要在 NAS 上建立 HFSS 工作目錄，也不要與另一個 HFSS 任務共用同一台機器。
原 worker watchdog 仍可能結束該機 HFSS 程序；一台只開一個本輪 worker。

```powershell
conda activate patch
$env:OMP_NUM_THREADS='4'
# 改成共享 dataset 的實際完整路徑；不要指向歷史主資料夾。
$dataset='T:/symmetry_filter_20261007/dataset'
python -m script.dedust --dataset-root $dataset jobs-ls --scope symmetry_filter_20261007 --all
python -m script.dedust --dataset-root $dataset worker --scope symmetry_filter_20261007 --selfgen 0 --poll 60 --stale 120
```

scope worker 只處理自己的 jobs，不執行歷史 probe 或啟動時全域 `_dedust_*` 清理。
`--stale 120` 大於本輪最長單筆 timeout（3600秒）加啟動緩衝，避免長測被另一台誤判為失聯。
也可在啟用 `patch` 環境後執行 `./script/start_scoped_worker.ps1 -DatasetRoot $dataset`。
同一 store 仍由原 claim 機制單一 worker 執行；中斷後同機恢復會先驗既有 tensor 與 hash。
無工作就等待；在共享 `dataset/jobs_state/STOP_symmetry_filter_20261007` 建立空檔可於 job 邊界停止本輪 worker。
不要刪舊 STOP，也不要使用 kill-python 或啟動 grind_loop。

## 派工順序

`check-dup` 與 `jobs-add` 分開執行。每個新 input 自含 config/measurement/score_spec；profile 不符時在 HFSS 前拒絕。
同一圖形跨 measurement 可以重測；同 measurement 的普通重複會拒絕，標明 repeat 的公證樣本另記。

初始只排 `dedust_r81smoke`（6）、`dedust_r81discrete`（2）、`dedust_r81mesh`（2）、
`dedust_r80b1`（48）與 `dedust_r80repeat1/2`（各2）。`dedust_r81b1` 的60筆只備妥。
等前三個 store 全數完成、hash replay 通過，再執行：

```powershell
python -m script.prepare_symmetry_filter check-wide --dataset-root $dataset
```

必須顯示 `engineering_check_passed: true`；否則先調查，不能修改門檻強行放行。
真實 HFSS 的這一步目前尚未執行。完成後才用同一 check-dup → jobs-add 流程送正式 dual 第一批。

## 回填與後續批次

將結果保留在共享 dataset，或完整複製到另一個明確 root；不要只帶 results.json 而漏掉 tensor/rad/markers。
下面在開發機執行，`$work` 是原備料輸出的 `single` 或 `dual`：

```powershell
python -m script.exploration feedback --work-dir $work --batch 1 --dataset-root $dataset --result-manifest "$dataset/dedust_r80b1/results.json"
python -m script.exploration train --work-dir $work --through-batch 1
python -m script.exploration select-batch --work-dir $work --batch 2
```

dual 改對應 store/work。先保存送測預測誤差才訓練；後續複製 `batches/batch-002_input` 至新 input 名稱、查重、排隊。
最多 batch3。重測/mesh 工程 store 不回填 SM。首批 cold start 的預測缺失會如實列示，不能寫成 SM 選優成效。

量測身分與 score 身分分開：改門檻可離線重新評分，但不改原 results、measurement 或 score 檔。
舊 SM loader 排除帶 `measurement.json` 的新 store；不把49點補成17點或以舊資料端值填補寬頻。

## 目前可證明與待完成

本機測試可驗 config/幾何/頻率邊界、fake COM、派工隔離、真值 hash、SM 假閉環與恢復。
它們不證明 HFSS 量測已成功。worker 尚待使用者啟動；本機 `script.status --factory` 因 T: 未掛無法取得 NAS 現況，
舊操作板的「三台已啟動」等記錄不是本次 live 狀態。
