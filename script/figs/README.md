# script/figs — 歸檔 round 的圖表生成腳本（可重現性存檔）

`docs/log/assets/round-NN/` 各圖的來源。一次性分析腳本、非模組（直接
`python script/figs/<name>.py` 跑,路徑寫死 repo 位置）；資料來自 NAS 的
dedust store / pool 快取。R6 的圖走 `python -m script.expected_best report`（正式模組,不在此）。

| 腳本 | 產出 |
|---|---|
| `r8_figs.py` | round-08 四圖（除塵/補洞/SM 體檢/random 基線） |
| `r8_pattern_gallery.py` | round-08 除塵前後 pattern 實例 |
| `pool_families.py` | round-09 池 top-300 家族普查圖 |
| `r9_figs.py` | round-09 四圖（校正/探索全景/SM 體檢/冠軍 gallery） |
| `r9_champ_curves.py` | round-09 冠軍三傑三標曲線 |
| `r10_figs.py` | round-10 三圖（w17 血統/三標曲線/承重熱圖） |
| `report_r1r10_style.py` | R1-R10 成果報告圖共用風格（下三支 import） |
| `report_r1r10_online.py` | 報告 F1-F3（紀錄時間軸/迴圈示意/線上線五輪疊圖）→ `docs/report/assets/` |
| `report_r1r10_batch.py` | 報告 F4-F11（R6 期望邊界/分布、R7 除塵、歸因、R8 四臂、R9 家族/校正/s05） |
| `report_r1r10_champs.py` | 報告 F12-F16（血統/承重圖/八冠軍/曲線/+0.48 案例） |
| `report_r1r18.py` | 總報告 E1-E6（每輪最佳 gallery/血統鏈/紀錄時間軸/帶外 Pareto/分組答案/新王 a024）→ progress-r1-r18 |
