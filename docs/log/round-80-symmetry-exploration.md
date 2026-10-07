# R80：金屬對稱下的場型與頻率響應探索

日期：2026-10-07。狀態：實作與本機備料；尚無本輪 HFSS 結果。沿用 Antenna，不遷移 emforge。

## 問題與歷史

本輪問的是「金屬左右對稱時，兩個主切面的實際場型如何變化，與 S11/Gain 有何關係」。
金屬對稱是本輪實驗條件；場型對稱、S11 與 Gain 都是待觀察的結果。
不將場型不對稱量放進最佳化 loss，也不用 4 dBi、舊 ±45° 餘裕或 S11 過線篩掉送測候選。

原單埠 spec 的 Gain 為正 z 軸 boresight **RealizedGainTotal**，26.5–29.5 GHz ≥4 dBi；
S11 同頻帶 ≤−10 dB。歷史 rad 則是 28 GHz 的 **GainTotal**、phi=0/90 兩切面；
舊 ±45° 相對 boresight 的 3 dB 窗與前者不是同一個量。
本輪保留這些背景，但不新增「場型對稱才合格」的正式 spec。

歷史脈絡：R54 明確化菱形橋；R55 的事後鏡射樣本屬舊目標下的局部探針，不能據此否定新的探索。
R73 線上迴圈未勝批次流程；R75–79 及 analysis-16/17 顯示家族與採樣偏差。
因此採用原有批次架構，但把探索、量測、評分與舊資料匯入分開。
歷史統計見 [analysis-18](analysis-18-symmetry-census.md)，其快取來源限制須一併閱讀。

## 固定設計

- 25×25，左 12 欄與右 12 欄精確鏡射；中欄自由、中心饋點保留。
- 橋寬 0.1 mm；用 simulator 同一份 `diag_bridge_sites` 檢查實體橋亦鏡射。
- S11/RealizedGainTotal：24–32 GHz、0.5 GHz，共 17 點。
- rad：28 GHz，phi=0/90，原始 theta −180..180、步距 2° 全保留；SM 預測 ±90° 的兩條 91 點曲線。
- 每批 48：random、幾何覆蓋、ensemble disagreement、預測曲線多樣性各 12，最多 3 批共 144 個新圖形。
- 另選兩個幾何相距較遠的首批代表，各增加兩次重測，共 4 次，分開保存。
- 首批沒有本 profile 真值模型：SM 兩臂明記 fallback；第二、三批才用前批模型。
- 每批先比較**送測時保存的預測**與新真值，再匯入訓練；分列 S11/Gain/phi0/phi90 MAE。
- 訓練兩個 CPU MLP；按 lineage 與 geometry profile 分组 holdout，不能用近親變體隨機拆分灌高分。

## 判讀與停止

主要報告金屬 XOR 對稱度、兩切面的線性功率對稱指標、波束質心、Gain28、S11/Gain 頻率曲線、
密度與幾何覆蓋。低 Gain、差匹配與場型不對稱樣本仍保留；數值損壞/缺失則另記錯誤。
比較探索臂的覆蓋與 prospective SM 誤差，不把「找到合格解」當作本輪成功條件。
完成三批即收輪；缺資料、模型無效或 HFSS 不穩時明記原因，不暗中擴充預算。

配置：[single_r80_symmetry_explore.yaml](../../configs/single_r80_symmetry_explore.yaml)。
操作：[共同 runbook](symmetry-filter-20261007-runbook.md)。沒有新實測前，不寫成「對稱已改善場型」。

## 2026-10-07 排程更正（append-only）

本次先執行首批48個圖形及兩組各2次重測，共52次；先前「最多三批」仍是研究設計上限。第二、三批未預先產生或混入首批佇列，會在前一批收檔、稽核與模型回填後依序產生。worker由使用者啟動，目前尚未開始HFSS。

備料預設為 `--phase single`，只讀34個R55 seeds。local controller為 `tmp/r80_symmetry_20261007_single/single`，NAS初始備份為 `controller_initial/single`。已部署並逐檔比對158個檔案；`-CheckOnly`確認三個single jobs共52次，combined queue會在worker前遭拒。完整pytest **544 passed / 361.29秒**，golden bytes不變；worker實作 `29a6834` 已推送Git `main`，舊ZIP不再使用。HFSS尚未開始，實際結果仍待正式機量測。

## 2026-10-07 分支更正

使用者確認 Antenna 開發與每台 worker 全部使用 `GAN`。初次交付只推送 `main`，導致尚未更新的 `GAN` 缺少啟動腳本；已補推至 `GAN`（`b3f1064` 包含已驗證的 worker）。此後以 `git checkout GAN`、`git pull --ff-only origin GAN` 更新；操作文件與私人 NAS 啟動說明同步採用 `GAN`。原始部署收據保留首次交付歷史，量測設定與佇列不變。

## 2026-10-07 18:16 台北：啟動與監看

使用者回報三台已啟動，並授權每30分鐘檢查直到對稱與後續spec任務完成。NAS首次檢查：218主批3/48成功；216重測1的2/2完成，完整sample/rad hash與observation重播通過；重測2一筆出現HFSS RPC錯誤，既有worker尚可重試。認領檔只觀察到216/218，不能由此聲稱第三台已實測。原始快照與限制見[監看收據](assets/r80_monitor_start_20261007.json)。目前尚未將首批局部結果用於SM訓練或後批選樣。

## 2026-10-07 18:34 台北：依使用者新指示擴大資料工廠

使用者明確將主目的改為至少10,000筆有效、唯一、精確金屬鏡射的HFSS資料，盡量利用SM優化S11、Gain與場型，允許歷史非對稱資料訓練及大量低tier預備池。這項新授權覆蓋前述「最多三批即停」的總預算；舊首批48及重測snapshot保留。工廠目標10,240筆，三批仍可作研究檢視節奏，並非資料停止條件。1.5週對三台等於每台平均每筆約272秒；這是吞吐需求，尚非實際完工預測。

依[論文§III](../paper/manuscript-draft-2026-10.html#method)與歷史`select-smpool`，採用資料回饋的批次學習：新資料先做預測前瞻稽核，再以累積新真值更新模型、挑下一批。初始更新間隔48筆有效唯一新資料；重複驗證不灌水計數。SM高分/分歧/盲選約40/30/30，保留完整曲線與模型版本，低性能有效資料也入學習池。v108歷史SM只作冷啟動先驗，其rad前瞻ρ=0.213不足以宣稱可靠，因此保留軟訊號和配額，隨新資料檢查。

原48筆不可拆搶；補充job各16筆、同一tier可供不同worker各自認領。初始盲選預備池2,048筆（34個R55 seeds多親本變異與全域新圖形），先釋出64筆，SM正式小批prio1；補池prio6。prio6低於worker預設tier2讓位門檻8，避免現存「求解後、存檔前讓位」路徑浪費剛算完的一筆；16筆job在批界交回排程。本次不改live worker。配置見[single_r80_symmetry_factory.yaml](../../configs/single_r80_symmetry_factory.yaml)。

18:34:56已將四個16筆小job `dedust_r80g00111`–`dedust_r80g00114`加入私人NAS佇列；每job包含8筆全域新圖形及8筆不同親本變異，無SM分數篩選。64筆均通過具名profile、橋接/鏡射/饋點、檔案hash及獨立check-dup；重複檢查成功後才發車。預備池不是已量測資料；目前未宣稱這64筆完成。[備料與派送證據](assets/r80_factory_buffer_20261007.json)。

## 2026-10-07 18:46 台北：首波SM選樣與三機實測

使用v108三個響應模型及rad head冷啟動，10,000個候選經精確鏡射、feed/橋接驗證後選32筆；配額為性能13、分歧10、盲選9。候選生成包含多親本變異、演化及全域探索；rad只作裁剪後的弱導航項。保留模型hash與量測前S11/Gain/兩切面曲線，拆成`dedust_r80b2a/b`各16筆、prio1，獨立查重後於18:45送出。此32筆與2,048盲選預備池及初始pilot均去重；SM預測不列入真實資料點數。

完整選樣/模型hash/分片/派送收據見[SM首波證據](assets/r80_sm_wave_20261007.json)。後續模型版本/量測身分/已訓唯一樣本數由checkpoint綁定，不接受呼叫端任意宣稱；舊v108固定冷啟動0筆。`snapshot_successes`可將長批已成功的資料另建不可變子集合（標記partial_snapshot），保持原工作繼續跑，使48筆新資料的更新節奏不必等待整個48長批結束。

18:46:31私人NAS實體稽核：23次有效量測、20個唯一圖形；主批14/48、補池兩job為2/16及3/16，兩組重測均2/2完成。216的RPC失敗已由既有重試恢復；216/218/37均有認領及實測進展。solver time中位174秒尚不含全部等待/重試/重開開銷，不能直接承諾1.5週。原監看只數wm會漏掉observation-only結果，已增加具名成功欄位與私人scope入口。[本次收據](assets/r80_factory_snapshot_20261007_1846.json)；下次例行檢查19:16。

### 18:56 工程驗證

完整既有回歸及本次選樣/分片/监看套件574 passed /329.30秒，golden原始bytes不變、零warning；該次明確排除仍在開發的`test_symmetry_training.py`。最終分片快照、模型身分绑定及catalog變更另跑53 tests /4.88秒全部通過，pyflakes/diff通過。第一輪曾發現catalog將新factory模式誤套舊三批validator，已改為專用完整factory白名單驗證；未修改live worker求解程式。訓練模組及實際新資料更新是下一個獨立里程碑，本段不宣稱已完成訓練。
