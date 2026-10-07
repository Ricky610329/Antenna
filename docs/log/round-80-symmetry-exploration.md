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

### 19:05 親本家族識別修正

對照34筆實際seed manifest發現，同一`t07_top`家族的多個鏡射變體具有不同seed ID。初版SM生成器以seed ID記ancestry，首波收據的13種ancestry因此是seed來源數，不能當作13個獨立家族。後续生成器改用明確canonical_group/lineage，再退回個別seed ID，保留source_id追溯個體；11項選樣回歸通過。已排入NAS的32筆輸入/預測不回寫；其訓練分組另以已知seed→家族對應處理，不藉修改原始資料補證據。

### 19:08 新資料批次訓練介面

`symmetry_training.py`已接上新資料：具名profile成功快照經hash/曲線重播後累積，更新觸發只計唯一圖形，重測可保留但不灌水；固定家族分組保留20%，已知34 seed→13家族映射與來源manifest hash綁入protocol。歷史非對稱資料僅作預訓練，先排除與當前保留集相同圖形/家族的訓練資料；共享由當前訓練子集估計的input/target標準化，避免兩階段換座標系，驗證集不參與選checkpoint。

每個版本重新初始化三個MLP（625–512–512–256–216），分別seed0/1/2；batch128、Adam lr0.001、歷史30ep＋當前100ep。216維為兩條17頻點響應＋兩切面各91前半球採樣；原HFSS完整181角度仍永久保存。這是當前工廠的新全曲線模型配置，**不是宣稱沿用v108權重或已比v108好**；前瞻誤差與選樣效果另行觀察。checkpoint保存Adam/RNG及epoch，模型成員、資料版、量測身分、唯一計數與歷史先驗hash均綁定，不能以複製同一成員偽造分歧。

訓練protocol已在`tmp/r80_factory_20261007/training_v2`初始化，[快照](assets/r80_factory_training_protocol_20261007.json)可查；先前未訓練的`training`保留。完整回歸585 passed /390.61秒，golden bytes不變、零warning；最後家族別名/模型介面再驗17 tests /3.13秒，pyflakes通過。另有Sol只讀整合檢查與真checkpoint形狀/複製成員拒絕測試。歷史資料掃描正在背景進行，尚未啟動真實資料SM訓練；滿48個新真值後才建立首個更新版本。

### 19:15 實測進度與歷史先驗登錄

三機實體稽核為50次有效量測、47個唯一圖形；主批22/48，兩個補池13/16及11/16，重測兩組均2/2，無警報。中位solver time175秒；重測不灌入唯一數，SM預測亦不計入。[實測收據](assets/r80_factory_snapshot_20261007_1915.json)綁定jobs與input/store metadata；下次例行監看19:45。

歷史備份只讀掃描791 stores，43,846筆通過圖形到response唯一配對、有限2×17響應及完整場型檢查；31筆配對不明/缺失、5個缺input store另列。依預先固定hash排序取12,000筆，含11,254個唯一圖形、4,323個lineages；登錄和實體快取61.14秒，原歷史檔不修改。固定分割為legacy train10,150、holdout1,850，但當前保留集圖形/家族仍在每版訓練前額外排除。來源條件混雜/未知，不能用作本profile真值或驗證。這是資料準備里程碑；尚未由登錄本身產生任何新SM準度結果。[登錄與來源hash](assets/r80_factory_legacy_20261007.json)。

### 19:23 首個新資料SM完成

19:19成功凍結52次量測→49個唯一圖形（3個實體重複移除），全部含有效S11/Gain及原始181角度場型。來源store繼續由worker寫入，凍結副本另行保存。當時送測前有效預測為0（23筆明示fallback、26筆未提供），因此前瞻ρ尚不可計算。data-v001固定家族分割為32筆train、17筆holdout，29/12個canonical groups互斥；20%是家族hash門檻，不保證每批剛好20%資料。

三成員CPU訓練162.895秒，CUDA未初始化；模型、資料、protocol及歷史先驗身分均綁定。Sol實體稽核與conductor獨立重算17筆保留集誤差一致：S11 MAE1.9606 dB、Gain3.7696 dB、phi0 3.1413 dB、phi90 2.6610 dB，總MAE2.8955。訓練/保留集差距大，這是首版更新工程里程碑，不能宣稱優於v108或盲選；後續仍保留40/30/30並以送測前預測評估新量測。[模型與重播收據](assets/r80_factory_first_fit_20261007.json)。

首版模型及完整49筆凍結資料已另存私人NAS `experiments/r80_symmetry_20261007/sm_versions/data-v001`，121個payload檔37,708,863 bytes，加上archive manifest共122檔；原子發布後逐檔SHA256重讀全部一致。含歷史先驗manifest/provenance，但不複製12,000筆本機快取。沒有動公共NAS或worker佇列。[備份收據](assets/r80_factory_first_fit_archive_20261007.json)。

49筆初始描述統計：金屬鏡射49/49；26.5–29.5 GHz目前沒有任何一筆Gain全帶≥4或S11全帶≤−10。±45°線性功率鏡射残差中位phi0=0.2685、phi90=0.0209。這一早期子集合以盲選/幾何探索為主，尚無同條件非對稱對照，不能據此推論對稱改善/損害性能。原地形圖的左右金屬差x軸在本輪恆為0；後续視覺化須改用獨立幾何特徵，不能沿用該軸而假造二維分布。[描述統計](assets/r80_factory_first_snapshot_statistics_20261007.json)。

### 19:42 滾動controller驗證

單輪controller完成physical snapshot→精確新唯一集合→凍結前瞻誤差稽核→累積重訓→候選池→分片→逐片獨立check-dup/jobs-add。prepare與dispatch都有具名profile/protocol/模型/資料綁定；guided通常16筆一片，總額末尾可明記不足16筆，待跑不超過96，實測加預留不超過10,240。局部訓練checkpoint、bundle與派送收據可恢復，kernel locks避免重複controller；正常results.json更新使用專用可延後例外，身分/manifest損壞仍停止。

watcher每1800秒啟動週期，支援私人scope/global STOP與本機STOP、設定/配置變更拒絕、重啟attempt保存及錯過tick跳過。只驅動本機SM及私人NAS派工，不開HFSS、不改live worker；完成資料額後停在對稱collection邊界，R81另走工程檢查。conductor整合74 tests /12.46秒通過；最後競態分類修正後cycle+watch再驗34 tests /6.81秒，Sol獨立34 tests /6.82秒與bounded review通過，pyflakes/diff通過。此段只記程式驗證，實際常駐啟動另記。

### 19:45 實際新SM選樣及常駐啟動

從已提交`e0716ba`實際prepare 18.798秒：當時73次有效量測、70唯一；主批26/48，兩盲選補池均16/16，冷啟動SM小批8/16及3/16。data-v001仍綁定49筆，未把新到的21筆誤稱已訓練。新SM評估10,000候選，選48（性能LCB19/分歧15/盲選14），凍結三個16筆片段；guided既有待跑42，加本波48為90，不超過96；真值＋全部預留192，不超過10,240。

conductor核對模型hash/配額/派送預算後，19:45:17以hidden process PID13976啟動watcher。第一輪逐片check-dup/jobs-add完成，加入`dedust_r80cbdb25da6g01`–`g03`，12個jobs總額；每片NAS bytes與queue重讀一致，沒有啟動本機HFSS。watch狀態waiting、completed_cycles=1，下次20:15:19；後續按48新增唯一量測觸發更新。啟動時監看code已凍結，三台worker不需重啟。

模型/資料首版已在私人`sm_versions/data-v001`，本輪action/SM audit/完成標記/設定/啟動/prepare收據另外逐檔核對存到私人`controller_receipts/bdb25da6e4e419869f1bef4da6d89b14efe28c4c658ac55ee8299cf8d34292a5`。[完整啟動與派送證據](assets/r80_factory_watch_launch_20261007.json)。48筆新候選尚待HFSS，不灌入70筆實測計數；新SM選樣效益仍待未來真值驗證。R81尚未派送，對稱collection與spec工作皆未宣稱完成。

### 20:09 首版資料的可重跑對稱與頻率圖

擴充既有`symmetry_analysis.py`的`profile`入口，對19:19凍結的49個唯一圖形逐項重播具名量測、橋接鏡射、sample/rad hash及source bindings；新分析不重新讀取live store。JSON保留來源、原始manifest/result與描述分位數，NPZ保留對齊的2×17響應與2×181場型。這49筆已經是資料版本v001的去重集合；不是20:09最新累計筆數，也沒有新增HFSS。

左右金屬差在49筆皆為0，因此地形座標改成金屬面積比例及上半部減下半部金屬比例，固定範圍[0,1]×[−1,1]、24×24六角格，只有有樣本格取中位數，不插值。此早期圖格多數近單點；顏色不代表連續曲面。頻率圖畫全部49筆與逐頻率中位數／25–75%分布，區間不是信賴區間或單一可製作圖形。兩個28 GHz GainTotal切面的±45°鏡射殘差不是完整3D對稱；去重後的單次圖也不估重測變異。

![v001幾何地形與指標](assets/round-80/data-v001_geometry_terrain.png)

![v001 S11與Gain頻率分布](assets/round-80/data-v001_frequency_responses.png)

conductor既有＋新入口9 tests /0.20秒通過；實際49筆以double precision獨立重算S11/Gain全帶餘裕完全一致，兩切面線性功率殘差相符至1e−12，49筆皆精確鏡射。Sol只讀review PASS，新入口3 tests通過，並核對全部row/hash與完整頻角軸。兩張PNG已目檢中文字、標籤與版面，pyflakes/compile/diff檢查通過。來源以不可變凍結副本為契約，不應在分析途中另行修改。

20:08:53前已存到私人NAS `experiments/r80_symmetry_20261007/analysis_versions/data-v001`，7個payload共1,370,732 bytes，加manifest共8檔；發布後逐檔SHA256重讀一致，程式來源一併保存。[完整來源、圖表hash與歸檔證據](assets/r80_profile_analysis_20261007.json)。本批S11/Gain全帶門檻均0/49、phi0/phi90殘差中位0.2685/0.0209，與先前初步統計一致；無配對非對稱對照，尚不證明SM選樣或金屬對稱的性能效果。watch/worker程式與排程未改動。

### 20:22 第二版SM自動回填及派工確認

常駐PID13976於20:15:19準時開始第二輪，20:17:09完成，全週期109.766秒（含稽核、訓練、選樣及派工，非單獨訓練時間）。20:16:48的實體快照為103次有效量測、99個唯一圖形；主批37/48、舊SM兩小批16/16與12/16，新SM首片2/16。相較19:43:51快照新增29唯一／32.94分鐘，短期產速52.82筆／小時；樣本時間仍短，不據此保證完工日期。

本輪先凍結50個新增唯一圖形，與v001的49筆無交集，再稽核送測前預測並累積訓練data-v002（99 unique）。30筆完整預測其實是28筆v108冷啟動＋2筆data-v001；其餘20筆缺失（包含14筆明示無效fallback）。分版本後，v108的S11/Gain前瞻MAE為2.7255/3.8740 dB、factory-score Spearman 0.8681（n=28）；data-v001為2.5430/8.7254 dB（n=2），不報兩筆的排序相關。混合30筆的高ρ不能當作新版SM有效的證據；未因低分或高誤差剔除真值。

data-v002固定家族分割59 train／40 holdout，52／13個家族完全互斥；v001的49筆與原分割保持不變。Sol只讀實體重播確認99唯一、50新圖形、別名／分割一致、歷史train與當前holdout圖形及家族無交集；三成員hash與40筆保留集重算完全相符。ensemble S11/Gain/phi0/phi90 MAE為2.6643/5.0645/3.0002/2.8407 dB，整體3.0690。保留集已由17增至40，不能直接將兩版總誤差解讀為同一測試集上的進步或退步；新模型效益仍待更多前瞻真值。

新SM從10,000候選選32（性能13／分歧10／盲選9），以兩片16筆prio1加入`dedust_r80c91d71565g01/g02`。派送前guided待跑61，加32為93≤96；實測＋全部待跑＋新派224≤10,240。conductor重新核對兩片NAS完整輸入tree/hash與jobs，總14 jobs；新32筆未算入99實測。後續輕量狀態讀取看到三台各有近期進展（主批39、舊SM第二片13、新SM首片4），無status警報；這些metadata不另冒充完整唯一數稽核。未結案job內HFSS錯誤保留給既有重試，未重啟worker或清掉錯誤證據。

第二版三模型、累積manifest、50筆新凍結曲線／場型、前瞻與派工收據已存私人`sm_versions/data-v002`，122檔、payload 28,906,941 bytes；发布後SHA256全數重讀一致，原49筆raw由已綁定hash的v001父歸檔保存。20:21再確認controller原PID存活且waiting，下次20:45:19，下一次更新門檻累積147唯一。[本輪完整收據、版本分層前瞻統計及歸檔hash](assets/r80_factory_cycle_v002_20261007.json)。尚未完成10,240筆或R81正WM目標。

### 20:31 兩版SM相同保留集的配對診斷

以原v001已固定的17筆／12家族作共同基準，另在v002的40筆／13家族上對兩版重播CPU預測。兩版current與實際legacy train均核對無任何比較圖形或家族交集；17筆選取規則未依v002表現改動。兩版模型已凍結、v002已派工，這次未改訓練、checkpoint或選樣政策。

| 相同資料上的MAE（dB） | 固定17：v001 → v002 | 擴充40：v001 → v002 |
| --- | --- | --- |
| 全216維 | 2.8955 → 2.9031 | 3.2708 → 3.0690 |
| S11 | 1.9606 → 1.9162 | 2.6793 → 2.6643 |
| Gain | 3.7696 → 3.9323 | 5.1492 → 5.0645 |
| phi0 | 3.1413 → 3.3355 | 3.2594 → 3.0002 |
| phi90 | 2.6610 → 2.4628 | 3.0418 → 2.8407 |

固定17總誤差幾乎持平，各分量有改善也有退步；擴充40有24/40筆總誤差下降，整體改善主要來自場型。全216維等權MAE內有182維場型，不能將這個總數當作S11/Gain改善。新增23筆屬於本次資料收集分布，擴充40是回溯診斷，並非獨立前瞻確認；家族內樣本相關，不宣稱統計顯著性或SM選樣效益。

實際CPU兩模型重播完成，CUDA未初始化；Sol另由綁定NPZ以float64重算，彙總差最多2.83e−7、逐筆差最多6.55e−7，review PASS。完整預測／真值矩陣、診斷程式及收據存私人`analysis_versions/model-compare-v001-v002`，4檔發布後hash核對一致；[配對診斷證據](assets/r80_sm_matched_holdout_20261007.json)保留兩組全部IDs、模型版本與來源hash。原HFSS及模型檔未變動。

### 20:51 第三輪巡檢：126唯一，沿用SM v002

controller於20:45:19開始、20:45:51完成第三輪（31.438秒）。20:45:32實體稽核130次有效觀測、126唯一圖形；相較模型訓練99筆只多27，尚未到147更新門檻，所以本輪不重訓、不新建前瞻訓練稽核。仍用data-v002從10,000候選選16（性能6／分歧5／盲選5、10個祖系），查重後以同tier prio1派`dedust_r80c28d601d2g01`。guided待跑66＋16=82≤96；全部實測＋待跑＋新派=240≤10,240，16個新派候選不是新實測。

Sol只讀核對三模型hash、binding、本機及NAS輸入tree、queue登錄，對訓練99及14個已綁定job manifests去重皆零交集；完整NAS歷史掃描及130份raw/rad重播未重做，前者依controller的逐片查重通過收據、後者依本次物理稽核收據。conductor另驗NAS tree與queue並執行status，三個v001小片均有近期進展，alarms為空。20:51原PID13976仍存活且waiting，下次21:15:19；未重啟worker、controller或啟動R81。

第三輪完整輸入及action收據已保存私人`controller_receipts/28d601d2...`，46檔、payload465,471 bytes，發布後逐檔SHA256重讀一致，綁定既有v002模型歸檔。這次沒有新模型或新訓練快照，原始觀測仍在來源store；[第三輪證據](assets/r80_factory_cycle003_20261007.json)保存稽核時間、限制及獨立review範圍。10,240對稱蒐集與R81正WM目標均未完成。

### 21:30 第四輪：155筆訓練v003，訓練後156唯一

controller於21:15:23開始、21:17:50完成（147.781秒）。先凍結56個新增唯一圖形，累積155筆訓練data-v003（100 train／55 holdout）；21:17:22更新後再次稽核得到160次有效觀測、156唯一。兩個數字是不同時間的快照，第156筆未進本次SM，留待下一更新；下一門檻203唯一。

conductor唯讀重播完整56筆凍結store，核對模型binding、三個member hashes與55筆保留集sample/rad hashes，再重新預測得到S11 MAE2.4814、Gain5.0850、phi0 3.1930、phi90 2.9185、全216維3.1703 dB，與模型收據一致。保留集由40增加至55，這些總數不能作配對版本改善結論。本輪45筆完整前瞻分為舊v108 4、v001 35、v002 6，保存原模型身分，不用混合誤差推論v003選樣效益；尚未完成v003的前瞻驗證。

新模型選32候選分成兩片16筆，同tier prio1派`dedust_r80c452a2930g01`／`g02`，本機與NAS輸入tree、queue及profile均核對。guided待跑52＋32=84≤96，實測＋全部待跑＋新派272≤10,240；新派不計HFSS成果。21:30執行`script.status --factory`無警報；部分job仍在已認領狀態，claim不等於即時worker心跳，未重啟worker或另開controller。

模型、56筆新凍結真值及收據已存私人`sm_versions/data-v003`，135檔、payload29,192,207 bytes，發布後逐檔SHA256重讀一致，父歸檔綁定data-v002（前99筆）。[本輪證據](assets/r80_factory_cycle_v003_20261007.json)保留模型重播、前瞻分層、派工及歸檔hash；未再次全掃所有live原始store與全部歷史，156數量依controller實體稽核收據。R80資料蒐集與R81正WM目標仍未完成。

### 22:01 第五輪巡檢：179唯一，沿用SM v003

controller於21:45:23開始、21:46:18完成（55.672秒）。21:45:42實體稽核183次有效觀測、179唯一圖形，含4次重複；相較v003訓練155筆新增24，未到203更新門檻，因此不重訓、不建立新的訓練前瞻稽核。以既有v003從10,000候選選32，分成兩片16筆同tier prio1派`dedust_r80c743c7eb1g01`／`g02`；guided待跑61＋32=93≤96，實測＋全部待跑＋新派304≤10,240。新派候選不計實測成果。

Sol只讀重算cycle ID、三成員模型與155筆實體訓練資料hash／100 train、55 holdout分割；32份保存預測及8個score欄位重播最大誤差≤2.9e−6。兩片本機／NAS輸入tree、queue及量測profile一致，與之前17 manifests的272唯一圖形及模型155筆無新圖形交集。本次未重做183份live raw/rad重播或全部歷史查重，依controller已綁定的實體稽核及逐片check-dup收據。

21:51確認之前`r80cbdb25da6g_00035_948bdee0`的RPC超時已由原worker重試成功，該job16/16且.done存在；沒有人工清理或重啟worker。22:00實際執行`script.status --factory`無警報，三個已認領job均有近期結果；原PID13976仍存活、waiting，下次約22:15:23。179是21:45快照，後續輕量metadata不另冒充更新的唯一計數。

本輪action／輸入／稽核收據已保存私人`controller_receipts/743c7eb1...`，82檔、payload891,732 bytes，發布後逐檔SHA256重讀一致。[第五輪證據](assets/r80_factory_cycle005_20261007.json)保存root與Sol驗證範圍、重試恢復及歸檔hash。10,240筆對稱收集與重測確認的新濾波器WM > 0目標均未完成；R81仍未派工。

### 2026-10-07 使用者調整監看方式

讓HFSS與SM背景循環持續運作，每30分鐘只做健康檢查；沒有需處理的結果時掛著即可，不持續主動喚醒。例行小批不再額外反覆全量重播、獨立review或發進度訊息；錯誤／停滯、重要結果驗證及階段邊界才介入。原controller補池、每48新增唯一量測的前瞻稽核及SM更新機制維持，原始資料與自動收據保留；不因此重啟worker、另開controller或提前派R81。

### 2026-10-07 22:28：目標改5,000，區分兩種停滯

使用者將對稱實測目標由10,240改為5,000個有效唯一圖形，覆蓋之前至少10,000的數量授權；性能停滯要通知，工作停滯自主處理。正式factory config只改target，量測／score／學習protocol不變；移除validator內過時的10,000硬編碼下限，仍驗正整數預算、40/30/30配額、同tier及48新真值更新。已派輸入與原測試資料不回寫，最後尾批可少於16，派工前逐片重查實測＋預留不超額。

舊PID13976完成22:15第六輪（202唯一、仍用v003，新增一個16筆job）後，在22:16:43依local STOP正常退出，process handle已消失；僅停止開發機controller以重新綁定config，不停HFSS。22:20實際status顯示三個已認領小批均有近期成功結果，佇列還有待跑；old state／attempt／10240 config及v003 model hashes保留於本機handoff，正式commit後以新5000 profile恢復唯一attempt。

conductor完整回歸677 tests通過（370.14秒，無warnings）；其後新增5,000目標的最後8筆尾批及4,999＋2超額邊界驗證，factory／cycle／watch共42 tests通過（7.54秒），正式程式未再變。pyflakes、diff通過，golden未改。本次沒有worker部署或HFSS producer變更。

性能判讀參照既有`stall-protocol`，只依綁定到選樣時SM版本的實測S11/Gain／場型，多軸及前緣共同判讀。三個成熟版本合計≥96真值才具備提醒窗口；cold-start、重測、無綁定、預測分數、SM保留集MAE或數量增長不能當性能證據。詳細政策保存私人handoff，屬conductor的advisory判讀，未宣稱controller已自動通知；目前沒有足夠成熟版本支持停滯宣稱。工作故障仍由conductor修復，性能提醒不擅自停HFSS或改物理搜索範圍。

### 22:35：5,000目標恢復首輪與工作故障

新controller PID16556於22:31從已提交`235a52e`隱藏啟動，launch `1e4bc90a746e42f9a52ec0be7a29f324`，與舊PID13976沒有重疊。首輪22:31:05至22:33:28（143.375秒），以211唯一圖形訓練data-v004（相較v003新增56），更新後live稽核212唯一；這兩個數字屬不同快照。新SM派`dedust_r80c2e16eb62g01`一片16筆、prio1，guided待跑76＋16=92≤96，全部實測＋預留＋新派336≤5,000。原量測／score與v003模型及protocol hashes未改；首輪功能恢復已驗證，未在本次額外宣稱完整v004模型性能重播或改善。22:35新controller存活且waiting，下次約23:01。

交接狀態、旧10240／新5000 config、STOP證據、新attempt、首輪action及性能停滯政策已保存私人`controller_handoffs/target5000_20261007`，14檔、payload83,246 bytes，manifest SHA256 `47f7af521e1f630bfb1d4ab9f05d576b3ef302d6a16425a282dfee73e56b27dd`，發布後逐檔hash重讀一致。[恢復收據](assets/r80_target5000_resume_20261007.json)綁定驗證範圍；不需為目標變更重啟三台HFSS。

22:35實際`script.status --factory`另報`dedust_r80c91d71565g02.fail`；原37號機本批15/16成功，剩餘`r80c91d71565g_00015_659f82ca`三次COM `0x80070223`，22:33:16因profile未完整而停worker。claim不是活機心跳，不能以它宣稱三台仍在線。原15筆成功與error／fail保存；這是工作故障，獨立於性能停滯，正在查閱歷史與恢復機制。歷史同錯曾涉及磁碟滿，但本次原因未驗證，不以舊事件直接歸因。

### 22:45：故障資料凍結，未改重試帳或成功結果

故障store、原16筆inputs／profile、claim／fail及凍結producer已保存私人`operational_incidents/worker37_20261007_223316`，59檔、payload491,104 bytes，manifest SHA256 `a25c1a599c5023dc797440126ec258ea5dbc8ab16b342ad183ee12a3f35946e3`，發布後逐檔hash核對。以`verify_completed(require_complete=False)`實體重播全部15筆成功的sample/rad與指標通過；副本與來源copy前後hash一致。[故障收據](assets/r80_worker37_fault_20261007.json)明記原error三次、沒有.done；不刪claim／fail、不修改live results、不重置attempts。

既有跨機接管會在下一個job邊界取此fail，初次todo對attempts≥3的error仍試一次，後續retry passes才限制<3，因此不需人工清帳或重跑15筆成功。37號的SSH／WinRM端點無回應，Windows DCOM唯讀disk inventory驗證被拒；没有遠端變更，已透過非阻塞問題請使用者在該機原終端重啟。其他兩機目前仍有已認領jobs，未干涉其HFSS。針對新scoped worker的單筆殘留錯誤容錯正在另行修正，尚未宣稱已部署。

### 23:13：scoped worker部分HFSS失敗容錯通過，待機台部署

本次工作故障修正僅放行具型別的部分HFSS終態失敗：全部manifest IDs均有終態、至少一筆成功經sample/rad hashes與原始曲線重播、殘留僅HFSS錯誤且attempts≥3。profile observation無效、錯scope、schema/hash損壞、缺／外來ID及全失敗不放行。連續scoped worker保留fail／claim／raw且不寫.done，本機跳過失敗store繼續其他jobs，記憶集合避免跨機接管換claim空窗再搶同批；三批連續部分失敗停機，完整成功才重置計数，`--once`與舊無scope仍回報失敗。

獨立Sol review發現部分失敗會跳過原正常暫存清理，可能持續吃滿磁碟，已在HFSS關閉及原始成功重播後修正：只刪本job預設`_dedust_<store>`，先驗resolved parent等於cwd且basename完全相符；自訂out／其他歷史目錄保留。路徑不符或清理錯誤轉普通停機，不能假裝修復後續跑。status保留批次失敗警報並分列續跑決定；決定不是心跳，已明記終態的partial claim不再重複當作仍在求解的stale job。

conductor最終完整回歸**695 passed／328.50秒**、無warnings；golden未改，pyflakes／diff通過，獨立Sol最終只讀審查無未解重要缺陷。新增10項HFSS替身整合測試及31項相關回歸通過，涵蓋補測attempts≥3一次、失敗續跑／第三批停機／成功重置、claim空窗記憶與當批暫存安全清理；status9項通過。先前完整回歸691通過／4項Windows暫存路徑過長，短目錄重跑factory-cycle20項後，再以短fresh basetemp完整通過；未改程式邏輯或golden以迴避失敗。新驗證器對22:33私人凍結故障的15筆成功與1筆error也通過相容重播。[修正收據](assets/r80_scoped_worker_recovery_20261007.json)綁定最終source hashes、測試log hash與部署限制；沒有在本機啟動HFSS或變更live重試帳。

23:02按30分鐘節奏做輕量health check：controller PID16556存活，第二輪23:01:05–23:02:04、58.625秒，稽核227唯一，沿用v004。派`dedust_r80cc154845dg01`一片16筆、prio1，guided76＋16=92≤96，全部實測＋預留＋新派351≤5,000；下次約23:31。另兩批12/16、8/16均有2–3分鐘內的新結果；37號fail仍在，未確認恢復。未額外全量重播本輪原始資料、模型性能或NAS歷史；227依controller具名稽核，原自動收據保留。R80／R81目標仍未完成，沒有性能停滯結論。

新worker需在正式機pull GAN並於job邊界重啟原行程才生效；本次只驗程式与凍結故障相容，沒有遠端權限替37號啟動，亦未停／重啟另兩台仍工作的worker。不以推送等同部署，不另開第二份controller或重複worker。

### 23:35：218完成跨機補測，使用者重啟37

使用者回報已重啟37並貼出22:33舊停機log，允許需要時打補丁後pull新指令；既有修正`c640fdd`已在GAN，不重寫或新增不必要的補丁。23:33實際status無警報，原fail已消失：218於23:11:26帶`prior_fail=[37]`接手`dedust_r80c91d71565g02`，23:15:11.done為16/16、零error。conductor重播全16筆原始sample/rad與指標通過，與私人故障副本對比原15筆results及實體hash均未改；只補原error的`r80c91d71565g_00015_659f82ca`，沒有重置attempts或重跑已成功15筆。[恢復收據](assets/r80_worker37_recovery_20261007.json)保存實際新結果與claim／done。這個圖形跨機可成功求解；不由此推定37的具體COM／磁碟病因。

37於23:31:35新認領`dedust_r80c9dfe3fc7g01`，與使用者重啟回報一致；第一筆新真值及執行中Git revision仍未驗證，不能僅憑claim宣稱補丁已部署。另兩機current jobs為4/16與2/16、均有近期結果。controller PID16556實際存活，23:31第三輪46.265秒完成，稽核246唯一、仍v004，下次約2026-10-08 00:01。例行health未重播全部live資料或模型；本次16筆物理重播只為驗證工作故障確實恢復，R80／R81目標仍未完成。
