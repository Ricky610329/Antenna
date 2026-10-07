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
