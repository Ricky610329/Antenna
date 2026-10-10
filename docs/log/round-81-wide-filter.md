# R81：新版濾波器 spec 與寬頻量測

日期：2026-10-07。狀態：實作與本機備料；尚無寬頻 HFSS 真值。

**目前性能目標（使用者2026-10-07補充）：新版spec的實測WM > 0 dB。** 完成程式改動、工程檢查或有限批次均不等同達標；WM仍為負或恰為0時，此正餘裕目標仍未完成。

2026-10-11使用者先明示後續自行接濾波器，再進一步指定「到5000筆就開始做」。最新時序以此為準：一確認R80有5,000有效唯一實測，立即開始R81備料、歷史先驗與工程準備，R80完整凍結、重測、真值稽核、統計與圖表同步完成，不等待全部圖表收尾。工程HFSS開始前仍須實際validated5,000／零pending的release證據、私人獨立dataset與worker入口部署；先6 Fast＋2 Discrete＋2 mesh，通過才接有限正式批次及正WM確認，不再等待額外研究計畫確認。工作仍限Antenna／GAN及鄒穎麒私人工作區，已驗證里程碑持續commit並push。目前尚無R81實測，有限批次結束不代表新spec達標。

## 規格

| 量 | 頻帶（GHz） | 門檻 |
| --- | --- | --- |
| S21 通帶 | 26–30 | ≥−3 dB |
| S21 左阻帶 | 16–20 | ≤−20 dB |
| S21 右阻帶 | 36–40 | ≤−20 dB |
| S11、S22 匹配 | 26.5–29.5 | ≤−10 dB |

使用者希望 20 以下/36 以上「盡可能」抑制；先採 16–40 GHz 作有限工程觀察窗。
不宣称 16 以下或 40 以上已達標。20–26 與 30–36 為過渡帶，不加入 worst-margin。
每段取最差點，總分取五段 margin 的最小值，並保存每段最差頻率。

2026-10-11使用者再次強調spec有「緩衝空間」：兩側各保留6 GHz過渡區間，即20< f <26及30< f <36 GHz，讓S21在通帶和阻帶之間過渡。過渡區間完整保存、畫圖，但不增加額外S21門檻，也不納入WM。现有`configs/dual_r81_wide_filter.yaml`已只列16–20／26–30／36–40三段S21 bands；`antenna.measurement.score_response`只評分各band的含端點切片，與此理解一致，因此不更改spec或runtime。20／36 GHz端點沿用阻帶門檻，26／30 GHz端點沿用通帶門檻；仍只有16–40 GHz有限觀察窗的實測能作達標證據。

同日使用者進一步更正：「這個緩衝空間也還是希望壓下去的……不參與wm，相對下壓就好」，並補充老師的用語是「平緩」。所以過渡帶是**有優化目標的軟性區間**，不是完全不在意：從26／30 GHz通帶邊緣往20／36 GHz阻帶方向，希望S21平緩衰減，盡量降低相對傳輸、波紋與突起。保留原五段WM與正WM確認；不把過渡帶硬設為−20 dB、不加入第六段hard margin，也不強制每個離散點嚴格單調。後續保存並比較相同頻率軸的實測過渡曲線、相對下壓與平順程度；WM表現可比時，將較好的過渡響應作為候選／親代的軟性偏好，並分列結果。現有五margin SM尚無過渡帶品質輸出，不能宣稱已學會或正在用此軟目標排名；需在R81真值回填與選樣時明確接入、保存來源與排名依據。

舊 `wm_mfg` 的 −15 dB 阻帶、24–32 GHz 量測不能換個門檻冒充新真值。
沿用 R69 後 p01 幾何與 0.075 mm 橋；p00/無橋資料不納入本 profile。
舊冠軍與專項候選只提供圖形起點，重新量測後才能評新規格。

## 量測前檢查

新量測為 16–40 GHz、0.5 GHz，共 49 點；setup=28 GHz、open-region=16 GHz。
先做 6 個 Fast 試跑（歷史、專項、隨機各兩個）。其中兩個再各做 Discrete 與較細網格
（MaxDeltaS=0.005、最多20 passes），分 profile、分 store 保存，總計 10 次。
比較每個點及五段 margin 的最大差值；任一 >0.3 dB 就先調查頻點、邊界與網格，不放行正式批次。
這是有限工程檢查，不是完整數值收斂證明。新 CSV 缺兩端點會拒收，禁止靜默端值外插。

## 有限探索

首批 60：20 個歷史折衷解、20 個專項解（舊 m3/m4 各10）、20 個新隨機圖形。
後兩批各 60：SM 預測最差 margin、ensemble disagreement、隨機各20。
最多三批共180，工程檢查10次另外記帳。初始6筆與正式重測分開，不能把檢查值混入訓練。
兩埠 5×5 饋墊固定，橋寬及量測 profile 固定；每批先做預測稽核再訓練。
五段達標數、最差頻率、各臂真實進步、負結果及樣本覆蓋均報告。

配置：[dual_r81_wide_filter.yaml](../../configs/dual_r81_wide_filter.yaml)。
操作：[共同 runbook](symmetry-filter-20261007-runbook.md)。正式60筆在檢查通過前不排隊。

## 2026-10-07 排程更正（append-only）

本輪R81整體延後。上列規格、工程檢查與有限探索保留為研究設計紀錄，但目前不得執行 `dedust_r81smoke`、`dedust_r81discrete`、`dedust_r81mesh` 或正式批次，也不得混入R80的52次佇列。

`prepare` 的預設 `--phase single` 不讀dual歷史、不產生R81輸入；待進入後續R81階段才使用 `--phase combined`。目前沒有R81 HFSS真值或新工程檢查結果。

## 2026-10-07 18:16 台北：接續執行授權

使用者要求每30分鐘監看，直到目前對稱任務與spec工作完成。R81在R80完成後接續，不與R80首批混跑；保留上述工程檢查、0.3 dB門檻與最多三批設計。工程檢查不通過須先調查；完成驗證流程不等同所有規格已達標，負結果如實記錄。目前尚未派送R81。

## 2026-10-07 19:55 台北：正WM目標補充（append-only）

使用者明確補充「濾波器spec希望wm是正的」。本輪的WM沿用具名`filter_26_30_stop20_36_v1`評分：五段各取全帶最差點，WM取五段margin最小值。S21通帶餘裕為`S21−(−3)`，兩阻帶為`−20−S21`，S11/S22匹配為`−10−Sii`；所以**WM > 0**代表目前五段門檻都有嚴格正餘裕。原逐點合格旗標允許等號，但這次優化目標要求正值；不能以WM=0、SM預測正值或舊`wm_mfg`正值代替新版HFSS真值。

五段頻帶維持上表，16–40 GHz之外仍沒有量測保證，工程檢查的0.3 dB一致性門檻也不替換性能目標。找到正WM候選後仍需按既有重測/工程驗證流程確認並保留每段margin及最差頻率。三批是原定階段性探索上限，若尚未正WM就記為未達成，判讀後規劃下一輪，不把本輪結束當作性能目標完成。此補充不改變R80保留低分有效資料的資料蒐集政策，也不提前混派R81。

## 2026-10-07：正 WM 確認工具完成（工程里程碑）

新增`script/filter_confirmation.py`，由指定新版profile唯讀重播原件與重測完整3×49響應，核對同圖形、獨立來源、repeatability親代關係、sample及metadata hashes，再確認兩筆實測WM皆嚴格>0。保存五段margin／最差頻率與pair最小WM，不改通用評分允許等號的行為。工具不評估smoke、Discrete或mesh，也不標記campaign完成；[操作指令與限制](symmetry-filter-20261007-runbook.md)。

Sol實作與conductor審查後，修正輸出可能寫入來源store及只查頻點數量的邊界；新增回歸驗證拒絕時不建立任何來源子目錄，以及錯誤的49點頻率軸。conductor新工具＋既有measurement/profiled_batch共32 tests通過（0.85秒）、pyflakes通過；均為生成fixture，尚無R81 HFSS量測、派工或正WM候選。

## 2026-10-07：歷史濾波器先驗備料 API（尚未串接訓練）

新增`script.filter_prior.prepare_filter_prior`，只讀呼叫者明示的歷史manifest與sample root。每筆須具備24–32 GHz／0.5 GHz、[S11,S21,S22]、p01／25×25／0.075 mm、饋墊及來源hash；原3×17曲線保留，只計算完整覆蓋的S11/S22匹配及S21通帶margin。兩個新阻帶mask=false，數值0只是被遮罩的佔位，不是量測或新版完整WM。五margin模型仍可沿用，不強制新增全曲線輸出頭。

當前保留集的lineage與canonical group共用排除身分集合，並排除圖形及所有同圖形別名；其餘資料再作實體圖形去重。manifest與sample均從同一份byte snapshot計hash及解析，避免驗證和讀取看到不同檔案版本。Sol及conductor審查修正跨欄位家族別名邊界；conductor新入口＋measurement/exploration共41 tests通過（6.64秒），pyflakes通過、無warnings。

這次只有生成fixture驗證，沒有掃描實際dual歷史、訓練模型或派送R81。後續仍須建立有來源證據的實際歷史清單、以固定當前保留集排除後串接masked loss、共用訓練正規化，再完成新版HFSS回填到下一批選樣的整鏈驗證。現行R80 controller及worker均未改動。

## 2026-10-07：工程檢查身分與對照覆蓋修正

發現舊`measurement_check`只信各input自帶config，可能把相同Fast設定冒充Discrete，或因parent重複而讓dict少做比較。新增`check-wide-inputs`在HFSS前核對正式60筆20/20/20來源、smoke各來源兩筆、三個具名量測設定及timeout、獨立圖形與親代、Discrete/mesh共用兩個smoke代表。combined preparation也先執行此檢查；目前仍不派R81。

收檔時完整重播三store、核對input/store身分與manifest、sample hashes、metadata前後一致，固定兩組各兩筆共四個曲線／margin比較，門檻維持0.3 dB。通過只允許工程階段後續放行，`performance_target_assessed=false`且`campaign_complete=false`，不等同正WM。

以完整60筆輸入與10筆生成觀測替換舊mock驗證；錯誤儀器、timeout、重複parent、缺來源組、不同代表、損毀曲線／profile、不完整結果及0.31 dB差異均拒絕。Sol只讀review無阻斷項並指出timeout未綁定，已修正；conductor44相關tests通過（3.87秒）、pyflakes通過。沒有新HFSS或實測工程通過結果。

## 2026-10-07：歷史先驗到新批次選樣的可選 SM 整鏈

新增`script/filter_training.py`，以`filter_masked_prior_v1`明示啟用於既有exploration生命週期，未改R81正式config。兩個五margin MLP成員每批重新做masked歷史預訓練，再訓練累積新版真值；兩階段共用只由current train擬合的輸入正規化。原始3×17與3×49曲線保留，舊資料的兩個新阻帶不計loss。保留集僅做誤差紀錄，無early stopping／checkpoint選擇；模型及Adam、RNG、正規化可由epoch邊界恢復。[啟用條件與操作](symmetry-filter-20261007-runbook.md#r81-歷史先驗與批次-sm-更新可選入口)。

Sol review指出並已修正：歷史保留集別名須作完整傳遞排除；舊split帳本須對已完成模型驗證；選樣／回填／訓練間須維持canonical親代；完整批次ID、數量與hash須被收據綁定，不能刪列、改batch或插入未稽核batch0；模型不可挪用其他批次或成員。另拒絕相對先驗路徑及非有限保留集輸出。final read-only review無剩餘阻斷項，未另跑測試或HFSS。

conductor生成CPU fixture驗證53項通過（23.64秒）：真實兩批累積回填／訓練到第三批選樣、預訓練中／階段切換／新版訓練中的恢復與不中斷結果逐tensor一致、masked阻帶零梯度、train-only正規化、保留集隔離及上述拒絕情境。最終完整回歸677 tests通過（415.30秒，無warnings），含golden檢查；golden檔無改動、pyflakes通過。這些是工程證據，尚未完成實際dual歷史清單或R81 HFSS量測；不能當作SM準度改善或新版WM > 0達標。

### 21:42 實際歷史先驗：R79／smp073的360筆

唯讀本機backup的12個具名store：R79三批各a/b/c與smp073 a/b/c。360個輸入均以相同二值圖形找到唯一原始sample，再以凍結commit `07de8e3`的原六項margin、energy與S11/S22 gap函式逐筆重播，八個rounded記錄均完全相同；所有input/store setup均只有0.075 mm橋、result均蓋p01章，pattern為25×25且兩饋墊完整。360/360配對、360唯一，沒有性能篩選；保留全部3×17原始響應及來源hash。

頻率與label來自凍結producer鏈：simulator固定24–32 GHz／0.5，三曲線按頻率對齊17點；PORT_SPECS及dedust依S11／S21／S22排序存檔。R79四個執行紀錄commit的相關producer blobs相同。但原sample未保存axis metadata、當年實際worker revision未逐筆記錄，所以這是有程式／artifact佐證的歷史profile重建，八個rounded指標也不能單獨證明axis身分；清單明記這兩項限制，只作歷史先驗，兩新阻帶仍mask。實際R81真值不採這种推定。

保守家族為DUAL_H79 180、DUAL_HS79 90、SMPOOL073 90；接到新版觀測後仍須依當前保留集排除相連歷史家族，不能先固定可用360筆。conductor原backup與凍結副本兩次loader陣列逐項一致；Sol另重播副本loader、核對全部sample及六個source SHA/blob，review通過，未再次重算八項指標。

完整原曲線、清單、重建腳本、六個凍結source及收據存私人`experiments/r81_wide_filter_20261007/historical_prior/r79_smp073_v001`，372檔、3,023,728 bytes發布後hash重讀一致；[實際先驗收據](assets/r81_actual_prior_20261007.json)綁定來源、manifest、review及歸檔。另由metadata盤點發現345個store約21,251個可能符合p01/.075的結果，尚未逐筆驗證，不計入訓練先驗。下一步擴大實體配對後再凍結正式訓練設定；本次沒有R81 HFSS、模型訓練或派工。

### 22:03 擴大歷史先驗：21,034個有效唯一圖形

沿用凍結`07de8e3`的profile重建與八項rounded指標重播，唯讀本機backup的345個p01／25×25／0.075 mm store。21,285個輸入中21,251個與唯一raw圖形及八項舊指標精確匹配；34個缺失／error結果另列，不作性能篩選。備料loader再排除217個實體重複，保留21,034個唯一圖形、105個canonical家族；來源與凍結副本兩次loader的圖形、3×17響應、三段margin與mask完全一致。當前保留集尚不存在，接到R81真值後仍需每版排除相連家族及圖形。

首個擴充掃描對每筆重讀metadata效率不佳，已中止並保留原腳本／log，當時尚未建立輸出；替代版本每個store只讀一次byte snapshot與hash，結尾核對metadata未變，再逐份保存原曲線。沒有改動原歷史、R80 controller或三台HFSS worker。全部21,251筆重播後的rounded energy_max介於0.394–0.923，沒有>1，也沒有依energy丟資料；這只是原資料一致性摘要，不是新的物理量測或axis身分證明。

原曲線、完整清單、345×4份來源metadata、六個凍結producer檔、重建及歸檔程式存私人`experiments/r81_wide_filter_20261007/historical_prior/full_p01_db075_v001`。ZIP共22,642 entries、payload143,578,836 bytes、壓縮36,802,240 bytes；全部entry SHA256及CRC本機核對，發布後NAS bundle與外部manifest hashes重讀一致。使用時解開ZIP，以原`manifest.json`和`raw_snapshot`交給既有loader；不直接改公共dataset。

[完整備料收據](assets/r81_full_prior_20261007.json)保留排除帳、345 stores的hash、源commit、唯一計數及歸檔。原sample仍未保存axis metadata，實際歷史worker revision仍未知；兩個新阻帶保持mask=false，不能由舊資料算新版完整WM。本里程碑只準備先驗，沒有R81 SM訓練、HFSS真值、派工或正WM結果；正式config仍未啟用。

Sol有界只讀review通過完整性與歷史先驗schema：全清單欄位／計數、六source SHA及Git blobs、360筆前版逐列完全包含，以及跨14 stores／家族的24筆實體tensor、八項舊指標和mask重播均一致。另核對本機bundle hash、22,642個安全且唯一的ZIP目錄項目；未再次跑完整21,034 loader、全ZIP payload／CRC或NAS，這些依conductor的完整歸檔驗證。沒有新增模型校準或效能證據。

★ **歷史重複圖形噪聲限制**：201個重複群組的217個later-vs-first對中，只有167對三段covered margins完全相同；40對差>0.01 dB、32對>0.1、11對>0.3、4對>1，最大2.559473 dB，原曲線單點最大差6.907406 dB。conductor另用保存的float32曲線轉float64獨立計算，計數與最大值均與Sol一致；Sol以loader的float32 margin及零容差判同。所有重複圖形tensor相同、舊指標各自可重播，尚不判為檔案毀損；實際差異原因未知。按store字典順序與原manifest順序取first而不挑高分，保留集別名在去重前作相連排除；先驗須視作有噪聲的舊量測，不能作當前profile驗證或校準結果。

兩份可重跑診斷程式／完整217對數值及發布helper另存私人`historical_prior/full_p01_db075_v001_review`，6檔、payload323,520 bytes發布後逐檔hash核對，綁定原先驗manifest及bundle。不回寫原ZIP與歷史量測。

### 2026-10-10：獨立R81 worker入口交付，尚未啟用

已提交`267af9c`，新增`r81_worker_entry.py`、`start_r81_worker.ps1`與focused tests，不改原R80 main／worker程式。R80與R81共用scope，故R81必須使用私人獨立dataset；不能用原R80-only launcher，也不用combined混合queue。入口要求R80精確5,000有效唯一／零pending的watch/action與最終validated census，綁定raw report、rows、proofs和producer。原始action/profile路径保留为producer_path，實際封存位置另hash綁定；census使用clone-relative Git blob，跨三台不同repo路徑不需修改原始證據。

先6 Fast＋2 Discrete＋2 mesh工程工作，四組完整曲線／五margin最大差異≤0.3 dB再放行Formal；正式依序最多三批各60，後批核對前批feedback／SM模型／manifest，positive-repeat只在正WM候選存在時另排。單機fail仍留給固定三機roster重試；完整done store要實體重播，所有roster耗盡才是終態失敗。release不允許unknown queue jobs，每次native --once前重驗；prefix更新須等待前版全部launcher成功退出，preflight到claim並非原子操作。

V1 generated focused lifecycle17項通過，但之後發現本機絕對路徑阻止跨clone，故保留V1 review、另修portable V2。V2實作者3個metadata案例及加強census案例通過，獨立靜態delta review通過，沒有重跑訓練或廣泛suite。cherry-pick後CRLF transport與原審查LF不同，三檔LF-normalized bytes／Git blobs一致；conductor在實際commit上驗證44個runtime blobs與clean filters，拒絕替換digest，並以真實3,462筆及目前watch/action證據確認R80未完成時拒絕入口。這是實際metadata gate，不是R81 HFSS或performance證明。[完整來源、審查、postcommit與限制](assets/r81_worker_entry_readiness_20261010.json)。

尚無R80最終completion、R81 release/dataset、三機新launcher、R81 SM或HFSS實測；historical masked-prior production recipe另待固定。首個包含synthetic training的實作者test命令曾在CreateProcess前因其有界task禁止training而遭自動審查拒絕，沒有執行／failure artifact；其後本次V2及conductor僅做metadata／靜態驗證。原R80 SM合法批次訓練繼續，不能把此有界test限制解讀為使用者禁止R80 SM。

### 2026-10-10：歷史masked-prior訓練配方固定，尚未訓練

`103517c`新增小型config-only builder `script/r81_prior_recipe.py`，不另造training/controller。相同R81 measurement／score／runtime保持不變，沿用60×3、候選池1024、ensemble seeds0/1及20/20/20選樣；只啟用既有`filter_masked_prior_v1`，歷史30ep＋當前100ep、holdout0.2與caller-supplied absolute prior paths，manifest固定SHA8fcd05614e7851e9c0f7043c00540068df37380423ee4469db0afed90505c777。CLI只檢查manifest hash／root存在、輸出新config並重parse，不讀樣本、不prepare、不train或dispatch。

V1獨立5個metadata tests通過，但存在exists-check後write_text可覆蓋並行寫入者的缺陷；V1候選及BLOCKED review保存。V2改exclusive open('x')，單一race-shaped metadata案例驗證sentinel bytes不變，獨立static delta PASS；不重跑training／廣泛suite。實際committed CLI已使用既有本機41,901,535-byte凍結manifest與raw_snapshot生成config SHA b3ff0084…9b6；科學ID與原21034-prior audit一致，兩檔LF-normalized Git blobs與審查候選相同。[完整來源、審查與設定](assets/r81_masked_prior_recipe_20261010.json)。

這只是設定備妥：沒有重新跑21,034 loader／current-holdout相連家族排除，沒有R81工作區、模型、release、NAS或HFSS。21,034是排除前上限；舊資料只覆蓋S11/S22 matching及S21pass，新16–20／36–40阻帶保持masked，須由當前R81真值學習。歷史噪聲／axis及worker revision限制不變。R80完成研究稽核後，才以實際paths與既有prepare→feedback→train→select-batch流程做完整preflight和分批更新。

## 2026-10-11：達5,000立即開始R81，R80分析同步收尾

使用者最新指定「到5000筆就開始做」。此指示覆蓋上方歷史段落的先完成R80全部分析才開始R81時序：達5,000有效唯一真值就開始私人R81備料、真實prior paths的完整preflight與工程準備，R80凍結、重測、稽核、統計圖表同步完成。工程HFSS仍使用已綁validated5,000／零pending的實際release與獨立worker入口；既有`_validate_r80`不要求統計／圖表完成，無需更改runtime。過渡帶相對下壓／平緩響應作軟性偏好、五段WM及原件＋獨立重測WM>0不變。沒有把達額、工程檢查或有限批次結束當成spec已達標。
