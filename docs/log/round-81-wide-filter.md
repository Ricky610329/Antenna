# R81：新版濾波器 spec 與寬頻量測

日期：2026-10-07。狀態：實作與本機備料；尚無寬頻 HFSS 真值。

**目前性能目標（使用者2026-10-07補充）：新版spec的實測WM > 0 dB。** 完成程式改動、工程檢查或有限批次均不等同達標；WM仍為負或恰為0時，此正餘裕目標仍未完成。

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
