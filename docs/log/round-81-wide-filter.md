# R81：新版濾波器 spec 與寬頻量測

日期：2026-10-07。當前狀態：工程任務已恢復，正式入口已交付；即時健康觀察保存在私人工作區。

10/12同機掃頻診斷交付：原工程真值另已在私人工作區重播與獨立審查，掃頻方法與機台因素仍須分開查驗。沿用歷史網格實驗的同機原則，新增`SweepProbe`：只在原Fast來源owner做兩個既有history圖形的Discrete重測；保持原Discrete量測／runtime／score spec逐bytes一致，原mesh工作不改。`0ee4948`新增入口、launcher與短指令`-Probe`；11項生成資料測試及独立審查通過，包括目錄junction、來源親代、native成功及skip終態。私人release已exclusive發布，真實CMD→PowerShell→native的`-Probe -CheckOnly`通過；尚未啟動HFSS。來源集合48檔，原44檔保持不變；probe不具有正式放行權限。公開[探針descriptor](../../configs/r81_sweep_probe_current.json)只含release檔名與SHA，未含私人NAS路徑或原始結果。完整交付與健康收據保存在私人工作區。

同機診斷啟動：只在原Fast來源owner機台，確認其舊worker沒有進行中的求解後結束閒置wrapper，在原環境／repo根目錄pull GAN再執行：

```powershell
git pull --ff-only origin GAN
.\script\r81.cmd -Probe
```

其他機台的原工程任務不切換；預設`.\script\r81.cmd`仍指向原工程版。這是兩筆診斷，不是正式60筆搜尋。

10/12程式交付：`95fe37c`讓短入口支援FormalSharded，preset仍為EngineeringOnly；`f99a083`新增原工程來源再用驗證，允許獨立正式dataset只派三個formal shard。每次preflight重驗原EngineeringOnly release、全部成功真值、四組≤0.3dB比較、原始檔案及claim／done雜湊；不複製工程markers為新派工，不允許資料根目錄重疊。同一資料夾的舊模式保持相容；實作與獨立審查各通過12項針對性流程測試。原44來源未改，正式來源集合47檔；這是入口交付，不代表實際工程通過或正式HFSS開始。Git遠端確認公開，新原始健康快照只保存私人工作區；下方13:08及更早觀察是歷史切點，不能當成即時狀態。[正式入口](../../script/r81_formal_sharded_entry.py)。

最新實際狀態（2026-10-12 13:08）：使用者明示「恢復了，開始吧」，v002 EngineeringOnly release仍通過原生驗證，queue active。37於13:04:04認領Fast6、216於13:04:40認領Discrete2、218於13:06:33認領mesh2；保存時均尚無results.json、done或fail，未觀察遠端PID。這證明新的派工認領進度，不能代替成功結果或工程比較。舊待切換條件已改變，已接續實際工作；Goal工具欄位仍回報blocked，工具無resume操作，不宣稱系統已改為active。下一例行約13:38；工程全數成功且四組最大曲線／margin差≤.3才放行正式分片。未改44 live來源、NAS queue、inputs或markers。[保存觀察](assets/r81_resumed_health_20261012_1308.json)。

10/12短指令入口：使用者希望啟動命令縮短，新增`script/r81.cmd`與ASCII PowerShell wrapper `r81.ps1`，預設值放在[configs/r81_worker_current.json](../../configs/r81_worker_current.json)。CMD及PowerShell皆可使用下方短指令；仍呼叫原`start_r81_worker.ps1`並驗證release SHA、Git runtime、inputs與queue，不自動pull、不派工、不關閉其他process。已實際沿CMD→PowerShell→原launcher路徑執行`-CheckOnly`通過，沒有HFSS。原受release綁定的44來源未改，私人NAS的v002 release與SHA不變。

最新交付（2026-10-12）：已依使用者要求將scoped profile毒樣本改為忽略而非中斷。`c6e19c1`保留已知HFSS失敗error／attempts，滿3次標`skipped`，續跑／跨機接手都不再開HFSS重試；worker正常寫`.done`的全部跳過ID並繼續其他job，`--once`正常返回。R81全隊已處理但有跳過時為`terminal_skips`，不冒充`all_success`；有效成功／SM只接受真值，缺資料仍須補齊，不能以skip通過工程檢查或合併。HFSS開不起來、嚴重連敗、profile/schema/hash資料錯誤仍停止。小批全skip是派工終態，不證明幾何毒性或機台健康。[補丁與驗證](assets/r81_poison_skip_patch_20261012.json)。

私人NAS已建立且原生preflight通過的新EngineeringOnly release：`r81_engineering_only_20261012_poison_skip_v002.json`，SHA `7d9b1a3b13643c5736cc68bc40abca90b53b661d2b385a0acbf5d3fe2d9d0292`，綁定完整commit `c6e19c18ca4636e909253d1f2a7277089cdffdaa`。只新增release metadata，原6+2+2 inputs、jobs、claim／done／fail與舊release未改；jobs SHA前後相同，candidate與published native report一致。發布時queue active，檢查未啟動HFSS，也未觀察遠端PID；11:08改動前的舊release唯讀觀察仍無markers。整體goal維持blocked，待三台實際切換或提供啟動錯誤。[實際release交付收據](assets/r81_poison_skip_release_20261012.json)。

最新狀態（2026-10-12 01:12）：同一EngineeringOnly release原生檢查仍通過，queue-status為active且沒有claim／done／fail；尚未確認R81 live worker／HFSS handle或工程真值。00:11、00:42、01:12三個連續goal turns都沒有實質進度，啟動情形問題未回覆；已核對三份原始健康紀錄，完成blocked audit並實際將goal更新為blocked。必要的工程／正式入口與備料已完成，目前需要216／218／37依下列現有EngineeringOnly指令切換，或提供啟動錯誤，才能做實際工程比較與後续派工。Timer `1641`已terminal，未再建立例行timer，也沒有停止、重啟worker或改NAS queue。R80完成不等同濾波器達標；R81工程真值、有限批次／SM更新及原次＋獨立重測WM>0仍待完成。[本次觀察](assets/r81_health_20261012_0112.json)／[blocked audit](assets/r81_worker_switch_blocked_20261012.json)。

最新健康觀察（2026-10-12 00:42）：EngineeringOnly release原生檢查仍通過，queue-status為active且沒有claim／done／fail；仍未確認R81 live worker／HFSS handle或工程真值。三台切換問題尚待回覆，這是同一條件的第2次連續no-progress goal turn；保存未變健康狀態不算實際實驗進度，也沒有改派、重啟、NAS寫入或模型更新。下一例行約01:12。[保存觀察](assets/r81_health_20261012_0042.json)。

最新健康觀察（2026-10-12 00:11）：同一EngineeringOnly release原生檢查仍通過，queue-status為active且沒有claim／done／fail；未確認R81 live worker／HFSS handle或工程真值，尚不能說已開始。正式三機入口已由GAN `58a7e62`整合並push，原工程44來源未改；目前待三台實際切換，啟動情形問題仍待回覆。這次狀態未變，沒有實質實驗進度，也沒有改派、重啟、NAS寫入或模型更新。下一例行約00:41。[保存觀察](assets/r81_health_20261012_0011.json)。

最新健康觀察（2026-10-11 23:41）：EngineeringOnly release不變，原生queue-status仍為active，沒有claim／done／fail，未觀察遠端PID；不能當作worker已啟動。已詢問三台切換情形，沒有重啟worker、改NAS queue或開始HFSS。正式3×20入口在隔離worktree已完成實作與6項fixture驗證，獨立審查中；尚未部署或執行。下一例行約10/12 00:11。[保存觀察](assets/r81_health_20261011_2341.json)。

目前狀態（2026-10-11 23:08）：R80完整5,000唯一資料凍結、統計與頻率／極座標圖已收尾，session `13178` exit0，[完成結果](assets/r80_final_closeout_20261011.json)。**R81金屬排列及場型對稱皆不設限制，對稱與非對稱圖形均可搜尋**，兩個port pad仍保留。已push的EngineeringOnly release不變；23:07原生queue-status為active，但保存時沒有claim／done／fail，尚無worker啟動或R81 HFSS真值證據。三個工程job待216／218／37按下列指令切換；先6 Fast＋2 Discrete＋2 mesh，正式60筆仍未入queue。緩衝帶軟性平緩下壓、不參與WM；新spec需原次及獨立重測WM>0才達標。例行檢查維持約30分鐘。

三台切換指令（216／218／37；EngineeringOnly，10/12毒樣本補丁版）：在原環境、repo根目錄讓舊worker閒置後Ctrl+C結束，再`git checkout GAN`、`git pull --ff-only origin GAN`並執行下方短入口。Pull不會更新已啟動的Python process。可加`-CheckOnly`只核對；實際啟動腳本本身必做release preflight，失敗會停止、不啟動HFSS。這次只會消耗已綁定的6 Fast＋2 Discrete＋2 mesh，正式60筆未入queue；沒有手動刪除markers或重新跑成功樣本。

```powershell
.\script\r81.cmd
```

新幾何不要求對稱；這份60筆正式備料全部非LR對稱是實際切點，並不代表日後禁止對稱候選。歷史prior的兩個新阻帶仍mask，不能把舊S21資料當成新16–20／36–40GHz真值。工程HFSS真值、三機preflight回覆與實際worker PID仍待啟動後確認。

最新release狀態（2026-10-11 22:46）：R81 assembler session `61209` exit0，私人NAS final EngineeringOnly release SHA為`c2bba9e8…bca2`，candidate bytes相同且保存native report passed；**release ready，但HFSS／worker仍未啟動**。實際queue僅3 jobs：6 Fast＋2 Discrete＋2 mesh；正式b1 60筆已staged未入queue，三arms各20。四組皆dual／`symmetry_filter_20261007`／49頻點；正式60筆實際幾何全為非LR對稱，符合本輪不限制對稱。Discrete與mesh逐bytes沿用前兩個history圖形，smoke每source group各2。Controller只讀stored prior preflight：21,251驗證、217排除、21,034保留，3舊band covered、2新stopband masked且沒有full-current WM；不當作SM fit或accuracy。R80 closeout session `13178`仍背景執行，不延後三機CheckOnly。下方22:42列保留為歷史啟動切點。[實際release審查](assets/r81_engineering_release_20261011.json)／[小型證據封存](assets/round-81/r81-engineering-release-20261011-v001.zip)。

最新實際啟動（2026-10-11 22:42、尚未完成）：R81 assembler session `61209`依本機log建立時間於22:42:13開始；並行R80 closeout session `13178`於22:42:57開始。兩者均使用PowerShell 5.1可讀的UTF-8 BOM transport；先前無BOM中文路徑失真及缺少`PYTHONPATH`兩次pre-output失敗已保留。此切點沒有terminal receipt，不能宣稱R81 release或R80完整freeze完成；也沒有HFSS、worker或model執行。原生完成與portable binding仍以commit `c736c2a7…`及binding SHA `4a5d14d4…b0a2`為準。[啟動紀錄](assets/r80_r81_actual_preparation_started_20261011.json)／[精確bytes封存](assets/round-81/r80-r81-actual-preparation-started-20261011-v001.zip)。

最新切換狀態（2026-10-11 22:34）：R80原生watch/action已正常完成5,000有效唯一／pending0／idle audit5,000，原main PID在terminal後消失；舊health helper因預期active PID而失敗，不視為新crash。以native44與commit `66dfc5c…`驗證的portable completion packet已在鄒穎麒私人NAS建立，共11檔，binding SHA `4a5d14d4…b0a2`。Native census使用Git-normalized blob SHA `1ce52488…957a`；Windows實體SHA `98f4b829…c835`只作來源bytes證明。這只解除R81組裝前的R80完成gate：**尚未執行R81 assembler、建立release、寫queue或啟動HFSS**。R81幾何仍不限制對稱，首步仍是獨立私人dataset的6 Fast＋2 Discrete＋2 mesh工程release。[完成紀錄](assets/r80_native_completion_20261011.json)／[小型證據封存](assets/round-81/r80-native-completion-20261011-v001.zip)。

2026-10-11使用者提醒：「跑新的濾波器spec可以不用對稱」。R81不限制金屬排列或場型對稱，對稱與非對稱圖形皆可作候選；R80天線的精確左右金屬對稱僅屬該輪實驗，不延伸為濾波器spec。已核對實際dual路徑：歷史seed不做對稱篩選、隨機seed生成完整25×25圖形，`enforce_dual_geometry`只保留p01所需的兩個5×5 port pad；後续全圖形mutation亦不鏡射。現有`dual_margin5`設定符合此要求，因此不改live runtime或既有release來源綁定。共同scope名稱不代表幾何對稱約束。[指示與來源核對](assets/r81_unrestricted_geometry_20261011.json)。

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

### 正式批次的三機分工備料（2026-10-11）

已核對原`dedust.worker`以store為單位原子claim，不允許三台同時跑同一store；原Formal入口只接受每個logical batch一個60筆store，會讓另外兩台沒有工作。Repo已有`profiled_shards.split_bundle`與`merge_stores`，因此沿用它們補控制入口，不另寫分片／合併方法。

實際備妥的首批60筆已在本機用原splitter分成三片各20；每片保持原row與pattern bytes，逐項hash核對通過。輪流分配使history／specialist／random分別為7/7/6、7/6/7、6/7/7，合計仍20/20/20，spec、候選、預測、模型身分與三批60的總量不變。這是已驗證的輸入分工，不是HFSS真值。[準備收據](assets/r81_formal_sharding_preparation_20261011.json)及[小型封存](assets/round-81/r81-native-three-worker-inputs-20261011-v002.zip)保存73個native split檔及5份來源／審查資料，共78個payload，另有archive manifest。

新增FormalSharded入口與launcher已整合到GAN，通過獨立靜態審查、現有舊base上7項fixture驗證（26.62秒）及PowerShell語法解析。[入口與驗證](assets/r81_formal_sharded_control_20261011.json)。這次只新增`script/r81_formal_sharded_entry.py`、`script/start_r81_formal_sharded_worker.ps1`與對應測試；EngineeringOnly原44來源逐一Git blob核對未變，release、NAS與jobs未改，隔離過渡帶commit未合併。FormalSharded release及私人NAS分片仍須先等6+2+2工程實測成功、四組差值≤0.3dB才建立；目前的EngineeringOnly啟動指令不變。

正式階段每片沿用原生三機claim／retry，三片全部零error成功才以coordinator的`--aggregate-batch`操作呼叫既有merger，建立原順序60筆logical store、逐筆來源及claim／done bytes／owner綁定，再回填及更新一次SM。Worker只模擬，不合併或訓練。缺片、錯誤、來源不一致或合併後原完成marker缺失／改寫都會拒絕重派或回填；既有成功片保持原樣，不重跑整批。先前完整logical60與回填／模型證據都通過才允許下一批prefix。尚無實際正式release、合併真值、模型更新或時間加速量測。

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

## 2026-10-11：實測過渡帶診斷與有界親代排序核心（備妥，未接正式選樣）

新增獨立`script/r81_transition_soft.py`，沿用既有已回填feedback的驗證及五段`score_response`。只接受目前R81兩埠3×49、16–40／0.5 GHz、p01／25×25／0.075 mm完整實測身分；舊24–32歷史與未實測候選不能冒充過渡帶真值。左右分別按26→20與30→36方向保存S21原曲線，分列開區間的平均相對衰減，以及完整定向曲線的上升RMS與曲率RMS（端點作曲線錨點）。端點仍按原WM頻帶處理；非單調響應不拒收，這些數值不增加hard margin或改變WM。

親代排序核心將六項實測診斷轉成同一pool內的midrank，平均為0–1軟品質；備妥的v1排序值是`measured WM + 0.1×soft quality`，原WM另列不變。0.1 dB只是一個明示的有界設計偏好，不是HFSS噪聲估計或0.3 dB工程門檻；軟品質不能推翻超過0.1 dB的WM差。純算術函式不認證任意mapping，正式`rank` CLI必須從目前bound feedback／sample重新計算整份sidecar、核對內容及來源hash後才寫出，修改數值再重算content ID也不能通過。

Sol與root審查曾發現初版rank只查self content hash不足，已修正為完整來源重播；最終實際ant環境九個focused tests通過（root1.77秒），包含生成prepare→store→observation→feedback生命週期、相同WM的平順／波紋曲線、排序tie／permutation與有界差值、舊軸／nonfinite／changed spec、篡改及stale來源拒絕／無output。config、state、feedback、sample及所沿用程式來源以解析bytes的SHA及前後不變證據綁定，output只能create-only。

工具使用形態（目前沒有實際R81工作區，以下不是worker啟動指令）：

```powershell
python -m script.r81_transition_soft sidecar --work-dir <R81工作區> --batch 1 --output <新版本sidecar.json>
python -m script.r81_transition_soft rank --sidecar <該sidecar.json> --output <新版本ranking.json>
```

两命令使用同一不變cutoff；訓練或新feedback更新state後，舊sidecar會被拒絕，須在新cutoff建立新版本。跨批親代pool的合併、geometry diversity與正式`select_batch`／release consumer仍待接入；目前五margin SM沒有transition輸出，未宣稱學會此軟目標，三批60／各臂配額均未更動。沒有改原R80 live來源或44個R81 runtime檔，沒有NAS／HFSS／模型訓練／forward或實際R81工程真值。這是軟目標的診斷與排序核心準備，不能當正式優化已啟用或有效性的證據。[最終source／測試與限制](assets/r81_transition_soft_readiness_20261011.json)。


### 2026-10-11：過渡帶親代偏好接點已提交隔離版本，尚未部署
隔離commit `7928468`沿用既有五餘裕SM候選排名與20/20/20配額，僅對已實測親代以WM及平緩／相對衰減軟偏好選出多樣親代；0.1dB偏好上限是設計選擇，不是noise估計，transition不進WM，也沒有新SM輸出head。選樣可重播完整1,024候選與精確60筆ordered manifest／pattern bytes，不接受被替換候選、額外truth aliases、重算hash的錯誤receipt或缺漏transition來源閉合。前次審查五個缺口及修正均保留；完成寫入最後，部分state沒有receipt不能放行。

生成資料的batch2／3生命周期包含60／120 feedback、實際CPU小MLP訓練與forward；主／獨立核心6tests及最後受影響lifecycle各1通過，限定regression14實際通過。提交後真正Git blob／scoped-clean原44及選用46來源閉合均通過，這不是full `validate_release`。Exact CPU predictions跨三台Torch／BLAS／threads重播尚無實證，正式Release前需核對；未調數值規則／tolerance或宣稱可跨機部署。

主GAN的19 live R80來源與原native entry未改、無新selection模組；完整源碼與審查封存於[隔離準備包](assets/round-81/r81-transition-selection-prep-20261011-v001.zip)，[版本／範圍／限制收據](assets/r81_transition_selection_readiness_20261011.json)。不得把封包解壓覆蓋live checkout；待R80原生收尾binding完成後再整合。R81仍無工程真值／派工／正式Release；達validated5,000時先開始備料，統計與圖表同步收尾。


### 2026-10-11：首輪工程 Release 組裝工具已隔離備妥
交接核對找到 production R81-only assembler 缺口，已補薄層並獨立審查，來源仍只放repo/tmp封存、不改live runtime。沿用原生44來源與R80 exact5000／零pending gate，建立輸出前驗證；單一masked-prior controller產生formal60並bound但不排入工程queue，再用既有函式取每arm兩筆共6Fast、相同前兩history樣本2Discrete／2mesh，native jobs_add只建三工作。Exact candidate先native validate_release、final release最後exclusive寫入，失敗prefix保留／不覆寫。

獨立review發現原工具只鎖4override、coherentSHA的seed漂移可通過；已改為完整canonical recipe equality（只有manifest／sample_root絕對路徑可換），pre-output拒絕fixture通過。初版3tests與独立2fixtures；窄修後實作者affected2pass10.77s、独立affected2pass8.95s＋独立adversary1pass2.06s，unchanged6+2+2不重跑。Complete-shape的prior/controller与tracked census lookup有generated fixture替換，不是實際5000或HFSS證據。Conductor核對12payload／CRC、七個native／profile來源SHA前後不變。[準備／版本與限制](assets/r81_engineering_assembler_readiness_20261011.json)／[保留原repo/tmp路徑的封存](assets/round-81/r81-engineering-assembler-prep-20261011-v002.zip)。

實際R80最新action4298（07:04切點），5000/finalgate、私人NAS/R81release、真實prior/controller、三機CheckOnly與HFSS均未執行。使用時將封存按manifest核對、保持documented repo/tmp位置。原初版helper bytes未封存，保留的初版review/receipt只屬歷史，不能當其完整replay包。先以native44完成R80 closeout／EngineeringOnly，isolated transition7928468及三機CPU prediction replay留Formal後段，不讓這個後段gap阻擋first engineering。達validated5000即開始實際R81備料，R80統計圖表同步。


### 2026-10-12：新版 S21 三段與三台探索工作池

新版探索以 S21 的 26–30 GHz ≥−3 dB、16–20／36–40 GHz ≤−20 dB 計算三段 WM，目標仍為 WM>0 並經獨立重測。S11／S22 繼續量測並保留作參考，不納入這個新版 WM。20–26／30–36 GHz 的緩衝區仍是相對下壓和平緩的軟偏好，不加入 WM。16–40 GHz 以外尚無量測承諾。

新增 `configs/dual_r81_s21_discrete.yaml` 與 `script/r81_provisional_filter_dispatch.py`，沿用原生 splitter 和 worker，將既有首批 60 個兩埠幾何改以 Discrete 掃描，分成三台各 20 筆、低優先序 6。完整來源幾何、順序、pattern bytes、三段 spec、round-robin 分割與機器指派均驗證；不用金屬對稱限制。這是探索資料工作池，沒有宣稱原五段工程門檻通過，也不能用執行完成或跳過毒樣本來宣稱達標。

原 Engineering／SweepProbe 的 48 個來源檔、輸入與 claim 保留；新增 release 固定 51 個來源檔。實作與 conductor 各通過九項 focused tests，conductor PowerShell 解析通過，私人目錄實際 release preflight 通過且保存全部準備證據。發布不會自行啟動 HFSS。舊 worker 結束後，三台使用相同短指令；launcher 核對本機身分並拒絕重複 worker。

```powershell
git pull --ff-only origin GAN
.\script\r81.cmd -Survey
```

短指令的兩欄 descriptor 僅含 release 檔名與 SHA；原預設 EngineeringOnly 指標保留作來源驗證。私人原始結果與路徑不隨這次里程碑上傳公開 Git。歷史資料按新版逐段重算，缺少兩側帶外量測的記錄列為待補測，不沿用舊合格標籤或補出不存在的 WM。SM 的三段 masked-prior 接點另外驗證，尚不能把這次輸入發布描述成 SM 自動回填已啟用。

歷史重算工具 `script/filter_history_rescore.py` 已提交；逐筆核對原始 sample／geometry／response hash，保留所有量測實例與重複幾何，不把重複資料視為獨立重測。缺少任一新版頻段時，完整 WM 留空；通帶達標但缺少帶外的全部候選另存排序清單與幾何輸入。完整 census、另外核對的寬頻觀察、可讀摘要及幾何已實際封存私人資料夾；公開 Git 僅放工具與八項 focused tests，不放新的原始研究結果。

`script/filter_s21_training.py` 已提交並通過主／獨立四項 focused tests、實際 profiled-store CLI 與真實歷史來源檢查。沿用原有 MLP、masked loss、family split 與 partial-success snapshot，兩個模型已在 CPU 各完成 30 epoch 歷史預訓練；完成收據、模型與 optimizer 有限性、來源及輸出 mask 核對通過，檔案亦封存私人資料夾。歷史只有通帶標籤，兩個帶外輸出保持未訓練且 `rank_ready=false`；模型完成不能當成新版性能達標。

累積至少 20 個有效、不同幾何的新寬頻觀察後，三段接點可凍結當前成功資料，以同一固定 family split 更新模型並按三段預測 WM 排名。缺少 held-out family、來源改變或模型／候選 receipt 不符時拒絕；holdout 不用於選 checkpoint。這個接點與首批工作池已可用，後續自動監看、子代輸入與 worker 交接仍需單獨驗證，未宣稱持續補派已部署。

新增 `script/r81_s21_survey_cycle.py`，沿用原生候選產生、成功資料快照、三段 SM 與 splitter。當至少 20 個尚未消費的新成功幾何出現，凍結模型實際使用的 pattern／sample hash，訓練並準備 20 個預測 WM、20 個 ensemble uncertainty、20 個隨機子代。訓練期間允許其他新結果追加，原已成功觀察改變或消失仍拒絕；只有已凍結的觀察進消費帳本。既有窄頻幾何仍可補測新版寬頻，並可將整理出的歷史 seed bundle 納入候選；不將缺少帶外的歷史資料當成已測新版幾何排除。

主測新 controller 五項通過，獨立組合十八項通過；實際 release 的 prepare／dry-run 通過並回報等待新真值。候選、模型、分割、來源與消費帳本綁定，失敗前綴保留，完成後的 ledger／pending 清理可恢復。已啟動本機 metadata watcher：達門檻後更新 SM 並準備下一批輸入，重用完成的 historical prior。此 watcher 不啟動 HFSS、不寫 NAS queue；下一批仍是 local-only，須完成獨立的 release／worker 交接，不能描述為自動持續派工已完成。

2026-10-12 20:42–20:43 健康檢查：原工程批次已成為 `terminal_skips`；兩筆 mesh 均達三次 HFSS 失敗後略過，Fast 六筆與 Discrete 兩筆成功結果保持不變。略過不等於成功，也不進 SM 真值或工程門檻。新的 S21 三台工作池均已有 claim，但尚無成功寬頻觀察；claim 只證明取得工作，沒有觀察遠端 PID。完整健康查詢與原始錯誤已驗證並封存原私人健康目錄，公開記錄不含新的原始結果或 NAS 路徑。下一個實作里程碑為可供 worker 自行接續的版本化 registry 與共享 successor release；現有 local-only watcher 的限制仍有效。

### 2026-10-12：SM 子代與 worker 的完整接續實作

新增 `r81_s21_live_controller.py`、`r81_s21_live_entry.py` 與 live worker launcher，原 51 檔保持不變，擴充 runtime 精確固定 56 檔。首批沿用原 60 筆的 jobs、結果與 markers；每台完成自己的 shard 後可讀取後續 immutable release，不必等待其他台整批完成。三台仍各 20 筆、低優先序 6，原生 worker 使用 `--selfgen 0 --once`，並拒絕同機重複啟動。

Controller 在至少 20 個尚未消費的有效寬頻幾何出現後，凍結累積真值、更新兩個三段模型並發布下一批 60 筆，分成 20 個預測 WM、20 個 uncertainty、20 個隨機子代。第三批另須至少 20 個新第二批真值，避免僅消費首批較晚到的資料。歷史 prior、固定 family split、當前模型及快照放在共享 cache；只有來源與 recipe signature 相同才重用 prior。沒有 train／holdout family 時等待追加真值，不更換驗證家族。這個階段含首批最多 180 筆；用完發布額度不等於 worker 已跑完或 spec 達標。

每批物理快照、模型、候選池、rank、選樣、canonical／split／machine partition 均綁定。Registry 最後才以原子、不可覆寫方式追加，發布前綴與未完成 transaction 可恢復；訓練中允許新結果追加，但已使用的樣本改變會拒絕。Pinned job 的失敗保留作人工／controller 修復依據，不透過跨主機 takeover 或刪 marker 重派；略過毒樣本可繼續取件，仍不算真值。Actor 或任一已發布批次的 STOP 均阻止後續取件與補派，包括等待下一批期間。

Conductor 20 項 focused tests 通過（37.67 秒），獨立 Sol 同組 20 項通過（36.69 秒）且完成原始碼審查；包含真正 Torch 訓練／排名／快照驗證與資料替換、恢復、STOP 測試，Git identity 使用 fixture mocks。PowerShell 解析及 diff check 通過。私人 NAS 小型原子發布實測通過，既有目標拒絕覆寫；完整 bootstrap／cache／三台狀態實測與背景 controller 部署留下一個交付里程碑，尚不把實作測試描述成新的 HFSS 性能結果。

完整私人 bootstrap 與 batch-001 registry 已實際建立，固定 `6657046` 的 56 檔來源。原工作池沿用，沒有重派首批；七個 prior cache 檔案逐一核對，模型與完成收據 SHA 一致，三個機器身分各自的 worker 狀態驗證通過。公開 descriptor 僅存 bootstrap 檔名與 SHA，私人路徑、模型及新原始結果不加入公開 Git。核對身分後只停止仍在 metadata 等待的舊本機 CPU watcher；新 live controller 已於 21:12 啟動並確認本機 PID 存活，首次回報七個新成功幾何，尚未達 20 筆更新門檻。沒有停止 HFSS／遠端 worker 或改動既有 claims，也沒有宣稱看見遠端 PID。

三台仍在原 `-Survey` 求解時，保留目前工作；讓舊視窗自然跑完、回到提示字元後，改用以下指令接續。新 launcher 會沿用已完成的第一批，並等候 controller 發布後續批次；不要同時保留舊 launcher 或原生 worker，重複啟動會被拒絕。Controller 的 120 秒 metadata poll 不等於 agent 主動喚醒；routine 健康檢查仍約每 30 分鐘一次。

```powershell
git pull --ff-only origin GAN
.\script\r81.cmd -Loop
```

部署證明自動派工程式與共享交接已就緒；實際下一批尚未產生，三台切到新 launcher 也須由使用者執行。新的三段正 WM 與獨立重測、原工程量測一致性均仍未證明，不能宣稱濾波器 spec 完成。
