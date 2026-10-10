# 進行中實驗追蹤（ONGOING）

> 這裡是 **live 操作板**：只記「**現在在跑 / 待跑**」，保持精簡、會搬走。完整「為什麼/學到什麼」在研究日誌。
> - 研究主線時間軸（append-only 歷史）→ [../docs/log/README.md](../docs/log/README.md)
> - config 全集（不刪）→ [README.md](README.md)
> **流程**：新實驗 → `docs/log/` 開 round 檔 + 這裡加「🔵 進行中」一行指向它；跑完結論寫進 round 檔，這裡只留「✅ 已歸檔」一行指標。

最後更新：2026-10-10

## 2026-10-07：R80 已啟動；每30分鐘監看，之後接續 spec 驗證

- **目前狀態（10/10 09:53 全量／09:52 健康）**：有效唯一對稱實測 **3,277/5,000（65.54%）**，比09:15增加36筆；沒有新最佳，WM仍 **+0.0216822624 dB**、獨立重測 +0.0364336967 dB。[全量及健康證據](../docs/log/assets/r80_best_status_20261010_0953.json)。09:52原主線exact live，SM報v016／完成21輪／running_cycle；三個claimed批次各4/16，最近結果2／0／0分鐘前，scope無警報；另四片LOW待跑。cached3,214不是新census；**下一例行檢查10:02**。資料增加，但近期多批未刷新最佳，性能推進仍停滯。
- **固定32筆對照完成並通過獨立raw核對**：32有效、0終態錯誤、0正WM；批內最佳−4.6554374695 dB，沒有刷新全域最佳。共同32筆LCB Spearman原v010為0.2221、current-only為0.0238；current-only共同樣本預測誤差也較高，先不替換主線。[完整結果與圖](../docs/log/assets/r80_currentonly_prospective_v010_completed32_result_20261010.json)。current-only擁有樣本的best較好、median較差，不能宣稱全面優劣或pretrain因果；不是v016對照。舊drivers不得重啟，兩片不得重派。
- **新多樣性48筆輸入與本機action已完成，未派HFSS**：20k固定v015排名；排除212個scoped輸入及main保留共3,376個圖形。random／smooth blob／pixel mutation／group mutation各12（各8LCB＋4disagreement），joint Hamming最小64／canonical cap2，3×16每片各來源4筆；完整來源／tensor／保存分數／排序／shards通過獨立核對。[輸入證據](../docs/log/assets/r80_structured48_preparation_readiness_20261010.json)。publisher與實際request/action獨立核對通過，本機cycle`1f498860…`狀態prepared、3片g01–g03；尚無NAS input／capacity proof／queue/readback。[發布準備](../docs/log/assets/r80_structured48_publisher_readiness_20261010.json)。背景等待driver於09:46:43實際啟動，PID64504；09:52 exact身分／claim／空stderr核對通過、waiting／publisher calls0。[啟動證據](../docs/log/assets/r80_structured48_deferred_launch_20261010.json)。只讀本機狀態30秒、最多2小時，等待elapsed≥300秒且remaining≥600秒後單次呼叫publisher，不自動重試；後續須fresh鎖內整批48的5000／96容量、duplicate與exact main guard。核心逐片append，不宣稱rollback atomic；partial prefix禁止自動重送。R80/R81未完成。

下列較早時間戳為歷史操作紀錄；已被上述新狀態取代的程序不得重啟。

- **10/10 06:03 主線健康／前瞻派工格式錯誤已定位**：三台claimed10/16、10/16、8/16，最新14／2／2分鐘、scope零警報；主線完成18輪、reported v013，cached3,052不是新全量census。phase1 driver9960於05:59單次呼叫publisher8，06:01在copy／append前exit失敗，兩action仍prepared、無capacity proof；不重啟舊driver。實際40,800B佇列檔使用Windows CRLF，helper計算LF bytes而誤判；診斷cutoff前後相同，正在窄修新版本，不改模型／選樣／容量限制。[失敗與診斷](../docs/log/assets/r80_currentonly_prospective_v010_publication_failure_20261010.json)。最新全量05:04的3,015／+.021682；下一routine06:33，R80/R81未完成。

- **10/10 05:40 前瞻phase1背景等待已實際啟動**：driver PID9960／creation1791581986675／argv／claim现场吻合，原main15944仍exact live；只讀本機狀態，每30秒、最多2小時，剩餘窗口至少360秒才單次呼叫原publisher v8。不自動重試／殺程序／改模型；所有global5000／guided96、來源、duplicate及append前180秒guard照舊。[啟動](../docs/log/assets/r80_currentonly_prospective_v010_deferred_publication_launch_20261010.json)。尚未派HFSS／改queue，phase2之後接續；05:33三台健康、舊三筆HFSS錯誤已由既有worker重試成ok。最新全量05:04的3,015、WM+.021682。下一routine06:03；R80/R81未完成。

- **04:46 CPU準備成功／04:50實際full-pool重播通過**：child45732 exit0／193.393秒，共同20k＝10k random+10k parent；32筆＝16 frozen-live v010+16 current-only v010，兩片各8+8、joint Hammingmin71／25 canonical groups／cap2，雙預測已保存。[結果](../docs/log/assets/r80_currentonly_prospective_v010_preparation_result_20261010.json)。原auditor的shard名字硬編碼不符已保留並只修一行，原準備／模型／選樣不改、不重訓。獨立actual pool review及下游auditor綁定尚待，未派HFSS／未讀新truth。原main在04:43完成v012週期；下次健康05:08，R80/R81未完成。

- **v010對照派工讀回入口已接續prepared_v6／未執行**：publisher v7／seal／truth沿用已核對的capacity、交易與owner來源，只改新輸出版本並綁定terminal guard及full20k replay；獨立static review及實際static validator通過，不重跑廣泛測試。[準備](../docs/log/assets/r80_currentonly_prospective_v010_publisher_rebound_20261010.json)。實際準備／pool replay／獨立pool review仍必需，未派HFSS、未讀新truth／改queue；guard38776等待主線空檔。04:38三台健康，下一05:08；R80/R81未完成。

- **04:38 routine健康通過**：原controller exact／running_cycle，三台claimed14/16、15/16、5/16，最新3／3／2分鐘，scope零警報；新增兩片guided待跑。[健康](../docs/log/assets/r80_health_20261010_0438.json)。reported v011／completed15，watch快照不當新的global census；最新全量仍04:16的2,979／+.021682。CPU guard v6 PID38776現場exact live，等待主線空檔、尚無preparation receipt／未派對照；不重啟／清claim。下一routine05:08，R80/R81未完成。

- **04:22:42 CPU準備修正版實際啟動**：versioned guard v6 PID38776／creation1791577362433／exact argv現場吻合，原main PID15944同時live。prepared_v6建立修正已独立review，固定同v010模型／20k池／32雙owner配方不變；單child600秒、最多等2小時，不自動publish／派HFSS。[啟動收據](../docs/log/assets/r80_currentonly_prospective_v010_guard_retry_launch_20261010.json)。原v5失敗完整保留，這不是新性能結果；最新全量04:16為2,979／最佳+.021682。原三worker與SM照常，下一routine04:38，R80/R81未完成。

- **10/10 04:16 使用者詢問／最新全量實測**：2,979/5,000 唯一有效圖形，較03:20 +49；沒有新最佳，最佳 WM +0.021682 dB。[收據](../docs/log/assets/r80_best_status_20261010_0416.json)。04:13三台claimed有近期結果、scope零警報；主線completed15／SM v011持續更新。新對照CPU準備於03:50因prepared_v5輸出資料夾未建立exit1，未派工／未改queue；原失敗輸出保留、資料夾窄修已review通過但尚未重跑，沒有影響HFSS；兩筆HFSS首次COM失敗非終態、不計有效筆數。下一routine04:38；R80/R81未完成。

- **v010兩片16前瞻派工鏈靜態驗證通過／未執行**：versioned publisher v6綁每筆model／prediction及immutable shards，global5000／guided96分開計main保留量，append前exact CIM／等待窗口再檢；partial-copy／exact-existing recovery與seal／capacity／queue轉移來源補齊，7窄測試及獨立review通過。[準備](../docs/log/assets/r80_currentonly_prospective_v010_publisher_ready_20261010.json)。actual prepared_v5／full20k replay／獨立pool review仍必需；未派HFSS、未改queue／主線source。原guard20232仍只準備CPU排名，下一routine04:08；最新全量03:20的2,930／+.021682。

- **03:38 routine健康通過**：原PID15944 exact／running_cycle，三台claimed13/16、13/16、8/16，最新結果3／2／2分鐘，scope零警報；completed14／reported v011，新guided兩片16排隊。[健康](../docs/log/assets/r80_health_20261010_0338.json)。watch fit2,921不是新全量census；03:20全量2,930／最佳+.021682維持。一次性guard20232已啟動、03:44 native確認仍等待main窗口／無CPUchild；其兩模型固定v010，與目前main v011區分。下一routine04:08，不重啟／清claim，R80/R81未完成。

- **03:34:43 一次性CPU排名guard實際啟動**：PID20232／creation1791574483613／argv現場吻合，03:44 native process核對為等待原main窗口、只有console host、無CPU準備child／stderr；guard v5獨立PASS後只讀本機狀態、最多等2小時，單child600秒＋bounded cleanup，不自動publish或派HFSS。[啟動](../docs/log/assets/r80_currentonly_prospective_v010_guard_launch_20261010.json)。原main仍持續SM／HFSS；準備結果需另replay後派兩片16，各owner8+8；global5000／guided96不變。03:38 routine已通過，下一04:08；最新全量仍03:20的2,930及+.021682。

- **v010前瞻排名入口來源修正完成／尚未執行**：每筆採用owner自己的scores／curves／model hash，joint audit區分live-base生成證據並移除未採用的內部selection標記；既有queue驗證會讀sample/rad，與新32筆truth未讀分開。固定20k／16+16／Ham64／cap2及兩片8+8不變，12窄測試及獨立scoring v5 review通過。[收據](../docs/log/assets/r80_currentonly_prospective_v010_scoring_ready_20261010.json)。旧v2–v4入口與guard不launch；修正版guard獨立review後僅做一次CPU排名，publisher需另核對最終等待窗口、capacity和immutable輸入，尚未派工／adopt。主線HFSS照常，03:38 routine；最新全量03:20為2,930、WM+.021682。

- **10/10 03:20 使用者詢問／全量 raw 核對**：2,930/5,000 唯一有效實測（較02:20 +51），沒有新最佳，最佳 WM +0.021682 dB。[收據](../docs/log/assets/r80_best_status_20261010_0320.json)。03:20 三台有近期結果、scope 無警報；SM 持續批次更新。v010 current-only 三模型實際訓練及物理審計完成，有限前瞻對照仍未派工／採用。下一例行03:38，R80/R81 未完成。

- **03:08 routine健康通過**：原controller exact／running_cycle，三台claimed近期有結果／scope零警報；completed13、reported v010。[健康](../docs/log/assets/r80_health_20261010_0308.json)。02:38結果較久的那片已推進到13/16、最新1分鐘，未重啟／清claim；v010 current-only CPU已完成，雙scorer同池準備review通過但未prediction／派工。背景一次性guard的窗口計算已在launch前發現需窄修，保留原版，原main／LOW繼續。全量仍02:20的2,879／+.021682；下一routine2026-10-10 03:38，R80/R81未完成。

- **v010前瞻對照SM實際CPU訓練完成**：dfb6fe6固定後單次session53209 exit0，02:40:47–02:41:24／36.830秒；三fresh seeds100epochs／Adam各1600，independent1944rawtrain-only normalization逐位吻合、有限參數／nonzero moments、兩frozen來源不變。[模型收據](../docs/log/assets/r80_currentonly_prospective_v010_training_20261010.json)。未做candidate pool prediction／NAS／HFSS派工，future scoring/publish另review；不把訓練完成當性能改善。下一routine03:08，全量仍02:20的2,879／+.021682。

- **02:38健康與v010有限前瞻對照準備通過**：原controller exact／running_cycle，三台claimed／scope零警報（結果最新2／15／2分鐘），未重啟／清claim。[健康](../docs/log/assets/r80_health_20261010_0238.json)。固定同2,840 frozen資料／1944train896hold、同20k候選池、16live+16current-only／joint Ham64／cap2，五窄邏輯測試及独立inventory/pretrain review通過；先固定commit再單次CPU，不自動採用。[準備](../docs/log/assets/r80_currentonly_prospective_v010_prepared_20261010.json)。尚未train/predict/dispatch，futurepublish需另review；global cap5000和guided96不變。最新全量仍02:20的2,879／+.021682；下一routine03:08，R80/R81未完成。

- **10/10 02:20 使用者詢問現況／全量raw核對**：2,879/5,000唯一有效實測（較00:51 +69），沒有新最佳，最佳WM +0.021682 dB。[收據](../docs/log/assets/r80_best_status_20261010_0220.json)。02:08三台有近期結果／scope零警報，controller報v010；v00848全數有效但0正WM。current-only排名稍好，準備有限前瞻對照、尚未派工或換live。下一routine02:38，R80/R81未完成。

- **01:38 routine健康通過／等v008最後1筆**：原PID15944 exact／running_cycle，三台claimed最新0／0／1分鐘、scope零警報；controller已報v009／11cycles。v008g01/g02 done，g03尚15/16且非terminal；同48 live/current-only讀回及獨立replay已備好／未執行。[健康](../docs/log/assets/r80_health_20261010_0138.json)。不補假真值、不追加解算／清claims；最新全量仍00:51的2,810／+.021682，下一routine02:08。

- **v007 first32真值及獨立raw重播完成**：32/32有效唯一／exactLR／0terminal error／0正WM；best−3.558715、median−10.154610，非新最佳。[結果](../docs/log/assets/r80_diverse_sm_v007_first32_result_20261010.json)。LCBρ0.0550、GainbandMAE3.9779，性能推進及排名仍弱；496對Hammingmin66、26canonical／cap2。只比原live v007，不混不同候選v006 shadow；v1/v2未執行、讀回gate/command窄修已核對。2810仍以00:51全量為準、不再加32；v00848待終態，01:38 routine。

- **01:08 routine健康通過**：原PID15944 exact／running_cycle，三台claimed近期結果／scope零警報，LOW及guided有待跑；v007兩片皆done，可在讀回窄修審查後核對32真值。v008g02／g03為12／7且非terminal，48對照繼續等。[健康](../docs/log/assets/r80_health_20261010_0108.json)。current-only v008已實際完成／未adopt，最新全量仍00:51的2,810／+.021682；下一routine01:38，不重啟／清claims。

- **current-only v008背景對照完成／結果混合**：固定2,739／1855train／884hold、三fresh current100／Adam1,500，CPU32.135秒；checkpoint/norm及獨立保存算術通過。[結果](../docs/log/assets/r80_sm_currentonly_shadow_v008_result_20261010.json)。LCBρ0.0768→0.1138、GainMAE5.5604→5.4897稍改善，但fullMAE3.6045→3.7277、S11與rad變差，保留原live。48shadow預測已保存、truth未讀且timing=false，待同批HFSS終態，不加解算。最新全量仍00:51的2,810／WM+.021682，01:08 routine。

- **current-only v008固定後備背景對照**：沿用v006 recipe，只換frozen2,739資料（1,855train／884hold）；5,478 cache hashes、48訓練排除與41項獨立prelaunch檢查通過。[準備收據](../docs/log/assets/r80_sm_currentonly_shadow_v008_prepared_20261010.json)。先commit固定config再執行一次CPU；此刻尚未train/predict/adopt，48預測明示non-prospective，原HFSS／SM／queue不變。

- 現況 **10/10 00:51：使用者詢問全量raw核對**。2,810/5,000唯一有效實測，較23:51增加54；全域最佳仍WM **+0.021682 dB**，既有獨立重測+0.036434。[收據](../docs/log/assets/r80_best_status_20261010_0051.json)。三台claimed有新結果／scope零警報，SM v008持續批次更新；worker狀態在controller掃描中變動已安全defer、未重啟。current-only v008準備及獨立審查通過、尚未訓練／採用；多樣批次尚無新最佳。下一routine01:08，R80／R81仍未完成。

- **10/10 00:38健康通過**：原PID15944／creation／argv不變、running_cycle／零警報；三片claimed最新4／2／3分鐘，v007g02尚14/16、v008g01/g02為12/3，g03排隊。[健康](../docs/log/assets/r80_health_20261010_0038.json)。v008資料／模型／48派工及完整20k獨立排名均已核對；current-only新資料對照準備中。最新全量仍23:51的2,756／+.021682，不把metadata或result entries冒稱新總數。下一routine01:08。

- **v008完整20k獨立保存排名核對通過**：10k fresh／10k parent，strict三臂各16、48selected／三staged片完全重播；36家族／cap2／minHamming71、32parents32groups、fit2,739 binding一致。[補充收據](../docs/log/assets/r80_diverse_sm_v008_selection_review_20261010.json)。只核對派工幾何／來源，不作HFSS性能改善宣稱，未改live政策。

- **v008新資料／模型／48實際派工核對通過**：新增48／fit2,739，舊2,691保留；1,855train／884hold，三fresh legacy30＋current100／Adam3,900、train-only norm逐位吻合。私人NAS三片各16 tree／queue及48保存曲線重播通過。[收據](../docs/log/assets/r80_diverse_sm_cycle_v008_20261010.json)。dispatch2,749與fit分開記帳，最新全量仍23:51的2,756／+.021682；獨立20k排名核對待做，current-only新資料shadow準備中，未改主線。

- **v006 first32終態真值及四SM比較完成**：32/32有效唯一／exactLR／0terminal error／0正WM，最佳−2.101229、median−17.323709；非新最佳。[結果](../docs/log/assets/r80_diverse_sm_v006_first32_result_20261010.json)。LCBρ live0.3138／current-only0.3952／CNN0.1074／weighted0.2786；CNN曲線誤差較低但排名較差，加權loss未改善本批Gain或LCB。先保留主線，不憑單批替換；32原預測prospective、三shadow仍false。窄round-robin讀回修正與独立raw重播已核對，v1保留；v006另16不混入，00:38下次routine。

- **10/10 00:08健康通過／首次RPC樣本已重試成功**：原PID15944身分不變，scope零警報，三片claimed最新4／2／2分鐘。v008已完成cycle（fit/model實體完整核對待做），正常waiting30分鐘；v006兩片於正確jobs_state皆done、原失敗樣本status ok。[健康及恢復](../docs/log/assets/r80_health_20261010_0008.json)。first32判讀工具因round-robin/concatenation順序假設拒絕，窄adapter審查中，不改原始資料或派工。下一routine00:38。

- **v007完整20k保存排名獨立核對通過**：10k fresh＋10k parent，strict11／11／10選32，26家族／cap2／minHamming66；兩個staged16與action／completion／fit2691吻合。[補充收據](../docs/log/assets/r80_diverse_sm_v007_selection_review_20261010.json)。只驗派工來源與幾何，不是新HFSS性能或SM改善；00:08再查終態，live流程未改。

- 現況 **10/09 23:51：依使用者詢問全量 raw 核對**。2,756/5,000唯一有效實測，較22:54增加43；最佳WM仍 **+0.021682 dB**，既有獨立重測+0.036434。[收據](../docs/log/assets/r80_best_status_20261009_2351.json)。23:50精確PID15944／零警報，三片claimed有待跑工作；SM最近完成v007、持續更新中。CNN及weighted-loss背景對照未adopt；R80與R81未完成，下一routine00:08。

- 現況 **10/09 23:38：routine健康通過，first32仍待終態**。原PID15944精確身分不變／running_cycle，scope零警報；v006兩片各14/16、最新1／0分鐘，第三台已認領v007。32讀回因尚未terminal拒絕、未產真值報告，保留現場不重啟／清claims。[健康](../docs/log/assets/r80_health_20261009_2338.json)。背景weighted-loss結果已保存且未adopt；全量最佳仍以22:54的2,713／+0.021682為準。下一routine **10/10 00:08**。

- 現況 **10/09 23:24：帶內加權SM背景對照完成**。固定current-only MLP只改loss，CPU54.91秒／三fresh100epochs／Adam1,400，實體norm/optimizer與獨立NPZ核對通過。factoryρ0.0749→0.1302／LCBρ0.0672→0.1175，但GainbandMAE5.5318→5.5920變差，仍未採用。[結果](../docs/log/assets/r80_sm_weighted_loss_shadow_result_20261009.json)。32預測已保存待HFSS真值對照，不加解算／不調coefficients；原SM/佇列續跑。下一routine23:38，最新全量2,713／最佳+0.021682。

- 現況 **10/09：v007實體模型／32派工核對通過**。新增50／fit2,691（1,812 train／879 hold），舊2,641完整保留；三fresh legacy30＋current100／Adam3,900，train-only norm與32保存預測重播通過，兩私人NAS input tree／queue吻合。[收據](../docs/log/assets/r80_diverse_sm_cycle_v007_20261009.json)。dispatch截點2,702與較早fit2,691分开記帳；latest全量仍22:54的2,713／最佳+0.021682。只核對資料／模型／派工，無新增完整20k獨立排名或HFSS性能宣稱。下一routine23:38；weighted-loss背景對照備料中。

- 現況 **10/09 23:09：v005 generation48完整實測核對**。三個不同cycle各16、同SM v005 fit2,578；全48有效唯一／exactLR／零terminal error，最佳WM−0.637783、median−12.490290、0正WM／0超過全域最佳。[結果](../docs/log/assets/r80_diverse_sm_v005_generation_result_20261009.json)。本代接近合格但尚非新最佳；combined48 minHamming20（各自16內≥64），LCBρ0.0671／GainMAE5.5967。23:08三台claimed有新結果／零警報，原PID15944不變；自動v007更新已觀察、完整核對待做。[健康](../docs/log/assets/r80_health_20261009_2308.json)。最新全量仍2,713／最佳+0.021682；下一routine23:38。

- 現況 **10/09 23:02：CNN背景對照完成**。三fresh current100／Adam各1,400、CPU194.92秒；同868保留集fullMAE3.759→3.582／Gain5.574→5.327，比live v006 Gain5.559也略低。factoryρ0.0749→0.0783仍弱、mean-WM排名反降，未採用。[結果](../docs/log/assets/r80_sm_cnn_shadow_result_20261009.json)。實體checkpoint／保存算術核對通過，32候選預測已保存；不用追加HFSS。最新全量仍22:54的2,713／最佳+0.021682；原controller及worker續跑。

- 現況 **10/09 22:54：使用者要求的全量最佳核對**。2,713/5,000唯一有效實測、較22:09增加38；最佳仍WM **+0.021682 dB**，既有獨立重測+0.036434。[查詢收據](../docs/log/assets/r80_best_status_20261009_2254.json)。22:53精確controller身分不變，三台claimed新v006工作／零警報。SM持續批次更新；current-only對照陰性未採用，CNN背景對照準備中尚未訓練。R80與R81仍未完成。

- 現況 **10/09 22:38：routine健康通過**。唯一PID15944 exact身分／argv不變、8cycles、三片claimed最新3／2／1分鐘、零警報；LOW及guided都有排隊。v006沿用再補一片16，新的48真值尚未完成。[健康收據](../docs/log/assets/r80_health_20261009_2238.json)。current-only背景对照已核對陰性／未採用，既有CNN架構對照正備料；不改live。最新全量最佳仍22:09的2,675／+0.021682；下一routine約23:08。

- 現況 **10/09 22:29：current-only SM背景對照完成／陰性**。固定2,641／1773train＋868hold、三fresh current100，31.34秒完成、獨立NPZ/JSON核對通過。fullMAE3.606→3.759／Gain5.559→5.574；factoryρ0.0726→0.0749變化很小，未採用。[結果](../docs/log/assets/r80_sm_currentonly_shadow_result_20261009.json)。32shadow預測已保存但真值未讀，不加HFSS；live SM及佇列照常。最新全量仍22:09的2,675／最佳+0.021682，下一routine22:39；R80及R81未完成。

- 現況 **10/09：v006完整模型／32派工審查通過**。新增63／fit2,641，train1,773／hold868；三fresh模型norm／Adam3,800及32保存預測重播通過。獨立20k strict ranking核對11／11／10、23家族、minHamming84；NAS兩片40-file tree與queue一致。[收據](../docs/log/assets/r80_diverse_sm_cycle_v006_20261009.json)。32真值待完成；最新全量最佳仍以22:09的2,675／+0.021682為準。背景current-only SM對照準備中，live配方未改；下一routine約22:39。

- 現況 **10/09 22:09：全量實測重播，最佳未變**。2,675唯一成功圖形／2,680成功觀測，較20:17新增83；WM仍 **+0.021682 dB**，既有獨立重測 **+0.036434 dB**。[查詢收據](../docs/log/assets/r80_best_status_20261009_2210.json)。第二批多樣48獨立核對完成，最佳−5.102279／median−12.183969／0正WM；兩批96皆由同v004提出，非兩代迭代或停滯定論。[第二批收據](../docs/log/assets/r80_diverse_sm_second_cohort_result_20261009.json)。SM v006 summary新增63／fit2,641，完整模型派工審查尚待；PID15944身分不變、三台有claimed jobs／scope零警報。約30分鐘例行檢查，R80 5,000及R81未完成。

- 現況 **10/09 21:28：首批多樣48真值／獨立算術核對完成**。全有效唯一／精確LR／零error，最佳WM **−4.820067**、median **−12.214764**，0正WM／0超過既有全域參考+0.021682。Hamming min65；保存LCB對WMρ0.151324、Gain MAE4.556437，排序仍弱。[結果](../docs/log/assets/r80_diverse_sm_first_cohort_result_20261009.json)／[頻率圖](../docs/log/assets/round-80/diverse-sm-first48-20261009-v001/frequency_responses.png)。僅單cohort，非因果改善／三代停滯；完整讀回posthoc、原預測prospective。原PID15944／零警報／stderr空，後續批次及SM持續；下一routine **21:58**。R80 5,000與R81未完成。

- 現況 **10/09：v005三fresh SM及實際16筆補派核對通過**。新增78、fit **2,578**，train1,720／hold858；舊2,500列保留，norm／Adam3,800／16保存預測重播通過。獨立20k排名與strict selection核對：global6／parent5／disagreement5、13家族、cap2、min Hamming72；私人NAS單片20-file tree／queue一致。[薄收據](../docs/log/assets/r80_diverse_sm_cycle_v005_20261009.json)。容量限制只補16；首批48於20:28仍未完整，無性能改善宣稱。Controller原PID15944 live／零警報／stderr空，下一routine **20:58**；最新已完成全量raw query為20:17的2,592，最佳未變。

- 現況 **10/09 20:17：最新實測未出現新最佳**。全部 raw 重算 **2,592 unique / 2,597 observations**，較17:46增加134筆；仍為 `r80localv1_00026_f6ffb938` WM **+0.021682 dB**，既有獨立重測 **+0.036434 dB**。首批多樣性48尚未全部完成；部分結果有一次COM例外，仍在worker重試流程，20:15 scope無警報。v005 summary已出現：新增78、累計fit2,578；本輪尚在dispatching，完整模型/派工審查待terminal receipt。[只讀查詢收據](../docs/log/assets/r80_best_status_20261009_2017.json)。唯一controller PID15944身分不變、stderr空；維持約30分鐘健康檢查，R80 5,000與R81未完成。

- 🔵 **10/09 19:04：多樣SM首輪v004／48實際派工核對通過**。新增81、fit **2,500**（1,653 train／847 hold），原2,419列保留；三fresh模型norm／Adam3,700與48 saved prediction重播通過。獨立20k幾何／strict排名重播：32親本／30家族，三臂各16、37入選家族、cap2／min距離65；私人NAS三片16完整input tree／queue核對。[薄收據](../docs/log/assets/r80_diverse_sm_first_dispatch_20261009.json)。較晚dispatch cutoff2,508分開記帳；HFSS未完整，不宣稱性能提升。18:58實體PID15944 live、下一cycle運作、三台有新結果／零警報，下一routine **19:28**。

- 🔵 **10/09 18:28：多樣SM controller已單次啟動**。已push source `5802b79`；唯一實體PID15944／creation1791541659267／launch42afe91c263f4bacb74f6d6c5c6bbcb7，exact CIM／argv、政策、14來源模型binding及初始running_cycle核對通過，stderr空白。[啟動收據](../docs/log/assets/r80_diverse_sm_controller_launch_20261009.json)。新controller root、同familydev training root，維持48–96新真值重訓／LOW補池；三台原worker續跑。初始啟動時第一live fit／派送未完成；現已另驗首輪v004。原18:58 checkpoint已完成，繼續約30分鐘健康；R80 5,000與R81仍待完成。

- ✅ **多樣SM實作／實體20k pool通過（啟動前核對）**。三臂各16選48，32親本／31家族，入選37家族、最多2同家族、最小距離65；v003真實模型與凍結圖形／排序／選樣重播通過。[收據](../docs/log/assets/r80_diverse_sm_implementation_20261009.json)。獨立focused27＋narrow1通過；候選preview不算實測或已派工，保持原worker／LOW，下一步單次接手。18:20 scoped health零警報，三台12/16、9/16、7/16持續有結果，另五片LOW排隊。

- ✅ **10/09：新familydev首批48實測已完整判讀**。`63c3efcdg01`–`g03`皆.done／零error、48互斥pattern及sample/rad完整重播；最佳WM **−5.617131 dB**，median−16.231790，0/48正WM、0/48超過既有最佳；saved LCB對實測WMρ=0.038645。整批幾何距離min5／median304，不能只看median宣稱沒有近親。保留19／15／14原臂及前瞻預測；[真值收據](../docs/log/assets/r80_familydev_first_cohort_result_20261009.json)。這是單批陰性結果，不是三世代性能停滯；多樣策略配方已先固定，實作審查中。

- ✅ **10/09 18:02：舊controller正常退出，HFSS續跑原佇列**。新配方已固定／push `3cf381c`，實作在隔離worktree審查中。精確PID60956／creation／argv與完成cycle核對後，18:01:20只寫其本機STOP；18:01:46正常退出，18:02 CIM核對舊PID消失、全機watcher數0，[交接收據](../docs/log/assets/r80_diverse_sm_controller_stop_20261009.json)。18:00 scope零警報；三台有已認領6/16、3/16、0/16與五片LOW待跑，共119筆metadata待跑，供交接期間持續工作。未kill／重啟worker或改queue／claims；新controller尚未啟動，完成review／實體pool核對後以同training root、新controller root單次接手。下一routine約18:30。

- 🔵 **10/09 17:49：第二輪SM更新／私人NAS派工核對通過**。自動cycle於17:34:16完成；v003新增75、累積fit **2,419**（1,583 train／836 hold），原2,344列完整保留。三個fresh member為legacy30＋current100 epochs／Adam3,700 steps，effective-train-only四norm tensors完全吻合；3×16 prio1私人input tree、唯一queue row、互斥48pattern及17／17／91／91預測曲線讀回通過。[薄收據](../docs/log/assets/r80_family_development_cycle_v003_20261009.json)。固定派工截點2,431與較早fit2,419分開；初版人工audit誤把兩截點之差當新增fit，在寫通過收據前拒絕，已用實體新舊manifest差75修正，並非controller失敗。HFSS結果仍待整批真值判讀。使用者新授權：維持SM排名並嘗試更多不同潛力對稱幾何，正準備前瞻批次，不改既有完成批次。

- ✅ **10/09 17:46：最新最佳渲染已完成**。只讀raw重播為 **2,458唯一成功pattern／2,463成功觀測**；最佳仍`r80localv1_00026_f6ffb938`，原次WM **+0.021682 dB**、已核對獨立重跑 **+0.036434 dB**。沿用既有renderer及極座標helper，金屬／S11／Gain／rad圖與raw排名核對通過；[最新圖](../docs/log/assets/round-80/best-symmetric-polar-20261009-v006/sample_card.png)／[收據](../docs/log/assets/round-80/best-symmetric-polar-20261009-v006/best_query_receipt.json)。截點保留一筆首次COM失敗，未計成功，尚非最終毒樣本判定；最佳未變不等於性能停滯。背景SM／HFSS流程繼續，routine維持30分鐘；R80 5,000與R81未完成。

- ✅ **10/09 17:17：依使用者要求更新最佳圖**。原只讀query逐store截點raw重播為 **2,431唯一成功pattern／2,436成功觀測**、截點零error；最佳仍為`r80localv1_00026_f6ffb938`，原次WM+0.021682 dB、已核對獨立重跑+0.036434 dB。沿用既有極座標renderer，PNG與前版byte-identical；[最新圖](../docs/log/assets/round-80/best-symmetric-polar-20261009-v005/sample_card.png)／[raw排名與出圖收據](../docs/log/assets/round-80/best-symmetric-polar-20261009-v005/best_query_receipt.json)。17:04已核對同一實體PID60956、第二cycle開始、LOW新補4×16、scope無警報／三台有新結果；下一routine約17:34。最新48筆SM引導HFSS尚待完整真值判讀，不宣稱性能停滯或改進；R80 5,000與R81未完成。

- 🔵 **10/09 16:44：新protocol首輪已完成且獨立核對通過**。Controller於16:40:08自然轉waiting，completed_cycles=1、receipt `dispatched / passed_per_shard`；familydev v002新增96、累積fit2,344（1,533 effective train／811 hold，c48全121筆train且保留reference hold）。三個fresh member的六欄binding、30＋100 epochs／Adam3600 steps與effective-train-only四norm tensors均核對通過。已派3×16 prio1，Conductor實際私人NAS完整60-file tree與三個唯一queue row讀回吻合、48個pattern互斥，全部保留17＋17＋91＋91預測曲線；[首輪模型／派工收據](../docs/log/assets/r80_family_development_cycle_v002_20261009.json)。**這是controller派工完成，HFSS三片尚待真值判讀**；固定dispatch截點2,373 unique與即時總量分開。16:36 scope零警報／三台有新結果，下一routine約17:04；不以模型更新宣稱性能改善。

- 🔵 **10/09 16:04：family development 新 controller 已單次接手**。舊 PID54712 在15:49:59透過本機 STOP 正常退出；新唯一 PID60956（creation1791532984209）於16:03:04從已push `b1d4c4a` hidden start，launch=`555a4dbe6a7d474e8ffc70a97e555de7`，新根 `controller_familydev_v1`／`training_familydev_v1`。固定2,248筆bootstrap、c48整家族110筆轉development train，三個fresh模型已訓練與獨立稽核；其fit誤差不是泛化證據。首輪 `running_cycle`、stderr空，尚無首輪完成／新模型派工證據；[交接紀錄](../docs/log/assets/r80_family_development_activation_20261009.json)。16:04 scope零警報，三個claimed jobs及三片LOW待跑；主線1800秒／LOW90秒、48–96新資料更新SM不變，下一routine約16:34。不重啟HFSS worker。
- ✅ **10/09 16:02最新最佳圖已核對**：逐store截點raw重播 **2,359唯一成功pattern／2,364成功觀測**，仍為`r80localv1_00026_f6ffb938`，原次WM+0.021682 dB、獨立重跑+0.036434 dB。[極座標圖](../docs/log/assets/round-80/best-symmetric-polar-20261009-v004/sample_card.png)／[排名與出圖收據](../docs/log/assets/round-80/best-symmetric-polar-20261009-v004/best_query_receipt.json)。R80 5,000與R81均未完成；repeat不計新unique。

- 🔵 **10/09 14:29：32筆局部變體全數零error終態，完整判讀／獨立核對完成**；scope零警報、三台有工作，下一routine約14:59。SM v034新增75、累積fit2,248（1,359 train／889 holdout），派工實體截點2,254唯一，送出3×16 prio1；[更新收據](../docs/log/assets/r80_factory_cycle_v034_20261009.json)。[獨立輸入／覆蓋診斷](../docs/log/assets/r80_sm_input_contract_diagnostic_20261009.json)未找到所檢查路徑的編碼／normalization bug；最佳家族在v033為0 train／105 holdout，歷史另0 train／17 holdout。覆蓋落差是假說，尚非因果結論；不改現有分割或32筆事前排序。R80 5,000與R81仍待。

- ✅ **10/09 最佳pattern原次與獨立重測均正WM**：`dedust_r80bestrep01`由37於13:54:34零error完成；原次216為WM+0.021682 dB，新次+0.036434 dB，金屬／profile相同。第一終態raw凍結、Conductor直接曲線重算與獨立Sol的35來源前後SHA／實際數值核對通過；[兩次結果](../docs/log/assets/r80_best_repeat_result_20261009.json)／[派工](../docs/log/assets/r80_best_repeat_dispatch_20261009.json)。餘裕很小，這是兩次觀測，不是三次公證或mesh／連續頻率／穩定性證明；重測unique增量0，不自動重跑。上次實體派工截點2,241唯一，與SM fit分開。後續完整32筆已核對；R80 5,000筆及R81仍待。
- 🔵 **10/09 最佳附近SM局部試驗32筆均已完成，完整raw與排名統計已核對**：完整52,650個d1／d2對稱變體，排重後52,635個由v031排名，相對WM／LCB雙門檻取32，分兩片16筆prio1；首片10:50、第二片12:17派工／input tree／唯一queue row讀回通過。[第二片及完整派工收據](../docs/log/assets/r80_local_variation_phase02_dispatch_20261009.json)／[備料收據](../docs/log/assets/r80_local_variation_prepare_20261009.json)。12:34 raw截點最佳為首片全域rank27，單次WM **+0.021682 dB**，比固定anchor提高0.268123 dB；完整32筆有6筆超過anchor、1筆正WM，LCB／真值Spearman −0.090176；該最佳的一次獨立重測亦正WM。[完整結果](../docs/log/assets/r80_local_variation_result_20261009.json)。候選／模型／審查／派工證據存本人私人NAS；controller／訓練分割未改，絕對SM校準誤差限制仍保留。
- 🔵 [R80 金屬對稱探索](../docs/log/round-80-symmetry-exploration.md)：10/09 12:32–12:34全量逐store截點重播為 **2,178唯一實測／5,000**（2,182含repeat）；最佳`r80localv1_00026_f6ffb938`單次WM **+0.021682 dB**，26.5–29.5 GHz最差S11−10.096088 dB／最低Gain4.021682 dBi，達兩項門檻但未重測公證。[最新極座標圖](../docs/log/assets/round-80/best-symmetric-polar-20261009-v002/sample_card.png)／[排名收據](../docs/log/assets/round-80/best-symmetric-polar-20261009-v002/best_query_receipt.json)。這是全量單次最大值推進，不替代成熟世代前緣／完整32筆判讀；既有v008停滯證據保留。SM v033累積納入2,173筆profile真值（本版新增59），與實測截點分開記帳。唯一controller PID54712持續運作，48筆新唯一真值門檻更新SM、prio1主線／prio6補池；12:56有效scope無警報，三個claimed jobs，第二局部片尚待跑。每30分鐘健康檢查，下一約13:26。R80／R81未完成；[同239保留集舊SM比較](../docs/log/assets/r80_sm_common_v013_v015_20261008.json)保留，版本更新不等於性能改善。
- **首個成熟性能窗口已驗證**：v001／v002／v003共176唯一實測（48／48／80），全部終態且零error。v001、v002有實質推進；v003無相關前緣增量，只累積一個未推進世代，未達三世代停滯判準。此225筆窗口最佳單次factory margin−0.246441 dB，帶內最差S11−9.76091、最低Gain3.75356，仍未達原天線雙門檻；不把這個窗口數字當作326筆全量最優。主基準49與非重測48敏感性結果一致；完整證據643檔存私人`analysis_versions/performance-window-v001-v003_20261008`，見[收據及適用限制](../docs/log/assets/r80_performance_window_v001_v003_20261008.json)。
- **延伸至v005的性能窗口**：新增v004／v005各48筆，17片共272唯一實測。v003、v004無實質推進，但v005有一個相關2D／3D前緣增量，重置停滯計數；綜合最佳仍−0.246441 dB。新增點S11最差−10.46454 dB、最低Gain2.66478 dBi，屬取捨而非雙門檻達標。49／48基準結論一致，完整新證據223檔存私人`performance-window-v001-v005_20261008`，見[延伸窗口收據](../docs/log/assets/r80_performance_window_v001_v005_20261008.json)。
- **v008窗口已觸發性能停滯**：新增v006／v007／v008各48／64／80筆，全部終態、零error；三世代相關點與2D／3D增量均0，固定判準的192筆未推進後綴成立。513／512基準分析及獨立192筆raw／全量前緣重算一致；本窗口最佳仍−0.246441 dB，不當作615筆全量最優。440檔含新真值、原三世代9個模型及審查存私人`performance-window-v001-v008_20261008`。停滯僅通知，不自動停HFSS或改佇列／spec；選樣誤差與歷史方法診斷正在另行處理。
- ✅ **15+1一次性pilot已判讀**：[固定protocol／審查](../docs/log/assets/r80_incumbent_shell_pilot_design_review_20261008.json)及[實測陰性結果](../docs/log/assets/r80_pilot_sm013_result_20261008.json)均保留；15格d1對稱變體＋1個blind對照全部有效，四項改善條件計數均0。選擇先於SM annotation，blind不能使pilot陽性；不自動重複pilot、不改門檻，所有有效低分真值保留供SM與資料集使用。
- ⏭️ [R81 新濾波器規格](../docs/log/round-81-wide-filter.md)：使用者19:55補充性能目標為**新版五段spec的實測WM > 0 dB**；改完程式/跑完批次不算達標，WM≤0仍未完成。對稱任務後接續，先做smoke、Discrete、mesh工程檢查再放行探索；正WM候選仍需重測確認。目前仍只派R80。
- **監看方式（使用者最新指示）**：讓現有背景controller持續補池與按批次更新SM，每30分鐘只做健康檢查；沒有需處理的結果／錯誤／停滯時保持掛著，不持續主動喚醒、反覆全量稽核或傳送例行進度。需要修正、驗證研究結果或階段切換才人工介入。
- 首版49筆凍結資料已有[幾何地形／S11與Gain頻率圖及分析紀錄](../docs/log/round-80-symmetry-exploration.md#2009-首版資料的可重跑對稱與頻率圖)，完整分析存私人`analysis_versions/data-v001`。這是19:19資料的描述統計，不是最新累計數；尚無對稱改善性能的因果結論。
- **最新目標覆蓋先前10,240上限**：對稱資料改為5,000筆有效、唯一的實際HFSS資料。採16筆獨立小job、最後不足16筆的尾批，正式SM臂同tier並行，盲選候選池預備2,048筆。重測與預測不計入唯一實測數。
- **兩種停滯分開處理**：SM迭代下的對稱pattern實測性能推進卡住時通知使用者；worker、佇列、訓練或controller等工作停滯自主修復。資料數增長／SM保留集誤差不是性能推進的替代判準，性能停滯不自動停止有效低分資料收集。
- **scoped worker容錯已驗證，使用者回報三台均重啟**：只在完整入帳且raw重播有效的部分HFSS終態失敗後，保留fail／claim並續跑其他jobs；當批預設暫存先驗路徑再清理。連續三批部分失敗、全失敗、schema/hash錯誤或清理失敗仍停機；不標假.done。完整695 tests與獨立Sol review通過，見[修正收據](../docs/log/assets/r80_scoped_worker_recovery_20261007.json)。補丁`c640fdd`已push GAN，使用者已回報全部重啟，37新結果亦已落檔；尚無遠端runtime Git核對，不以重啟回報宣稱版本稽核通過，不另開重複worker。
- 歷史controller（已於10/09 15:49:59正常退出）PID54712（creation1791505204815），2026-10-09 08:20由已push `ae63e1f`一次hidden start，launch=`b126f3badf3649a9bc9c4ea0f28ab37f`；固定`controller_shell_v1`／training_v2／原settings。原PID16536於07:13因已完成候選筆數與fresh名額比對失敗，原失敗status已綁入新attempt，候選／模型／raw保留。完整817 tests、獨立Sol及實際16筆本機重播通過；08:35首輪已完成並派原16筆prio1；08:50核對CIM／scope無警報，截止1,950唯一實測與v029 fit1,851分開記帳。[實際恢復收據](../docs/log/assets/r80_guided_recovery_completed_20261009.json)。[修補](../docs/log/assets/r80_guided_recovery_fix_20261009.json)／[啟動](../docs/log/assets/r80_guided_recovery_launch_20261009.json)。主線1800秒、LOW90秒；08:20 scoped status無警報，三台claimed／新結果及三片LOW待跑，下一routine健康約09:20。不啟動本機HFSS、不重啟worker；唯讀健康檢查不得取得live controller／dataset鎖。
- 歷史先驗12,000筆已登錄至本機`tmp/r80_factory_20261007/training_v2`，來自43,846個嚴格配對合格項目。data-v003累積155唯一（新增56），100 train/55 holdout；模型與55筆實體sample/rad hash及保留集預測重播通過，S11/Gain MAE2.48/5.09 dB。保留集由40增至55，不能直接用總誤差比較版本進步；本輪45筆完整前瞻依原模型分為v108 4／v001 35／v002 6。下一更新門檻203唯一。見[模型收據](../docs/log/assets/r80_factory_cycle_v003_20261007.json)、[歷史登錄收據](../docs/log/assets/r80_factory_legacy_20261007.json)。
- 18:34已派4×16盲選補池prio6；18:45派2×16 v108冷啟動SM選樣prio1。19:45已用data-v001新SM從10,000候選選48（19性能/15分歧/14盲選），逐片查重後加入3×16 prio1 jobs `dedust_r80cbdb25da6g01`–`g03`，共12 jobs；3片NAS bytes及queue均驗證。新候選未算入實測數。模型首版與派送收據另存私人NAS `sm_versions/data-v001`、`controller_receipts/bdb25da6...`，不改公共資料。
- 20:17以data-v002選32、派兩片16筆prio1；模型與50筆新凍結資料保存私人`sm_versions/data-v002`，122檔hash核對。20:45再派`dedust_r80c28d601d2g01`一片16（6性能/5分歧/5盲選），總15 jobs；派送時guided待跑82≤96，實測＋全部待跑＋新派240。最新cycle輸入／收據保存私人`controller_receipts/28d601d2...`，46檔發布後hash一致；新派不計入126實測。
- 21:17以data-v003再派`dedust_r80c452a2930g01`／`g02`各16筆prio1；guided待跑52＋32=84≤96，實測＋全部待跑＋新派272。模型、56筆凍結真值與收據存私人`sm_versions/data-v003`，135檔、29,192,207 bytes發布後hash核對；R80目標尚未完成，R81仍未派工。
- 21:45沿用v003再派`dedust_r80c743c7eb1g01`／`g02`各16筆prio1；guided待跑61＋32=93≤96，實測＋全部待跑＋新派304≤10,240。第五輪收據私人歸檔82檔、891,732 bytes發布後hash核對；一次RPC超時已由原worker重試成功，原job全16筆完成，未重啟worker。
- R81可選SM整鏈已完成：53項針對性及677項完整回歸通過，Sol review通過。歷史先驗已由360擴充至345 stores的21,034唯一圖形：21,251原曲線精確配對及八項舊指標重播，34缺失／error與217實體重複另列；[完整備料收據及限制](../docs/log/assets/r81_full_prior_20261007.json)。原曲線／metadata／凍結producer存私人`historical_prior/full_p01_db075_v001`；舊axis與worker版本屬明示重建，兩新阻帶仍mask。正式config未啟用，接到新真值後仍需排除保留集家族，尚無R81 HFSS真值或派工；工程檢查與正WM重測仍待完成。
- [啟動及回填指令](../docs/log/symmetry-filter-20261007-runbook.md)：開發與每台 worker 均使用 Git `GAN`；依序執行 `git checkout GAN`、`git pull --ff-only origin GAN`、`conda activate patch`，再以 `powershell -NoProfile -ExecutionPolicy Bypass -File .\script\start_scoped_worker.ps1` 啟動。
- 啟動器自動解析私人 `ROOTDIR` 下的 `experiments/r80_symmetry_20261007/dataset`；加 `-CheckOnly` 只列出佇列並檢查single-port身分，不啟動HFSS。scope=`symmetry_filter_20261007`，固定 `--selfgen 0 --poll 60 --stale 120`，且拒絕dual jobs。
- local controller 為 `tmp/r80_symmetry_20261007_single/single`；NAS初始備份為 `controller_initial/single`。158個檔案已部署到私人NAS並逐檔比對；預設 `-CheckOnly` 成功列出三個single jobs共52次，combined queue拒絕檢查也通過。worker實作 `29a6834` 已包含於遠端 `GAN`；完整pytest **544 passed / 361.29秒**，golden bytes不變。部署時worker與HFSS均尚未啟動。
- 舊ZIP、62次量測與「T:未掛所以NAS不可查」只屬先前交付紀錄，不再是目前操作依據。以下2026-08的worker/王者段落是歷史快照；舊grind/selfgen不恢復。

## 2026-08-31 歷史快照

> **✅ 2026-08-31 換機遷移完成**（舊機碩一_電腦退役 → 新機 i5-14600K）。
> 環境：conda `ant` / **torch 2.7.1+cu128**（golden 綁此版）/ RTX 3060。
> **⚠ 跑測試一律 `OMP_NUM_THREADS=4 python -m pytest tests/ -q`**——golden 綁**執行緒數 4**
> （舊機預設值），新機預設 14 緒會讓 `sc_loss`/迴圈漂 0.003–1.09%；設 4 即 482 passed 零漂移。
> **不要重錨 golden、不要在本機加 `CI=1`**（CI 的相對 1% 容差會讓 conftest 把漂移值寫回
> `tests/golden.json`＝靜默重錨）。詳見 [[project_golden_thread_count]]。
> NAS `T:` 已掛、量測備份 37.7 GB 已對帳通過（八類逐項相符）。三台正式機 worker 已 pull 並重啟。
> 現任王 `smp073_d_040` wm_mfg −2.39 未變；`tmp/grind_loop.STOP` **仍在**＝無人值守磨機維持暫停。
>
> **📦 2026-08-31 送板交付（學長鄭國宏要安排送板）**：天線 25 張 + 濾波器 14 張，
> 39 筆全部重跑 HFSS 產模擬檔（零失敗；**37 筆與原量測位元級一致**、2 筆差 ≤0.08 dB
> ＝順帶完成一次交付集公證）。工具＝`script/handoff_pack.py`（`index`/`report`/`pdf`/
> `export`/`collect`/`zip`）；輕量包（`.aedt`＋2 份 PDF，5 MB）已產出，含解完整版
> 在 NAS `dataset/handoff_package/`（7.5 GB）。
> 橋寬：天線 21 張 @0.1mm ＋ 學長池頂系 4 張 @0.075/0.05mm；濾波器 14 張全 @0.075mm。
> ⚠ **待人工清**：三台的 `keep_handoff_*` 工作目錄（`keep_*` 不符 `_dedust_*`，三道自動清理
> 都不碰＝刻意的 opt-in 設計，防整批線重演 2026-07-15 磁碟滿事件）；
> **218 那個 `keep_handoff_flt_db075` 是跑錯的單埠版，務必刪**。
> 教訓：dual 批發車漏帶 `--config` 被當 single 量完 14 筆 → 已補埠數守門（`bf1271f`），
> 誤打資料保留於 `dedust_xport_probe_20260831`（附 README，**不入鍋**）。見 [[project_dual_dispatch_config]]。

**批次線現況（2026-08-01）**：R33/R34/R35 已連續收輪（⚠ 開輪漏加本板 🔵 行三次——
時間軸真相以 [log/README](../docs/log/README.md) 為準,本板此段補指標）:
- ~~R33 反王朝結構輪~~（✅ 07-18:CNN 排序器制度化/顯微鏡爬山過 0）
- ~~R34 tier 架構元年~~（✅ 07-22:兩軸毫米線 8.95・rad −0.53/champ 判死/+1,554 筆）
- ~~R35 新節奏首輪~~（✅ 07-23:**新節奏常駐**〔批 75,產出 1.8×∧週期 <半〕/影子 CNN 轉正判準成立/
  tier 再平衡 2.37× 二連讀/expert best 口徑 2/2/兩堵牆=oob 9.0 抗線+G free 外推區）
- ~~R36 抗線輪~~（✅ 07-23:嫁接 70 分破紀錄 8.97→左右拆帳制誕生;批 50 常駐;c4lo 定讞王系左側斜率零）
- ~~R37 左側大陸殖民輪~~（✅ 07-23:殖民開工三選三;tri 前緣 −0.31;三提案全兌現）
- ~~R38 影子二號輪~~（✅ 07-24:★★★左側合格解首例達陣〔36hr,雙紀錄 7.78/−2.63〕/影子二號轉正=架構換代/鏈店大鍋）
- ~~R39 左側家族化輪~~（✅ 07-25:**批線孤點警訊成立**〔F 臂 48 席三批合格 0,族窄到只有鏈 d1 搆得到→F 臂撤〕;two 五批連勝→R40 換裝;lohead 進鍵;selfgen 換種王朝 0%）
- ~~R40 換裝與空洞輪~~（✅ 07-25:**two 換裝定案**〔三批全勝,ρ+0.856 全史最高,G 臂 adv 100%→0〕;V 臂 2/3 存續;多樣性警報結構解;佇列原子化零事故）
- ~~R41 組測試輪~~（✅ 07-26:**★usable_lo −3.46 易主〔組算子第二包破紀錄,公證3/3〕·「組=變異單元」定案鏈線常駐**;刀鋒解結構學;第四筆合格解 p06_11 單次;V 4/5;鏡射警訊）
- ~~R42 常態輪~~（✅ 07-26:**★★margin 王 +0.73〔全史最大單跳,公證3/3〕**;可製造化三連負定案〔清潔死路→生成端〕;家族帳5;組文法立項+前置全套）
- ~~R43 組文法首航~~（✅ 07-27:**「對角=左側門票」方向性定案〔GDd −3.34 vs GD +2.44 三批,混雜解開〕**;文法vs舊互有勝負續測〔GC汰/GA2進〕;V 7/8+首帕累托;cnn2 ens 上線）
- ~~R44 文法二輪~~（✅ 07-27:**文法六批判定=組文法 v1 未勝舊文法〔old 天然對角=左傾 9/12〕,GC/GA2 收案,GDd 留任**;苗子 84 筆全零=結構性斷層→接力）
- ~~R45 接力輪~~（✅ 07-28:**★深水右爬大成立——g 線 +4.70〔p09 單包 +2.91〕wm 貼苗子線,苗子斷層被鏈跨過**;d 線 +2.07 高原=起點均衡度>深度）
- ~~R46 接力二輪~~（✅ 07-29:**g 系接力全線終點 −2.72〔總帳 +5.32〕止步作戰區前 1.7 dB=高原判定條件①成立**;均衡型新原礦 d46b3_008〔rad +1.21〕出土;v85 準度新低 1.227）
- ~~R47 接力三輪~~（✅ 07-30:**盆地個體性定調**〔rad 正穿 −7 牆但終點 −6.21 vs g 線 −2.72 差 3.5dB〕→廣度勝深耕;**儀器換代:lr 修=凍結尺 1.10→0.53**;two 轉正斷;GPU/30ep 基建）
- ~~R48 定向嫁接輪~~（✅ 07-31:**嫁接當錨+爬山=換盆地可重複**〔嫁接單步主判準未達,B 式嫁接體當錨→c48nq1 鏈 5 包爬 +0.94〕;**margin 王易主 +0.79**〔r48n2 3/3〕;雙空間雙閘首案例;lohead 三連過線;儀器 1.5hr 常態化）
- ~~R49 兩段式制度化輪~~（✅ 07-31 一日輪:**兩段式條件化定論**〔主流聲錨×px 有效/碎片語言錨死;裁決懸置〕;**I 臂爆發 33→58→67%**;合格 27 三連升;凍結尺 0.48;lohead 六連〔首航挪 R50〕）
- ~~R50 型態體系軸元年~~（✅ 08-02:**儀器元年**〔OOD 尺/雙頭制/雙外軸產線〕;主錨 0.47 全史新低;冷啟動 4 點非單調;學長錨銀行 6+3;文法外未開山〔−5.88〕）
- 🟡 **戰略決策點(Ricky 裁,2026-08-16)**:−3.14 高原定讞(五算子+兩桿全出清,
  R72b3-R74 證據鏈)——①慢磨模式 ②帶成績單回規格對話(帶內已解/S21 兩側各差 ~3)
  ③50×50 重議 ④E2E 純自產線(觸發已到)。裁定前=**SM 常態補池模式**(Ricky 2026-08-16:smp chunk 週期送+c 臂防馬太;首 chunk smp002 90 筆已發)。
- ✅ R74 縫合二輪(2026-08-16 提前收,1 批+橋寬探針):**−3.14 高原定讞;SC 二連敗
  (SM 弱在 m3);輻射/橋寬兩桿出清(0.075 已最適)**
  →[round-74](../docs/log/round-74-dual-suture2.md)。
- ✅ R73 線上迴圈首航(2026-08-16 收,15 迭代 135 筆):**基建成立但效率輸批次;
  破王 −3.14(公證)=現任王;四敗因→v2 回爐;治理合約首次執行**
  →[round-73](../docs/log/round-73-dual-online-suture.md)。
- ✅ R72 SM v5 上崗輪(2026-08-15 收,3 批+2 公證):**王 −3.15(七跳);分軸定向常態化;
  軸紀錄爆發(m3 −1.01 三軸體/m4'+2.32);harvest 遠親出清;縫合=最後一關**
  →[round-72](../docs/log/round-72-dual-skirt-farpeak.md)。
- ✅ R71 王朝續採+演算法對照(2026-08-14 收,3 批+2 公證+2 探針):**王 −3.51→−3.37
  (兩跳跨機 bit 級;★首雜交王=池產苗×王脈);演算法增益帳(選拔+3.63/開採+2.08);
  池產迴路成立;出圖必帶橋定案;缺口=兩條裙擺**
  →[round-71](../docs/log/round-71-dual-dynasty-mining.md)。
- ✅ R70 可製造時代奠基輪(2026-08-13 收,單日 3 批+3 公證):**A 臂繼承路線勝;
  ★wm_mfg 開帳 −5.90 → 易主 −4.26(跨機 bit 級 3/3×2 次),−3.82 公證中;
  帶內近收乾;通濾匯率 ρ−0.16=近獨立**
  →[round-70](../docs/log/round-70-dual-mfg-founding.md)。
- ✅ R69 哥白尼輪(2026-08-13 收,單日 41 筆探針):**p00 對角零厚度接觸=數值幻影定讞
  (三線證據);儀器換代 p01(follow single)+era 三閘;存活地圖=部分重置(乾淨層
  Δ0.16 存活/湯值報廢);可製造前緣 kn_16+0.075 橋=−5.96;50×50 凍結**
  →[round-69](../docs/log/round-69-dual-hd50.md)。
- 🟢 **SM 篩選補池上線**(2026-08-13,Ricky 核):`dedust select-smpool --dispatch`=
  auto 池升級(SM 粗篩+配額防馬太 L40/d30/c30,prio 6;對照臂出頂率=馬太量測);
  首 chunk smp001(90 筆,預測 max −2.84)已生成、**待三台重啟後與煙測一起發**;
  selfgen 退居佇列全空備胎→decisions「Auto 池升級」節。
- ✅ R68 零點歸因輪(2026-08-13 收,3 批 145 筆單日):**歸因成功手術失敗——敏感度地圖
  (零點旋鈕區/通帶命脈/中央脊柱)+通濾非鎖死;但六種位元粒度+亞像素縫劑量全滅於 m3
  (整像素匯率 1:17/細縫非單調)→25×25 對規格 v2 判死,50×50 匯合轉正**
  →[round-68](../docs/log/round-68-dual-zero-attrib.md)。
- ✅ R67 多王朝→★規格 v2 事件輪(2026-08-13 收,3 批+公證):**規格 v2(學長:帶內 −10/
  阻帶 −15/帶外退場)缺口 6.1→3.05;新尺首王 kn_16 wm_r2 −3.05 公證(m4' 過標);
  m3=位元語言死軸(九判準零過)**→[round-67](../docs/log/round-67-dual-dynasty.md)。
- ✅ R66 蛇形輪(2026-08-12 收,1 批答定):**兩假說雙殺(細蛇形線=meander 天線,不降輻射;
  摺疊只調 m3 不救 m4);第三次定向工程輸有機演化→結構工程收線**
  →[round-66](../docs/log/round-66-dual-meander.md)。
- ✅ R65 六軸全尺首輪(2026-08-12 收):**wm_full 主尺換代(學長裁定六軸全硬),首帳 −6.21
  (0549 公證 3/3);平頂 −6.1~−6.2**→[round-65](../docs/log/round-65-dual-fullruler.md)+
  [analysis-13](../docs/log/analysis-13-dual-ceiling.md)(天花板 EVT −6.08;★修正=採樣語言極限≠域極限)。
- ✅ R64 實戰考(2026-08-12 收):**SM v3 細排不過(實戰 ρ+0.118;離線考假訊號三度應驗);
  線上學習觸發撤回;m4 +7.39 七連破(公證)**→[round-64](../docs/log/round-64-dual-v3-live.md)。
- ✅ R63 sel 續採(2026-08-12 收)+R62 sel 開採(2026-08-11 收):**sel 王朝三代均衡開採,
  王 −5.62→wm_dual −4.67(kn_23 公證);分歧採樣 3/3 出頂=常規武器入 decisions**
  →[round-62](../docs/log/round-62-dual-sel-mining.md)/[round-63](../docs/log/round-63-dual-sel-2.md)。
- ✅ R61 王朝開採輪(2026-08-11 收,3 批):**王 −5.62(d 臂高分歧出土,公證);m4 四連破 +6.73;
  假象攔截 r61n3;SM v2 過閘 ρ+0.50**→[round-61](../docs/log/round-61-dual-smv2-guided.md)。
- ✅ R60 dual 縫寬輪(2026-08-11 收,一日輪 1 批):**slot_spec 儀器成立(單調+括號全過)但
  量子化假說否證——守恆帶=5×5mm 板結構極限(裝不下多節/輻射損)**
  →[round-60](../docs/log/round-60-dual-slotw.md)。
- 🟢 **dual auto 池上線**(2026-08-11):三台閒時自產 dual(symr+錨鄰域各半,錨池=NAS
  dual_anchors.json 免部署輪換);資料量累積=前期主目標(Ricky)。
- ✅ R59 dual 開縫輪(2026-08-11 收,一日輪 3 批 153 筆):**開縫=首個實證結構旋鈕(m4 撬
  11dB)/sel 前緣 −9.25/三構型全撞 m3+m4≈−17 守恆帶=像素量子化天花板→出路=亞像素縫寬
  /50×50(與 single R56 跨線收斂)**→[round-59](../docs/log/round-59-dual-slot.md)。
- ✅ R58 dual SM 冷啟動輪(2026-08-11 收,一日輪 3 批):**SM 過閘=粗篩非細排;王 −6.04 三批
  停滯(盆地枯竭);m4 紀錄 +3.13(公證)→+6.67(公證中)全出對稱系;wireO 存在證明+
  「m3 穩 m4 僵」→R59 切路成諧振段**→[round-58](../docs/log/round-58-dual-sm-coldstart.md)。
- ✅ R57 dual 元年(2026-08-11 收,一日輪 3 批+公證):**設施復用全綠 240 筆 0 error/三判準全過/
  harvest 天花板 −7.20 推至 −6.04(單次,公證中)/生成時對稱升預設/事後對稱化死/殺手軸=S21 選擇性**
  →[round-57](../docs/log/round-57-dual-maiden.md)+[analysis-12](../docs/log/analysis-12-harvest-dual-audit.md)。
- ✅ R56 組消融方法論輪(2026-08-10 收):**歸因圖成立+Δn8 偵測器轉正 6.8×;c8trip=25×25 局部最優,收穫=無(誠實帳)**→[round-56](../docs/log/round-56-block-ablation.md)。
- ✅ R55 微調輪(2026-08-06 當日收):**對稱化判死(置中 100% 達成但代價中位 −17)→置中=血統性質**;
  微調菜單(50×50/參數化/波束舵)續議中,P4 smoke 等下次 pull→[round-55](../docs/log/round-55-finetune.md)。
- ✅ R54 菱形橋輪(2026-08-10 閉帳):**「可製造對角」成立——稅≲噪音/三閘全保 79.1%/rad·lo 零損;設計三規=buffer 0.3·n_sites<17·出圖帶菱形**;交付集 943→[round-54](../docs/log/round-54-diamond-bridge.md)。r51 橋池不還原(機時轉 dual)。
- ~~R53 低對角左側軸首輪~~(✅ **08-05 收檔,3 批**):**雙軸鏈同撞 rad −2.1 高原/5k 消融翻案(OOD 2.5×)/轉正制議廢**;lo 深水穩(峰 −6.41 單次)。
- ~~R52 錨銀行攻堅輪~~(✅ **08-04 收檔,3 批**):**網格定讞〔S0 未收斂/天花板 0.837〕+兩閘裁定〔生成路+軸首批錨〕+rad 鏈 −0.55→−0.08〔p03 跨輪〕**;lo 鏈學費盡;學長臂退役。
- ~~R51 橋接與進鍵輪~~（✅ **08-03 收檔,3 批**）:**橋接 2.5k 全上架/lo 首航退鍵〔0/150,生成端沒貨〕/two 首度三尺全贏〔轉正 1/2〕**;可用帶外連零 3=收割期確認;事故三連全清零。
- ✅ 主軸=低對角左側(decisions 08-04;兩閘已裁:生成路+手術窄門/t07 帶 lo 過驗)——R54-55 的菱形/微調線即其延伸。
- 🟢 **grind_loop 無人值守迴圈上線**(2026-08-21;pid 起於 detached 進程):
  dual 慢磨全自動(chunk→判讀→破王公證→每 3 chunk 重訓),**與對話 session 無關、
  不自己加冕**(待審寫 tmp/pending_records.jsonl)。看狀態=`tmp/grind_loop_status.json`;
  煞車=建 `tmp/grind_loop.STOP`;詳見 decisions「無人值守慢磨迴圈」。
- 🟡 **線別切換決策點(預先登記,2026-08-20)**:R78 保和再平衡的結果決定主力——
  **A 成立**→dual 續戰追 −2.3 級;**B 脆弱定讞**→**主力轉 single**(Ricky 裁定),
  開場=三個零 HFSS 分析(單一栽培診斷/守恆框架移植/距離-品質剖面),詳見
  decisions「線別切換的預先登記分支」。
- 🔜 **single 線多起點移植**(Ricky 2026-08-20:「這個方法很不錯,之後 single 用用看」;
  觸發=single 線重啟或 dual 主線收斂後):把 dual R76-77 的整套方法移過去——
  ①**單一栽培診斷**(歷代冠軍兩兩 Hamming + top300 家族數;dual 實測:十五代王互距
  中位 10 bits、top300 零外族=前緣是一個家族)②**距離-品質剖面**(按距冠軍 Hamming
  分帶取最佳,找遠域種)③**遠域種多代開採**(dual 實測 S2 系四代 −4.83→−3.36,
  距王 257 bits 的獨立家族)。
  why:single 線同樣長期單錨開採(王系凍結/dyn-frac 0.2 是既有的多樣性措施,但沒做過
  「遠域種扶正」);且 dual 已證**兩獨立家族的飽和點對照可區分「家族極限 vs 域極限」**
  ——single 的 +0.79 平頂是哪一種,這個方法能回答。
  ⚠已知限制(dual 實測):遠×遠雜交與遠×王朝雜交**都判死**(結構互斥),遠域種只能
  各自單線爬;起點品質決定一切(dual 四種只有兩種爬得動)。
- 🔜 **single 線儀器問題:橋從未升為預設**(Ricky 2026-08-20 指出;觸發=single 線重啟/
  交付前/要做可製造宣稱時)。現況:single 量測幾何=**隱式 0.01mm 橋**(像素盒 +0.01
  重疊的副產物),R54 的 0.075mm 菱形橋只當「出圖時的可製造化處理」,**從未成為量測
  預設**——所以整張 single 榜(含 margin 王 +0.79/rad 王/左側 c8trip03_01)都是
  0.01mm 尺量的。
  ⚠**dual 實測的警訊**:同一顆王 0.01 橋 −6.86 vs 0.075 橋 −3.82=**差 3 dB**
  (round-71 x1 探針)——**換標準橋後 single 的排名可能重排**,不是小數點後的事。
  三個層次(由輕到重):①**王冠三檢回補**(現役冠軍跑 fill/切角/裸三態+跨尺度括號,
  ~15 筆機時)②**標準橋重測榜前段**(看排名重排幅度)③**儀器換代**(single 也走
  p01+0.075 標準橋=量測即可蝕刻;代價=6 萬筆 harvest 與全部冠軍轉「舊代」,
  比照 dual 2026-08-13 的做法,含 era 三閘與紀錄分鍵)。
  價值:dual 的可製造代四天做出十五代王+第二王朝,**證明換代後系統照樣跑得動**;
  single 若要做實體驗證或可製造宣稱,這是必經之路。
- 🔜 探索方向存量:①生成端知識化（per-組義容忍度→bias 組變異）②嫁接兩段式系統化（R49 吸收）③GNN 封存待命（重啟=Ricky 點名;exam+142 考卷+7 ckpt 全在）。


**全域變更（2026-06-28）**：① 驗證預算改為**跑到 500 epoch**（約 3 天；原 250）→ Round-2 config `epochs: 500`。② **回滾機制已移除**（對 generator-free + K 候選 + 線上 SM 不合身、且原實作有 off-by-one + 覆蓋最佳檔兩個 bug）→ Round 1 的「不收斂」有它一份；探索改靠 K 候選 + SM 引導 (+ trust)。最佳 pattern 仍安全存 `patterns/`。

---

## 🔵 進行中 / 待跑

### 資料工廠（🟢 **基礎設施就緒**,2026-07-10）
- 正式機常駐:`python -m script.dedust worker`（認領 NAS `dataset/jobs.json` 佇列;原子 claim 防互踩;
  單筆 watchdog 900s 殺卡住的 HFSS;連敗 5 筆標 .fail 停機）。派工:`jobs-add --input X_input --store X --prio N`。
- 停止=建 `dataset/jobs_state/STOP`;stale 接管=45 分無進度可被別台接手。
- **哨兵（零 token 純腳本）**:`python -m script.status --factory --alert --notify-topic <主題>`——
  掃佇列進度+卡住(30分無結果)+停機(.fail);排程 `schtasks /Create /TN AntennaFactory /SC HOURLY /TR "..."`。
  **待 Ricky:定 ntfy 主題(隨機長字串)+手機裝 ntfy app+回報後我出完整註冊指令**。
- **流程全 skill 化（2026-07-12 弱模型化）**:主入口 **/batch-cycle**（收檔→判讀→公證→重錨→發車→
  補池→掛偵測→記帳）;/notarize（公證/換王）/new-round /close-round /gain-check /stall-protocol。
  工具面:`analyze batch`（判讀一鍵）`dedust watch`（偵測）`sm_reanchor train --add`（重錨一鍵,
  清單=configs/clean_stores.txt）;紀錄門檻真相源=`docs/records.json`。
- worker 三機常駐（2026-07-11 起;個性見 memory）;**自產 tier-2（--selfgen 預設開）=佇列全空自動翻
  歷史 bit 產資料,任何 job 入佇列即讓位——HFSS 制度上不停**。

### diffsim — 可微模擬器（🔵 **2026-08-03 晚間翻案重開**）

> **★ 現況一句話（2026-08-04 收）**：出貨值 **`l3fld` + `a_quad=1`**（對角連通已進，`build_l2`
> 與所有 CLI 的預設）。**同池對照**（正片 `clean_OOS` n=817，先前所有跨池比較作廢——
> fit 分割比評估池容易）：`l3fld` 命中 **41.7%**／`P(勝隨機)` 61.7%／ρ **+0.8248**，
> 對上 SM-cnn 41.7%／80.0%／+0.84、SM-two 45.0%／86.4%／+0.86、SM-mlp 51.7%／87.5%／+0.84。
> 負片 n=1200：`l3` 50.0% → **`l3fld` 83.3%**（ρ +0.46→+0.65）。
> ⇒ **ρ 已追平 SM；差距在 `P(勝隨機)`。命中率＝整批品質（追平 SM-cnn），
> `P(勝隨機)`＝能不能出一個王（明顯落後）—— 對「推紀錄」這用途，落後的正是要緊的那一半。**
> ⚠ **出貨值有一個已量化的數值錯誤**（對角×軸向互耦高估 69%）而表現最好；
> 假說是它剛好在模擬未建模的 10µm 頸縮（[§51](../docs/log/analysis-10-sm-vs-physics.md)，待驗）
> ⇒ **可以說「排序有用」，不能說「對角物理是對的」。**
> **下一步（判準已寫死）**：`a_quad=4` + 顯式頸縮阻抗掃 25–500 pH，能否重現 55%。

- ★★★★ **收線的理由被推翻了**（2026-08-03 19:26，[analysis-10 §37–§38](../docs/log/analysis-10-sm-vs-physics.md)）。
  下面的收檔內容**原文保留**，但兩個關鍵結論已撤：
  1. ❌ **「離散化天花板」是統計假象**。真網格收斂（固定貼片改 dx、未知數 ×8.7）
     `Re(Zin)/閉式` 只從 0.369 → 0.387，**完全平的**。舊那組「7→23 格單調上升」是
     **固定 dx 改貼片物理尺寸**，量到的是埠因子隨貼片大小變。
  2. ❌ **「diffsim 在選批上沒有任何場景勝過隨機」不再成立**。真根因是**埠模型**
     （delta-gap 壓在貼片邊 0.2mm 一格上，對同一模態振幅注入 **2.2 倍**電流；
     HFSS 的埠在 22.5mm 饋線的**遠端**）。把真饋線建進格網 + 駐波法萃取 Γ
     （`l2.SOLVERS['l3fl']`）之後，**正片 clean_OOS 的 top-60 命中率 10.0%（＝隨機）→ 26.7%**，
     Δ 的配對 bootstrap **P(>0)=100%**（判準寫死 ≥95%），`P(勝隨機)` 18.6% → **54.7%**。
- ⭐ **兩個零參數的改動疊起來到 36.7%**（隨機 10%，SM ~52%）：饋線埠 + 讀出換
  `−mean(S11[5:12])`。⚠ 讀出方式**尚未採用**——它是在評估集上量的，
  必須先在 `fit` 分割確認（判準已寫死）。
- ★ **機制不是「變準」**：絕對準度全面**變差**（谷更淺、誤差 std 更大、系統偏差更負），
  贏在**收回了 `min(S11, Gain)` 被 Gain 通道毒害的損失**（作弊檢查：把模型 Gain 邊界
  換成真值，集總埠 10.0% → 25.0%）。
  ⇒ ★★ **`worst_margin` 這把尺本身是風險**：真值 top-60 有 **97%** 是被 Gain 卡住的，
  而模型只有 25–33% ⇒ 模型的 Gain 通道系統性太樂觀。**剩餘空間幾乎全在 Gain 半邊。**
- ✅ **在完全不相交的 `fit` 分割上重現**（§42，這是最重要的一節）：
  clean(680) **18.3% → 28.3%**（Δ+10.55%，**P=99.0%**）、neg(306) **28.3% → 46.7%**（Δ+17.20%，**P=100%**）。
  ⇒ 負片的命中率改善在評估池上未達門檻（P=81.7%）只是**功效不足**（基線已在天花板）。
- ★★ **一個與模型對錯無關的結構性結論**（兩池一致）：真值 top-60 裡 Gain 主宰的比例
  **正片 97% / 負片 12–15%** ⇒ **正片的好 pattern 被「增益」卡住、負片的被「匹配」卡住**。
- ❌ **機制敘事不重現**：`−mean(S11[5:12])` 那個讀出在 fit 分割上**四格全輸**（**不採用**）；
  「`min()` 被 Gain 毒死」也是評估池特有的。⇒ **結果重現、解釋不重現。**
- ✅ **四個 agent 全部收工**（`Z_c` 判定／Gain 半邊／對抗式複核／fit 確認）→ §39–§42。
  其中對抗式複核**推翻我五條宣稱**（含把「跨實作對帳」這個招牌證據拆掉）、
  找到三個我沒列進候選的東西、以及一條我**漏記的有利證據**（共振頻率誤差 −7~−8% → −1.5~−3%）。
- ★★★★★ **08-04 凌晨：又一個規格不符，是整輪最大的單一改善**（§45/§47）。
  Ricky 指示「先審 simulate 的規格再繼續」，第一眼就撞到：**HFSS 的像素盒是 `pixel + 0.01mm`**
  ⇒ **對角相鄰的像素在角落重疊 0.01×0.01mm、`Unite` 後是同一塊導體**
  （那 0.01mm 的註解只提「邊相鄰不要只共邊」，對角是沒被注意到的副作用），
  而我們的 rooftop 基底只有邊相鄰 ⇒ **全史 57.1% 的樣本幾何與 HFSS 不符**。
  修法＝方向從布林推廣成**電流矩向量**、對角**只在它是唯一通路時**啟用（實心區逐位不變、成本近乎零）。
  判準發車前寫死，fit 分割 n=986（與評估池零重疊）：
  **ρ(wm) +0.4571 → +0.7070（Δ+0.2504，P=100%）、top-60 命中率 26.7% → 40.0%（P=99.6%）**；
  幾何不一致子集 **ρ +0.0515 → +0.5177** ⇒ §45 的因果歸屬證實（是連通性不是破碎度）。

  | 組態（fit 分割 clean，隨機 10%） | 命中率 | ρ(wm) | 每點 S11 MAE |
  |---|---|---|---|
  | `l3` 今天之前實際在用的 | 18.3% | +0.5172 | 2.30 dB |
  | ＋饋線埠＋eta＋排除饋線遠場（`l3fl`） | 26.7% | +0.4571 | 1.95 |
  | ＋對角連通（`l3fld`） | 40.0% | +0.7070 | 1.74 |

  ⚠⚠ **06:0x 更新：上面那一列已經不是出貨值。** 對抗式複核找到**我實作裡的兩個 bug**
  （反對角基底因角落格指派錯而**全史零活化**、A 項求積點對對角不是反射協變）。
  修完之後 **ρ 暴漲到 +0.9114 而選批命中率崩到 21.7%**（`l3fl` 是 +0.4922 / 35.0%）
  ⇒ **本輪第九次「指標漲、用途跌」，而且打的是我自己的頭條**。
  而且那兩個修正**一起下**、A 項改動連 `l3fl` 都動到 ⇒ **無法歸因**（2×2 消融未完）。
  ⇒ **出貨暫回 `l3fl`；引用任何 08-04 凌晨的 `l3fld` 數字前先看 analysis-10 §49。**
- ★★ **誤差從「結構性有害」翻成「結構性有利」**：模型的實際 ρ 現在**超過它自己的純噪音版**
  （+0.7070 vs +0.7029）、命中率贏純噪音 10 個百分點（§36 當時是**輸** 10 點）；誤差 std 6.38 → 4.54。
- ⚠ **今天三個最大進展，兩個是「規格不符」、一個是「實作 bug」，沒有一個是物理更精深**
  ⇒ **「有沒有在模擬 HFSS 實際做的事」比「物理夠不夠深」重要得多**，而這一類還沒挖完。
- ⚠⚠ **一條與 diffsim 無關、但影響你們所有以 wm 為準的判定**（§46.2，已送留言板）：
  掃頻是 `Interpolating` ⇒ **遠場只在真正解過的頻點存在**，
  **26.1% 的樣本至少一個 Gain 值是 `np.interp` 捏出來的**；
  **5.0% 的樣本其 `worst_margin` 由捏出來的點決定**，改用最近真解點 margin 中位動 **2.28 dB**。
- 🔜 **還開著**：**Gain 通道仍系統性太樂觀**（真值 top-60 有 **97%** 由 Gain 決定、模型只有 22%）
  ⇒ 剩餘空間幾乎全在這裡；`gamma_from_line` 的 `|w|=1`（~19% `Re(Zin)` 敏感度，兩個選項都不明顯正確）；
  「源端洩漏 40%」vs「駐波擬合高估 |Γ|²」（決定 §37 的阻抗帳平不平）。
  ⚠ 我先前說「有限基板是最可疑的」——**已自我修正**：η 只比 Jackson–Alexopoulos 差 1.3–1.7×
  （換算 Gain 約 1.1 dB），而我們差 4 dB ⇒ 不太可能是主因。

<details><summary>以下為 08-03 白天的收檔內容（原文保留，上面兩條已撤）</summary>

- **L1 腔模型是唯一有實用價值的產物**（clean 層內 ρ **+0.418**、**84ms/筆**）；
  L2/L3 的 MoM 路線**依發車前寫死的判準收線**——G-L3a 未過（clean +0.194 < +0.40）。
- ★★ **「停」的理由不是 bug**：四條獨立證據排除核錯／自項錯／精度不夠／沒做完
  （empymod 四管道 1e−12＋鑑別力測試、真積分 b_eff=0.3363 對上實測最佳 0.30、
  G_V 擾動 Δρ≈0.005、S1 上界兩路徑都估 +0.04）⇒ 是**離散化的天花板**。
- ⚠ **真正的瓶頸可能在真值端**：→ [proposal-mesh-convergence](../docs/discuss/proposal-mesh-convergence.md)
  （HFSS 網格收斂實驗，~1 小時機時，已在下方 🔜 候選區）。**優先於任何模型改良。**
- ✅ **與現役 SM 的比較已做**（Ricky 08-03 授權唯讀口）→ [analysis-10](../docs/log/analysis-10-sm-vs-physics.md)：
  **不接主管線**（正片域 SM 領先 **+0.61**）。
  ★ **原本提的「L3 當新域冷啟動排序器」已於收檔 20 分鐘後撤回**（analysis-10 §5）——
  全域 ρ 與 **top-K 選批效益反向**：L3（ρ +0.43）在 K=30 只有 **30%** 機率勝過隨機，
  影子 CNN（ρ +0.078）反而 71%；兩兩配對 CI 全跨 0＝**沒有方法顯著勝出**。
  ★★★ 進一步用同一把尺量**正片域**（主戰場）：**L1 39%／L3 17% 勝隨機＝比隨機挑還差**，
  而 SM 三個都 80–92% ⇒ **diffsim 在「選批」這唯一實用途徑上沒有任何場景勝過隨機**。
  ★★ 最強的一條證據是縱向的：修掉一個**真正的物理 bug**（饋電模型）讓 ρ 改善 **+124%**，
  而**選批 P(勝隨機) 18%→17% 完全沒動** ⇒ 兩指標**幾乎正交**。
  ★ 我一度據此警告「五軸 KPI ① 改善不能預設會轉化成選批」，**已實查、不成立、撤回**（見下）。
- ✅ **順手替批次線驗了五軸 KPI ①**（analysis-10 §8，唯讀 `sm_reanchor` v50–v100 十三版，
  同一批 OOS 樣本 817 筆逐版重量）：
  **ρ vs P(勝隨機) = +0.720、ρ vs top10% 命中率 = +0.775、|err| vs 命中率 = −0.664**
  ⇒ **KPI ① 是有效的代理，重錨把 SM 訓準、選批能力確實跟著上去。**
  ★ 真正的分界不是「哪個指標」而是「改善的性質」：**全域均勻**的改善（加資料/調容量/修訓練 bug）
  ρ 可當代理；**結構性**的改善（修某一項物理/加某一特徵/換某一層）**必須直接量 top-K**。
</details>

- ⭐ **順帶量到：v85 → v88 是 4 倍跳躍**（ρ 0.567→0.823、|err| 1.52→0.73 dB、
  **top10% 命中率 15%→58%**；v50–v85 六個版本幾乎是平的 12–18%）。
  與「2026-07-29 SM 儀器換代日（lr bug 修）」對得上 ⇒ **那次修復在「選批命中率」這把尺上值 ~4 倍**，
  比凍結尺的 −51% 更能說明它值多少。⚠ 單一 OOS 樣本集的重量，非官方口徑，僅供對照。
- 🔧 **工具已交付**：`python -m script.sm_selection_audit --versions 88,94,100`（唯讀、可重跑）。
  ⚠ 讀數注意：平手算輸 ⇒ **上限 ~86% 不是 100%，50% ＝ 與隨機無異**。
- ⚠ **一個觀察（不是結論）**：換代後 12 次重錨，在該 OOS 池上四個指標**全無顯著趨勢**
  （p=0.36–0.88, n=7 版）。**但該池的定義是「不在 `CLEAN_STORES` 內」⇒ 新重錨吃到的資料
  永遠不會落在裡面**——它只量得到**泛化外溢**，量不到重錨在自己覆蓋區內的收益（那才是主要價值）。
  ⇒ 正確讀法：「**v88 後，重錨對永遠吃不到的區域沒再帶來額外幫助**」，對 R50 開新域直接相關；
  **不能**讀成「重錨投報遞減」——要判那個得在**新加入的店**上量（你們的口徑，我不越界）。
- ⭐ **兩條對批次線直接有用的產出**：
  ① **主 SM（mlp）在完全未見過的域選批比隨機還差**（P = 21–26%，K=30/60 都是，B=4000）
     ⇒ **開新域時不要用主 SM 挑候選**（R50 雙外軸產線正在開新域）；影子 CNN 與 L1 至少不害。
  ② ⚠ **`_load_clean_stores()` 自動納入 `dedust_auto*`/`dedust_c*`** ⇒ SM 訓練集是 **587 店**
     而非 `clean_stores.txt` 的 513 行。任何「SM vs 其他方法」的比較沒扣這層都會
     **系統性高估 SM**（我們量到 clean 層 97% 是 SM 見過的）。
- ✅ **`docs/diffsim.md` 的鏈已全部走完**（含原計劃最後一步「殘差 head ＋最終 SM 比較」）：
  學習曲線 A/B/C ⇒ **物理當特徵也沒價值**（物理特徵沒讓曲線左移，小樣本反而更差）。
  ⭐ 但同一個 L3 輸出「直讀 8% → 學習節重讀 22%」（×2.75）⇒ 問題是**響應到真值的映射不是恆等**，
  不是「什麼都沒算對」。執行結果摘要已補在 `docs/diffsim.md` 頂端。
- ★ **若日後要重啟，第一步已經指出來了**：病在**共振頻率完全抓不到**（谷位置 ρ +0.078
  ＝隨機水準），而不是 wm 或層內 ρ。已試的半屋頂埠（`MoML2(half_port=True)`，預設關）
  把它拉到 **+0.188**（六指標全同向、Im(Zin) 中位 −50.9→−8.9 Ω）但**未達發車前判準 +0.30**
  ⇒ 還缺 **表面波高估 2–3 倍**那條（analysis-09 §7/§8 #6）。
  ★★ **再往下追之後，病因收斂成單一根因**（analysis-10 §13–§16）：
  **`Re(Zin)` 低估 ~4.8 倍**（共振點 178 Ω vs Balanis 積分式 862 Ω）
  ⇒ |Γ|≈1 ⇒ S11 很淺 ⇒ 谷很平 ⇒ 谷位置 ρ 低 ⇒ wm/選批跟著壞。**一個根因解釋所有症狀。**
  沿途排除七個候選；**共振頻率其實很準**（只低 8–9%、四尺寸一致——我原本說的「低 20%」
  是漏了邊緣延伸 ΔL 修正）。
  **驗收指標＝共振點 `Re(Zin)` 對上 Balanis 積分式**（比值 0.05–0.09 → O(1)），
  ★★ **再修正兩次之後的最終數字**：真實差距是 **~3.3 倍不是 14 倍**（我的閉式對照
  兩度漏了修正項：共振頻率的 ΔL、輻射電阻的互導納 G₁₂）；「Zin 不隨埠寬度變」是**對的物理**。
  離散化解釋一部分（5→15 格 0.018→0.333）但 **≥15 格飽和在 0.33**。
  ⇒ **終點＝輻射電阻低 ~7 倍**（Re 低 3.3× × η 只有 0.43），而模型的 R_sw≈101Ω
  其實接近文獻反推的 ~134Ω ⇒ **偏差幾乎全在 R_rad，嫌疑在電流分佈**（遠場積分已驗證正確）。
- 完整脈絡 → [analysis-08](../docs/log/analysis-08-diffsim.md)＋[analysis-09](../docs/log/analysis-09-diffsim-l3.md)；
  code `script/diffsim/`（32 條物理測試）。

### ~~網格收斂實驗~~(✅ **R52 收檔定讞 2026-08-04**)→ [round-52](../docs/log/round-52-anchor-assault.md)
- S0 未收斂(谷位移中位 0.985GHz)/S1→S2=收斂;rank 天花板 ρ=0.837=SM 已頂;t 帶 lo 被 S0 低估。

### ~~Round 13 — 組數階梯~~（✅ **2026-07-08 收檔**）→ [round-13](../docs/log/round-13-block-ladder.md)
- 組數=真設計軸但報酬有取捨:4-5 塊甜蜜點(5 塊買 rad/4 塊買選擇性)、6 塊遞減;margin 天花板僅微升。

### ~~穩健 bake-off~~（✅ **2026-07-08,製造冠軍=x00**）
- 缺陷 k1×18:x00 存活 72% > c25 56% > c21 28%。**送製造首選=x00**（wm +0.19/rad +0.19,公證✓）。

### ~~Round 32 — 海峽輪~~（✅ **2026-07-17 收檔,3 批**）→ [round-32](../docs/log/round-32-strait-crossover.md)
- **海峽幾何可填（92%×2）電性不可雜交（判死）,教材→期望閘 4.05→1.50**;影子三讀定型
  「CNN=排序器/MLP=回歸器」;同框 wm 首過 0（殘閘=rad 單軸）;B 泵血統 B>A;工作模型 v45。

### ~~R34 候選~~(✅ 已於 07-22 開輪並收檔 → [round-34](../docs/log/round-34-tier-era.md))

### ~~Round 33 — 反王朝結構輪~~（✅ **2026-07-18 收檔,徹夜 3 批+2 顯微鏡包**）→ [round-33](../docs/log/round-33-anti-dynasty.md)
- **CNN 排序器制度化**（8.5 倍檢定→雙 rank 進鍵→O 臂 62%;尺度邊界定案）;**爬山三步過 0**
  （s2_18=兩線同破在望）;表型過濾常駐（40% 線→R34 錨組解）;rad↔lo 梯度不可兼得三確認;
  平王 +0.560（I 系）;工作模型 v48。

### ~~Round 31 — 王系凍結輪~~（✅ **2026-07-16 收檔,3 批**）→ [round-31](../docs/log/round-31-dynasty-freeze.md)
- **梯度變現不轉移中繼帶（bridge 判死）、鄰域變異轉移（L 同框帳 1→4→7）**;深水實體 oob_bad 3.10;
  **SM 利用率元年**（std LCB 進鍵+rad 鍵常駐+漏斗+誤差錨池）;殘閘=同框系 rad 全負∧wm 最後一哩。

### ~~Round 30 — SM 準度輪 2＋低側據點擴張~~（✅ **2026-07-16 收檔,3 批+X 臂**）→ [round-30](../docs/log/round-30-lowside-beachhead.md)
- ★ 可用帶外 **9.0=真天花板**（窮舉 6/6）;**中繼帶發現**（lo −4.2∧wm +0.11∧oob 7.77=破天花板
  實體差 rad 閘）;champ adv 47% 首過門檻;深淵據點判死;→R31 王系凍結。

### ~~Round 29 — G 臂主力輪~~（✅ **2026-07-16 收檔,3 批+公證;戰略換軸首輪**）→ [round-29](../docs/log/round-29-gradient-inversion.md)
- **SM 可信半徑=d≤25**（champ 連兩批 realized 三標=準度變現通道）;adversarial training 循環成立
  （ikpi +4.50/吹牛歸零/rad 頭 +0.621）;★ **低側 gap 區破冰**（8+ 筆 −8~−2,三路踏進）;
  **margin 王 o29b2_011 +0.56**（24hr 冷支連兩王）。

### ~~Round 28 — 塊內 rad 手術~~（✅ **2026-07-15 收檔,3 批+雙公證**）→ [round-28](../docs/log/round-28-inblock-rad-surgery.md)
- **塊內手術修不回 rad（三批確證）**;正資產=**承重圖方法論**（族通用 4/6,魔法塊/大畫布/次可加）;
  **雙紀錄:margin 王 s28b3_005_a024 +0.50（a024 冷支,加壓稅輪）＋帶內 0.61**;worker 三層死亡判定實戰全通。

### ~~Round 27 — 加厚雙主軸：網架×R26 延續~~（✅ **2026-07-14 收檔,2 批提前收**）→ [round-27](../docs/log/round-27-mesh-arch.md)
- **雙假設答畢:H2=低側住網布＋塊內塵可實心化;H1 否決=網布逐像素承載**;學長解結構定案;
  三顆半成品（全卡 rad）→R28 塊內手術;216 壞死插曲=容錯全鏈驗證。

### ~~Round 26 — 帶外前瞻復活驗證~~（✅ **2026-07-14 收檔,3 批+公證**）→ [round-26](../docs/log/round-26-oob-foresight.md)
- **主判準=續觀察帶**（oob ρ 三讀中位 +0.107 蹺蹺板）;三標率大回收;**D 判決過**（51.7 新低）;
  **F 收官=單軸可/雙軸敗→網架接棒**;雙穩態 +1（0.53→定 0.44 榜 2）;重驗屍 L2b 紅→R27 換分布。

### ~~Round 25 — 多樣性加碼~~（✅ **2026-07-14 收檔,3 批+公證**）→ [round-25](../docs/log/round-25-diversity-dial.md)
- **近期成本換遠期資產**（三標率↓、L2b 單位效率近翻倍 −0.71、帶外前緣 9.04、探索類 40%）;
  wm 前瞻預言 0/3 否證、**oob 前瞻 +0.577 首現顯著**→R26;帶內紀錄 **0.58**;I 臂 61% 質變;停滯協議首戰=非真停滯。

### ~~Round 24 — 降根計畫~~（✅ **2026-07-13 收檔,3 批硬上限首例**）→ [round-24](../docs/log/round-24-root-diversity.md)
- 根稅買到覆蓋（根系分散/非王朝三標榜首）,帶外因果未證（9.0=王朝樂透,轉 L2 跨輪長評）;
  雙紀錄=可用帶外 **9.0**+帶內 **+0.51**;margin 連三批無新高→升稅 0.6;D 學費滿待裁決。

### ~~Round 23 — 價值軸主戰~~（✅ **2026-07-13 收檔,四批+公證**）→ [round-23](../docs/log/round-23-selectivity-axis.md)
- sel 鍵增值成立(O>M 四批)+rad 頭進鍵;可用帶外逼近未穿 9.09=深血統打轉→R24;
  **一輪四紀錄**（margin 雙躍 +0.41→**+0.49 c18 奪回**/rad 王 +1.00/帶外王 8.61）=軸相關枯竭實證。

### ~~Round 22 — 分布組合批~~（✅ **2026-07-12 收檔,三批**）→ [round-22](../docs/log/round-22-distribution-portfolio.md)
- **一天三公證紀錄**（rad +0.89/帶外 8.61/帶內 +0.46）+死區告破;C 冷支=新主產線;S 槽鏈首批成立;
  Q/H 收臂=低側蓋棺;oob 鍵壽終→sel_score。

### ~~Round 21 — 收割管線~~（✅ **2026-07-12 收檔,五批 774 筆**）→ [round-21](../docs/log/round-21-harvest-pipeline.md)
- 量產成立（M 臂 17-27%）+帶外王 o1_035 8.65 公證;SM 帶外過濾紅利=一次性;馬太確診+探索稅止跌;
  rad 候選 0.89 單次→R22 公證;制度收穫=自癒/切片/tier-2 搶佔/gain 儀表/機器個性檔案。

### ~~Round 20 — 模型線終審~~（✅ **2026-07-11 收檔,Ricky 拍板 (a)**）→ [round-20](../docs/log/round-20-evolution-loop.md)
- ③帶外 GA 19:10 顯著優=SM 有效僅帶外;①②隨機優+GA 逐代衰退=分布收窄;F 碎片族三代 0/85 蓋棺。
- **雙王易主**:margin 王 r2_016 +0.39/帶外王 vg0338 8.84(c18 王朝,每代 1px);全能型 r3_001 +0.36。
- 新知:vg0258 雙穩態、HFSS 敵意血系;前瞻性驗證制度成立(看無偏臂)。

### ~~Round 19 — 模型線第一批~~（✅ **2026-07-10 收檔**）→ [round-19](../docs/log/round-19-model-line.md)
- **門檻未過(wm ρ 0.493<0.5)=飽和是本質**;帶外排序 0.603 可用;rad 王易主 cc_r9s2 +0.62;vg0338 帶外 8.84 單次。
- vargen 品質 ✓(全譜 −20.5~+0.4,各帶單調;26px+ 全滅=尖銳最優再證);**單次破紀錄待公證:vg0338_c18 帶外 8.84**。
- 進行中:SM v5 重錨(開發機,CLEAN_STORES+R15-R19a;**r19b 整夾=門檻 held-out**)→ ρ≥0.5 → R20 GA。
- 待補:218 重跑一筆 error `run --input dedust_r19b_input --store dedust_r19b`(只補 vg0795_g14,~3 分)。

### ~~R19 發車紀錄~~（原 🔵 條目收合）→ [round-19](../docs/log/round-19-model-line.md)
- Ricky:「基於王結構做組件級隨機 variation,各 400 不重複,為訓練輪作準備;diff 像素要合理分布」。
- 806 筆已備（vargen 800+公證 6）:九王錨點×算子鏈(1-3),**diff_px 五帶配額** 1-3px 15%/4-10 30%/
  11-25 30%/26-60 18%/61-120 7%;交錯分夾兩機同分布;雙夾查重 0;搭載 cc 三筆紀錄級公證(r19a)。
- **發車:37 `run --input dedust_r19a_input --store dedust_r19a`;218 `run --input dedust_r19b_input --store dedust_r19b`**
  （各 ≈13-15hr,過夜+半天）。收檔後=sm_reanchor v5 → held-out 排序門檻(ρ≥0.5) → GA(R20)。

### ~~Round 17 — 帶外主目標~~（✅ **2026-07-09 收檔 45/45**）→ [round-17](../docs/log/round-17-oob-primary.md)
- **換王 a024 +0.35(公證3/3)**;低側可動(hslot)但三標內不可負擔=張力;分組=集中≫分散;尺寸峰3×3;
  hslot=rad 大旋鈕(+1.7);i12/a017 卡線蓋棺(rad/wm −0.01)。

### ~~Round 18 — 帶外二批:挖礦落地~~（✅ **2026-07-09 收檔 34/34**）→ [round-18](../docs/log/round-18-oob-mining.md)
- **b20 假象(3/3 −0.19,鐵則第二次救命)**;vb43 9.24/x20 9.15 真(入帶外榜);低側救援 0/20=粉塵本體定案;
  c18_sm 9.04 紀錄守住;三標內帶外地板≈9.0=帶外戰役定案。

### ~~Round 15 收尾批 + Round 16~~（✅ **2026-07-09 全收檔**）→ [round-15](../docs/log/round-15-pushbutton-vs-toolbox.md)｜[round-16](../docs/log/round-16-addition-map.md)
- R15:換王 i02 +0.29 公證✓;push-button 至少打平=「空間即知識載體」。
- R16:單塊近全負(空間飽和),唯一正點 r9c11×3×3;配對正交互(貪心不夠);g14 rad 兇手=g3;
  翼修邊/再分配雙雙因果否決。

### ~~Round 14 — 組件級軸~~（✅ **2026-07-08 收檔**）→ [round-14](../docs/log/round-14-component-axis.md)
- 翼=帶內引擎(+6dB)且付 rad/帶外=張力機理;冠軍在尖銳最優;細旋鈕=小塊;像素級退役。兩台現空。

### ~~Round 12 — 收斂 × 破單一化~~（✅ **2026-07-08 收檔**）→ [round-12](../docs/log/round-12-consolidate-diversify.md)
- crown 8 候選全公證;新王 c25 +0.22;family2 否決第二山頭=w17 特殊性確立;穩健王 c21。

### ~~Round 11 — 冠軍公差穩健化 × 規則普適性~~（✅ **2026-07-08 收檔,五批**）→ [round-11](../docs/log/round-11-robustness.md)
- occl2 @37 ✅ 48/48（規則普適性過關:底排承重跨家族 ρ+0.53~0.72,低成本區/rad 旋鈕重現）。
- tol @218 ✅ 60/60（整面蝕刻全滅;局部缺陷存活=margin 函數:c21 10/18 穩健王/a15 2/18/w17 1/18）。
- ref3 @37 ✅ 159/159（**三標過 27 筆;新王候選 c25 +0.22/+0.34=5 塊翼對,組數階梯大成功**;
  SM 帶外排序 ρ+0.21 偏弱）。
- probes @37 🔵（56 筆 ≈2.8hr:c25 公證 6+全對稱冠軍 8+搭橋 6+t07 構造化 4+底緣精修 32）;
  重啟:`run --input dedust_probes_input --store dedust_probes`。
- wide @218 過夜 🔵（160 筆,Ricky 提議:W 遠距 k48-128 高原半徑 64+X 對稱必要性（不再對稱化）48+Y SM 遠距導引 48）。
- 重啟指令：37 `run --input dedust_ref3_input --store dedust_ref3`;218 `run --input dedust_tol_input --store dedust_tol; run --input dedust_wide_input --store dedust_wide`（分號串接,tol 收完自動接 wide）。

### ~~Round 10 — 精修 × 物理歸因~~（✅ **2026-07-07 收檔,八冠軍 certified**）→ [round-10](../docs/log/round-10-refine-attribution.md)｜報告 [round-10-report](../docs/log/round-10-report.md)｜名鑑 [champions](../docs/champions.md)
- **★ w17 公證後修正（2026-07-06 晚）**：十次公證 8/8 = **wm −0.06**（原單次 +0.48 為 Gain context 個案;S11 +0.83✓ rad +0.26✓）→「三標全過」收回,w17=可製造新紀錄（−0.29→−0.06,差全過 0.06）。**新規則:紀錄級結論一律公證後才算數**;g24 的 rad+0.44 也是單次、待公證。X 臂 4/4 規則、承重圖（已補成 48/48）、SM 重錨 1.41 不變。
- **進行中（2026-07-06 傍晚發）**：37 → **ref2 過夜**（`run --input dedust_ref2_input --store dedust_ref2`,122 筆 ≈6hr：A w17 密掃 48/B 承重圖知情編輯 36/C 重錨 SM 導引 32/D y05 線 6,目標=帶緣餘裕推高+第二冠軍）；218 → **雜項鏈**（ref1 補 3 error → `dedust_w17rep` w17 十次公證 → occl 補 5 error,共 18 筆 ≈1hr）。
- **收檔後待辦**：ref2 判讀（任何紀錄級候選→先公證再宣稱;B vs A=承重圖知情是否贏盲掃）；g24 公證＋w17 在 37 補公證（確認 −0.06 跨機）；round-10 §5 結論＋README 索引；規則→generator（R11）。
- ⚠ 監看掛在開發機 Claude session,session 沒了就沒監看——進度隨時可用 `python -m script.dedust report --input dedust_ref2_input --store dedust_ref2`（其餘 store 同理）查 NAS 真相。

### ~~Round 5 — 滑動視窗 SM 訓練量~~（✅ **2026-07-10 收 E 臂,線上線收束**）→ [round-05](../docs/log/round-05-window-sm.md)
- gap 1.24=訓練量修好;best −3.65@401ep 未破紀錄、trust 未解鎖=瓶頸在泛化/搜尋。**216 釋出→掛 worker（工廠第三機）**。

### ~~Round 9 — 池頂端重驗＋乾淨前緣探索~~（✅ **2026-07-06 晨收檔 159/162**）→ [docs/log/round-09](../docs/log/round-09-pool-revalidation.md)
- **oracle 活著（8/18,t00 +0.44）**；漂移家族依賴（頂帶 ±0.4 可信,fit −0.26+1.13x σ0.77）；**可製造紀錄 −2.68 → −0.29**（s05=F2×10-5-10 對稱化,S11 已過差 Gain 0.29；g24 rad 已過差 wm）；F3=可製造沃土（top-10 佔 7）；SM 分布外+4.3 樂觀但排序有訊號（G 贏 E 2.4dB）→ 批次 guided loop 成立。
- **2026-07-06 補完**：error 3 筆補收=162/162；**重複性公證**（s05×41 次,37+218 兩機）=模擬雜訊地板≈0、
  跨機 bit 級一致（詳見 round-09 §4 附錄）→ 37/218 均已釋出待派工。

### ~~Round 8 — 乾淨子空間測繪~~（✅ **2026-07-05 收檔 97/97**，斷電中斷一次續跑收完）
- 判讀完整版 → **[round-08-report](../docs/log/round-08-report.md)**（附圖）：A 崩（除塵 |Δ| 中位 1.17、通則不成立）/ B 敗（補洞非因果,rad 四筆全負）/ C 半亮（SM 池內 1.5-2.4dB、池外 4-5.5）/ D 實錘（uniform 輸池抽樣 ~5dB）＋ ⚠ 池值漂移警訊 → 催生 R9。正式歸檔（README 索引/§5 結論）待 R9 一起。

---

## 🔜 候選 / 待排
- **P1b 鏡像對噪音儀器（~10 筆 S0）**:非對稱合格解 Y 鏡像對重測——底板對稱 ⇒ 理論恆等,
  實測差=純數值抖動(連 mesh 手性都測,比 repeat 嚴格);順帶驗證「SM 鏡像資料增強」合法性(36,990→~74k)。
  觸發:R54 戰役收檔後的批間閒時。→ round-55 §1 押注/analysis-11。
- ~~HFSS 自適應網格收斂性實驗~~（✅ 已執行,R52 定讞 → [round-52](../docs/log/round-52-anchor-assault.md)）
- ~~R24 降根計畫（預開檔）~~（✅ 已執行並收輪 2026-07-13 → [round-24](../docs/log/round-24-root-diversity.md)）
- ~~sm_denovo 萃取路徑~~（✅ 2026-07-13 落地=`sm_reanchor train-denovo`;sm_denovo1 已訓、D 臂 b2 復航）
- ~~圖 4-4「骨架+網布」塊級 variation~~（✅ **確定進 R27**——Ricky 2026-07-14:「27 可以做厚一點,
  網架和 26 的延續」=加厚授權〔3 批上限本輪放寬〕＋雙主軸定案;N 網架臂已實裝
  （select-r27 --mesh 24,四式變體,t07 目檢過,commit f080c38）;R26b3 判讀完即開輪）。
- **de novo 先導臂 ≤15（2026-07-12 帶外側拆解討論產物）**：池外隨機構造＋可製造閘＋sel_score/rad 頭
  預篩——R8 uniform/R15 理論模板兩次否證都是舊感知時代,新濾網值得再賭一次「池=唯一沃土」。
  **觸發=R23 可用帶外連三批零推進的回檢選項之一**（與 d1 窮舉並列）。
- **d1 殼層窮舉（2026-07-12 期望模型討論產物）**：紀錄鏈 c18→vg0338→r2_016→r3_001 全是 1px 子代
  ＝紀錄來自王鄰域局部滲透而非樂透尾巴 → 直接窮舉 r3_001 可製造單翻殼層（~150-250 筆,1-2 批量）,
  把「破紀錄機率」降維成「有限驗證」:找到新王或證明局部最優。**觸發=R22 b1 收檔後,或 M 臂連兩批 best<+0.35**。
- **[使用者] 模型線接棒帶外戰役（GA/線上學習＋資料回灌,2026-07-09）**：構造法乾涸就換模型抓耦合——
  ①SM v5 重錨（CLEAN_STORES 補 R17/R18 手術+池頂族 ~80 筆,驗新區域 oob/lo 排序）→②rad head 盤點
  （張力拓撲 (wm+lo)↔rad,模型線最缺 rad 預測器）→③GA over 組件-算子空間（非像素;fitness=字典序）。
  **觸發=R17/R18 收檔判死低側 or 構造法連兩輪無紀錄**。詳見 scratch 2026-07-09 塊。
- **理論模板探針（局部最佳討論產物,2026-07-09）**：教科書 28GHz 貼片離散化當錨點（唯一不從學長池衍生的
  種子來源）套構造化管線,~10-15 筆;觸發=R15 收檔後有空機。詳見 scratch 2026-07-09 塊。
- ~~元件消融 / 像素級→組件級~~ ✅ 升 R14（上方 🔵;resize_component 已實作）。
- ~~probes＋帶外批~~ ✅ 已發車（c25 公證臂併入;SM 帶外訊號弱=P4 未上預篩）。
- **🔜 R12 已備妥（2026-07-08,R11 收檔後接跑）→ [round-12](../docs/log/round-12-consolidate-diversify.md)**：
  收斂線 crown@37（8 top 候選公證+缺陷穩健,48 筆）＋破單一化線 family2@218（非 w17 家族深掘,45 筆）;
  判準寫死於 round-12;兩批已生 NAS＋查重過。指令見 round-12 §3。
- **[使用者] 組數階梯探索（3→5→7 塊）**：多塊=多共振器=選擇性潛力;ref3 先掛 add_block 先導臂
  （承重圖低成本區放鏡射塊對）,有訊號升 R12 系統對比。詳見 scratch 2026-07-07 塊。（**[使用者] = 你提的**；看 benchmark + Round 2 結果再決定優先序）
- ~~[R7] R7.5 乾淨前緣重驗~~ → **併入 Round 8 A 臂**（見上方 🔜 R8 區塊/[round-08](../docs/log/round-08-clean-mapping.md)）。
- **[R7] 乾淨子空間 warm-start 精修 round**：起點= p03_d3（可製造最佳已知,-2.68/rad+0.24）+ R8 A 臂產物；線上學習回鍋當精修器（差距 ~2.7dB 正在 analysis-01 局部射程內）,候選生成端掛 `strip_small` 無粉塵修復＋B 臂驗過的編輯算子。**觸發：R8 收檔＋SM 重錨完成**。
- ~~[R7] SM 乾淨區重錨~~ → **✅ 2026-07-06 完成**（`script/sm_reanchor.py`,held-out 3.20→**1.41** 進 2dB 帶、無遺忘;權重 `sm_reanchor.pth`）＝R8 C 臂判準結案＋「週期 harvest 重錨」第一次落地。詳見 round-08 §4。
- **[analysis-01] 去洞/平滑先驗「服務 Gain」**：analysis-01 實錘 S11/Gain 結構配方不同（S11←少組+feed連通、Gain←少洞、共同敵人=細碎）→ 現有 `island_suppression`/`tv` loss 剛好對應「去洞/平滑」,但從未以 Gain 視角調權重；sigmoid 只修 S11 側結構＝R3-D Gain 卡住之謎的解。**觸發條件：R5 收檔判讀時一起看**（若 Gain 側仍是 worst_margin 瓶頸即試）。動 loss 權重前依規矩討論。詳見 [docs/log/analysis-01](../docs/log/analysis-01-pattern-anatomy.md) §3。
- **[討論] 選擇端 known-bad 鄰域懲罰（治 R4 E ping-pong）**：acquisition 罰「採過且證實爛」的鄰域；SM 續走 elite-only（CartPole 論點：只學好的保地形、盲區問題在選擇端解）。**觸發條件：R4 結束時 trust_t 未升離 0.05 且 ping-pong（flips 雙峰）未消**；若 trust 升了它自癒、本條作廢。analysis-01 佐證：~300 翻轉的跨區跳落在不相關區＝重抽。詳見 `docs/discuss/scratch.md`「ping-pong」塊。
- **[使用者] DIP + 探索 → 已成 Round 3（config ready）**：E(lr↑)/D(sigmoid DIP)/E+D factorial,見上方 Round 3 區塊與 [docs/log/round-03](../docs/log/round-03-explore-dip.md)。待 Round 2 判讀完後發。
  - direct-only 探索子臂（UCB `selection.uncertainty_weight`↑ / diversity↑）留待 Round 3 之後（候選式旋鈕、sigmoid 用不了,不進本輪 factorial）。
- ~~**[使用者] val-早停**~~ → **已成 Round 4**（`mode:adaptive`）：用「下一個 held-out HFSS 點」評 member0 快照、自調每輪 SM 重訓 epoch 數。見上方 Round 4 區塊與 [docs/log/round-04](../docs/log/round-04-adaptive-sm.md)。
- **[使用者] 可解釋性 / SM 歸因（AlphaFold-like）**：用 SM 做屬性分析，找「哪些像素對好 pattern 貢獻最大」→ 當設計先驗/引導。先記錄、之後測。
- **[使用者] 把「對稱」做對（下一次想試）**：現行硬 mirror（`MirrorGenerator`，**12-1-12** = 對中央 1 欄做完整左右鏡射）表現普通、可能太死。試**部分對稱**：例如 **10-5-10**（外側 10 欄左右對稱 + **中央 5 欄自由**，給饋電/中央共振區自由度），或改成**軟對稱 loss**（鼓勵而非硬鎖）。做之前先定哪種（generator 結構切法 vs loss）+ 中央自由帶寬度。動 loss 前依規矩討論。
- **[R6 分析] harvest 池頂端 warm-start（候選/初始 pattern，不只 SM）**：R6 實錘達標 pattern 已在池內（oracle **+0.38**、池內 18 筆 ≥0）且「分布≫策略」（池抽樣等效預算領先 200-450×）→ 讓搜尋**從池內 top 樣本出發/混入候選**，而非從頭找。**觸發條件：R5 收檔後討論排程**。詳見 [docs/log/round-06](../docs/log/round-06-offline-expected-best.md) §5。
- **[使用者] 週期 harvest 重錨（更極致 refit）**：把過往好樣本（含 harvest）週期性整批重訓 SM，讓資料越跑越多、暖啟動越來越好（現在 run 的資料不回灌中央池，這條補那塊）。R6 分析 +1 佐證（池=最有價值資產）。
- **[使用者] 結構性先驗 → 走架構、不走 loss（主題）**：**連通** 和 **對稱** 是同一類——都是 pattern 的**結構性先驗**,適合用 **generator 架構(DIP)** 內建,而不是靠 loss 硬拉。
  - **連通**：不動 `sc loss`（**已驗證有效**）;連通交給 **DIP**（sigmoid 架構天生連通,r_feed 0.62 vs direct 0.2）→ **Round 3 D 正在測**。
  - **對稱**：10-5-10 部分對稱（見上方對稱候選）——同樣走 generator 結構切法。
  - 洞見：pattern 的結構約束（連通/對稱）架構做比 loss 做乾淨、不跟主目標搶梯度。
- **[我/發現] loss 對齊 worst_margin**：sim_loss 最低 ≠ 天線最好（Round 1 發現）；潛力大但動 loss 前討論。
- **[使用者] rad 塑形 = 弱推力（走 a；設計已定 2026-06-30）**：radiation 透過 SM rad 預測影響 pattern（beam loss 算在預測上、反傳到 logits；絕對增益歸 Gain target）。**實測現有 head 窗內 ±45° ~3.5dB**（形狀歪、非高度偏 → 改吐相對形狀沒用；是**凍 trunk 容量限制、非 n_basis**）≈ 3dB 門檻 → 不夠精確驅動 3dB 覆蓋。**(a) 走弱推力**：覆蓋項改 **worst-angle（soft-min ±45°，對齊 worst_margin）** + **低權重 nudge** + 課程化（S11/Gain OK 後升）+ rad 收尾（實際 `sm_min_loss`、`n_basis`=8）。**(b) 容量投資**（週期解凍 trunk⚠NaN / 物理 FFT）延到 radiation 變主角。詳見 [[project_radiation_pattern]]。動 loss 前討論。

---

## ✅ 已歸檔（一行指標，完整結論在 round 檔）

- **Round 07 — 除塵驗證** → [docs/log/round-07](../docs/log/round-07-dedust.md)：**粉塵=共振的一部分（4/5 崩 -4.7~-16.9dB）→ 乾淨可製造解要用搜的、不能用修的**；例外 p03 整塊型近零代價＝可製造最佳已知點（wm -2.68、rad +0.24）；R6 oracle 重驗真（p00 +0.44 達標）；rad ±45°=獨立第三關（p00 rad -2.71）且與可製造同向；SM 乾淨區低估 5-15dB；rad 15 條入袋（Stage-3 解鎖）。批次驗證實測 **3 分/筆**。2026-07-03 當日完成。

- **Round 06 — 離線期望基準（零 HFSS）** → [docs/log/round-06](../docs/log/round-06-offline-expected-best.md)：**期望爬升到不了 spec**（fit -9.18+0.75·ln k、躍遷主導 46%）；**達標 pattern 已在 harvest 池（oracle +0.38）**；學長同預算贏 1-2dB（KM 500 輪內達標 6% vs 我們 0%）；**分布≫策略** → 池頂端 warm-start 升候選。工具 `script/expected_best.py`（每 round 收檔可重跑疊圖）。圖 `docs/log/assets/round-06/`。2026-07-03 當日完成。
- **Round 04 — 自適應 SM 訓練量** → [docs/log/round-04](../docs/log/round-04-adaptive-sm.md)：**E+D 破專案紀錄 -2.89@154**（探索躍遷,+2.80 vs R3）；主假設未驗證（探測自鎖 3-5ep、fit_loss 仍 8-11、trust 全鎖;E/D 輸 R3 ~0.9dB）→ R5 滑動視窗。2026-07-03 停（E@208/D@222/E+D@201）。圖 `docs/log/assets/round-04/`。
- **Round 03 — 探索 × DIP factorial** → [docs/log/round-03](../docs/log/round-03-explore-dip.md)：**E(lr↑)最佳 -3.63@89（¼ epoch 追平②）**;DIP 連通成功(r_feed~0.95)但停滯(best@8);三臂被 SM 欠訓汙染、factorial 不乾淨 → R4 修瓶頸重跑。2026-07-02 停(E@189/D@101/E+D@132)。圖 `docs/log/assets/round-03/`。
- **Round 01 — SM 訓練量 A/B** → [docs/log/round-01](../docs/log/round-01-sm-training-ab.md)：**訓練量非 bottleneck**(dlf −4.18≈refit −4.21 > dlf_fit −5.58、皆差 spec ~4dB)。圖 `docs/log/assets/round-01/`。
- **Round 02 — ensemble + trust 治本** → [docs/log/round-02](../docs/log/round-02-ensemble-trust.md)：**治本微幅、未決定性**(②③ trust 微贏 Round-1 ~0.3-0.5dB、① ens-only 輸、皆未收斂;trust_t 卡低)。2026-07-01 提早停(未到 500)釋放機器給 Round 3;② ~417ep 當 Round-3 reference。
- 🔜 **雲內爬升（tier2,速度紅利④）**:worker 閒時自產改「SM 引導虛擬爬山→只測 top」——動 worker 端;觸發=R48 嫁接試點收檔後/或 Ricky 點名。
