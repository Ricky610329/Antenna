# R80 操作：金屬對稱資料工廠與獨立 worker

目前先執行 R80 金屬左右對稱量測；使用者已啟動worker，並於2026-10-07授權每30分鐘監看，完成對稱任務後接續spec調整與R81驗證。R81工程檢查及正式批次待R80完成才派送。最新量測狀態以[操作板](../../configs/ONGOING.md)為準。

scope 固定為 `symmetry_filter_20261007`，worker 固定使用 `--selfgen 0 --poll 60 --stale 120`。目前使用的 `start_scoped_worker.ps1` 只接受 R80，會拒絕 scope 內任何 dual/filter job；後續 R81 必須用獨立 dataset 與下述新入口。

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

新版scoped worker對「本批每筆皆已有終態、部分成功經raw/sample/rad重播有效、殘留僅用盡重試的HFSS錯誤」保留`.fail`／claim／原結果後，跳過本機失敗批並續跑其他jobs；別台仍可依原機器名單補測。连续三批部分失敗、整批失敗、資料／profile驗證故障及原嚴重連敗保險絲仍停本機，`--once`也不把部分失敗回報成成功。狀態中的`worker_continues`僅是失敗當下的續跑決定，需要另查後續進展；不是活機心跳。此修正要在正式機pull後重啟worker才生效，不能熱套用在原行程。

通過上述部分成功重播後，僅清理該job預設暫存`_dedust_<store>`：先確認resolved路徑是目前repo工作目錄的直接子目錄且名稱完全符合；自訂`--out`保留，其他歷史暫存不掃。HFSS已關閉後才清理；路徑不符或清理失敗會停本機並保留NAS結果，避免容錯續跑反而累積專案吃滿磁碟。

## 本機備料與 controller

本次預設備料只做 single phase，只讀歷史 `dedust_r55sym_input` 的 34 個幾何 seeds：

```powershell
$env:OMP_NUM_THREADS='4'
python -m script.prepare_symmetry_filter prepare `
  --history-root C:/Users/ricky/antenna_nas_backup/dataset `
  --out tmp/r80_symmetry_20261007_single
```

預設 `--phase single`；目前不使用 `--phase combined`。它會混合 R80/R81 engineering jobs，不作後續 R81 的 worker 交付方式。single controller 位於 `tmp/r80_symmetry_20261007_single/single`；NAS初始備份位於 `controller_initial/single`，已納入逐檔部署比對。

備料輸出不得覆蓋；若需重建，使用新的輸出名稱並保留舊證據。歷史 response 不匯入為新真值，橋寬維持 0.1 mm。

## 原 pilot 回填與目前工廠更新

首批完成後，保留完整共享 dataset，不可只帶 `results.json`。在開發機以 local controller 回填：

```powershell
python -m script.exploration feedback --work-dir tmp/r80_symmetry_20261007_single/single --batch 1 --dataset-root <私人 dataset 完整路徑> --result-manifest <私人 dataset 完整路徑>/dedust_r80b1/results.json
python -m script.exploration train --work-dir tmp/r80_symmetry_20261007_single/single --through-batch 1
```

以上指令僅回填原pilot controller。目前5,000筆工廠使用獨立`tmp/r80_factory_20261007/training_v2`及`symmetry_training.py`，每48筆有效唯一新結果觸發更新（單次最多96）；`snapshot_successes`可凍結長批已成功的子集合，不必等整個長批跑完。原store與送測前預測均保留，低性能有效資料也入訓練。新policy見`configs/single_r80_symmetry_factory.yaml`及R80最新追加紀錄。R81接續對稱階段，先通過工程檢查再擴大量測。

## 開發機的30分鐘controller

三台正式機只需執行原worker；SM/controller僅在開發機啟動一份。`symmetry_factory_cycle --once`先產生具名、hash綁定的action receipt，`--commit-receipt`再逐片查重及加入私人佇列。一般guided小批為16筆，接近總額時允許明記的最後不足16筆尾批。待跑guided最多96，原48長批及claim不改寫。新資料先稽核保存的前瞻曲線再訓練；沒有預測的初始盲選資料明示缺失。

常駐入口讀取本機已準備的設定，設定內明確指定私人dataset、factory profile、seed快照、blind預備池、`training_v2`及`expected_retry_workers`。2026-10-08的設定版本`watch_settings_retry_v1.json`沿用原工作區，增加固定名單`["140.123.106.216", "140.123.106.218", "140.123.106.37"]`；原設定保留作歷史證據。設定不得從目前少數claims推估名單：

```powershell
$env:OMP_NUM_THREADS='4'
$env:PYTHONUTF8='1'
python -m script.symmetry_factory_watch --settings tmp/r80_factory_20261007/watch_settings_retry_v1.json
```

此指令為開發機恢复用，**不要在三台worker上另開controller**，也不要重新初始化已訓練的protocol。每1800秒按起始節奏檢查；長週期錯過時跳過過期tick，不立即連跑。模型訓練只用本機CPU，HFSS照原佇列運作。controller與NAS派送均有kernel lock，中斷後鎖自動釋放，原資料/收據保留；重啟另記attempt，不抹掉之前錯誤。

狀態位於`tmp/r80_factory_20261007/controller/watch_status.json`。該資料夾的`STOP`只在週期邊界停止controller，不會殺HFSS；NAS全域/本scope STOP也會阻止新的prepare/dispatch。正常worker寫入`results.json`造成快照競態時記deferred receipt，下個30分鐘重試；profile/hash損壞仍停止並保留診斷。使用者已將唯一真值目標降為5,000；達額後只標記對稱資料蒐集完成，R81仍須依工程檢查流程接續，不會由此腳本直接啟動。目標不是16的倍數，既有最後尾批可少於16筆，不以補整批灌成5,008。

`.fail`只代表已列名機台失敗，別台仍能接手；因此保留尚未成功圖形的名額，直到fail名單確實涵蓋全部指定worker才釋放，並永久排除原失敗圖形以補派不同圖形。未知／缺失／格式錯誤的名單不當作全機終態。掃描期間claim／done／fail有變動會延後重試，dispatch重新核對容量；只有有效唯一數精確等於5,000且無剩餘預留才標記蒐集完成，超額會報錯。新增機台或改名單前須在controller週期邊界停止、保存舊state，再以新的設定版本重啟一份；不重建protocol或修改live worker結果。

使用者2026-10-07最新監看指示：平時讓背景流程掛著，HFSS依私人佇列工作、SM依設定的新資料批次慢慢更新。conductor每30分鐘只做健康檢查，確認原controller狀態／最近週期是否完成、佇列有工作及worker近期進展；一切正常且沒有需處理的結果時，不持續主動喚醒、不逐小批反覆重播預測／全量hash，也不送例行進度。出錯、停滯、研究結果需驗證或到達階段邊界才介入。controller原有snapshot、模型批次更新與派工收據仍自動保存，不因減少人工監看而停用。

## 性能停滯與工作停滯

工作停滯包含worker無近期成功結果、佇列未補上、controller失敗與SM訓練／恢復故障，由conductor自主診斷及修復；保留錯誤與恢復證據，不把工作故障解讀為pattern性能天花板。

性能停滯由conductor在SM迭代有足夠實測回填時按既有`stall-protocol`分軸判讀，通知使用者。只用送測前綁定的當前profile `data-vNNN`版本及HFSS真值：S11/Gain全帶margin、兩者最小值及實測場型；初始盲選／v108、重測、未綁定或後補預測不算成熟SM版本。SM保留集MAE、候選預測、資料數及吞吐都不是性能進步。

預設提醒窗口為三個連續成熟SM版本、合計至少96個有效唯一真值；每版至少24筆，已派圖形均terminal且有效率≥90%，窗口前至少24筆同profile基準。以0.30 dB的S11/Gain實質解析度及0.50 dB的場型解析度檢查最小margin與二／三維前緣；有指標或前緣推進就分列報告，不能只因單一最小margin持平判停滯。詳細判讀政策與不足證據規則保存於私人交接證據；這是conductor的advisory判讀，不是controller已接入的自動通知器。

確認性能停滯時回報版本／真值數、前後最佳值、前緣與分臂／家族進展，且保留資料繼續收集，不擅自停HFSS、改觀測spec或改物理搜索範圍；有效低分資料仍入學習池。未成熟或故障版本先標不足證據並處理工作問題，不發性能停滯通知。

## 凍結資料的對稱與頻率統計

先選定 `snapshot_successes` 的完整凍結副本；不直接分析仍在寫入的原始 store，也不以性能篩掉低分有效資料。下列入口重播具名 profile、實體橋接/金屬鏡射與 sample/rad hashes，依量測身分＋實體圖形去重，優先保留非重測項目：

```powershell
python -m script.symmetry_analysis profile --store tmp/r80_factory_20261007/first_fit_snapshot --out-json <新的 analysis.json> --out-npz <新的 curves.npz>
python -m script.figs.symmetry_profile --analysis-json <analysis.json> --data-npz <curves.npz> --out-dir <新的圖表目錄>
```

JSON 保存來源、原始觀測與分位數；hash 綁定的 NPZ 保存對齊的完整頻率／181角度曲線。新入口不改舊歷史 census CLI，輸出拒絕覆蓋。地形圖採固定金屬面積比例與上下金屬差，六角格只取有樣本格的中位數、不插值；精確左右鏡射的左右差恆為0，故不用它作橫軸。頻率圖的25–75%是資料分布，不是信賴區間。selection arm只表示來源，不能視作隨機因果對照。

首版49筆的完整分析及圖已保存私人 `experiments/r80_symmetry_20261007/analysis_versions/data-v001`；來源 raw snapshot 仍在 `sm_versions/data-v001`。這只是早期描述統計，不表示全部5,000筆已完成或對稱改善性能。

### 5,000筆收尾：重測資料另留，不能只凍結unique

`snapshot_successes`與`_snapshot_successes_from_proofs`每個pattern只保留一個代表，優先非重測；將全5,000個hash傳入也不會保留其餘repeat的sample/rad。最後穩定cutoff須保存完整`pair_proofs`、各store的results cutoff及source hashes，先建立5,000 unique代表的凍結store，再將每個repeat源store**分開**凍結。每store內pattern須先確認唯一；若同store有同hash多次觀測，不可靜默去重，也不可裁切proof後沿用原metadata hashes冒充完整cutoff，應沿用觀測級raw-freeze/readout保留。

主分析依序指定unique store及各repeat snapshot；既有profile入口會把repeat保留在`duplicate_rows`的audit，NPZ、分位數與人口圖仍只計unique代表。重測量化則對各repeat snapshot單獨使用相同profile入口，再按pattern hash與主分析代表比較S11/Gain、場型曲線與scalar差值，明列原件／repeat來源。**`duplicate_rows`本身不是已計算的repeat residual，場型mirror residual也不是重測誤差。**

```powershell
python -m script.symmetry_analysis profile --store <full_unique_frozen> --store <repeat1_frozen> --store <repeat2_frozen> --store <best_repeat_frozen> --out-json <新combined.json> --out-npz <新combined.npz>
python -m script.symmetry_analysis profile --store <某個repeat_frozen> --out-json <新repeat.json> --out-npz <新repeat.npz>
```

10/10已用既有最佳原件＋獨立repeat的本機凍結資料驗證這個分層呼叫：combined為2觀測／1unique／1duplicate，repeat單獨分析保留1觀測，來源hash不變，無NAS／HFSS／訓練；[證據與限制](assets/r80_repeat_closeout_reuse_20261010.json)。實際小例是schema-v2原件＋schema-v1 repeat；完整单pair proof的schema-v2 repeat路徑另經只讀source核對，尚非5,000正式凍結／全repeat驗證。15:59保存cutoff的三個repeat stores分別2／2／1觀測，store內hash皆唯一；final cutoff仍須重新核對。所有輸出用新路徑、不覆蓋原資料，原19個live runtime來源不改。

## R81 正 WM 候選的確認入口（待後續濾波器階段使用）

2026-10-11新增研究偏好：20–26／30–36 GHz過渡帶仍希望相對下壓、往阻帶平緩衰減並減少波紋／突起，作軟性優化與曲線比較，**不參與WM**。門檻與正WM確認仍用原五段；WM可比時可依具名、保存來源的過渡響應偏好選候選／親代。五margin SM目前沒有此軟目標輸出，不能把未接入的偏好說成已訓練；後續R81選樣須明確記錄如何利用實測過渡曲線。[最新解釋與限制](round-81-wide-filter.md)。

R81派工前先執行`python -m script.prepare_symmetry_filter check-wide-inputs --dataset-root <R81備妥dataset>`：只驗證正式60筆與6+2+2工程輸入、來源組、親代及Fast／Discrete／mesh設定，不啟動HFSS。完成量測後才用同模組`check-wide`重播完整觀測並比較四組曲線／margin，任一差異>0.3 dB不放行。

`python -m script.filter_confirmation --profile-config configs/dual_r81_wide_filter.yaml --candidate-store <原件store> --candidate-id <原件id> --repeat-store <重測store> --repeat-id <重測id> --out <來源store以外的新收據.json>`

唯讀重播完整3×49量測及五段margin，核對原件與repeatability重測的parent、圖形、profile和hash。只有兩筆HFSS實測WM均嚴格>0才輸出`performance_target_confirmed=true`；WM=0仍可是通用評分的等號合格，但不達本次正餘裕目標。輸出不覆寫既有檔，也不寫入來源store。相同response bytes可來自兩次獨立觀測，來源獨立性依可信worker/snapshot metadata判定。此入口固定保留`campaign_complete=false`：smoke、Discrete及0.3 dB網格一致性仍須另驗證。目前沒有R81真值，不能把生成fixture測試當作達標。

## R81 歷史先驗與批次 SM 更新（可選入口）

沿用`script.exploration prepare → feedback → train → select-batch`，不新增controller。明示設定`exploration.training_protocol: filter_masked_prior_v1`、正的`pretrain_epochs`及介於0與1的`holdout_fraction`，並提供`exploration.filter_prior`三個欄位：`manifest`與`sample_root`必須是絕對路徑，`expected_manifest_sha256`固定歷史清單hash。prepare先完整驗證歷史樣本再建立工作區；禁止同時使用舊`old_pretrain`。

每批先保存選樣時的前瞻稽核，再以累積新版3×49真值訓練五段margin模型。歷史3×17資料只監督三段完整覆蓋margin，兩新阻帶不計loss；原曲線保留。每次重新由歷史預訓練接續當前真值，兩階段共用只由當前train擬合的輸入正規化。固定親代／canonical家族／相同圖形的相連群組，保留集永不回流；完整排除其歷史別名。epoch恢復保存模型、Adam、RNG與正規化。保留集只記誤差，沒有early stopping或模型挑選。

完成回填後用`python -m script.exploration train --work-dir <工作區> --through-batch <N>`，再用`select-batch --work-dir <工作區> --batch <N+1>`產生候選。回填收據綁定選樣manifest、各批完整ID集合與sample hashes；後續訓練核對分割帳本，選樣核對模型所屬批次、成員及hash。改資料、少資料或誤放其他批次模型會拒絕。生成候選仍須通過既有查重與工程檢查才能派HFSS。

生成CPU fixture的整鏈與中斷恢復已驗證，最終完整回歸677 tests通過（415.30秒，無warnings），含golden；沒有R81 HFSS真值。實際歷史清單已擴充至21,034個唯一圖形，[完整備料收據](assets/r81_full_prior_20261007.json)保留來源hash及axis／worker重建限制。私人`experiments/r81_wide_filter_20261007/historical_prior/full_p01_db075_v001/prior_bundle.zip`含全部原曲線與來源metadata；使用時先核對外部`archive_manifest.json`與bundle hash，解開後以`manifest.json`及`raw_snapshot`作兩個絕對路徑，固定manifest SHA256 `8fcd05614e7851e9c0f7043c00540068df37380423ee4469db0afed90505c777`。21,034是當前保留集排除前的上限，不能當作每版實際先驗訓練筆數。

`dual_r81_wide_filter.yaml`仍未啟用此protocol；待正式設定凍結及新版真值回填後，才以current train正規化做masked預訓練。不要為了背景備料先用全歷史估計正規化而破壞兩階段共用座標。R80仍為唯一派工項目；濾波器完成條件仍是原件與重測均確認的新版實測WM > 0，不能用SM預測或舊wm_mfg替代。

歷史先驗含量測不一致：217個重複圖形對中11對的covered margin差>0.3 dB、4對>1，最大2.559473 dB；原因未確定。固定保留first而不挑高分、保留全部候選來源，屬帶噪聲先驗，不可當作新profile的真值或模型校準證據。

## 可證明範圍

完整回歸 **544 passed / 361.29 秒**（`OMP_NUM_THREADS=4`），golden 原始 bytes 不變；pyflakes 通過。worker 實作 `29a6834` 已包含於遠端 `GAN`。部署收據中的 `main` 為首次交付的歷史紀錄；現行更新與啟動一律使用 `GAN`。

程式測試、`-CheckOnly` 與部署稽核只證明工程邊界；實測數量由後續NAS樣本、場型和hash重播稽核確認，見操作板及R80日誌。三機已持續產出，但目前不宣稱金屬對稱已改善場型，或新SM已優於歷史模型。

## R81 獨立 worker 入口（2026-10-10已提交，尚未啟用）

`script/start_r81_worker.ps1` 與 `script/r81_worker_entry.py` 已提交到 GAN。現有三台 R80 worker 繼續用原入口；不要現在切換。R81只使用鄒穎麒私人 `experiments/r81_wide_filter_20261007` 內的獨立 dataset，不將 dual jobs 混入 R80 queue。[版本、審查與限制](assets/r81_worker_entry_readiness_20261010.json)。

啟用時由開發機交付實際 `Release` 檔、SHA256及 `Stage`，三台先pull GAN再執行新入口；目前尚無可執行 release 或 worker 指令。release會綁定Git版本、完整輸入、queue與R80最終證據：精確5,000唯一有效／零pending、已驗證的最終census及原始rows/proofs/producer。封存action/profile保留原producer_path，NAS位置另行綁hash；tracked census採repo-relative Git blob，各電腦clone路徑不同也可核對，不回寫原始證據。10/11使用者最新指定「到5000筆就開始做」：達5,000立即開始R81備料，R80重測、完整凍結稽核、統計與圖表同步完成，不將全部圖表完成額外設為工程release前提。HFSS仍以實際validated5,000／零pending／獨立dataset及release部署為必要交接證據；既有入口本來只驗證這些真值與交接條件，無需改其runtime。

先釋出`EngineeringOnly`的6 Fast＋2 Discrete＋2 mesh；正式60筆保留未排隊。四組實測曲線與五個margin的最大差異均≤0.3 dB後，才交付`Formal`。正式批次限三批、每批60：b1為history／specialist／random各20，b2/b3為best_min_margin／uncertainty／random各20，後批須綁定前批回填與SM模型。不預排三批，也不要求尚不存在的正WM重測才允許首批。

每個release immutable；前一版所有launcher回報all_success並退出後，才能更新queue prefix和交付下一版。單機fail仍可由固定216／218／37 roster接手；全roster耗盡才視為終態失敗。入口不pull、不排新job、不刪claims，只在每次native `--once`前重驗release。唯讀preflight與claim並非同一原子transaction，所以prefix交接必須由開發機串行完成。若找到正WM原件，再另排獨立重測；只有兩次實測WM均嚴格>0才算性能達標，三批完成或SM預測正值不算。

R81正式訓練配方已由`script.r81_prior_recipe`固定：以已解開的真實manifest／raw_snapshot絕對路徑產生新config，固定歷史清單hash、masked-prior30ep＋current100ep、holdout0.2；其他基礎量測／spec／runtime及三批設計不變。CLI只做config和位置/hash檢查，既有exploration prepare才會完整驗證先驗；每版還須排除當前holdout相連家族。尚未啟動R81模型。[實際config-only生成與限制](assets/r81_masked_prior_recipe_20261010.json)。
