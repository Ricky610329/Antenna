---
name: project_sm_training_redesign
description: SM 線上更新：單樣本擬到收斂=反模式→replay/DLF;但我們 dlf 只訓 elite 1 epoch=under-trained(2026-06-27 修:dlf_fit/refit 訓到 fit);B1 確認 sm_harvest 對得上現在 HFSS(MSE 1.56)
metadata: 
  node_type: memory
  type: project
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
---

正式機 profiling（`script/profile_training.py`，2026-06-20）找到瓶頸：每 epoch ≈278s，
HFSS 求解 61%（外部難加速）、**SM 方向圖擬合 24%（撞滿 20000 步、純浪費）**、SM S11/Gain 擬合 12%（暴衝到 6634 步）、SC 0%（之前誤以為重，其實 0.1s）。

**根因＝反模式**：`train_one_data`/`train_one_data_rad` 把「單一最新樣本擬到 loss<0.1、最多 20000 步」
→ catastrophic forgetting（為擬最新一筆洗掉舊知識）+ 速度大頭。rad 凍 trunk 單層 head 容量不夠、永遠擬不到 0.1 → 撞滿。

**文獻範式**（查證）：這套系統＝**Neural Adjoint / Tandem Network**（凍 forward surrogate 反傳到 generator）+ 線上 surrogate-assisted optimization。好做法：
- **Experience replay**（streaming-regression 版，PMC8720895）：定容 FIFO 緩衝、minibatch、**每次 1 epoch**（不擬到收斂）；prioritized 抽樣 ∝ exp(預測誤差²）。
- **Boundary loss**（NA 最大改進）：把 G 限制在 surrogate 可信流形內，防 G 鑽 SM 盲區。
- 學長舊 code 有雛形：`legacy/data.py` 的 `DataManager` + `dynamic_loss_filter`（DLF＝留低 loss 好樣本）+ rollback 用 `filter(upper=平均loss)` 重訓。現有 `online` SampleStore 的「< 平均 loss 才收」就是 DLF 精神，但**緩衝只在 rollback 用、每 epoch 仍走單樣本擬到死**。

**Phase 1（已做，opt-in、golden 零漂移）**：
- `ReplayBuffer`（`antenna/utils/replay.py`，定容 FIFO in-memory + Dataset 介面）。
- `sm_train.mode: single|replay`（**預設 single ＝原樣**）；replay = 對最新跑 `newest_steps`(50) 步 + `train_by_datas(replay, epochs=1)` 一遍。緩衝收「好+壞都收」(SM 在 G 探索處都準)。
- `radiation.sm_max_epoch`（rad 擬合上限，`single_sc_rad.yaml` 設 1000 → 直接砍 rad 那 24%）。
- config：`single_sc_rad_replay.yaml`（mode replay A/B）。測試 `tests/test_replay.py`。

**Phase 2（已做，2026-06-21）＝移植學長論文 §3.5 的真 DLF**（agent 翻 docs/Paper.pdf 確認：DLF 是論文核心，獨立消融 §4.4 比 baseline 改善 >50%）：
- 機制＝**全收（不在寫入端篩）+ 每輪 SM 重訓用累計門檻 λ_t = mean(歷史 sim_loss, 含本筆) 重新過濾出「loss ≤ λ_t」菁英子集訓 SM**。門檻隨訓練自動收緊（前期多樣、後期精準）。
- **⚠ 關鍵發現：舊 `online` store「< 平均 loss 才收」＝論文點名要打倒的 baseline（[29] 寫入即保留 → 偽收斂：早期次佳樣本永久滯留稀釋資訊密度）。** replay/dlf 模式改走「全收」緩衝、品質篩移到取用端。
- `ReplayBuffer` 加 `loss` 欄 + `elite(threshold)→Subset`；`sm_train.mode: single|replay|dlf`（dlf 走菁英過濾）。config `single_sc_rad_dlf.yaml`。測試 `tests/test_replay.py`（6 測）。
- λ_t 的精確定義（ℋ 是每輪一筆還是所有樣本）論文在 Algorithm 3「圖」裡、文字沒給數學式；目前實作用「每輪 sim_loss 的累計平均」(合理近似)，要更忠於論文可再讀那張圖。

**Phase 3a（已做，2026-06-21）＝Boundary loss（trust-region）**：`antenna/losses.py` 的 `boundary_loss(pattern, seen)`＝懲罰「生成 pattern 與最近一個『已見』pattern 的均方距離」（已見＝replay 緩衝；`d>1e-9` 排除自己）。動機（NA 文獻最大改進）：G 靠下降凍住 SM 的預測 loss 優化，會鑽 SM 盲區產生「SM 說好、HFSS 說爛」的設計→每個白燒一次昂貴 HFSS 評估；boundary 拉 G 回 SM 已見鄰域→少浪費評估（＝加速）。NA 原版是連續參數 box、二元 pattern 改用最近鄰距離。config `loss.boundary`（預設 0＝關、golden 安全；需 replay/dlf 緩衝，否則 warn 停用）。`ReplayBuffer.patterns()` 取已見。config `single_sc_rad_boundary.yaml`。**權重待 A/B 調**（太大壓死探索、太小沒效）。測試 `tests/test_replay.py`（9 測）。

**Phase 3c（2026-06-27）＝修「under-trained DLF」+ refit + B1 驗證**：
- **關鍵修正**：我們的 `dlf` 模式每輪只 `train_by_datas(elite, epochs=1)`（1 epoch）＝ **under-trained DLF**；使用者確認**學長原版 DLF 是「訓到 fit（收斂）」**。釐清誤框：反模式是「**單一樣本**擬到收斂」（過擬合一點、忘掉其他）；把 **elite 集合** 訓到收斂是正常 batch 訓練、**不是反模式**——Phase 1-2 當初矯枉過正把 SM 訓殘,很可能是「輸 random」的一大主因（那些輸的 run 都用 `mode: dlf`）。
- **B1 spot-check**（`script/check_harvest_consistency.py`）：抽 15 筆 harvest 用「現在的」HFSS 重模擬比對,中位 MSE **1.56**（≪ val 13、遠離隨機 35-38）→ **sm_harvest 與現在 HFSS 對得上、暖啟動有效** → SM「來源」沒問題,「**訓練強度**」才是頭號嫌疑。
- 新模式：`sm_train.mode: dlf_fit`（elite 訓到 fit + **丟掉單筆 step** ＝學長原版）/ `refit`（訓**整個 buffer**、不挑 elite → SM 也學「爛 pattern 長怎樣」、避對抗洞;使用者「不一定要 elite」的版本）。`train_by_datas` 加 `min_loss` 早停（訓到 fit）+ NaN-epoch 跳過/空回傳防護（避免 `average('loss')` KeyError）。`sm_train.elite_epochs` 旋鈕。`sm_train.mode` 加「值」驗證（打錯字會靜默退 single 害 A/B 白跑→fail-fast）。
- **判準＝gap（對「新點」的準度,`gap_ema`/sm_target vs sim_loss）,不是 training loss**（後者可靠 memorize 壓到 0、不代表 generalize）。真正風險是 over-fit buffer（train loss 低但 gap 高）。
- 3 config A/B：`single_guided_{,dlffit_,refit_}harvest`（唯一變因 mode＝dlf/dlf_fit/refit）。**待正式機 A/B**（看 worst-margin vs HFSS-call + gap_ema）。⚠ 跑基準 A 前先刪舊 `pixel_single_guided_harvest` 結果夾（同名會續跑）。
- **更極致（待做）**：週期性把「harvest+online 全部」重訓（re-anchor 防遺忘;24k 每輪太貴→只能週期/rollback）。

**待做（Phase 3b+）**：prioritized 抽樣（依 SM 誤差；論文用硬篩 DLF 而非 PER 加權）、**SM 不確定性導引探索/聰明重啟**（ACP 現在停滯是「盲目原地加熱」＝SGDR warm restart，不用 SM 判斷往哪探索；文獻 SOAR/uncertainty-first 用 surrogate 選重啟位置更省 HFSS 評估；需 SM ensemble/MC-dropout）、便宜版＝停滯時 `generator: latent` 撒隨機 z 多點重啟、rad head 真正 replay（需先修 `forward`/`forward_rad` batch 契約 `reshape(-1, *num_response)`）、rollback 也走 DLF。驗證靠 A/B，不靠 golden。相關 [[project_radiation_pattern]] [[feedback_audit_existing_first]]。
