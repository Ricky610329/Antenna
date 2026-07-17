---
name: project_generator_hyperfeature_pivot
description: zbatch 砍掉;G=單一 pattern 超特徵;SM-only guided 搜尋已實作 (single_guided* 三 config:direct+ensemble+trust)
metadata: 
  node_type: memory
  type: project
  originSessionId: 46183b31-ca6a-4a82-9ee0-98dfff4174bc
---

使用者決定(2026-06-26):**停用 batch_latent(zbatch)**。洞見:`G(spec)→pattern` 在固定 spec 下本質是「**單一 pattern 的過參數化(超特徵)**」——不是真的生成模型;靠同批 K 個 latent 找「更多 pattern」策略上就錯(latent 空間會塌縮)。

**已實作(2026-06-26)** 成「generator-free SM-guided 搜尋」三 config 階梯(`configs/single_guided{,_ens,_ens_adapt}_harvest.yaml`),全 golden 零漂移、228 tests 綠、3 個 reviewer agent 過(無高/中 bug):
- Exp1 `direct`(`models/generators.py:DirectPatternGenerator`):pattern logits 本身即 `nn.Parameter`(無 MLP),K 個獨立候選=pattern 空間多樣;走既有多候選路徑(已通用化:`is_multi_candidate` 旗標,BatchLatent 行為不變)。
- Exp2 `ensemble`(`models/surrogates.py:EnsembleSurrogate`):K 成員,`uncertainty()`=成員分歧 → 信任懲罰 `loss.uncertainty`(λ_trust·u) + acquisition `selection.uncertainty_weight`(κ·u,主動學習)。
- Exp3 `trust.enable`(`training.py:TrustController`):gap=|SM 預測−真實|(**SM 線上訓練前**量)→ t∈[0,1] → 調 tau 乘子/λ_trust/κ;收斂湧現(SM 修準→gap↓→t↑→tau 自動銳化)。語意設計 `docs/guided_search_design.md`。
- 待辦:正式機真 HFSS A/B(對標 random best-of-N)、調 g0(gap 尺度,csv 已落 `gap_ema` 供稽核)、radiation.weight 已依使用者指示降到 0.1。

**實測發現(2026-06-27,r_feed 連通性)**:generator-free(direct)**丟掉了 sigmoid 的隱式連通先驗(DIP)**。線上 A/B 三臂 r_feed(=與饋電同連通塊的金屬佔比,`losses.FeedReachability`)**mean ~0.20**(=~80% 是浮空孤島),對照 sigmoid 基準 `single_sc_rad_boundary_harvest`(254ep)**mean 0.62、max 0.97**=**3× 差距**;而兩者 `loss.spectral_connectivity` **都只 0.0005**(direct 沿用 sigmoid 的值)→ 證實:**同樣低 sc,sigmoid 的架構自己會連通、direct 沒架構先驗就碎**。意涵:①generator-free 的具體代價 = 失去 DIP 連通先驗;②「direct vs sigmoid」要公平比應**對齊 r_feed(拉高 direct 的 sc)而非對齊 sc**,否則混淆「有沒有連通先驗」這個變因;③碎 pattern = 更大更吵的設計空間,可能更難給 SM 學、且難製造。**不影響當前 SM 訓練量 A/B 比較**(三臂都 direct+同 sc → r_feed 同低=常數)。**Round-2 lever**:若續走 generator-free,拉高 `spectral_connectivity` 補回連通。r_feed 已在 csv/TB `index` 追蹤(本來就有)。

**Why:** benchmark 顯示 G 式搜尋輸 random([[project_benchmark_vs_random]]);crux:**SM 不準則 guidance 沒用**→ 設計含 ensemble 不確定性門控 + active learning(治本=[[project_litreview_direction]] 的 C)。
**How to apply:** 動這條前先讀 `docs/guided_search_design.md` + `docs/senior_method.md`。關聯 [[project_sm_training_redesign]] [[reference_paper_terminology]] [[project_radiation_pattern]]。
