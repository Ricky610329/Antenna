# 探索/搜尋設計 roadmap（cold-start 反向設計）

> 收斂這幾輪討論的主線：把 per-task 反向設計做成「**不依賴歷史資料**、用神經網路當退火器」的受控搜尋。
> 文件若與 code/`tests/` 不一致，以 code 為準。對照表在 [`../configs/README.md`](../configs/README.md)。

## 北極星
- **終局**：不靠歷史資料（cold-start）也能對任意 spec 找到好 pattern。SM 是可微分加速器，但**最終品質不該被一顆爛 SM 卡住**。
- **現階段**：spec 允許預訓練 → `sm_harvest.pth` 當「好 SM」的 oracle/拐杖，量天花板；真正要交付的是「現場把局部 SM 養準 + 受控探索」。

## 問題診斷（首輪實測）
- 三個 run 都「**早收斂、之後躺平**」（best 在前段出現，之後上百 epoch 不動）。
- 根因兩個，正好是我們**還沒正面打**的：
  1. **SM 品質**：`old_sm.pth` 對我們資料 ≈ 隨機 → GEN 經 SM 的梯度是垃圾 → 退化成隨機搜尋 → 早躺平。（對策：sm_harvest / 現場養 SM。）
  2. **探索是盲目的**：ACP 盲目週期加熱，plateau 後在 SM 盲區亂跳、白燒 HFSS。
- **但有 headroom**：zbatch best 4.32 < sigmoid 6.61（同 SM/budget）→ plateau 是**搜尋提早卡住、非全域最佳** → 值得做。

## 探索三層框架（「有意義的隨機性」怎麼來）
隨機性必須**穿過結構**才有意義（純像素亂翻 = 不連通垃圾、對 SM 無用）：
1. **注入**（隨機從哪進來）：tau 退火重結晶 / latent 噪聲 / 重啟種子。
2. **結構**（變成合理 pattern）：multiscale 生成器（多尺度上採樣 → 連通平滑先驗）。
3. **防崩塌 + 導向**（真的不同又有用）：排斥項（逼解碼器用 z、不塌）+ 往 SM 不確定處取樣（active learning）。

## 設計決策（已定）
- **latent = γ**：`q(z)=N(z*,σ²I)` reparam + σ 退火（`BatchLatentGenerator`）。
- **selection = 閘門 + 排序**：可行性閘門（SC）+ `sm_target + λ·boundary` 排序；best-of-K 送 HFSS。
- **聚合 = mean-over-K**（reparam 下 = 最小化雲的 E[loss]）。
- **boundary：loss → 控制變數**。開 boundary-gate 就**關 loss.boundary**（兩者衝突：loss 把 G 拉回、壓掉 gate 需要的出界訊號）。

## 兩種骨幹
- **(B) 動力學式（cold-start 推薦主線）**：**一個** multiscale 生成器 + 狀態驅動的退火/重啟（boundary-gated ACP）。探索 = 退火重結晶穿過結構先驗。**不依賴 z 不崩塌。** → config `single_sc_rad_multiscale_bgate_harvest.yaml`。
- **(A) latent 並行（加速選配）**：`batch_latent` 同批 K 候選 + **排斥項**（防塌）。每步 best-of-K。 → config `single_sc_rad_zbatch_div_harvest.yaml`。

## 路線 items 與狀態
| # | 想法 | 狀態 | 落地 |
|---|---|---|---|
| 1 | 骨幹 (B)：multiscale + boundary-gated ACP | **已可跑** | `single_sc_rad_multiscale_bgate_harvest.yaml`（兩零件皆已實作，本 config 合一） |
| 2 | boundary 退出 loss、專任控制 | **已做** | `single_sc_rad_bgate_harvest.yaml`（不設 loss.boundary + `scheduler.boundary_gate`） |
| 3 | 排斥項治 batch_latent 崩塌 | **已做 v1** | `selection.diversity_weight`（有界 RBF 核罰候選太像）；config `single_sc_rad_zbatch_div_harvest.yaml` |
| 4 | 控制變數 boundary → SM 不確定性 | **待辦（規格）** | 見下「待辦」；先用 boundary 把閘門價值驗出來再升級 |

### item 3 細節（排斥項）
- 在 batch_latent 的 mean-over-K 聚合 loss 加：`L_div = λ_div · mean_{i<j} exp(−‖logits_i−logits_j‖²/h)`。
- 作用在**連續 logits**（梯度平滑）；**有界 RBF**：靠太近才罰、拉遠飽和（不用「最大化距離」以免炸 logits）。
- `h`（帶寬）= 兩兩平方距離中位數（detached、SVGD median heuristic）；`λ_div`（`selection.diversity_weight`）預設 0 → 與原樣相同（golden 安全）。
- 驗證：`select/score_spread` 維持 >0（不塌）、best-of-K 優勢持續。

### item 4 規格（SM 不確定性當控制變數，待辦）
- **動機**：boundary（離最近已見距離）只是信任的**代理**；SM 不確定性是**直接**度量。
- **取得**：起步 **MC-dropout**（HFSSNet 加 dropout，p 預設 0 = no-op/golden 安全；推論跑數次取預測方差）；進階 small ensemble / multi-head。
- **接入**：閘門刻意設計成「吃純量 + 門檻」→ 把 `boundary_now` 換成 `sm_uncertainty(pattern)`、τ_b 換成不確定性門檻，**scheduler 一行不改**。
- **雙重用途**：同一不確定性同時當「控制（何時探/固化）」+「取樣導向（探 SM 最不確定處 = active learning）」。
- **順序**：先用便宜代理（boundary）驗閘門價值，再升級不確定性（估計需校準、有成本）。

## 待辦（本文件為固定紀錄，持續累加）
- **鏡像對稱改「軟約束」**：`MirrorGenerator` 是**硬約束**（強制左右對稱），可能限制太死/不一定最佳。改成**軟約束**（例如對稱性 loss / 半邊+可學殘差），讓對稱是「偏好」而非「強制」。待辦、規格待寫。
- **item 3 排斥項升級（若 median 帶寬療效不足）**：審查確認 median heuristic 在「少數塌縮+多數散開」時對塌縮 pair 推力弱（飽和區）。先用現版量 `cand_similarity`/`score_spread`；壓不下去再升級 **min/per-point 帶寬** 或 **hinge(margin) 排斥**。good SM 下塌縮未必嚴重，量了再說。
- **item 4 SM 不確定性**（控制變數升級，見上）。

## 其他待辦（既有，見 configs/README.md 已知缺口）
- **方向圖 FFT 物理近似**取代/輔助 NN rad head（cold-start、零資料、可微）。
- **cold-start SM bootstrap / warmup**：前 N epoch 凍 generator、撒多樣候選養局部 SM，再優化（active-learning DoE）。
- **ACP 多參數整體重構**：tau/lr/週期耦合、旋鈕太多 → 重寫成「狀態（boundary→SM 不確定性）驅動」的瘦控制器。
- **學習式 latent 空間**（用收集到的 pattern + 響應虛擬標籤學流形）——ambitious、資料飢渴，等 (B)+multiscale 不夠再說。

## 驗證方法論（每個改動都照做）
- 比較**只看 target-only `best_loss`**（HFSS criterion）＋**對齊 epoch（≈HFSS 次數）**，不比牆鐘、不比合成 `gen_loss`（各 config 權重不同、不可比）。
- 每改動 **opt-in、預設關 → golden 零漂移**；TB 看對應診斷（`acp/*`、`select/*`、`sched/sigma`）確認「真的在做我們以為的事」。
- 增量、單變因、用 TB 診斷歸因；別一次堆太多分不清誰有效、誰把搜尋帶歪。
