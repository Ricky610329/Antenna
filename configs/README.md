# configs/ — 實驗對照表

> **一個 `*.yaml` = 一組實驗的完整設定。** 這份檔案是所有訓練實驗的索引（全目錄、accumulating）：每個 config 在「對標哪個 baseline、改了什麼、想看什麼」一目了然。
>
> 📍 **正在跑 / 待跑 / 收尾中的實驗看 [ONGOING.md](ONGOING.md)**（精簡的 live 狀態板，跑完搬去歸檔；這份 README 維持全目錄不動）。
>
> ⚠ **硬規則（CLAUDE.md 慣例）**：新增/修改任何 `configs/*.yaml` 或訓練腳本，**必須同步更新這份表**。產生新實驗前先掃這份表，避免重複造輪子。

## 怎麼跑

```bash
conda activate patch          # 正式機才有 HFSS
python train.py configs/single_base.yaml
```

- 入口固定是根目錄的 **`train.py`**（已取代舊 `train_single.py` / `train_dual.py`）。
- config 鍵名有白名單把關：打錯鍵直接報錯，不會默默吃預設值。
- 斷點續跑：用**相同 config**（→ 結果夾名相同）再跑即自動續；要重頭跑就改 `name` 或清掉 result 夾。
- 設定/結構對照見 [`../docs/training.md`](../docs/training.md)；系統架構見 [`../docs/architecture.html`](../docs/architecture.html)。

## baseline 是誰

| port | baseline config | 說明 |
| --- | --- | --- |
| single | `single_base.yaml` | **學長單埠基準**（所有 loss 正則化＝0）。新單埠實驗都對標它。 |
| dual | `dual_base.yaml` | 學長雙埠基準。新雙埠實驗都對標它。 |

## 單埠 single（對標 `single_base.yaml`）

| config | 測試重點 | 與 base 的差異 | 舊編號 |
| --- | --- | --- | --- |
| `single_base.yaml` | 基準 | — | 1 / 2 / 5 |
| `single_sc.yaml` | **論文主方法：圖譜連通度（單埠版）** | `loss.spectral_connectivity: 0.0005` | （新增，對標 base 看 SC 是否有幫助） |
| `single_sc_rad.yaml` | **SC + 方向圖 loss** | 多 `radiation:` 區段（`enable: true`，beam_coverage_loss）；SM 多 rad head，`pretrained: old_sm.pth` 以 strict=False 部分載入暖啟動（與 base 同速起步） | （新增，對標 `single_sc` 看方向圖是否有幫助；**需正式機 HFSS**） |
| `single_sc_rad_mirror.yaml` | **SC + 方向圖 + 鏡像對稱** | 在 `single_sc_rad` 上 `generator: mirror`（唯一變因） | （新增，對標 `single_sc_rad` 看對稱約束在 rad 前線上是否更快/更好；**需正式機 HFSS**） |
| `single_sc_rad_replay.yaml` | **SC + 方向圖 + SM 經驗回放** | 在 `single_sc_rad` 上 `sm_train.mode: replay`（唯一變因；最新少數步＋回放整個緩衝取代「擬到收斂」反模式） | （新增，對標 `single_sc_rad` 看防遺忘/削暴衝是否更好；**需正式機 HFSS**） |
| `single_sc_rad_dlf.yaml` | **SC + 方向圖 + 動態損失過濾 DLF** | 在 `single_sc_rad` 上 `sm_train.mode: dlf`（學長論文 §3.5 驗證法：全收＋每輪用累計門檻 λ_t 篩菁英子集訓） | （新增，對標 `single_sc_rad`／`_replay`；論文消融 >50% 改善；**需正式機 HFSS**） |
| `single_sc_rad_boundary.yaml` | **DLF + Boundary loss（trust-region；old_sm）** | 在 `single_sc_rad_dlf` 上 `loss.boundary`（拉 G 回 SM 已見分布；權重待 A/B 調） | （old_sm 版；sm_harvest 版見 `_harvest`；**需正式機 HFSS**） |
| `single_sc_rad_boundary_mirror.yaml` | **最強 recipe + 鏡像對稱（old_sm）** | 在 `single_sc_rad_boundary` 上 `generator: mirror`（唯一變因） | （old_sm 版；sm_harvest 版見 `_mirror_harvest`；**需正式機 HFSS**） |
| `single_sc_rad_zbatch.yaml` | **最強 recipe + 同批多候選 Z（old_sm）** | 在 `single_sc_rad_boundary` 上 `generator: batch_latent`（旋鈕 `num_candidates/sigma/sigma_min` + 選用 `selection:`） | （old_sm 版；首輪觀測候選塌縮 `score_spread`→0，待加多樣性項再跑；**需正式機 HFSS**） |
| `single_sc_rad_boundary_harvest.yaml` | **【下一輪】sigmoid 基準 / oracle（sm_harvest）** | = `single_sc_rad_boundary` 換 `surrogate.pretrained: sm_harvest.pth`；**name 含 harvest → 全新結果夾**（不續跑舊 old_sm 夾） | （generator-prior 三代 A/B 基準；量「好 SM 的天花板」；**需正式機 HFSS**） |
| `single_sc_rad_boundary_mirror_harvest.yaml` | **【下一輪】鏡像對稱（sm_harvest）** | 對標 `_boundary_harvest`，唯一變因 `generator: mirror` | （三代 A/B 之一；**需正式機 HFSS**） |
| `single_sc_rad_boundary_multiscale_harvest.yaml` | **【下一輪】多尺度淺層（sm_harvest）** | 對標 `_boundary_harvest`，唯一變因 `generator: multiscale`（1×1/5×5/13×13 上採樣相加、淺層；cold-start 連通先驗） | （三代 A/B 之一；⚠ 最細 13×13、細節不足就把 `scales` 加大；**需正式機 HFSS**） |
| `single_sc_rad_bgate_harvest.yaml` | **boundary 當控制（非 loss）：boundary-gated ACP** | 對標 `_boundary_harvest`：**不設** `loss.boundary` ＋ 開 `scheduler.boundary_gate`（plateau 出界抑制加熱、冷卻固化；區內卡住才探）。開 gate 就關 loss（兩者衝突） | （A/B：boundary 當控制 vs 當 loss；看是否少燒 plateau 後遊蕩；⚠ 配 sm_harvest 才發揮；**需正式機 HFSS**） |
| `single_sc_rad_multiscale_bgate_harvest.yaml` | **【探索組 A】骨幹 (B)：multiscale + boundary-gated ACP** | 結構化生成器（multiscale）＋ boundary 當控制（不設 loss.boundary + `scheduler.boundary_gate`）；cold-start 主線 | （探索三層骨幹；對標 `_boundary_multiscale_harvest` 看狀態驅動探索是否破 plateau；roadmap item 1；**需正式機 HFSS**） |
| `single_sc_rad_zbatch_div_harvest.yaml` | **【探索組 B】Z + 候選排斥（治崩塌）** | 在 zbatch 上 `selection.diversity_weight: 1.0`（有界 RBF 排斥，唯一變因；boundary 維持當 loss 以隔離） | （驗排斥項能否維持 `select/score_spread`>0、best-of-K 持續；roadmap item 3；⚠ λ_div 起步值待調；**需正式機 HFSS**） |
| `single_sc_rad_zbatch_div_bgate_harvest.yaml` | **【探索組 C】Z 全載：排斥 + boundary 當控制** | batch_latent + `selection.diversity_weight` + boundary-gated ACP（不設 loss.boundary） | （疊注：對標 `_zbatch_div_harvest` 看 boundary 當控制是否再加分；roadmap item 2+3；贏了再 ablation；**需正式機 HFSS**） |
| `single_guided_harvest.yaml` | **【新主線 1】generator-free SM-guided 搜尋** | 對標 `single_sc_rad_boundary_harvest`（sigmoid G），唯一變因 `generator: direct`（pattern logits 本身即可學參數、無 MLP）+ `num_candidates: 8`（pattern 空間多樣）+ `selection.diversity_weight`。radiation.weight 1→0.1（重點 S11/Gain）| （隔離「有沒有 G」；G＝過參數化超特徵，瓶頸在 SM；對標 random best-of-N；**需正式機 HFSS**） |
| `single_guided_ens_harvest.yaml` | **【新主線 2】+ ensemble SM 不確定性** | 在 `_guided_harvest` 上 `surrogate: ensemble`（K=5 成員）+ `loss.uncertainty`（信任懲罰 λ_trust·u(x)，推離 SM 盲區）+ `selection.uncertainty_weight`（acquisition κ，挑不確定點主動學習）| （隔離「ensemble 不確定性 + 信任懲罰 + UCB acquisition」；攻 SM 品質＝命門；λ_trust/κ 為靜態；**需正式機 HFSS**） |
| `single_guided_ens_adapt_harvest.yaml` | **【新主線 3】+ 閉迴路信任控制** | 在 `_guided_ens_harvest` 上 `trust.enable`（gap=SM-vs-HFSS 落差 → 信任標量 t → 調 tau 乘子/λ_trust/κ）| （隔離「閉迴路 vs 開迴路排程」＝把 ACP 升級成 gap 驅動；收斂湧現：SM 被修準→gap↓→t↑→tau 自動銳化；起步 g0/ema/tau_inflate 待 A/B；**需正式機 HFSS**） |
| `single_guided_dlffit_harvest.yaml` | **【SM 訓練量 A/B】把 DLF「訓到 fit」** | 對標 `single_guided_harvest`，唯一變因 `sm_train.mode: dlf → dlf_fit`（elite 訓到收斂 + 丟掉 50 步單筆 step）| （經查現行 `dlf` 只訓 elite 1 epoch ＝ under-trained DLF；學長原版是訓到 fit。B1 已確認 sm_harvest 對得上現在 HFSS(中位 MSE 1.56)→ 訓練強度是頭號嫌疑。判準看 `gap_ema` 非 training loss；**需正式機 HFSS**） |
| `single_guided_refit_harvest.yaml` | **【SM 訓練量 A/B】訓整個 buffer（不挑 elite）** | 對標 `_dlffit`，唯一變因 `sm_train.mode: dlf_fit → refit`（訓「整個 buffer」含非 elite，不只菁英）| （測「要不要挑菁英」：guidance surrogate 也學「爛 pattern 是爛的」→ 避得開對抗洞。⚠「整個」受 `replay_size` FIFO 上限；真「全部過往」靠之後的週期 harvest 重錨。**需正式機 HFSS**） |
| `single_r2_ens_harvest.yaml` | **【Round 2 ①】治本:ensemble（dlf 底）** | 在 `single_guided_harvest`(dlf) 上 `surrogate: ensemble`(K=5)+`loss.uncertainty`(信任懲罰)+`selection.uncertainty_weight`(acquisition)；rad `n_basis` 16→8（老師）。用 Round-1 A 當 baseline | （Round 1 結論：SM 訓練量非 bottleneck（dlf≈refit>dlf_fit、皆差 spec ~4dB）→ 改測文獻治本＝不確定性門控；對照 Round-1 A 看 ensemble 有沒有用；**需正式機 HFSS**） |
| `single_r2_enstrust_harvest.yaml` | **【Round 2 ②】+ trust 閉迴路（dlf 底）** | 在 Round2① 上 `trust.enable`（gap→t→調 tau 乘子/λ_trust/κ）| （隔離「閉迴路 gap 控制 vs 靜態 ensemble」；對照 Round2① 與 Round-1 A；**需正式機 HFSS**） |
| `single_r2_refit_enstrust_harvest.yaml` | **【Round 2 ③】全治本 × refit 底** | = Round2② 但 `sm_train.mode: dlf → refit`（廣覆蓋訓法、Round-1 sim_loss 最佳）| （測「完整治本＋最強 SM 底」上限；對照 Round-1 C；⚠ refit×K 成員每輪 SM 訓練量最大；**需正式機 HFSS**） |
| `single_r3_explore.yaml` | **【Round 3 E】探索臂** | 對標 ②（`single_r2_enstrust_harvest`），唯一變因 `lr 0.005→0.015`（3× 步長解凍） | （Round 2 實測搜尋凍住=每 epoch 才翻 ~6 像素；測加大步長探索能否解凍+變好。判準：像素翻轉數+worst_margin；**需正式機 HFSS**） |
| `single_r3_dip.yaml` | **【Round 3 D】DIP 臂** | 對標 ②，唯一變因 `generator: direct→sigmoid`（架構連通先驗、單候選，去 num_candidates/selection） | （generator-free 丟連通先驗 r_feed 0.2 vs sigmoid 0.62 → 帶回 sigmoid+治本救連通/S11；對照 ②；**需正式機 HFSS**） |
| `single_r3_dip_explore.yaml` | **【Round 3 E+D】DIP+探索加乘** | = `single_r3_dip`（sigmoid）+ `lr 0.015` | （測 DIP×探索加乘；E+D vs D=sigmoid 上探索效果、vs E=探索下 DIP 效果；factorial 需 E/D 都跑；**需正式機 HFSS**） |
| `single_sc_rad_smharvest.yaml` | **改用自訓 SM 初始化** | 在 `single_sc_rad` 上 `surrogate.pretrained: sm_harvest.pth`（唯一變因；old_sm.pth 對我們資料 ≈隨機，自訓的準 ~3 倍） | （新增，對標 `single_sc_rad` 看好的初始化是否讓早期收斂更快；**需正式機 HFSS**） |
| `single_sc_rad_flat15.yaml` | **±45 平整：收緊容忍** | 在 `single_sc_rad` 上 `radiation.floor_db` 3 → 1.5（唯一變因；窗內容許 ripple 收到 1.5dB，更平更高） | （新增，「±45 高且平整」對照組之一；對標 `single_sc_rad`；**需正式機 HFSS**） |
| `single_sc_rad_flat10.yaml` | **±45 平整：容忍 1dB** | 在 `single_sc_rad` 上 `radiation.floor_db` 3 → 1.0（唯一變因；比 flat15 更緊，掃 floor_db 一個點） | （新增，「±45 高且平整」對照組；看容忍收太緊會不會逼降峰值/難收斂；**需正式機 HFSS**） |
| `single_sc_rad_flatloss.yaml` | **±45 平整：主動壓平 loss** | 在 `single_sc_rad` 上 `radiation.flatness_weight: 0.5`（唯一變因；beam_coverage_loss 加 ③ `mean((G−G0)^2)` 主動壓平，floor_db 維持 3） | （新增，「±45 高且平整」對照組；主動壓平 vs 收緊容忍 A/B；flatness_weight 起步值待調；**需正式機 HFSS**） |
| `single_tv.yaml` | TV 正則化 + KuoHung SM 暖身 | `loss.total_variation: 0.01`、`surrogate.warmup: "1"` | 3 / 4 |
| `single_tv50.yaml` | 強 TV | `loss.total_variation: 50` | 7 |
| `single_island.yaml` | 孤島抑制（強） | `loss.island_suppression: 100` | 8 / 9 |
| `single_island1.yaml` | 孤島抑制（弱） | `loss.island_suppression: 1` | 10 |
| `single_peak.yaml` | ACP plateau 策略 | `scheduler.on_plateau: peak` | 6 |
| `single_mirror.yaml` | **左右鏡像對稱生成器** | `generator: mirror`（MLP 出半邊 25×13→flip 成 25×25；對稱+搜尋空間砍半） | （新增，對標 base 看對稱約束是否更快/更好；學長舊法是資料增強、不保證對稱） |
| `single_sc_mirror.yaml` | **SC + 鏡像對稱** | 在 `single_sc` 上 `generator: mirror`（唯一變因） | （新增，對標 `single_sc` 看對稱約束在 SC 上是否更快/更好） |
| `single_sc_mirror_boundary.yaml` | **SC + 鏡像 + DLF + Boundary（無方向圖）** | 在 `single_sc_mirror` 上 `sm_train.mode: dlf` + `loss.boundary`（進階 SM 訓練疊在對稱生成器上） | （新增，mirror 進階版；對標 `single_sc_mirror`；無 rad） |

## 雙埠 dual（對標 `dual_base.yaml`）

| config | 測試重點 | 與 base 的差異 | 舊編號 |
| --- | --- | --- | --- |
| `dual_base.yaml` | 基準 | — | 6 / 7 / 8 |
| `dual_sc.yaml` | 論文主方法：圖譜連通度 | `loss.spectral_connectivity: 0.0005` | 9 |
| `dual_tv100.yaml` | 強 TV | `loss.total_variation: 100` | 1 / 4 |
| `dual_tv1.yaml` | 弱 TV | `loss.total_variation: 1` | 2 |
| `dual_island.yaml` | 孤島抑制 | `loss.island_suppression: 100` | 3 / 5 |

## 已知缺口 / 可補的實驗

- ~~單埠沒有 `spectral_connectivity` config~~ → 已補 `single_sc.yaml`（2026-06-19）。
- ~~方向圖 loss 尚未有 config~~ → 已補 `single_sc_rad.yaml`（2026-06-19，Stage 2 整合完成：`radiation:` 區段 + SM rad head + `beam_coverage_loss`）。**僅正式機可跑**（需 HFSS 取方向圖）。
- **方向圖 rad head 冷啟動**：`harvest_single` 沒有方向圖標籤 → rad head 線上從零學。優化（Stage 3，未做）：收 `harvest_single_rad`（好 pattern 補方向圖標籤）預訓練 rad head。⚠ **`radiation.offline_dataset` 這個鍵尚未實作**（不在 `training.py` 的 `radiation` 白名單內 → 現在寫進 config 會被驗證器擋下、run 起不來）；要做 Stage 3 時需先把該鍵加進白名單並接上載入邏輯。
- **方向圖物理近似（FFT / array-factor）取代/輔助 NN rad head（待辦，規格待寫）**：cold-start 視角——遠場 ≈ 電流分布的傅立葉轉換，最粗近似「方向圖 ≈ |FFT(pattern)|²」。可微（`torch.fft`）、**零訓練資料** → 直接 guide GEN 梯度，且能取代現在「凍 trunk + 冷啟動擬不動」的 NN rad head。用 #1 收集的 `<結果夾>/radiation/` 資料**校準少數自由參數**（低維擬合、非訓大網路）並**量「對 HFSS 的誤差與偏差」**。⚠ **偏差 > 準確率**：粗但不偏才有用，系統性偏掉會把搜尋帶歪、白燒 HFSS。可選 hybrid：物理當骨幹 + NN 學殘差（physics-informed、省資料）。先拿現有 HFSS 資料量實際 % 再決定投入；S11/Gain 的解析近似更難（任意 pixel 無良定義等效長度），「70%」別預設、先量。
- **方向圖訓練預設凍 trunk**（`radiation.freeze_trunk: true`）：隨機 rad 頭 + 不凍 trunk + 極端 dB target 曾把 S11/Gain backbone 帶歪、爆 NaN。現在 rad 頭只更新自己、rad target 會 clamp（±60dB）、SM 訓練有 NaN 防護網。要放梯度回 backbone 才設 `false`。
- **⚠ `train_one_data_rad` 實測沒擬合（2026-06-27 使用者回報，下次改）**：即使換了平滑 cosine 基底，凍 trunk + K=16 係數**仍擬不到 `min_loss`（0.1，繼承 `sm_train`）** → 每筆 fresh 都跑滿 `radiation.sm_max_epoch`（1000）卻沒收斂、`rad_predict` 對真值不夠準（rad 在 `weight 0.1` 下對 S11/Gain 影響有限，但 rad 訊號本身不可信）。修向（下次）：①最省＝設**實際的** `radiation.sm_min_loss`（別沿用 0.1、設成頭真能到的值 → 先止損、不空跑滿 1000）；②加容量＝拉高 `radiation.n_basis` 或週期性解凍 trunk（⚠ 解凍有 NaN 病根）；③Stage 3 離線預訓 rad head。診斷盯 TB `components/rad_fit`（會發現它不降）。動 loss/訓練前先討論。
- **rad head ＝ 平滑 cosine 基底頭**（`radiation.n_basis`，預設 16）：head 不直接吐 `n_theta` 個獨立值（裸 `Linear` 無平滑先驗 + 凍 trunk 下擬不到收斂 → 預測鋸齒），改吐 K=`n_basis` 個 cosine 係數，乘固定基底展開成 `n_theta` 點 → 預測 band-limited、**結構上必平滑**，且只擬 K 個數收斂快。基底是不可訓 buffer、用 `set_rad_theta` 依實際 HFSS θ 網格（整 run 固定）逐欄重建 → 對位正確、HFSS 匯出序未排序也 OK。K 越小越平滑（K=1＝常數）。⚠ 改了 head 形狀 → 舊 rad-run 的 `sm.pth` 不能續跑（freq-only checkpoint 不受影響、golden 零漂移）。
- **SM 初始化（old_sm.pth ≈ 隨機）**：量過 `old_sm.pth` 對 `harvest_single` 預測中位 MSE 38（≈全新隨機 SM 的 35，且吐 +40/-88dB 亂值）→ 現在的「暖啟動」等於沒用、線上學習每次從近乎隨機重學。`script/train_sm_offline.py` 自訓一顆（val MSE 13、準 ~3 倍）= `sm_harvest.pth`，config 用 `surrogate.pretrained: sm_harvest.pth`（`single_sc_rad_smharvest.yaml` A/B）。⚠ 前提：harvest 要與「現在的 HFSS」對得上（正式機驗證：幾張 harvest pattern 丟現在 HFSS 比對存的 response）；不對就要重收。
- **SM 線上更新重設計（profiling 找到瓶頸後）**：`train_one_data` 把「最新一筆擬到收斂（≤20000 步）」是 catastrophic-forgetting 反模式、也是速度大頭。`sm_train.mode`：`single`(原樣/反模式) → `replay`(全收＋回放整個緩衝) → `dlf`(全收＋每輪用累計門檻 λ_t 篩菁英子集訓，＝學長論文 §3.5 驗證法、消融 >50%)。opt-in，預設 single 保 golden。`radiation.sm_max_epoch`(rad 擬合上限,1000)。⚠ 舊 `online` store「<平均才收」＝論文點名的 baseline（寫入端篩→偽收斂），replay/dlf 改走「全收」緩衝。`loss.boundary`（已做）＝trust-region 正則（拉 G 回 SM 已見分布、防鑽盲區、少白燒 HFSS；需 replay/dlf 緩衝；權重待 A/B 調）。未來：prioritized 抽樣、SM 不確定性導引探索/聰明重啟（取代 ACP 盲目加熱）、rad head replay。
- **同批多候選生成器（batch_latent / Z，已做 v1）**：`generator: batch_latent` —— 高斯雲繞可學中心 z\*（reparam `z_k=z*+σ·ε_k`、σ 隨 epoch 退火），同一輪生成 K 個候選 → 在 SM 上「閘門（SC≤feas_max，選用）＋排序（sm_target＋λ·boundary）」選最佳 → **只送選中那張去 HFSS**（HFSS 預算同單張、搜更廣）→ **mean-over-K** 聚合「四現役 loss（sm_target/SC/boundary/rad）」反傳。config：`single_sc_rad_zbatch.yaml`。TB 監控：`sched/sigma`＋`select/{score_best,score_mean,score_spread,fresh_frac}`（`score_spread→0`＝候選塌縮、Z 失效；`fresh_frac` 低＝探索停滯）。**待 A/B / 回顧的選項**：① K（`num_candidates`，起手 8）；② σ 排程（`sigma` 0.5→`sigma_min` 0.05、目前線性，可試指數/cosine）；③ 聚合改 **CEM-style elite 加權**（`w_k∝softmax(-loss_k/temp)`，好候選多給梯度，v2 加 config 開關）；④ `selection.feasibility_max`（SC 閘門門檻，等正式機看 SC 真實尺度再設）；⑤ `selection.boundary_weight`（λ，目前沿用 `loss.boundary`，尺度不同時再單調）；⑥ 是否把 σ 設成可學 `nn.Parameter`（reparam 天然可學）。⚠ 已知地基：`BiScaleNorm` 已加 `clamp_min(eps)` 防退化候選 grad-NaN（golden-neutral）。**觀測（首輪）**：zbatch 候選 `score_spread` 由 3.34 塌到 ~0.13（MLP 學會忽略 z、mode collapse）→ best-of-K 紅利消失，**待加多樣性/排斥項**（v2，碰 loss 先討論）。
- **訓練時順手收集方向圖資料（已做）**：rad run 每筆真跑過的 pattern 連同真實方向圖存進 `<結果夾>/radiation/`（`SampleStore`，`(3,n_theta)=[theta,phi0,phi90]`，hash 去重，**零額外 HFSS**）→ 供日後離線重訓 rad head／backbone（Stage 3）。⚠ backbone 重訓要**離線、freq+rad 一起**（線上放梯度回共用 trunk 會把 S11/Gain 帶歪→NaN）。
- **boundary-gated ACP（已做 v1，opt-in）**：boundary 從「loss」升級成「ACP 的控制依據」。plateau 時若 `boundary≥τ_b`（衝出 SM 可信區）→ **抑制 warm restart**（不加熱、冷卻就地固化、SM 在此被訓）；`boundary<τ_b`（區內卡住）→ 放行往外探。`scheduler.boundary_gate: true` ＋ `boundary_kappa`（τ_b=κ·replay 典型 NN 間距）/`boundary_recompute_every`/`boundary_suppress_cap`（連續抑制上限、防餓死）。需 replay/dlf 緩衝；預設關 → 現行 ACP（golden 安全）。TB 看 `acp/boundary`、`acp/restart_suppressed`。⚠ 配像樣 SM（sm_harvest/warmup 後）才發揮，純冷啟動早期「近≠SM 準」打折。待續：① ACP 多參數（tau/lr 與週期耦合）整體重構（旋鈕太多、之後想）；② boundary 改用 **SM 不確定性** 當控制訊號；③ 軟性 tau 偏置（連續調溫，非離散 gate）。
- **metrics.csv 補欄（已做）**：`rad_loss / sigma / score_best / score_mean / score_spread / fresh_frac`（rad/batch_latent run 才有值，其餘留空、向後相容）→ 離線可直接分析，不必解 TB 事件檔。⚠ 加欄改表頭：對**全新結果夾**生效，別拿新碼 append 舊表頭 csv。
- **SM 初始化換 sm_harvest（下一輪，另開 harvest 命名）**：下一輪 generator-prior 三代 A/B 用**全新的 `*_boundary*_harvest.yaml`**（`boundary_harvest` / `boundary_mirror_harvest` / `boundary_multiscale_harvest`），`surrogate.pretrained: sm_harvest.pth`（old_sm≈隨機）。**為何另開新 name 而非改舊檔**：同名 config 在既有結果夾會「續跑」載回舊 old_sm checkpoint、吃不到 sm_harvest；name 含 harvest → 強制全新結果夾。舊 `boundary`/`_mirror`/`zbatch`（old_sm）保留為「已跑過的 old_sm 實驗」記錄。⚠ metrics.csv 加欄也只對全新夾生效，正好一起。
- **multiscale 加更細尺度 25×25（待辦，視 A/B 結果再投）**：`single_sc_rad_boundary_multiscale_harvest.yaml` 目前 `generator.scales: [1, 5, 13]`，最細只到 13×13、band-limited 偏連通 → **單像素細節可能擬不出**。若 A/B 觀察到 multiscale 在需要細結構的 task 上 loss 卡住/擬不動，把 25×25 加進去（`scales: [1, 5, 13, 25]`，25＝side 上限、最細頭直接吐全解析）。⚠ 代價：最細頭參數 = `in_dim·25²`（其餘尺度的數倍），淺層加性「連通平滑先驗」會被沖淡；先確認是表達力不足、不是 SM/loss 問題再加。
- **rad head cosine 基底數試 8（`radiation.n_basis`，待辦 A/B）**：目前預設 16（見上「rad head ＝ 平滑 cosine 基底頭」）。可試 `n_basis: 8` —— K 越小→預測越平滑、擬合的係數越少→凍 trunk 下收斂越快，代價是高頻 lobe 細節表達力下降。適合方向圖本來就平滑（窗 ±45°/floor 3dB、boresight 為主）的場景。⚠ 改 `n_basis` 動到 head 形狀 → 舊 rad-run 的 `sm.pth` 不能續跑（freq-only checkpoint 與 golden 不受影響）；要與 16 公平比就各開全新結果夾。
- **`direct` generator（generator-free，已做）＋ ensemble SM ＋ 閉迴路信任控制（新主線階梯）**：per-task 單 target 下 generator 不是學「spec→pattern」映射、而是把這一張 pattern **重參數化**（≈deep image prior）；G＝過參數化超特徵，**真正瓶頸是 SM**。新主線把優化變數直接換成 pattern logits（`generator: direct`，`generators.py:DirectPatternGenerator`，logits 即 `nn.Parameter(K, out_dim)`、forward 只過 `BiScaleNorm`、K>1 走既有多候選路徑），並沿「攻 SM 品質」往上疊三級（configs `single_guided{,_ens,_ens_adapt}_harvest.yaml`）：
  - **Exp1 direct**：對照 sigmoid G，量「拿掉 G」。多候選 K 個獨立 logits＝pattern 空間多樣（取代會塌縮的 batch_latent latent 雲）。
  - **Exp2 + ensemble**（`surrogate: ensemble`，`surrogates.py:EnsembleSurrogate`）：K 個獨立成員，`uncertainty()`＝成員分歧（花 HFSS 前的便宜可信度 proxy）；驅動 **信任懲罰** `loss.uncertainty`（λ_trust·u(x)，推離 SM 盲區）+ **acquisition** `selection.uncertainty_weight`（κ·u(x)，挑不確定點主動學習）。資料缺乏非障礙：成員共享全部資料、多樣性來自不同 init + 暖啟動擾動（`init_perturb`）。
  - **Exp3 + 閉迴路控制**（`trust.enable`，`training.py:TrustController`）：gap=|SM 預測−真實 HFSS|（訓練前量）→ 信任標量 t∈[0,1] → 同軸調 **tau 乘子 / λ_trust / κ**（t→0 放軟+收緊+探索；t→1 純 ACP 銳化+收割）。= 把學長 ACP 的開迴路退火升級成 gap 驅動的閉迴路；收斂湧現（SM 被修準→gap↓→t↑→tau 自動銳化）。語意設計：**docs/guided_search_design.md**。
  - 對偶觀點仍成立：「架構先驗（multiscale/mirror，隱式）」⇄「顯式 loss（SC/boundary/uncertainty）」；direct＝最無先驗端，靠顯式 SC/boundary/trust 補回（拿掉 G 的 DIP 平滑先驗，由 SC 連通性顯式涵蓋）。⚠ 純 direct 無先驗 → 易破碎/鑽 SM 盲區，正是信任懲罰/控制要治的。全部 golden 零漂移（旗標/權重 gate；enable=False 且 base=0 → 逐位元同原樣）。對標：worst-margin(dB) vs HFSS-call、對比 random best-of-N。
- **軟對稱取代硬 mirror（待辦；先量 mirror vs sigmoid 再決定）**：`mirror`（`generators.py:98`）是**硬對稱**——G 只出半邊、`flip+cat` 強制精確左右對稱、搜尋空間砍半。隱憂：問題有對稱性時，**對稱解常是優化駐點（鞍點/退化），最優可能在對稱破缺處**；硬對稱把解鎖在對稱子空間 = 鎖在鞍點 →「容易塌」。同「架構先驗 ⇄ 顯式 loss」對偶，對稱也有硬→軟光譜：硬 `mirror`（weight-tying）→ 半硬「對稱骨幹＋小殘差 `p=mirror+ε·free`」→ 軟「對稱正則 loss `λ·‖p−flip(p)‖²`（自由出 625、允許自發破缺、λ 連續可調、與任何 generator 解耦）」→ 最軟「資料增強（學長舊法）」。軟化的好處＝允許先破缺逃鞍點、不對稱沒幫助時自己收回。⚠ 對稱群是**物理決定**的——單埠饋電底部中央 → 只有左右鏡像成立，上下/旋轉對稱不該硬加；equivariant 網路太重、排除。**先量再決定**：拿 `single_mirror` vs `single_base`（或 `single_sc_mirror` vs `single_sc`）比最終 sim_loss + 看 pattern 是否卡在某對稱形狀不動；mirror 砍半空間在 cold-start 早期可能反而更快收斂、塌可能只是後期/階段性 → 確認真的塌再做。要做就軟對稱 loss 優先（加 loss 前先討論）。**【2026-06-27 觀察】使用者回報 mirror 實測表現普通（無明顯優勢）→「軟對稱正則 loss `λ·‖p−flip(p)‖²`」升為實際待辦選項**：與任何 generator（含 `direct` generator-free）解耦、λ 連續可調、允許自發破缺逃鞍點、不對稱沒幫助時自己收回；比硬 mirror 更安全。實作前仍須先討論（動 loss 規則）。

- **可解釋性:SM 屬性分析 → SM 診斷 + 好解局部先驗（待辦，多為「跑出好 pattern 後」）**：對一張 pattern 用 SM 問「每像素對 S11/Gain 的貢獻」。二值最忠實＝**遮擋法**（翻第 i 格、看 SM 預測變多少；625 次 SM forward ≈ 亞秒）；梯度版近乎免費（guidance 本來就在算 ∂response/∂pixel）。用途（依價值）：
  1. **SM 品質診斷（現在就相關、近乎免費）**：好 pattern 的貢獻熱圖該像 EM 物理（饋入/輻射邊緣/共振長度重要、角落 don't-care）；**物理上講不通＝SM 壞掉的紅旗**。把 guidance 梯度當熱圖落 TB 即可，與 trust/ensemble 同目標（知道 SM 可不可信）。
  2. **好 pattern → 局部先驗（主實驗，需先有 HFSS 驗證過的好解）**：屬性把像素分「關鍵 vs don't-care」→ 鎖關鍵格、只搜 don't-care 格 → 在已知好解附近做**有根據的降維局部搜尋**找鄰近變體（比盲目多候選有方向）。
  3. **跨多好解抽設計 motif → 顯性先驗/loss（AlphaFold 式，最投機）**：需先有一堆好解；把學到的規則回注成 init/loss/縮小參數化（接「把先驗轉成 loss」那條線）。
  ⚠ 陷阱（同專案病根）：屬性＝**SM 的信念**，只在 SM 準的地方可信 → 只對 **HFSS 驗證過**的好 pattern 做、關鍵格用 **HFSS 遮擋抽驗**接地、配 **ensemble** 得「屬性不確定性」（成員屬性不一致＝連重要性都沒把握）。定位：2/3 需先有好 pattern（現階段搜尋還贏不過 random，非當務之急）；但 (1) 現在就能順手加。

## 資料集標記（before rad / *_rad）

NAS（`DATASET_PATH = T:\碩一_鄒穎麒's\antenna\dataset`）上的離線資料集分兩代，**勿混疊**：

| 資料集 | 標籤 | 說明 |
| --- | --- | --- |
| `harvest_single`（24189）/ `harvest_dual`（10023） | **before rad** | response 只有 S11/Gain(/S21)，**無方向圖標籤**。各夾內有唯讀 `_BEFORE_RAD.md` 標記 |
| `harvest_single_rad` 等 `*_rad`（**Stage 3，由 `script/collect_radiation.py` 累積**） | with rad | 每筆 (pattern, rad)；rad=(3,n_theta)=[theta, phi0, phi90 gain(dB)]。任何 pattern 丟 `SinglePortRadSimulator` 即可取得（**不管有沒有 rad loss**）。**獨立新名**、不疊進舊集 |

- **約定**：rad 標籤資料一律走新名 `*_rad`，**永不**寫回 before-rad 舊集。
- marker 用 `.md`（`SampleStore` 只認 `*.pt`，且 init 會清 `*.tmp` → marker 不能用這兩種副檔名）。

## 新增實驗 SOP

1. 複製最接近的 baseline（`single_base.yaml` / `dual_base.yaml`）。
2. 改 `name`（決定 result 夾名，避免撞到別的實驗）。
3. **只改一個變因**（A/B 才乾淨）：通常是 `loss:` 區段開一個權重，或 `scheduler:` / `generator:` / `surrogate:` 換一項。
4. **回來這份表加一行**（測什麼、與 base 差在哪）。← 這步是硬規則。
5. 跑：`python train.py configs/<新檔>.yaml`（正式機）。結果夾＝自我說明的檔案制資料庫（`metrics.csv` / `patterns/` / `tb/`…，見 training.md §6）。

## 相關訓練腳本（`script/`）

| 腳本 | 是否訓練 | 功能 |
| --- | --- | --- |
| `../train.py` | ✅ 主入口 | config 驅動的單/雙埠訓練閉迴路 |
| `script/verify_radiation.py` | ❌ 驗證 | 正式機驗證 `SinglePortRadSimulator` 能否把方向圖資料抓出來（不訓練、不碰核心） |
| `script/collect_radiation.py` | ⚙ 資料 | **累積方向圖資料 → `DATASET_PATH/harvest_single_rad`**（Stage 3）。任何 pattern 丟 `SinglePortRadSimulator` 跑一發 HFSS（**不管有沒有 rad loss**）；hash 去重、可重複跑只補新的。來源：`--runs`（result 夾各取最佳 K）/ `--dataset`（harvest 取最佳 N）/ `--pattern`。**需正式機 HFSS** |
| `script/kuohung.py` | ⚙ 資料 | KuoHung 參考圖樣載入（SM 單筆暖身用，對應 `surrogate.warmup`） |
| `script/harvest_legacy.py` | ⚙ 資料 | 從學長舊資料收割成自有 NAS 資料集（`harvest_single` / `harvest_dual`） |
| `script/train_sm_offline.py` | ⚙ 初始化 | 在 harvest 資料上 minibatch 訓一顆 SM 當「好的初始化」→ `DATASET_PATH/sm_harvest.pth`（old_sm.pth 對我們資料 ≈隨機；自訓 val MSE 13 vs 38）。開發機可跑、不需 HFSS |
| `script/check_harvest_consistency.py` | ❌ 驗證 | **抽驗 harvest 與「現在的 HFSS」對不對得上**(暖啟動前提):抽 N 筆 harvest pattern 用現在 HFSS 重跑、比對 MSE。中位 ≈ val MSE(~13)→ 可信;≈隨機(~35-38)→ sm_harvest 須重收。**需正式機 HFSS** |
| `script/benchmark_vs_random.py` | ❌ 分析 | **worst-margin(dB) vs HFSS-call 客觀 benchmark**(離線):從 run 的 metrics.csv+patterns/ 畫 best-margin-so-far 曲線、多 run 疊圖、對比 random best-of-N。一天驗一版(~250ep)用「曲線誰升得快」判斷,非「有沒有達標」。worst-margin 定義 = in-band(中央平台)對 spec 最差餘裕(與 custom_loss_minmax 一致)。開發機可跑 |
| `script/round_report.py` | ❌ 分析 | **round 結果歸檔**(reuse benchmark):吃一個 round 的 runs → 每臂「最佳 pattern + S11/Gain vs spec」圖 + worst-margin vs HFSS-call 疊圖(可加 random)+ 可貼進 `docs/log/round-NN` §4 的 markdown 數字。圖落 `docs/log/assets/round-NN/`。開發機可跑 |
| `script/status.py` | ❌ 監控 | **掃 NAS result/ 各 run 即時狀態**(取代手動猜、減 ONGOING churn):機器/epoch/每epoch耗時/**alive或卡住**/最佳 worst_margin/skip。`--match` 篩、`--md` 出可貼 ONGOING 的表。純讀、開發機可跑 |
| `script/analyze.py` | ❌ 分析 | **可重現診斷工具**(把散在對話的一次性分析收成子命令):`volatility`(每 epoch 像素翻轉+波動=探索量)/`rad-repr`(方向圖 K 個 cosine mode 最佳擬合殘差=表達力上限)/`rad-error`(已訓 rad head 窗內 pred-vs-real 誤差)。純讀、開發機可跑 |
| `script/convert_dataset.py` | ⚙ 資料 | 舊 `.dataset` 格式轉換 |
| `script/img2video.py`、`check_gpu.py`、`get_local_ip.py`、`kill.py`、`process_files.py` | ❌ 雜項 | 視覺化／環境／程序管理工具，與訓練無關 |
