# 學長原始方法整理（timmy90928/Antenna, branch `GAN`）— 單埠完整版

> **目的**：把學長最早期論文版的**單埠**反向設計方法整理成 self-contained 參考——讀完這份就**不必再回去翻他的 code 或論文**。
> 由 3 個分析 agent 唯讀分析其 repo（branch `GAN`）後彙整；行號指向**他的 repo**。
> 我們現在的 repo 是它的重構；「核心 vs 我們做的（可能無效）工作」+ redesign 啟示見 **§四**（最重要）。
> 聚焦單埠（S11 要夠低、Gain 要夠高）；dual/ris/selection 是埠變體，骨架相同、一句帶過。

## 術語對照（論文 ↔ 程式碼，先讀，極易誤解）

| 論文用語 | 程式碼裡實際是什麼 | 位置 |
|---|---|---|
| 「GEN / 生成」 | **代理模型 SM**（Pattern→Response 的可微 HFSS 近似），**不是生成器** | `smodels.py:422 OldSM` → `HFSSNet`（變數常叫 `model_ge`/`_ge`，誤導性極強） |
| 真正的生成器 G | `SigmoidGEN`（Response→Pattern） | `models.py:470` |
| 「ACP」 | 排程器（lr + tau 循環退火），**不是** rollback | `AdaptiveCyclicalScheduler`（`functions.py:248`） |
| rollback | 與 ACP **獨立**：early-stop 觸發後把 G 權重退回歷史最佳 epoch + 重訓 SM | `train_single.py:238-256` |

---

## 一、整體架構與訓練迴圈

### 1. 方法總覽
**per-task 線上 GAN 式反向設計**：對「一個固定目標響應」(target S11 + target Gain)，用梯度法找一張能達成它的 25×25 金屬 pattern。三件組成：
- **生成器 G**（`SigmoidGEN`）：輸入目標響應向量 → 輸出二值化 (0/1) pattern。
- **代理 SM**（`HFSSNet`，純 MLP）：「可微分的假 HFSS」，pattern → 預測響應；讓 loss 能反傳回 G（真 HFSS 不可微）。
- **真 HFSS**（`SinglePortSimulator`）：迴圈內每步對 G 產的 pattern 跑一次真模擬 → 拿真響應 **線上修 SM** + 評估真實好壞。

資料流（一個 epoch）：
```
target ─► G ─► pattern ─┬─► 真 HFSS ─► real_response ─► real_loss（評估/排程/rollback）
                        │                          └─► 訓 SM（把這筆 pattern→response 教給 SM）
                        └─► SM ─► pred_response ─► fake_loss ─► backward() ─► 更新 G
```
**G 只透過可微的 SM 反傳更新**；真 HFSS（不可微）只做 (a) real_loss 評估、(b) 提供新樣本訓 SM。非對抗（無 discriminator），而是 G 追著「被真值持續校正的 SM」跑。

### 2. 生成器 G（`SigmoidGEN`，`models.py:470-488`）
- 輸入：target response 攤平（`target.concat()`）。層結構：
  ```
  Linear(in → 1024) → PReLU → Linear(1024 → 1024) → PReLU → Linear(1024 → 625) → BiScaleNorm()
  ```
- 輸出 + 二值化：`forward` 結尾呼叫 `AntennaPattern.binarization(x, tau)`（STE，詳見 §二-2）。
- 優化器：**Adam, lr=0.005, betas=(0.5, 0.999)**（`train_single.py:153-154`）。
- 變體（單埠未走，死碼候選）：`OldGEN`、`GumbelSigmoidGEN`、`SPGEN`、`CVAE`、`MirrorCVAE`。

### 3. 代理 SM（`OldSM`→`HFSSNet`，`smodels.py:177-205, 422-438`）
- 純全連接 MLP（無卷積/UNet；`EnhancedHFSSUNet`/`UNetSM` 是未用變體）：
  ```
  625 → 2048 → 1024 → 512 → 128 → 64 → 34(=2×17)  ，每層 PReLU，輸出 reshape (2,17)
  ```
- criterion `MSELoss`；優化器 **Ranger**（lr=0.001）；排程 `ReduceLROnPlateau(min, factor 0.5, patience 10, min_lr 1e-6)`。
- 兩種訓練介面：`train_one_data`（單筆擬到收斂，詳 §三-3.3）、`train_by_datas`（整批）。

### 4. 主訓練迴圈 pseudo-code（`train_single.py:215-405`，標明更新對象）
前置：建 G/SM/ACP；SM 冷啟動三選一（續跑載回／載 `old_sm.pth` 後用 KuoHung pattern 跑一次 `train_one_data`／用離線 `data_manager` `train_by_datas`）。`lower` = 固定貼底的 5×5 全 1 饋入區。
```text
while epoch < epochs(=1000):
    epoch += 1; generator.change(epoch); simulator.start(epoch)
    G.requires_grad(True, train=True); G.optimizer.zero_grad()

    # (A) 生成（含 rollback 判斷）
    if TEMP.early_stop('real_loss', patience=10):       # 近 10 步 real_loss 沒進步
        best = TEMP.find('real_loss', TEMP('min_loss'), 'epoch')
        G.change(best, save=True, load=True)            # ★ G 權重退回歷史最佳 epoch
        smodel.train_by_datas(online_dataset)           # ★ 更新 SM：用 elite 線上樣本重訓
    pattern = AntennaPattern(G(target.concat())) + lower

    # (B) 去重 + 真 HFSS + 訓 SM
    if pattern 未出現過:
        real_response = pattern.simulate()              # ★ 真 HFSS 一次
        real_loss = real_response.criterion()
        smodel.train_one_data(pattern.series, real_response.stack())  # ★ 更新 SM（擬到 loss<0.1 或 20000 步）
        smodel.save()
        if real_loss < TEMP.average('real_loss'):       # ★ DLF：只收「優於平均」的精英樣本
            online_dataset.add_and_save([~pattern, stack])
    else: 取快取結果, jump += 1                          # 重複 pattern 不重跑

    # (C) 更新 min_loss/best_epoch（評估，不訓網路）
    # (D) 更新 G（唯一更新 G 處）
    pred = smodel(pattern.series)                        # ★ SM（可微）預測
    loss = pred.criterion() + TV + island + sc*w + gap*w   # 結構正則預設權重 0
    loss.backward()                                      # ★ 梯度經 SM 反傳回 G
    G.step(scheduler_param=real_loss)                    # ★ Adam 更新 G；ACP 吃 real_loss 調 lr/tau

    # (E) 收尾：simulator.end()/clean()；TEMP.save()；存 6 宮格圖
```
**誰更新誰**：更新 SM＝(B) `train_one_data` ＋ rollback 時 `train_by_datas`；更新 G＝只有 (D) 經可微 SM 反傳；ACP＝(D) 內 `G.step` 調 lr/tau；rollback(A) 與 ACP 各自獨立計數器。

---

## 二、ACP 排程、二值化與 DLF

### 1. ACP 排程器（`AdaptiveCyclicalScheduler`，`functions.py:248-468`）
OneCycle(暖身)+CosineAnnealingWarmRestarts(週期退火)+ReduceLROnPlateau(自適應重啟) 三合一，**同時排 lr 與 tau**。單埠參數（`train_single.py:156-169`）：
`T_0=100, T_mult=1, lr_max=0.005, lr_min=1e-6, temp_max=4.0, temp_min=0.1, warmup_ratio=0.2, patience=25, factor=0.7, on_plateau='linear'`。

退火公式（`warmup_steps=int(T_i·0.2)`、`T_cur`=週期內計步）：
- 暖身（`T_cur < warmup_steps`）：`lr/tau = min + (max−min)·(T_cur/warmup_steps)`（線性爬升）。
- 餘弦（`p=(T_cur−warmup_steps)/(T_i−warmup_steps)`）：`lr/tau = min + (max−min)·(1+cos(π·p))/2`。

→ **lr 與 tau 完全同步、同形狀**：週期開頭衝到 (lr_max, temp_max)、再餘弦降到 (lr_min, temp_min)。**探索強度 = tau 高低 = 週期相位**。tau 透過 `AntennaPattern.tau = get_temp()`（`functions.py:425`）寫進全域類別屬性傳給二值化。

warm restart：`metric=real_loss` 連續 `patience(25)` 步沒贏過 best → `T_i=max(int(T_i·0.7), 50)`，依 `on_plateau` 決定重啟位置（`'linear'`=從目前高度線性續爬回峰值）。

### 2. 二值化 / tau（STE，`__init__.py:677-723`）
給 logits `pattern`、溫度 `tau`：
1. tau 夾下限 1e-4；logits clamp `[−10, 10]`。
2. `threshold = pattern.mean().detach()`（**動態用該張的均值當門檻，非固定 0.5**）；`steepness = 1/tau`。
3. `soft = sigmoid(steepness·(pattern − threshold))`；`hard = round(soft)`。
4. STE 接合：`binary = (hard − soft).detach() + soft` → **forward 走 hard(0/1)、backward 走 soft 梯度**。tau 越小 → 越陡 → 越接近硬階梯。

### 3. DLF（Dynamic Loss Filter）—— 單埠 vs 雙埠不同（關鍵）
- **單埠 = 「寫入時」elite gating**（`train_single.py:276`）：`if real_loss < TEMP.average('real_loss'): online_dataset.add(...)`。門檻 λ = 累計平均 real_loss；只有優於平均才進池。rollback 時 `train_by_datas(online_dataset)` 用這個精英集重訓 SM → SM 被導向好 pattern 區。
- **雙埠 = 論文版「重訓時」filter**（`train_dual.py:245`）：`train_by_datas(online_dataset.filter(upper=TEMP.average('real_loss')))` → 每次 rollback 對整池動態重切「loss ≤ 當下平均」的子集。
- ⚠️ 對應我們 CLAUDE.md 記的「論文版 DLF rollback filter 未移植到單埠」——它本就**只在雙埠**；單埠用簡化版。論文「>50% 改善」**code 內查無實證**（屬論文宣稱）。

### 4. Rollback（`train_single.py:238-256`，獨立於 ACP）
- 觸發：`TEMP.early_stop('real_loss', patience=10)`（最近 10 步全部沒贏過視窗前最佳）。
- 動作：G 權重 `change(找回 real_loss 最小的 epoch, load=True)` → `train_by_datas(online_dataset)` 重訓 SM → 用回滾後的 G 重新生成。（`mutate` 那行被註解，mutation 只記錄不施作。）
- 關係：rollback↔ACP **獨立計數器**（rollback=掃 10 筆視窗、ACP=累計 25）；rollback↔DLF **串聯**（rollback 觸發時用 DLF 篩過的精英集重訓）。

---

## 三、損失函數、規格與 SM 訓練

### 3.1 損失函數
**(A) Response loss（進 G + 評估）**
- **`custom_loss_minmax`（單埠主力，`patch/__init__.py:60-83`）**：只罰沒達標的一邊。
  - `'low'`(S11)：取 `target.min()`(中央 −10dB)，只罰該 mask 中 `pred > target_low` 部分（S11 要夠低）。
  - `'high'`(Gain)：取 `target.max()`(中央 +4dB)，只罰該 mask 中 `pred < target_high` 部分（Gain 要夠高）。
  - 基礎 `SmoothL1Loss`；完全達標 → 0。
- G 總 response loss = `MultiResponses.criterion()` = `custom_loss_minmax(S11,low) + custom_loss_minmax(Gain,high)`（**等權相加，無 λ**，`__init__.py:187-190`）。
- 其他（單埠未掛）：`custom_loss_r/g`（雙邊舊版）、`interval_loss`/`custom_loss_interval`（容許帶區間）。

**(B) 結構正則（進 G，壓制破碎，單埠預設權重多為 0）**
- `total_variation_loss`：罰相鄰像素跳動（連續色塊）。
- `island_suppression_loss`：罰像素與 5×5 鄰域均值的 L1 差（抑制孤島）。
- `SpectralConnectivityLoss`（`functions.py:471`）：Fiedler value λ₂（拉普拉斯第二小特徵值）衡量連通性，`loss=1/λ₂`（重，每筆做 eigendecomp）。
- `GapClosingLoss`：形態學閉運算填裂縫，罰原圖與閉運算差。

**(C) `FeedReachability`(R_feed，`functions.py:546`)**：連通元件分析判斷饋電點是否在同一金屬塊；**評估指標、無梯度、不進 loss**。

### 3.2 規格 spec / target（`__init__.py:211-244`）
目標曲線 = 對稱梯形 `[side×w0 | 斜坡 | center×w2 | 斜坡 | side×w4]`，五段點數總和 = 17。單埠實際值（`train_single.py:122,129`）：

| label | side | center | width | 含義 |
|---|---|---|---|---|
| S11 | 0 | −10 | (5,0,7,0,5) | 中央 7 點(≈27–29GHz) ≤ −10dB（斜坡 0=硬階梯） |
| Gain | −19 | 4 | (5,0,7,0,5) | 中央 7 點 ≥ 4dB |

頻率 24–32GHz、17 點、中心 28GHz。**沒有獨立 spec-pass 判準**：「達標」≡ response loss → 0。

### 3.3 SM 線上訓練（`smodels.py`）
- **`train_one_data`（單筆過擬合到收斂，`:123-175`）**：對同一筆反覆訓到 `loss < min_loss(0.1)` 或 `epoch ≥ max_epoch(20000)`。每次 HFSS 跑完新樣本即呼叫，讓 SM 立刻「背起來」。**= 我們 CLAUDE.md 標的「激進過擬合單點」源頭**（易梯度爆炸/NaN）。
- **`train_by_datas`（整批，`:63-121`）**：DataLoader mini-batch、預設 100 epoch、early-stop(patience=epochs//2)。用於離線預訓練 + rollback 重訓。

### 3.4 資料管理
- `DataManager`（`data.py:150`）：`list[(pattern, response)]`，是 PyTorch Dataset；`add_and_save` 用 `make_hashable`(Tensor→bytes) 去重；`filter(upper=...)` 回 Subset（雙埠 DLF 用）。
- `Record`（`utils.py:719`，**核心非 legacy**）：`defaultdict(list)`，`rec['k']=v` 是 append、`rec('k')` 取最後值；`find/index`（rollback 找最佳 epoch、pattern 去重查 buffer）、`early_stop`、`average`。
- `Checkpoint`（`types.py:105`）：`{model, optimizer, scheduler, record}_state_dict`，逐 epoch 存。

### 3.5 評估 / 收斂判斷
- `real_loss` = HFSS 真響應代入 `criterion()` → **評估天線好壞的唯一客觀數字**（越接近 0 越達標）。
- `min_loss`/`best_epoch` 追蹤；`early_stop('real_loss', 10)` 觸發 rollback；最終 `Min Loss: min(real_loss)` 當成績。
- **無自動停機門檻、無 spec margin 指標**；收斂判斷靠人看圖 + real_loss 趨勢。

---

## 四、核心 vs 我們的工作（哪些可能是無效工作）+ redesign 啟示

> 這節是我（彙整者）對照我們 repo 後加的判斷，供 redesign 決策。

### A. 學長方法的「真正功臣」（三份分析交叉印證）
1. 單埠主路徑**只有 `SigmoidGEN` + `OldSM`（純 MLP）兩個類別在動**；其餘一堆 GEN/SM 變體全是死碼。
2. 讓它「能動」的不是 G 的花俏，是 **SM 資料管線**三件套：
   - **SM 線上即時更新**（`train_one_data` 把每筆新 HFSS 真值背起來）；
   - **elite 篩選**（`real_loss < 累計平均` 才進 `online_dataset`）；
   - **rollback**（連 10 步沒進步 → G 退回最佳 epoch + 用 elite 集重訓 SM）跳出停滯。
3. ACP（lr+tau 同步退火 + warm restart）只控探索節奏，是相對次要的旋鈕。
4. spec 很硬（中心 7 點 S11≤−10 且 Gain≥4），達標＝loss→0，**他也沒有客觀 benchmark**（人看圖）。

### B. 我們相對學長新增的，哪些有效 / 可能無效
- **有效（基礎工程，留）**：config 驅動、`RunState`/`SampleStore`（檔案制資料層）、TensorBoard、HFSS 容錯、golden 測試、方向圖 rad head、**`sm_harvest`（更好的 SM 初始化，實測把 best_loss 砍半 6.6→3.1）**。
- **可能無效（這次 benchmark 打臉）**：
  - **生成器變體（mirror / multiscale / batch_latent=zbatch）**：benchmark 顯示三支全部**輸給 random best-of-N**，且學長單一 `SigmoidGEN` 效能與我們相近 → 在「生成器架構」上的功**大半無效**。G 本來就是單一 pattern 的超特徵，換骨幹不改命。
  - **boundary-gated ACP / candidate_repulsion / diversity**：建立在「靠 G/latent 找多樣 pattern」的前提，而該前提（zbatch）已被否定；bgate 實測幾乎沒觸發（restart_suppressed≈0）。價值待重估。
- **還沒做、但 benchmark 指出才是關鍵**：**SM 品質 / active learning**。學長靠「elite + rollback 重訓」勉強讓 SM 局部可信；我們加了 replay/dlf 但**還沒做「主動探 SM 不確定處」**。

### C. 對 redesign（SM-only / diffusion-guidance）的啟示
- 既然 G 是單一 pattern 的過參數化、且不是功臣 → **SM-only（直接優化 pattern logits、loss 當 guidance）是合理簡化**；「要更多 pattern」用**多個獨立 pattern 並行**（pattern 空間多樣，非 latent，避免 zbatch 的塌縮）。
- **crux 不變**：guidance 來自 SM，SM 不準就沒用（benchmark 已證 SM 在誤導）。學長的 **elite + rollback** 就是「讓 SM 局部可信」的最小機制——SM-only redesign 必須**保留並強化這條**（active learning / 探不確定處），否則一樣輸 random。
- **可直接沿用**：STE 二值化（+ 動態 mean 門檻 + tau 退火）、單邊 `minmax` loss、spec 梯形定義、`real_loss` 評估。

### D. 待確認（誠實標註）
- 響應維度：single 註冊 2 label → `size()`=(2,17)=34，但有處 `target.concat()` 出現 51=3×17、`HFSSNet` 預設 (3,17)——對齊張量時需實測釐清（不影響方法描述）。
- 論文「DLF >50% 改善」code 內無實證。
- mutation 在此版未啟用（只記錄）。

---

**追溯檔案**（皆 senior repo, branch `GAN`）：`train_single.py`、`train_dual.py`、`antenna/models.py`、`antenna/smodels.py`、`antenna/functions.py`、`antenna/__init__.py`、`antenna/patch/__init__.py`、`antenna/patch/patch_simulator/single_port.py`、`antenna/utils/data.py`、`antenna/utils/utils.py`(Record)、`antenna/ranger.py`、`antenna/types.py`、`KuoHung.py`。
