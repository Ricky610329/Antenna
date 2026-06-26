# Generator-free SM-guided 搜尋 + 閉迴路信任控制（設計）

> 新主線的設計文件。對應 configs：`single_guided_harvest.yaml`（Exp1）/ `single_guided_ens_harvest.yaml`（Exp2）/ `single_guided_ens_adapt_harvest.yaml`（Exp3）。
> 程式：`antenna/models/generators.py:DirectPatternGenerator`、`antenna/models/surrogates.py:EnsembleSurrogate`、`antenna/training.py:TrustController`。
> 對照文件：`senior_method.md`（學長方法）、`research_landscape.md`（文獻地景）。

## 0. 一句話

把優化變數從「生成器 G 的權重」換成「pattern 本身的 logits」，對 surrogate（SM）做 guided 梯度下降；SC/boundary/不確定性當控制項；用「SM vs 真實 HFSS 的落差」閉迴路調節探索力度。**不一定要 generator。**

## 1. 為什麼（病根）

- **瓶頸是 SM，不是 G。** 固定 spec 下 `G(spec)→pattern` 只是「一張 pattern 的過參數化（超特徵）」——不是真生成模型。靠同批 K 個 latent 找「更多 pattern」策略上就錯（latent 會塌縮，實測 `score_spread→0`）。
- **benchmark 顯示我們輸 random best-of-N**（同 HFSS 預算）。文獻定論：單一 spec + 同一模擬器，對「學來的 NN surrogate」做梯度下降會收斂到 surrogate 的對抗洞（SM 說好、真 EM 爛）。**「直接找 SM 認為最好的 pattern」字面上就是鑽洞的定義。**
- 所以救命的是「**SM 對不上真實時就調整探索力度**」這個閉迴路——不是配菜，是主菜。

## 2. generator-free 的 reframe

- 優化變數 = `nn.Parameter(K, out_dim)`（K 個獨立 pattern logits），forward 只過 `BiScaleNorm`（與其他 G 同尺度 → ACP 的 tau 語意一致）。**無 MLP。**
- 下游全沿用既有管線：STE 二值化（tau 由 ACP 控）→ SM → loss → HFSS → DLF/replay 線上更新 SM。K>1 走既有多候選路徑（生成 K → `_select_best` 選一張送 HFSS → 聚合 loss + 候選互斥）。
- **K 個獨立候選 = pattern 空間的多樣**（非 latent 雲）；逐列各自被 SM 梯度引導、彼此用 `candidate_repulsion` 互斥。
- 取捨：拿掉 G 失去 DIP 隱式平滑先驗（CNN 重參數化的好處），但也甩掉它「放大鑽 SM 假洞」的副作用；平滑/連貫改由**顯式 loss（SC 為主）**補回。對二值金屬圖，物理上重要的連貫＝連通性，SC 已顯式涵蓋。

## 3. 控制迴路設計（核心）

**不要把 tau、λ_trust、κ 當三個獨立旋鈕。** 它們都在同一條「探索↔利用 / 不信↔信」軸上（＝學長 ACP「lr+tau 綁同一曲線」的精神），差別在 ACP 用**預設時間表**開迴路地走，我們用**真訊號（SM vs HFSS 落差）**閉迴路地走。

### 狀態變數：單一信任標量 `t ∈ [0,1]`

```
gap      = |SM 預測 loss − 真實 HFSS loss|        # 必須在「線上訓練 SM 之前」量
                                                  # （訓練後 SM 已擬合本筆、gap 被抹平成假 0）
gap_ema  = EMA(gap)                               # 信任是逐步累積的信念，不被單筆抖動牽走
t        = clamp(exp(−gap_ema / g0), [t_min, t_max])   # 永不全信、永不全凍
```

### 致動器（全部是 t 的單調函數，同軸耦合）

| 旋鈕 | t→1（SM 可信 → 利用） | t→0（SM 失準 → 探索） | 語意 |
| --- | --- | --- | --- |
| **tau 乘子** | 1（純 ACP 退火銳化） | `tau_inflate`（放軟、保持 pattern 可塑） | 對當前二值決策的承諾程度——只在 SM 可信時才敢鎖定 |
| **λ_trust** | 0（長牽繩） | `base`（短牽繩，拉回可信區） | 多緊地待在 SM 有信心的區 |
| **κ**（acquisition） | 0（純收割 SM 最佳） | `base`（去獵不確定點修 SM） | 花 HFSS 去利用 vs 去學習 |

公式：`tau_mult = 1 + (tau_inflate−1)(1−t)`、`λ_trust = base·(1−t)`、`κ = base·(1−t)`。
`enable=False` → t 恆 1：tau_mult≡1、λ_trust/κ 退化成靜態 base（Exp2）。`enable=False 且 base=0` → 三者皆 0、tau_mult≡1 → 與原樣逐位元相同（golden 零漂移）。

### 兩個不確定性訊號，別混為一談

- **逐候選、花 HFSS 前** = ensemble 成員分歧 `u(x)=std_k SM_k(x)`：驅動內圈信任懲罰 `λ_trust·u(x)`（把每個候選推離「它自己沒把握」的地方）。語意：「我對**這張** pattern 多沒把握」。
- **全域、花 HFSS 後** = gap → 信任標量 t：驅動上面三個耦合旋鈕。語意：「SM **現在整體**多失準」。

### acquisition 的符號叉路（值得點出）

不確定性在「挑哪張送 HFSS」是**雙向**的：想安全收割 → 送 SM 說好且有信心的（`−κ·u`）；想主動學習修 SM → 送 SM 最沒把握的（`+κ·u`）。解法：**讓符號跟著信任走**——`acquisition score = sm_loss − κ·u`（分數越低越優先；高 u 候選分數被拉低 → 被選中），`κ=base·(1−t)`。**SM 不可信（t↓）→ 去獵不確定點修 SM；SM 可信（t→1）→ 直接收割預測最佳。** 同一個 t 統一了 acquisition。

### 收斂是湧現的（最漂亮的性質）

**不**用「計時器到了就退火」。系統會收斂，是因為主動學習把 SM 修準 → gap↓ → t↑ → tau_mult→1 → ACP 退火接手把 pattern 銳化鎖定。**也就是說：系統恰好在「surrogate 贏得信任的時刻、且正因為它贏得信任」才開始銳化收斂。** 這比 ACP 固定時間表更有道理，也是 DIP 缺的那個「何時停止信任 SM」的煞車。

## 4. 三個實驗（階梯，逐一隔離變因）

| | config | 在前一個上加什麼 | 隔離的變因 |
| --- | --- | --- | --- |
| Exp1 | `single_guided_harvest` | G 權重 → K 個 pattern 參數（`generator: direct`）；其餘沿用 SC/boundary/DLF/sm_harvest/rad（開迴路 ACP，單一 SM） | **有沒有 G**（對照 `single_sc_rad_boundary_harvest` 的 sigmoid G） |
| Exp2 | `single_guided_ens_harvest` | 單 SM → ensemble（`surrogate: ensemble`）＋信任懲罰 `loss.uncertainty`＋acquisition `selection.uncertainty_weight` | **ensemble 不確定性 + 信任懲罰 + UCB acquisition**（tau 仍開迴圈、λ_trust/κ 靜態） |
| Exp3 | `single_guided_ens_adapt_harvest` | 開閉迴路 `trust.enable`（gap→t→調 tau/λ_trust/κ） | **閉迴路 gap 控制 vs 開迴路固定排程** |

舊的 3 個 G-based run（`boundary`/`multiscale`/`zbatch_div`）留著當 **G-based 基線**對照。全部用同一把尺：**worst-margin(dB) vs HFSS-call 曲線，對比 random best-of-N**（贏不過 random 不算進步）。

## 5. 誠實的限制

1. **不確定性是命門。** 「SM 不準就調」若只靠事後 HFSS 落差，已花掉那次昂貴呼叫。ensemble 在花 HFSS 前給便宜的可信度 proxy；但小資料 + 共用架構時成員可能「一致地錯」（共同盲點）→ ensemble 低估誤差。緩解：成員間略改架構/加噪、且**外圈真 HFSS 仍是最終裁判**（ensemble 只當早期預警）。
2. **拿掉 G 失去 DIP 平滑先驗**：用顯式 SC（連通）補回。對二值金屬圖夠用，但若觀察到破碎解卡住，再考慮加回弱 G 或 TV。
3. **起步超參是猜的**：`g0=1.0`（落差參考尺度）/`ema=0.3`/`tau_inflate=3`/`λ_trust=0.05`/`κ=0.02`——都要在正式機用真 HFSS A/B 調（開發機 mock 估的數字不可信）。
4. **改 loss 前先討論**（CLAUDE.md 慣例）：信任懲罰/acquisition 是新增的 loss 控制項，已先討論定案才實作。

## 6. golden 安全

所有新路徑都用「旗標 / 權重 gate」隔離：`generator: sigmoid` 走單張原路；BatchLatent 仍由 isinstance 命中（行為逐位元不變）；ensemble/trust/uncertainty 在 `hasattr/weight>0/enable` gate 後才生效。golden 測試（sigmoid 單張、無 ensemble、無 trust）逐位元同原樣。實測 `pytest tests/ -q` 全綠、golden 零漂移。
