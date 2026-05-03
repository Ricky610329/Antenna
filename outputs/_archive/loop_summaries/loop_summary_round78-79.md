# /loop Round 78–79 兩輪總結 — Function MAE ≠ Gradient Quality (核心發現)

> R76 寫 patch methodology doc，R77 暴露 surrogate-in-loop adversarial gap，
> R78 確認不是 OOD 問題，R79 直接量測證實 **gradient quality 是真正 bottleneck**。

## TL;DR

**Surrogate 預測函數值 MAE 3.3 dB（OK），但 gradient cosine similarity 0.001（隨機）**

→ 「random test MAE 好」≠「GD-through-surrogate work」

→ Patch methodology 必須加 **Sobolev training** 或 **gradient supervision**

## R78 — 排除 OOD hypothesis

把 surrogate-in-loop GD 改成 in-distribution hard binary STE input：

| Setup | Mean gap |
|-------|----------|
| R77 soft sigmoid (OOD continuous) | +13.54 dB |
| **R78 hard binary STE (in-distribution)** | **+15.30 dB** （更差）|

→ NOT OOD issue。Hard binary 也失敗。

## R79 — 直接量測 Gradient Quality

對 6 個 test configs × 5 in-distribution optimized patterns 跑 30 samples，
比較 surrogate gradient vs real-sim gradient（autograd ground truth）。

### 結果

| 配置 | cos sim | rel err | func err |
|------|--------|--------|----------|
| 38G n=31 0° w=20 | +0.014 | 1.000 | 4.74 |
| 38G n=31 -30° w=10 | -0.008 | 1.000 | 3.25 |
| 28G n=31 0° w=20 | -0.013 | 1.000 | 2.96 |
| 38G n=41 0° w=10 | -0.005 | 1.000 | 2.66 |
| 38G n=31 30° w=30 | +0.016 | 1.000 | 3.37 |
| 28G n=21 0° w=20 | -0.009 | 1.000 | 3.30 |
| **Mean** | **+0.001** | **1.000** | **3.30** |

**Cosine similarity ≈ 0** = surrogate gradient 跟真實 gradient 完全 uncorrelated。

**Function error 3.30 dB** 匹配 R72 test MAE 2.57 dB → surrogate 預測 value 是 OK 的。

### Pass Criteria（R76 protocol）

- cosine similarity > 0.7 → **FAIL**（達 0.001）
- relative error < 0.5 → **FAIL**（達 1.0）

R72 surrogate **完全不能拿來做 GD optimization**，但**可以拿來評估設計**（function value OK）。

## 為什麼 Function 準但 Gradient 隨機

```
Standard NN training:
  loss = MSE(f̂(x), f(x))

Optimization minimizes:
  ‖f̂(x) - f(x)‖²  ← function value 準

但 NOT minimize:
  ‖∇f̂(x) - ∇f(x)‖²  ← gradient 沒 constrain

NN gradient 是 architecture × initialization 的 byproduct，
跟真實 ∇f 沒直接關係。

物理理解:
  RIS response 對 binary pattern 是 high-frequency function
  (一個 bit flip 改變整個 grating lobe distribution)
  CNN with 32 channels smooth 掉高頻
  → mean function value OK, gradient (high-freq 成分) 完全錯
```

## 對 Patch Methodology 的核心結論

R76 doc 說「surrogate MAE < 1 dB → 可 deploy」是**錯的**。
R77/R78/R79 證明 even in-distribution 也是 random gradient。

正確的 protocol（已加進 PATCH_METHODOLOGY.md R78 補充）：

```python
# Necessary AND sufficient surrogate validation
test_x = sample_diverse(n=20)
for x in test_x:
    f_true, ∇f_true = compute_via_hfss(x)
    f_pred, ∇f_pred = compute_via_surrogate(x)
    
    # Function quality (necessary):
    function_mae = ‖f_true - f_pred‖
    
    # Gradient quality (sufficient for GD-deploy):
    gradient_cos = cosine_similarity(∇f_true, ∇f_pred)
    gradient_rel = ‖∇f_true - ∇f_pred‖ / ‖∇f_true‖

# Pass criteria for surrogate-in-loop deployment:
#   function_mae < 1 dB
#   gradient_cos > 0.7
#   gradient_rel < 0.5

# 如果 gradient quality fail:
#   選項 A: Sobolev training (loss = MSE(f) + λ·MSE(∇f))
#   選項 B: 大量 + diverse 資料 (高頻成分被 fit)
#   選項 C: Physics-informed NN (EM 方程 inductive bias)
#   選項 D: Active learning, 不靠 pure surrogate-in-loop
```

### Sobolev Training 是患核心解

訓練時加 gradient supervision:

```python
# Forward
f_pred = surrogate(x)
loss_value = MSE(f_pred, f_true)

# Gradient supervision (需 finite-diff or autograd ground truth)
g_pred = autograd(f_pred.sum(), x, create_graph=True)
g_true = compute_gradient_ground_truth(x)
loss_grad = MSE(g_pred, g_true)

total_loss = loss_value + λ * loss_grad
```

對 HFSS 直接 finite-diff 算 gradient 慢但可行。每個 sample 需 (geometry_dim) 次
HFSS 跑。Patch geometry 通常 < 30 維，可接受。

## 紀錄歷程更新

| 階段 | 結果 | 教訓 |
|------|------|------|
| R57-R63 | +30.99 max-max | 虛胖 metric |
| R64-R65 | +6.88 worst-case | metric 必須 reflect physics |
| R66-R72 | scaling N^-1.62 | 200-300 entries 達 < 1 dB MAE |
| R71 | hamming 51.72% | multimodal mapping |
| R73-R75 | generator 失敗 | binary RIS hardest case |
| R76 | methodology distill | 5 原則 + 4 templates |
| R77 | surrogate-loop adversarial | function MAE ≠ deploy quality |
| **R78** | **rule out OOD** | **hard binary 也失敗** |
| **R79** | **gradient quality 量測** | **cos sim 0.001, gradient 隨機** |

## 累計（79 rounds, 116+ commits）

關鍵 deliverables:
- `script/PATCH_METHODOLOGY.md` (R76, R77/R78/R79 update)
- 2 datasets, 4 surrogate variants
- Comprehensive negative findings: surrogate MAE 不夠 → 必須 Sobolev / active learning

## 下一階段建議（更新）

1. **試 Sobolev training**: 用 RIS playground 驗證 gradient supervision 能否關 cosine 0 → 0.7+
   - 可行：RIS sim 是 differentiable，autograd 給 ∇f true
   - 訓練 surrogate with loss = MSE(f) + λ·MSE(∇f)
   - 重跑 R79 量測

2. **Active learning baseline**: 不 deploy surrogate-in-loop, 用 surrogate guide HFSS sampling
   - Surrogate 預測 worst_supp + uncertainty
   - 用 Bayesian Optimization-style acquisition 選下個 HFSS 跑
   - 比 random sampling 更 sample-efficient

3. **直接 patch transition**: 帶上完整 R76+R77+R78+R79 diagnosis
   - 不要過度信任 surrogate
   - 用 active learning + HFSS verification
   - Surrogate 只用 evaluation 不用 GD

我推薦 (1) — Sobolev training 在 RIS 上 quick test。如果改善 gradient quality，
就是 patch 移植的 critical 突破。
