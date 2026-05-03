# /loop Round 80–81 兩輪總結 — Sobolev Fails + Ranking Fails (Cascade Negative Results)

> R79 證實 surrogate gradient 無用。R80 試 Sobolev training 失敗（架構限制）。
> R81 進一步發現 surrogate **連 ranking 都失敗**（Spearman 0.031）。
>
> **最終結論：R72 surrogate 對任何 downstream task 都不可用**。dataset 必須 cover
> 整個設計空間，不只 optimal points。

## TL;DR

| 用途 | R72 surrogate 表現 | 結論 |
|------|-------------------|------|
| Function value evaluation | MAE 2.57 dB (~32% signal range) | 看似 OK 但實際 mean predictor |
| GD gradient | Cosine 0.001 (R79) | 完全 random |
| Sobolev training fix | grad_mse stuck (R80) | NN 架構無法 represent |
| **Ranking / filter** | **Spearman 0.031 (R81)** | **跟 random 差不多** |

## R80 — Sobolev Training（試圖 Fix Gradient）

### 設計

```python
# Pre-compute ground truth ∂loss/∂pattern via autograd through real RIS sim
# Train surrogate with:
loss = MSE(f̂, f) + λ · MSE(∇f̂, ∇f)
```

### 結果（108 entries, 100 epochs, λ=1.0）

| Metric | Value |
|--------|-------|
| value_mse trajectory | 347 → 53 ✓ (function 學會) |
| **grad_mse trajectory** | **0.174 → 0.174 ✗ (完全 stuck)** |
| Cos sim final | 0.005 (沒比 R72 的 0.001 好) |

### 為什麼失敗：架構限制不是 loss 限制

```
RIS response 對 binary pattern 是 EXTREME high-frequency function
(1 bit flip 改變整個 grating lobe pattern)

CNN 32-channel 4-layer:
  - Conv kernel 內建 spatial smoothing
  - Gradient field 受 kernel 結構限制
  - 即使 supervision 提供 gradient ground truth，NN 仍 fit 不出來

→ 不是 loss bottleneck (Sobolev 已是正確 loss)
→ 是 architectural bottleneck (CNN 表達能力不夠)
```

## R81 — Surrogate as Ranking Filter（試圖 Salvage）

### 動機

R77-R80 證明 GD-through-surrogate 不行。但如果 surrogate **predicted ranking**
跟 true ranking 一致，仍可用於 BO acquisition / pre-filter。

### 結果

| Metric | Value | 解讀 |
|--------|-------|------|
| Pearson correlation | -0.027 | random |
| Spearman correlation | +0.031 | random |
| Per rw=0 Spearman | +0.021 | random |
| Per rw=2 Spearman | +0.093 | random |

### Top-K filter quality

| K | Pred top-K avg true | Random K avg | True top-K avg |
|---|---------------------|--------------|----------------|
| 5 | +1.13 | +0.62 | **+5.07** |
| 10 | +1.22 | +0.32 | +4.62 |
| 50 | +0.26 | +0.09 | +2.02 |

→ Surrogate top-5 預測 (+1.13) 比 random (+0.62) 略好，但**離真 top-5 (+5.07) 非常遠**。

### 為什麼 Ranking 也失敗

```
True worst_supp range in dataset_v2: ~-3 to +5 dB (range ~8 dB)
R72 surrogate MAE: 2.57 dB
→ MAE ≈ 32% of signal range (不是 small!)
→ Surrogate 預測 variation 跟 true variation 沒 correlation
→ 等同 mean predictor + noise

R72 surrogate 訓練資料只覆蓋 OPTIMIZED patterns:
  - 都是經 GD + SA fine-tune 的高品質結果
  - patterns 統計分佈很窄
  - response 統計分佈也窄
  - NN 學到「不論 input 都輸出 mean」的 trivial 解
```

## 最終 Patch Methodology Update

R76 doc 經 R77/R78/R79/R80/R81 五輪驗證後的**最終版本**：

### 1. Dataset Diversity 是基礎 (R81 教訓)

不能只用 optimized patterns 訓練。必須包含：
- Optimized geometries (current approach)
- Random sampled geometries (uniform / Latin hypercube)
- GD trajectory snapshots (中間 step 不只 final)
- Edge cases (extreme parameters)

對 patch: HFSS 跑 200 random + 200 optimized + 100 trajectory snapshots = 500 entries

### 2. 多層次 validation (R77/R78/R79/R81 教訓)

```python
# Necessary AND sufficient surrogate validation
def validate_surrogate(surrogate, real_sim_or_hfss):
    # 1. Function value MAE
    func_mae = test_set_mae(surrogate, real_sim_or_hfss)
    # Pass: < 1 dB (necessary)
    
    # 2. Ranking quality (R81)
    spearman = ranking_correlation(surrogate, real_sim_or_hfss)
    # Pass: > 0.7 (for use as filter)
    
    # 3. Gradient quality (R79)
    cos_sim = gradient_cosine_similarity(surrogate, real_sim_or_hfss)
    # Pass: > 0.7 (for GD-through-surrogate)
    
    # 4. Adversarial (R77)
    gap = adversarial_optimization_gap(surrogate, real_sim_or_hfss)
    # Pass: < 2 dB (for deployment)
    
    return {
        "deployment_ready": all 4 pass,
        "filter_only": ranking pass but gradient fail,
        "evaluation_only": function pass but ranking fail,
        "useless": none pass,
    }
```

### 3. Active Learning (R81 推薦)

如果 surrogate 沒過 deployment_ready 但 dataset 大小不允許大幅 expand：

```python
# Active learning loop
for iteration in range(max_iters):
    # 1. Random sample N candidates
    candidates = sample_geometries(N=1000)
    
    # 2. Use surrogate to RANK + estimate uncertainty
    pred_means, pred_stds = ensemble_predict(surrogate, candidates)
    
    # 3. UCB acquisition
    ucb = pred_means + κ * pred_stds
    
    # 4. Run HFSS on top-K (most promising)
    selected = candidates[ucb.argsort()[-K:]]
    true_responses = [hfss_run(c) for c in selected]
    
    # 5. Add to dataset, retrain
    dataset.extend(zip(selected, true_responses))
    surrogate = retrain(dataset)
    
    # Stop when surrogate validates
    if validate_surrogate(surrogate)["deployment_ready"]:
        break
```

### 4. Final Deployment Strategy

對 patch antenna：

```
PRIMARY (slow but reliable):
  - HFSS-direct optimization with worst-case loss
  - Multi-restart 5+ seeds
  - 用 finite-difference gradient (一次 ~30 HFSS runs/iter, 3000 iters = 90k runs/seed)
  - 慢但 ground truth, 結果可信

ACCELERATION (use surrogate carefully):
  - Surrogate 預測 + uncertainty 用於 BO acquisition
  - 篩選 candidate 跑 HFSS (1-10x speedup vs random)
  - NEVER 直接 GD-through-surrogate (R77/R79/R81)
  - NEVER 信任 surrogate scalar metric output (R69 mean collapse)
```

## 紀錄歷程最終更新

| Round | 結果 | 教訓 |
|-------|------|------|
| R57-R63 | +30.99 max-max | 虛胖 metric |
| R64-R65 | +6.88 worst-case | metric 必須 reflect physics |
| R66-R72 | scaling N^-1.62 | 但 MAE 對 GD 不夠 |
| R71 | hamming 51.72% | multimodal mapping |
| R73-R75 | generator failures | binary RIS hardest case |
| R76 | methodology distill | 5 原則 + 4 templates |
| R77 | surrogate-loop adversarial | function MAE ≠ deploy |
| R78 | rule out OOD | gradient quality 是真 bottleneck |
| R79 | gradient quality 量測 | cos 0.001, 完全隨機 |
| **R80** | **Sobolev 失敗** | **架構限制不是 loss 限制** |
| **R81** | **ranking 失敗** | **MAE 2.57 = mean predictor + noise** |

## 累計（81 rounds, 118+ commits）

完整 negative findings cascade:
1. Function MAE alone 不夠 (R77)
2. OOD 不是主因 (R78)
3. Gradient quality 是 bottleneck (R79)
4. Sobolev 救不回 (R80, architectural)
5. **連 Ranking 都不行** (R81)
6. → 必須 dataset diversity + active learning

## 對 Patch Team 最終 Action Items

```
Week 1-2: Dataset Construction
  - HFSS run 500 patches: 200 random + 200 optimized + 100 trajectory snapshots
  - Cover Pareto frontier (multiple ripple_weight)
  - Class balance + edge cases

Week 3: Surrogate Training
  - CNN forward (full S-curve)
  - Sobolev attempt (可能 fail like RIS, 但 patch smoother)
  - 多 architecture 試 (Fourier features, skip connections)

Week 4: Validation
  - 4-tier validation (function/ranking/gradient/adversarial)
  - 如果 pass deployment_ready: 用 surrogate-in-loop optimization
  - 如果 only filter pass: 用 BO acquisition + HFSS
  - 如果 only evaluation pass: surrogate 只做 design space exploration
  - 如果 useless: 加 dataset, retrain

Week 5+: Deployment
  - Per-target HFSS optimization (primary)
  - Surrogate-accelerated active learning (where applicable)
```

## 結論

75 rounds RIS playground exhaustively explored surrogate-in-loop methodology。
**R81 的 ranking failure 是最嚴厲的 negative result**——連 BO/active learning
這個 backup plan 都不能保證 work。

對 patch team 的核心訊息：
1. **不要相信 random test MAE** — 必須 4-tier validation
2. **準備 large diverse dataset** — 不只 optimized geometries
3. **HFSS-direct 才是 primary deployment** — surrogate 是加速但不可信任 alone
4. **Active learning 不是 free lunch** — 需要 surrogate 至少 ranking 成功

如果 ranking 都 fail，patch 必須走 HFSS-heavy 路線。
