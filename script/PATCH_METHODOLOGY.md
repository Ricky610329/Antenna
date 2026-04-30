# Patch Antenna Methodology — Distilled from RIS Playground (R1–R75)

> 這份文件是 RIS optimization 75 rounds 的精煉 transferable rules，給 patch
> antenna 設計與 surrogate-in-the-loop 開發直接參考。
>
> **一句話總結：用 dense supervision 訓 spatial CNN forward surrogate，
> 用 worst-case loss + multi-restart 做 per-target optimization，
> 拿 Pareto frontier dataset 涵蓋 use case 多樣性。**

---

## 核心原則（5 條）

### 1. Loss = Use Case，不是 abstract metric

❌ **錯**：用 max(main) - max(side) 作 loss。
- 在 RIS 達 +30.99 dB max-max 時，main 區 75/80 角度 < -3 dB（94% 違反帽蓋）
- 一根尖峰騙 metric，不是 deployable

✅ **對**：用 worst-case loss = min(main) - max(side)
```python
soft_main_min = -(1/β) · logsumexp(-β · resp[main])
soft_side_max =  (1/β) · logsumexp( β · resp[side])
loss = -(soft_main_min - soft_side_max)
```

對 patch：S11 跨 band 用 worst-case，不是某個 sweet 頻點過關。

### 2. Dense Supervision >> Sparse Supervision

R68/R69 實證：

| Surrogate output | training signal | result |
|-----------------|----------------|--------|
| 361-dim response (dense) | rich | MAE 4.89 dB |
| 6-dim scalar metrics (sparse) | thin | mean collapse |

✅ **對 patch**：surrogate 預測 full S(f) curve / radiation pattern，
不是 (worst_S11, BW, gain) scalars。Metrics 從 curve 後處理算。

### 3. CNN Spatial Inductive Bias 必要

R68 實證：MLP overfit (MAE 11.64 dB) vs CNN (MAE 4.89 dB)。

✅ **對 patch**：
- 把 patch geometry 編成 2D image（mask, slot positions, layer maps）
- 過 CNN 而非 flatten 餵 MLP
- Conv kernel 對應局部 EM 耦合

### 4. Multi-Restart 是必須

R44/R56 確認：lucky GD init 約 1/10 機率。Single restart 不可信。

✅ **規則**：
```python
best = None
for seed in range(5):  # minimum 5
    pat, supp = optimize(seed)
    if best is None or supp > best.supp:
        best = (pat, supp)
return best
```

### 5. 識別 Infeasible Region

R67 實證：binary RIS × wide main beam (w=30°) 物理做不到 flat-top。
不要把 loss 壓到 -∞，要承認某些 (config, target) 組合超出 hardware 能力。

✅ **對 patch**：
- 訓 surrogate 時標註 infeasible cases（HFSS 收斂失敗、impedance mismatch 太嚴重）
- Dataset 主動 cover edge cases（極窄 BW、極小 ground plane、高頻）

---

## Architecture Templates

### A. Forward Surrogate (`(geometry, config) → response`)

```python
class Surrogate(nn.Module):
    def __init__(self, n=41, config_dim=6, response_dim=361, channels=32, depth=4):
        super().__init__()
        c_in = 2 + config_dim  # pattern + mask + config_broadcast
        layers = []
        c = c_in
        for i in range(depth):
            layers += [nn.Conv2d(c, channels, 3, padding=1), nn.GELU()]
            c = channels
            if i < depth - 1:
                layers += [nn.Conv2d(c, c, 3, padding=1, stride=2), nn.GELU()]
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, 256), nn.GELU(),
            nn.Linear(256, response_dim),
        )

    def forward(self, geom, mask, config):
        b, h, w = geom.shape
        cfg_map = config[:, :, None, None].expand(-1, -1, h, w)
        x = torch.cat([geom.unsqueeze(1), mask.unsqueeze(1), cfg_map], dim=1)
        return self.head(self.conv(x))
```

**訓練**：MSE on full response。Adam lr=1e-3。

**預期 MAE**（基於 RIS surrogate scaling 外推）：
- 100 entries: ~3-4 dB MAE
- 200-300 entries: < 1 dB MAE
- 500 entries: < 0.5 dB MAE

### B. Per-Target Optimization (deployment workflow)

```python
def optimize_for_target(target_spec, n_restarts=5, surrogate=None):
    """surrogate=None → use real HFSS (slow); surrogate provided → fast."""
    best = None
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        params = nn.Parameter(init_geometry(seed))  # continuous, free-form
        opt = torch.optim.Adam([params], lr=0.05)
        for step in range(3000):
            opt.zero_grad()
            if surrogate is not None:
                resp = surrogate(params)  # fast surrogate forward
            else:
                resp = hfss_run(params)   # slow ground-truth
            loss = worst_case_loss(resp, target_spec)
            loss.backward()
            opt.step()
        # final HFSS verify (always)
        true_resp = hfss_run(final_params)
        true_supp = compute_worst(true_resp, target_spec)
        if best is None or true_supp > best.supp:
            best = (final_params, true_supp)
    return best
```

### C. Generator E2E (amortized prediction)

```python
class Generator(nn.Module):
    def __init__(self, config_dim=6, n=41, channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config_dim, 256), nn.GELU(),
            nn.Linear(256, 512), nn.GELU(),
            nn.Linear(512, channels * 8 * 8), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
    ...

# Training (E2E through surrogate, NOT direct supervised)
for batch in train_loader:
    geom_pred = generator(config_with_mode)
    resp_pred = surrogate(geom_pred)  # frozen, MAE < 1 dB ideally
    loss = worst_case_loss(resp_pred, target_spec)
    loss.backward()
    opt.step()
```

**重點**：generator 訓練不能用 supervised BCE on geometry（R73 失敗）。
必須用 surrogate-in-loop 物理 loss。

### D. Worst-Case Loss

```python
def worst_case_loss(resp, main_lo, main_hi, beta=20.0, ripple_weight=2.0):
    main = resp[..., main_lo:main_hi]
    side = torch.cat([resp[..., :main_lo], resp[..., main_hi:]], dim=-1)
    main_min = -(1/beta) * torch.logsumexp(-beta * main, dim=-1)
    side_max =  (1/beta) * torch.logsumexp( beta * side, dim=-1)
    loss = -(main_min - side_max)
    if ripple_weight > 0:
        main_max = (1/beta) * torch.logsumexp(beta * main, dim=-1)
        loss = loss + ripple_weight * (main_max - main_min)
    return loss
```

**β=20 比 β=5 更逼真 min/max**（R64 實證）。

**ripple_weight Pareto trade-off**（R65 實證）：
| rw | worst_supp | ripple | flat-top achievement |
|----|-----------|--------|---------------------|
| 0  | high (sharp peak OK) | high (10+ dB) | 0% |
| 0.5-1 | medium | medium (5 dB) | 20-30% |
| 2 | lower | low (2 dB) | 60-70% |
| 5 | minimal | minimal (1 dB) | 100% |

---

## Dataset Design

### Schema

```jsonl
{
  "entry_id": 0,
  "config": {
    "freq_ghz": 38.0,
    "geometry_params": {...},  // patch-specific
    "target_spec": {"theta_c": 0, "width_deg": 20, ...},
    "ripple_weight": 2.0
  },
  "pareto": [
    {
      "ripple_weight": 0.0,
      "metrics": {worst_supp, headline_supp, main_min, main_max, ripple, side_max, ...},
      "best_seed": 2,
      "geometry_file": "geometries/entry0000_rw0.0.npy",
      "response_file": "responses/entry0000_rw0.0.npy"
    },
    {"ripple_weight": 2.0, ...}
  ]
}
```

### 變化維度（patch-specific 估計）

```python
# 對 patch
freqs = [2.4, 5.8, 10, 28, 38]  # GHz
geometry_variants = ["rect", "circ", "slot", "cross", "U-slot"]
target_band_widths = [50, 100, 200, 500]  # MHz
target_S11_levels = [-10, -15, -20, -25]
ripple_weights = [0, 1, 2]
seeds = [0, 1, 2]

# 估算: 5 × 5 × 4 × 4 × 3 × 3 = 3600 runs
# 假設 HFSS run 5 min/sample: 300 hours = 12 days
# → 用 active learning 把 dataset 縮到 200-500 entries 達同 MAE
```

### Class Balance（必要）

R68 揭露 surrogate 對 rw=0 systematic under-predict。

✅ **規則**：dataset 必須 covered:
- 多 ripple weight（不能只 rw=0 或只 rw=2）
- Off-axis target (-30, 0, +30°+)
- 寬窄 main beam（10°-30°）
- 不同 frequency

---

## RIS-Only vs Transferable

### ✅ Transferable to Patch

- Worst-case loss（核心）
- Dense supervision（核心）
- CNN architecture（核心）
- Multi-restart workflow（核心）
- Pareto frontier dataset（核心）
- Mode conditioning（核心）
- Surrogate-in-loop generator architecture（核心）

### ❌ RIS-Only（patch 不需）

- Bit-flip augmentation（patch 沒 binary symmetry）
- Free-phase parameterization（patch 用 geometry params 直接）
- BinarySTE（patch 不需 quantization）
- 1-bit phase quantization Pareto（patch 連續）

---

## 75-Round Highlights（reference）

| Phase | Round | Result |
|-------|-------|--------|
| Generator path (lab original) | R1-R30 | conditioning failure (-0.46 max) |
| Sigmoid GD path | R30-R56 | +13.44 (max-max) — 卡 attraction basin |
| **Free-phase breakthrough** | **R57** | **+21.31 (max-max)** ← +7.87 algorithmic |
| Aperture scaling | R57-R63 | +30.99 (max-max, n=41) |
| **Critical revelation** | **R64** | **+30.99 真實 worst = -18.21 dB** |
| Worst-case loss | R64 | +6.88 deployable (n=41 × w=30) |
| Pareto trade-off | R65 | rw=2 達 0/30 main < -3 dB |
| Dataset_v1/v2 | R66-R72 | scaling power N^-1.62 |
| **Multimodal discovery** | **R71** | **同 config rw=0 vs rw=2 hamming 51.72%** |
| **Surrogate POC** | **R68** | **CNN MAE 4.89 → 2.57 dB (v2)** |
| Conditional generator | R73 | mode separation 40%, but BCE collapses |
| E2E generator | R74-R75 | continuous works, binary STE fails |

---

## 對 Patch Team 的具體建議

1. **Day 1**: 用上面 forward surrogate template 訓 small dataset（30-50 patch HFSS samples）。確認 CNN > MLP, dense > sparse supervision。

2. **Day 2-7**: 收 200-300 entries 涵蓋 Pareto frontier。Dataset_v1 schema 直接 reuse。

3. **Week 2**: 訓 forward surrogate 達 MAE < 1 dB（基於 N^-1.62 scaling）。

4. **Week 3**: 實作 per-target optimization through surrogate。**比 HFSS-in-loop 快 1000x**，且預期 quality close to HFSS。

5. **Week 4**: E2E generator (config + mode → geometry) 訓練。Patch 連續 geometry 應 strongly better than RIS binary。

**避免的陷阱**：
- 不要用 max-max metric 評估或當 loss
- 不要訓 surrogate 預測 scalar metrics
- 不要用 supervised BCE on geometry 訓 generator
- 不要相信 single seed 的結果
- **不要相信 surrogate test MAE 就直接 deploy**（R77 critical）
  - Surrogate test set MAE 是**隨機 sample 統計**
  - GD 過程**主動找 surrogate 弱點** → adversarial-like exploitation
  - 必須 add: Compare surrogate-loop vs HFSS-loop on small test set
  - 如果 gap > 2 dB → 不要 deploy，加 uncertainty/ensemble/active learning

---

## Surrogate Validation Protocol（必加，R77 教訓）

R77 在 RIS 上實證: surrogate test MAE 2.57 dB 看起來夠好，但 GD-through-surrogate
比 GD-through-real-sim **差 7-25 dB worst_supp**。原因是 GD 主動 exploit surrogate
prediction errors（adversarial overfitting）。

### 驗證步驟（patch deploy 前 mandatory）

```python
# 1. Train surrogate, achieve MAE < 1 dB on random test set ✓ (necessary, not sufficient)

# 2. CRITICAL: Adversarial validation
test_targets = sample_diverse_targets(n=10)
gaps = []
for target in test_targets:
    # Path A: surrogate-in-loop optimization
    pred_geom_A = optimize_through_surrogate(target, surrogate)
    # Path B: HFSS-in-loop optimization (slow but ground truth)
    pred_geom_B = optimize_through_hfss(target)
    
    # Both eval through HFSS
    true_supp_A = hfss_run(pred_geom_A).worst_case_supp(target)
    true_supp_B = hfss_run(pred_geom_B).worst_case_supp(target)
    gaps.append(true_supp_B - true_supp_A)

print(f"Mean gap: {np.mean(gaps)} dB")
# < 2 dB: surrogate-in-loop OK to deploy
# 2-5 dB: borderline, add uncertainty methods or active learning
# > 5 dB: don't deploy, expand dataset or retrain

# 3. Mitigation if gap > 2 dB:
#    a. Ensemble surrogates (variance as uncertainty proxy)
#       → reject patterns where ensemble disagrees > threshold
#    b. Active learning: HFSS-verify surrogate-loop outputs, add bad ones to training
#    c. Retrain with continuous augmentation if surrogate trained on discrete only
```

### 為什麼 random test MAE 不夠

Surrogate test set 是 i.i.d. samples。GD trajectory 不是 i.i.d. — 它沿著
gradient 方向走，主動尋找：
- Surrogate 預測高 main 但 real sim 預測低 main 的點
- Surrogate 預測低 sidelobe 但 real sim 預測高 sidelobe 的點

這些是 surrogate prediction error 的 worst-case 方向，跟 random test set 統計
分佈完全不同。

### Patch Antenna 預估

Patch surrogate 可能比 RIS 好（continuous geometry, smoother prediction），
但 adversarial gap 可能仍 1-3 dB。**必須驗證**。

---

## R85 Critical Update: Active Learning Greedy Fails

R85 直接實證 active learning **greedy acquisition** 比 random sampling 還差:

| Method | Final best worst_supp |
|--------|----------------------|
| Greedy (predict top-K) | +1.59 dB |
| Random sampling | +4.79 dB ★ |
| Pool max (oracle) | +5.57 dB |

Greedy 重複 exploit surrogate prediction errors → 越選越爛。

### MUST DO for Patch BO Loop

```python
# CORRECT: UCB with ensemble uncertainty
def patch_bo_iteration(surrogate_ensemble, dataset, candidates, k=10, kappa=1.5):
    means = []
    for surrogate in surrogate_ensemble:
        means.append(surrogate.predict(candidates))
    means = np.stack(means)
    pred_mean = means.mean(axis=0)
    pred_std = means.std(axis=0)  # ensemble variance as uncertainty
    
    ucb = pred_mean + kappa * pred_std  # acquisition
    selected_idx = ucb.argsort()[-k:]
    selected_candidates = candidates[selected_idx]
    
    # Run HFSS on selected
    new_labels = [hfss_run(c) for c in selected_candidates]
    dataset.extend(zip(selected_candidates, new_labels))
    
    # Retrain ensemble with new data
    surrogate_ensemble = retrain(dataset)
    return dataset, surrogate_ensemble
```

### NEVER DO

```python
# WRONG: greedy (R85 fail)
selected = candidates[surrogate.predict(candidates).argsort()[-k:]]
# 沒 uncertainty → 重複 exploit surrogate 弱點 → worse than random
```

### Ensemble Setup

```python
# 訓 5-10 surrogates with different seeds
ensemble = []
for seed in range(5):
    torch.manual_seed(seed)
    s = SurrogateCNN(...)
    train(s, dataset, epochs=200)
    ensemble.append(s)

# 部署時 mean + std
def ensemble_predict(ensemble, x):
    preds = torch.stack([s(x) for s in ensemble])
    return preds.mean(0), preds.std(0)
```

### R78 補充：問題根源是 GRADIENT quality 不是 function quality

R78 把 surrogate-in-loop GD 改成 in-distribution hard binary input（不是 R77 的
soft continuous），gap 反而**更大**（15.30 vs 13.54 dB）。

**這證明 R77 失敗不是 OOD issue，是 surrogate 的 gradient quality 不夠。**

```
Surrogate quality measures:
  ‖f - f̂‖∞       function value MAE        ← random test set 評估
  ‖∇f - ∇f̂‖∞     gradient MAE              ← GD 用, 才是 deployment 真實 bottleneck

NN 預測值 OK 但 gradient 可能完全錯。
GD 走的是 ∂loss/∂params, 不是 loss 本身。
```

### Surrogate Gradient Quality 評估方法（patch deploy 前必跑）

```python
# 用 finite difference 測 ground-truth gradient
def true_gradient(x, sim, eps=1e-3):
    """∂response/∂x via finite difference."""
    grads = []
    for i in range(x.shape[0]):
        x_p = x.clone(); x_p[i] += eps
        x_m = x.clone(); x_m[i] -= eps
        grad = (sim(x_p) - sim(x_m)) / (2 * eps)
        grads.append(grad)
    return torch.stack(grads)

# Compare surrogate gradient vs true gradient
test_points = sample_geometries(n=20)  # diverse
for x in test_points:
    g_true = true_gradient(x, hfss_run)
    x_t = torch.tensor(x, requires_grad=True)
    y_pred = surrogate(x_t).sum()
    g_pred = torch.autograd.grad(y_pred, x_t)[0]
    cos_sim = F.cosine_similarity(g_pred.flatten(), g_true.flatten(), dim=0)
    rel_err = (g_pred - g_true).norm() / g_true.norm()
    print(f"cos sim {cos_sim:.3f}, rel err {rel_err:.3f}")

# Pass criteria for deployment:
#   cosine similarity > 0.7 (gradient direction OK)
#   relative error < 0.5 (gradient magnitude OK)
# 否則 GD-through-surrogate 不可信, 即使 function MAE 好看
```

### 改善 gradient quality 的方法

1. **Sobolev training** ⚠️ R80 在 RIS 失敗：
   - Loss = MSE(f̂, f) + λ MSE(∇f̂, ∇f)
   - RIS binary 上 grad_mse 完全不降（CNN 架構限制）
   - Patch 連續幾何上**可能**有效（smoother gradient field, 待實證）
   - **不能假設 work**

2. **大量資料**: 經驗上 ~10× function MAE 級的 gradient 需要 10-100x 更多資料

3. **Physics-informed NN**: 把 EM 方程當 inductive bias

4. **Architecture 改變**: 標準 CNN 有 smoothing bias，可能需要：
   - Higher capacity (channels 32 → 128)
   - Skip connections (preserve high-frequency)
   - Fourier features (explicit high-frequency basis)

5. **不依賴 gradient: Active learning (推薦)**
   - 用 surrogate 預測 function value（OK）
   - 用 BO acquisition (UCB / EI) 選下個 HFSS sample
   - 不直接 GD-through-surrogate
   - 比 Sobolev 更可靠且 dataset-efficient

### R80 在 RIS 上的結論

| Method | Function MAE | Cos sim | 結論 |
|--------|-------------|---------|------|
| R72 vanilla CNN | 3.3 dB | 0.001 | function OK, gradient 隨機 |
| R80 + Sobolev λ=1 | 5.3 dB | 0.005 | 沒改善, function 還變差 |

**RIS binary gradient 是 pathologically high-frequency。標準 CNN 無法 represent。**

**Patch transition 不能假設 Sobolev 可救 — 可能仍需 active learning fallback。**


---

## 文件版本

- v1.0: R76 (2026-04-30)
- 基於 RIS 75 rounds 完整探索

任何問題參考 `script/RIS_RESEARCH_REPORT.md` 全紀錄 + `outputs/loop_summary_round*.md`。
