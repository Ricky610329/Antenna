# /loop Round 74–75 兩輪總結 — E2E Generator on RIS = Hardest Case

> R73 結論「需 surrogate-in-the-loop」實作。R74 用真實 RIS sim 訓 E2E generator,
> R75 加 STE 試解 quantization gap。**兩者都暴露 binary RIS 的根本困難**，但對
> patch (continuous geometry) 是正面結論。

## TL;DR

| Round | 設計 | Continuous worst | Binary worst |
|-------|------|------------------|--------------|
| R74 E2E continuous | free-phase + worst-case loss | -0.07 to -1.04 ✓ | -5.55 to -16.96 ✗ |
| R75 E2E STE | + binary STE | **-12 to -40 ✗** (loss 不收斂) | **-14 to -28 ✗** |
| R64 per-target | GD-from-scratch | n/a | **+6.88 ✓** |

→ **RIS binary 場景，per-target GD >> generator amortization**

→ **對 patch (continuous geometry)，這是 positive 結論：amortization 應 work**

## R74 — E2E Generator (continuous training)

### Architecture

```python
config (3-dim) → MLP encoder → 8×8 latent
              → conv upsample 2x → 16×16
              → conv upsample 2x → 32×32 → crop n×n
              → free-phase output ∈ ℝ (no constraint)

# Forward
free_phase = generator(config)
response = simulator(free_phase)  # differentiable, sim does pattern * π
loss = worst_case_loss(response, main_lo, main_hi, ripple_weight)

# Eval with quantization
phase = (free_phase * π) % (2π)
binary = ((phase > π/2) & (phase < 3π/2)).float()
real_response = simulator(binary)
```

### 結果

| Test config | cont worst | binary worst | binary ripple |
|-------------|-----------|--------------|---------------|
| θc=-25 w=12 rw=0 | -0.22 | -8.03 | 3.12 |
| θc=0 w=20 rw=2 | -0.56 | -16.96 | 21.81 |
| θc=20 w=25 rw=1 | -1.04 | -8.53 | 8.68 |
| θc=-15 w=10 rw=2 | -0.07 | -5.55 | 2.40 |
| θc=15 w=30 rw=0 | -0.56 | -14.29 | 12.03 |

**Continuous performance 接近 0 dB（perfectly balanced main vs side），
binary 量化破壞 carefully-balanced 解 5-17 dB。**

## R75 — STE Binary Training

### 設計

```python
class FreePhaseBinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, free_phase):
        phase = (free_phase * π) % (2π)
        return ((phase > π/2) & (phase < 3π/2)).float()
    @staticmethod
    def backward(ctx, grad):
        return grad  # straight-through

# Training
binary = FreePhaseBinarySTE.apply(free_phase)
response = simulator(binary)  # real binary deployment response
loss = worst_case_loss(...)
```

### 結果：失敗

- Loss 1500 epochs 不收斂（throughout ~+40）
- Binary worst -14 to -28 dB
- **比 R74 continuous training 還差**

### 為什麼失敗

STE 提供 gradient via free_phase，但 simulator 對 binary {0,1} 的 response 在連續
空間 highly non-smooth。STE 的 identity backward 給 generator 的 signal 是「對
某個 binary pixel 有什麼梯度」，不是「該往 0 還是 π 推」。

Gradient variance 太大，generator 無法收斂。

## R64 per-target GD 才是 RIS binary 最佳工具

R64 per-target GD-from-scratch with worst-case loss:
- 38 GHz × n=41 × broadside × w=30 → worst +6.88 dB
- 5 seeds × 1500 GD steps + SA = 5-15 min per target

對比：
- R74 generator: -5 to -17 dB binary worst (failed)
- R75 STE: -14 to -28 dB binary worst (failed worse)

**結論：對單一 target 部署，per-target GD 是答案。Generator 在 binary RIS 場景
fundamental 困難。**

## 關鍵 insight：RIS 是 hardest case，patch 應 strongly better

### RIS 特殊困難（patch 不存在）

| 因素 | RIS | Patch |
|------|-----|-------|
| Geometry 連續性 | Discrete binary {0, π} | Continuous (lengths, widths) |
| 同 spec 多 optimum | Hamming ~50% (R71) | 通常 single |
| Quantization gap | -5 to -17 dB | 不存在 |
| Loss landscape | discontinuous near 0/1 | smooth |
| Generator amortization | failed (R74/R75) | 應 work |

### 對 patch antenna 移植的策略

```
RIS playground 學到:
- Worst-case loss design ✓
- Multi-restart workflow ✓
- Pareto frontier dataset ✓
- Spatial CNN > MLP ✓
- Dense supervision (full curve) ✓
- Mode conditioning ✓

不能 transfer (RIS-specific):
- BinarySTE / quantization tricks (patch 沒)
- Free-phase parameterization (patch 用 geometry params 直接)
- Bit-flip augmentation (patch 沒對應)

Patch 的 generator workflow:
config + mode → generator → patch_geometry (continuous)
            → trained surrogate (CNN, MAE < 1 dB after dataset_v3 size)
            → predicted S11 / radiation curve
            → worst-case loss vs target spec
backprop → generator update
```

## 紀錄歷程更新

| 階段 | 焦點 | 結果 |
|------|------|------|
| R57-R63 free-phase | max-max steering | +30.99 (虛胖) |
| R64 worst-case loss | flat-top deployment | +6.88 (per-target GD) |
| R66-R67 dataset_v1 | 多 use case 涵蓋 | 36 entries Pareto |
| R68-R69 surrogate POC | forward / metric | dense > sparse supervision |
| R70-R71 visualization | symmetry / multimodal | hamming 51.72% |
| R72-R73 v2 + cond gen | scaling + multimodal hypothesis | mode separation 40% |
| **R74 E2E continuous** | generator + sim | **continuous good, binary fail** |
| **R75 STE binary** | quantization gap fix | **STE 失敗** |

## 累計（75 rounds, 112+ commits）

- 27+ scripts
- 2 datasets (v1: 72, v2: 108 Pareto rows)
- 4 surrogate variants + 2 generator variants
- 完整 methodology 文檔
- 37 round summaries

## 下一階段建議

RIS playground 的 methodology 探索 saturation。下一步應該：

1. **Patch antenna methodology transition**（用同套設計搬到 patch）
   - Build patch surrogate trainer 架構
   - 用 R72 的 dataset 設計準則收 patch HFSS data
   - E2E generator 應在 patch 上 strongly better

2. **或: dataset_v3 (200-300 entries) 達 < 1 dB MAE**
   - 確認 surrogate scaling 預測
   - 但邊際收益遞減（已知 scaling power 1.62）

3. **或: 收斂這個探索鏈，寫 paper-style 總文檔**
   - 75 rounds 的 systematic 結果都有
   - "Worst-case loss + dense supervision + spatial CNN + per-target deployment"
   - 可作為 patch 移植的 reference manual

我建議 (1) — 開始 patch transition。playground 已給夠 lessons。
