# /loop Round 142–143 兩輪總結 — Surrogate scaffolding negative results

## TL;DR

Phase 2 開場 — Surrogate-in-the-loop 第一次嘗試 **失敗**。R142 標準 CNN 學不到
analytical RIS sim（R² ≈ 0），R143 物理感知架構修正 input transform 與 lr 後
training loss 開始下降但仍 overfit 嚴重（R² = -0.74）。

這是 **scientifically valuable 的負面結果**：surrogate-in-the-loop 比預期難，
這種 risk 應在投入 patch sim 之前發現。

## 為何 Phase 2 開於此？

Phase 1 (R94-R141) 完成了 1-bit RIS 端到端 deployment-ready API。下一步
patch transition 的 critical question：**recipe + loss + joint early-stop
在 surrogate gradient 下還 work 嗎？**

R142-R143 嘗試最簡形式：
- 在 RIS 上訓練一個 surrogate（學 analytical sim）
- 用 surrogate gradient 跑 optimizer
- 比較產生的 pattern 在 analytical truth 上的 metrics

如果連 RIS 上 surrogate-loop 都 work 不了，patch transition 風險就高很多。

## R142 — 標準 CNN 失敗

設計：3-layer CNN (16-32-32 channels) + AdaptiveAvgPool + 2-layer MLP
→ 1.2M params (initial) / 356K params (downsized)。

第一次：1K samples, 1.2M params → 嚴重 overfit（train MSE 1.5, test 73, R² = -1.32）
第二次（修）：10K samples, 356K params, dropout 0.3, batch 128, early-stop

| 結果 | 第一次 | 第二次 |
|------|--------|--------|
| R² | -1.32 | **0.00** |
| Mean abs err | 6.7 dB | 4.3 dB |
| 訓練收斂 | overfit | **完全 stuck on mean** |

第二次 training loss 從 epoch 1 (45) 落到 32，然後 stuck。Model 直接學「預測平均響應」。

## R143 — 物理感知架構

### 設計 motivation

Analytical RIS sim 結構：
```
phase = pattern * π        # binary 0/1 -> phase 0/π
amp = exp(j*phase)         # +1 or -1
F(θ) = sum amp[i,j] * steering_complex[i,j,θ]
power(θ) = |F(θ)|²
response(θ) = 10 * log10(power)
```

對應神經網路結構：
```python
input: (1 - 2*pattern)        # binary -> +1/-1 amplitude
real_lin, imag_lin: (n*n -> 361)
power = real² + imag²
output = 10 * log10(power) + db_bias
```

理論上 model 完全有能力學出 analytical sim 的精確 weights。

### 實驗

| Variant | R² | Train MSE | Test MSE |
|---------|------|-----------|----------|
| lr=3e-3, raw input | -1.69 | 113 | 117 (diverging) |
| lr=3e-4, (1-2*p) input | **-0.74** | 26 | 55 |

第二次 training loss 從 86 → 26 over 300 epochs，但 test loss 卡在 55，
train/test gap 大 → 嚴重 overfit。

## 失敗原因 diagnosis

1. **數據分佈問題**：random binary patterns 產生極變化大的 response（main beam 隨機落點），
   多數 patterns 是「壞解」。Optimization trajectories 的數據分佈完全不同
   （都是「往好解前進」的 pattern）。surrogate 學了「平均隨機 pattern 的響應」對
   optimization 沒幫助。

2. **架構雖對但太大**：695K params 對 10K samples 就是 70:1，過擬合無可避免。
   即便結構正確，從 random init 找到 analytical sim 的精確 weights 是困難 cold start 問題。

3. **Log gradient 病態**：response = 10·log10(power)，當 power 接近 0 時（response = -∞），
   gradient 爆炸。這在 sidelobe 區域常發生。

4. **沒有 inductive bias**：analytical sim 有空間 translation symmetry、conjugate symmetry
   等結構，model 完全沒利用。

## Patch Transition 的 Implication

這是非常 valuable 的早期警告：
- 訓 surrogate（不論 RIS, patch, 或 HFSS）需要 careful data curation
- 隨機數據點 train 出的 surrogate 在 optimization-loop 中可能完全 useless
- 應該訓練 on **optimization trajectory snapshots** — 與真實 inference distribution 對齊
- 或從 analytical 模型的 weights 做 warm-start
- HFSS data 是 expensive 的，必須 carefully selected

R141 的 deployment API 仍然 valid — Phase 1 結論不受影響。但 Phase 2 的進度
比預期慢，需要 R144+ 多輪 iterate。

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R142 | 標準 CNN surrogate 失敗（R² ~ 0, predicting mean） |
| R143 | 物理感知架構：訓練 loss 下降但 overfit (R² = -0.74) |

## 下一階段建議 (R144+)

優先順序：

1. **R144**: 用 optimization trajectory data 重訓 surrogate
   - 跑 100 次 R141 optimization，收集中途 50 步 snapshot pattern
   - 5000+ trajectory snapshots，distribution 對齊
   - 重訓 R143 的 physics-aware 模型
   
2. **R145**: 若 R144 過 R²>0.8，跑 surrogate-loop optimizer
   - 用 surrogate gradient 跑 R141 recipe
   - 評估產生的 pattern 在 analytical truth 上的 worst/flat-top metrics

3. **R146+**: 如果 surrogate-loop work，繼續 patch surrogate 訓練 + HFSS validation

4. **降級 fallback**：如 surrogate 一直 train 不起來，可考慮
   **gradient-free optimizer** (CMA-ES) 直接用 analytical sim — 對 patch 來說
   trade-off 是 sim cost vs surrogate quality。

## 結論

Phase 2 開場碰壁不是失敗，是 **methodology validation 提早 surface 真實 risk**。
這個 risk 在 patch transition 才被發現的話，已經投入大量 HFSS sim time 了。
現在發現相對便宜（兩輪 RIS surrogate 嘗試，~80s 訓練 GPU 時間）。

R141 deployment API 仍然 production-ready 對 analytical RIS sim。Phase 2 的
正確 framing 是「在 RIS playground 上 derisk surrogate transferability，
再進 patch」— 而不是「順利套到 surrogate」。
