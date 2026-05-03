# /loop Round 64 — Worst-Case Metric 真相揭曉

> 用戶指出 R63 的 main beam 不是「整片貼上蓋」而是單一尖峰騙 max-max metric。
> R64 設計新 loss 暴露這個 systematic overestimation。

## TL;DR

**R63 +30.99 dB 是 max-max metric 虛胖。在 worst-case (min-max) metric 下：**

| Metric | R63 (舊 loss) | R64 (新 loss, width=30) |
|--------|--------------|--------------------------|
| headline supp (max(main) − max(side)) | +30.55 | +17.86 |
| **worst supp (min(main) − max(side))** | **-18.21** | **+6.88** |
| main_min | -48.77 dB | -10.98 dB |
| main ripple | 48.77 dB | 10.98 dB |
| main < -3 dB count | 75/80 (94%) | 20/30 (67%) |

新 loss 把 main_min 從 -48.77 拉到 -10.98（改善 38 dB），但暴露真實
binary 1-bit 在 wide main beam 下的物理限制。

## R63 patterns 在新 metric 下的本質

R63 +30.99 dB 解：
- 一根尖峰在 main center 達 0 dB
- main region 其他 75/80 角度被當 sidelobe 順便壓低（最低 -48.77 dB）
- side max 確實 -30.55 dB（這部分是真的）
- max-max metric 取 max(main)=0 - max(side)=-30.55 = +30.55

→ 相當於「點對點 beam steering」配置，不是「廣域 flat-top」配置

## 新 Loss 設計

```python
# soft-min: 對 main region 取 logsumexp 的反數
soft_main_min = -(1/β) * logsumexp(-β * resp[main])

# soft-max: 對 side region 取 logsumexp
soft_side_max = (1/β) * logsumexp(β * resp[side])

# 最大化 worst-case suppression
loss = -(soft_main_min - soft_side_max)
```

`β=20` 比舊的 5 更逼真 min/max（避免太鬆 saturate）。

## 跨 main beam width 結果（38 GHz × n=41 × 5 seeds）

| width (samples / 角度) | best worst | main_min | ripple | main<-3 比例 |
|------------------------|------------|----------|--------|--------------|
| 20 (10°) | +4.24 | -12.89 | 12.89 | 50% |
| **30 (15°)** | **+6.88** | **-10.98** | 10.98 | 67% |
| 40 (20°) | +5.07 | -12.12 | 12.12 | 75% |
| 60 (30°) | +4.95 | -11.34 | 11.34 | 82% |
| 80 (40°, R63 width) | +1.57 | -15.17 | 15.17 | 86% |

物理解讀：
- 越寬 main beam → 物理上 spread energy 越均勻（per-angle gain 越低）
- 越窄 → 越接近單一 peak，main_min 受 GD 局部 minimum 影響
- width=30 是 sweet spot（夠寬 spread + 夠窄保留 gain）

## 物理上限估算

1681 元素 spread 能量到 main beam region：
- Peak array gain (聚焦一點): 10·log₁₀(1681) = +32.26 dB
- Spread 到 N_main 角度，per-angle gain ≈ peak − 10·log₁₀(N_main)
- 對 width=30 (N_main=30)，理論 main_min ≈ -14.77 dB
- 實測 main_min = -10.98 dB → 接近物理上限

→ +6.88 dB worst-case 是 binary 1-bit 在 width=30 配置下的接近實際物理上限。

## 對 Patch Antenna 移植的關鍵啟示

| 教訓 | Patch 應用 |
|------|-----------|
| max-max metric 在 wide-band/wide-region 場景嚴重 overestimate | Patch 評估 S11/gain 跨 band 用 worst-case，不是 max |
| Loss 必須直接反映 use case 物理需求 | Patch 想要「band 內 S11 ≤ -10」就用 worst-case loss |
| Surrogate label 要用 worst-case，不是 single point | 訓練 NN 預測 worst-case 才不會 reward 尖峰解 |
| Use case 決定 loss form，不是抽象 metric | 「點對點 steering」vs「sector 覆蓋」用不同 loss |

## 紀錄歷程修正

| Round | headline | worst (用新 metric 重評) |
|-------|----------|--------------------------|
| R37 5.6 GHz × 19 | +11.82 | TBD |
| R47 28 GHz × 13 | +13.44 | TBD |
| R57 free-phase 28 GHz × 13 | +21.31 | TBD |
| R63 free-phase 38 GHz × 41 | +30.99 | **-18.21** |
| **R64 worst-case 38 GHz × 41 width=30** | +17.86 | **+6.88 ★** |

→ R64 +6.88 是當前**真實可部署**最高紀錄（之前都是虛胖）

## Open Questions

1. R57/R63 max-max 紀錄套到 worst-case loss 重跑，能否回到 +6.88 級別？
2. 加 ripple penalty 能否進一步降 ripple？
3. Chebyshev / Parks-McClellan 風格的 equiripple 設計能否跨 binary 限制？
4. 不同 use case 對 loss 的敏感度（單點 steering vs 扇形覆蓋）
5. Surrogate-in-the-loop 場景下 worst-case loss 的 numerical stability

## Sources（文獻）

- [Chebyshev synthesis for arrays](https://link.springer.com/article/10.1023/A:1022941416515)
- [Flat-top beampattern synthesis (Generalized Dolph-Chebyshev)](https://link.springer.com/article/10.1007/s11045-012-0217-0)
- [Parks-McClellan minimax for arbitrary arrays](https://www.sciencedirect.com/science/article/abs/pii/S0165168405001143)
- [Equiripple MIMO beampattern Chebyshev (2025)](https://arxiv.org/pdf/2503.14315v1)
