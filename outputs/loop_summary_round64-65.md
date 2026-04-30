# /loop Round 64–65 兩輪總結 — Worst-Case Loss + Pareto Frontier

> R64 設計 worst-case loss 暴露 R63 max-max 虛胖。R65 確認 aperture scaling
> 仍然成立 + 找到 ripple penalty 達成完全符合帽蓋的設計。

## TL;DR

- R63 +30.99 dB 是 max-max metric 虛胖（worst-case 實為 -18.21）
- 新 worst-case loss 在 width=30 達 **+6.88 dB**（n=41）
- Ripple penalty rw=2.0 達 **0/30 main 點違反帽蓋**（真實 flat-top）
- 大 aperture 在 worst-case metric 下也有用（n=11→41 改善 +8.44 dB）

## R64 — Worst-Case Loss 設計

```python
soft_main_min = -(1/β) * logsumexp(-β * resp[main])
soft_side_max = (1/β) * logsumexp(β * resp[side])
loss = -(soft_main_min - soft_side_max)
```

R63 record 在新 metric 下：
- headline (max-max):  +30.55 dB
- worst (min-max):    **-18.21 dB**
- main_min: **-48.77 dB** (75/80 點 < -3 dB)
- ripple: 48.77 dB

→ R63 +30.99 是「一根尖峰」解，不是 flat-top。

## R64 width sweep（38 GHz × n=41）

| width | worst_supp | ripple |
|-------|-----------|--------|
| 20 (10°) | +4.24 | 12.89 |
| **30 (15°)** | **+6.88 ★** | 10.98 |
| 40 (20°) | +5.07 | 12.12 |
| 60 (30°) | +4.95 | 11.34 |
| 80 (40°, R63) | +1.57 | 15.17 |

→ width=30 是 binary 1-bit 的 worst-case sweet spot。

## R65a — N Sweep 確認 aperture scaling 仍成立

width=30, ripple_weight=0:

| n | worst_supp | main_min |
|---|-----------|----------|
| 11 | -1.56 | -10.05 |
| 21 | +0.61 | -11.04 |
| 31 | +3.35 | -11.53 |
| **41** | **+6.88** | -10.98 |

從 n=11 到 n=41 worst-case 改善 **+8.44 dB**。
→ 大 aperture **真的有用**，不是 max-max 假象。

## R65b — Ripple Penalty Sweep（重要發現）

n=41, width=30:

| ripple_w | worst_supp | ripple | main<-3 |
|----------|-----------|--------|---------|
| 0.0 | +6.88 | 10.98 | 67% 違反 |
| 0.5 | +3.67 | 4.50 | 20% 違反 |
| 1.0 | +2.81 | 5.09 | 27% 違反 |
| **2.0** | **+0.22** | **1.75** | **0% ★** |
| 5.0 | +0.66 | 1.21 | 0% |

**rw=2.0 達成 0/30 main 點 < -3 dB**——真實的 flat-top，整片貼上蓋。

## Pareto Frontier 對 Patch 移植的核心啟示

| 部署需求 | rw | worst | ripple | flat-top? |
|----------|----|----|--------|-----------|
| 單方向 high-gain steering | 0 | +6.88 | 11 | ❌ |
| 中等容忍 (5 dB ripple) | 0.5-1 | +3 | 5 | △ |
| 嚴格 flat-top (2 dB ripple) | 2 | +0.22 | 1.75 | ✅ |
| 極嚴格 plateau (1 dB ripple) | 5 | +0.66 | 1.21 | ✅ |

**Use case 決定 loss weight**——不是抽象 metric。

## 紀錄歷程修正

| Round | Metric | 紀錄 |
|-------|--------|------|
| R63 | max(main) - max(side) | +30.99 (虛胖) |
| R63 in worst-case 重評 | min(main) - max(side) | **-18.21** |
| R64 worst-case loss width=30 | min(main) - max(side) | **+6.88** ★ true |
| R65 ripple penalty rw=2 | flat-top compliance | **0/30 main < -3** |

## 對 Patch Antenna 移植的方法論確立

R64-R65 確認的設計原則（直接搬到 patch）：

1. **Loss = use case** — 不是 max-max 抽象 metric
   - Wide gain coverage → worst-case + ripple penalty
   - Narrow steering → max-max 即可
   - Multi-target → vector worst-case

2. **Surrogate label = worst-case** — 訓練資料用 honest metric
   - 不然 surrogate 學到 reward 尖峰解
   - HFSS verify 會嚴重失真

3. **Dataset 記錄 Pareto frontier** — 不只一個 best
   - (worst_supp, ripple) 兩維 trade-off
   - 多個 ripple weight 提供 deployment options

4. **Aperture scaling 仍成立** — 大 N 給更多 DoF
   - 跨 metric 一致

## 下一步：R66 起改 Dataset Generation Mode

Loss 設計到此 saturate。下一階段價值在批量生成「對的 dataset」：

- Schema: (target_shape, target_pos, target_width, freq, n, inc, ripple_w)
- 每 entry: worst-case loss + 5 seeds 取最好
- 保存: (config, binary_pattern, response, all metrics, full Pareto curve)
- 第一批 ~50-100 entries 看分佈

這 dataset 直接訓練 patch surrogate / generator。

## Sources

- [Chebyshev synthesis](https://link.springer.com/article/10.1023/A:1022941416515)
- [Flat-top Dolph-Chebyshev](https://link.springer.com/article/10.1007/s11045-012-0217-0)
- [Equiripple MIMO Chebyshev (2025)](https://arxiv.org/pdf/2503.14315v1)
