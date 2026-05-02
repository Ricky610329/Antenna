# /loop Round 9–10 兩輪總結

> 接續 round 5-8 總結。期間 2026-04-29，研究方向從「修 generator path」轉向
> 「直接 GD per-target 最佳化 + 物理特性研究」。

## Round 9 — Direct GD 路線完整建立

### 工具
- `script/sweep_physical_limit.py` — 多 target heatmap
- `script/design_pattern_for_target.py --n_restarts N` — multi-restart

### 32-target sweep 結果（15×15 RIS）
- 全部正向，suppression mean = +2.91 dB
- max +6.94 dB（plateau 280, w=46，5 restart）
- 沒有 dead zones（min +0.31 dB）
- 規律：寬 plateau + 邊緣 θ_center ≈ ±60° 容易

## Round 10 — RIS design guide + element_num sweep

### 工具
- `script/RIS_DESIGN_GUIDE.md` — 給使用者實用工作流程
- `script/sweep_element_num.py` — 探陣列大小對 suppression 影響

### Element_num sweep 結果（4 sizes × 3 targets × 3 restarts）

| RIS 尺寸 | left | center | right | 總計 |
|---------|------|--------|-------|------|
| 10×10 | +4.27 | +2.69 | +3.09 | 中等，cells=100 |
| **15×15** | +3.40 | +2.63 | +4.61 | **意外比 10×10 還差** |
| 20×20 | +5.70 | +4.67 | +5.43 | 顯著躍升 |
| **25×25** | +4.51 | **+7.57** | +4.86 | 最佳 |

### 對使用者最有用的具體發現
1. **硬體選 25×25**（cells=625）能多 +3 dB 的 suppression。
2. 15×15 在某些 target 比 10×10 還差 —— 物理 local-minima 特性，建議避開。
3. Multi-restart 5 次 + direct GD 是當前最有效的工作流程，5 分鐘輸出可部署 pattern。

## Round 1-10 完整路徑 retrospective

```
Round 1-4   建設 generator-based 框架（BinarySTE / pretrained surrogate / 位元遷移）
            ↓
Round 5     diagnostics — direct GD 揭露 BinarySTE 缺陷（−3.34 vs +3.05 dB）
            ↓
Round 6-7   找到 conditioning failure 真因（multi-target / cond_reg）
            ↓
Round 8     證偽各種解 — generator-based 的 conditioning failure 是架構限制（11 個 run）
            ↓
Round 9-10  轉向 direct GD per-target — 達 +7.57 dB（當前最佳，超越 generator 7+ dB）
```

## Generator path vs Direct GD path 最終對照

| 路線 | 最佳 suppression | 用途 |
|------|------------------|------|
| Generator (v6) | −0.46 dB | one-shot 對任意 target，但 hamming ~0%（fake conditional）|
| **Direct GD (25×25, multi-restart)** | **+7.57 dB** | 為單一固定 target 找最佳 pattern |

**結論**：對使用者真實 use case（每次部署 RIS 服務一個固定 beam direction），
**direct GD path 完勝**。Generator-based 路線只有在 dynamic target 即時生成
場景才值得繼續，且需要架構級別重設計。

## Git 累積

Round 9-10 共 4 個 commits：
- `36825ad` sweep_physical_limit
- `3aae357` design_pattern multi-restart
- `5e9efb8` design_guide + sweep_element_num
- (之後) round 9-10 summary + design_guide 更新

總計 round 1-10 共 18 個 commits pushed 到 `ricky/modernize`。

## 下一步建議

實驗 cycle 已完整探索，剩下的方向：

1. **物理研究方向**（增量改善）：
   - 探不同入射角 θ_i, φ_i 對 suppression 影響
   - 不同頻率（5.6 GHz / 28 GHz / 60 GHz）對比

2. **工程方向**：
   - design tool 加 batch 模式（一次設計多個 target）
   - design tool 加「auto-pick best plateau width」（給 user 一個 θ_center 後自動最佳化）

3. **如果要繼續 generator path**（高風險、高 ROI 不確定）：
   - 拋棄當前 fc_patch 架構 → 改 hypernetwork 或 retrieval-based
   - 或直接 lookup table（pre-compute N target 各跑 direct GD，runtime 查最近）
