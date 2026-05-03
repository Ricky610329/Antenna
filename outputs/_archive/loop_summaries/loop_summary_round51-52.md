# /loop Round 51–52 兩輪總結

> 接續 round 49-50 總結。Round 51-52 進一步驗證 +13.44 dB physical record
> 的 sweet spot 結構。

## Round 51 — n=13 是 28 GHz Sweet Aperture

28 GHz × +51° × width=80 × broadside × n sweep:

| n | aperture | suppression |
|---|----------|-------------|
| 11 | 5.5λ | +9.86 |
| **13** | **6.5λ** | **+13.44 ★** |
| 15 | 7.5λ | +11.12 |
| 17 | 8.5λ | +9.65 |
| 19 | 9.5λ | +9.70 |

確認 n=13 是這個配置的 sweet aperture，未破紀錄。

## Round 52 — 不同 Plateau 位置（執行中）

5 plateau positions × 28 GHz × 13 × +51° × width=80 × SA-per-restart：
- left, center_left, broadside, center_right, right

看是否其他 plateau 位置也達 +13 級，或 +13.44 是 broadside-only。

（結果回來後補表）

## 6-維度 Sharp Peak Configuration（截至 R51 確認）

**+13.44 dB physical record requires ALL 6 dimensions match**:

| 維度 | Sweet value | 偏離影響 |
|------|-------------|---------|
| 1. **freq** | 28 GHz | 5.6 GHz: 11.82 / 60 GHz: 10.52 |
| 2. **n** | 13 (6.5λ aperture) | 11: -3.58 / 15: -2.32 |
| 3. **inc** | +51° (knife-edge ±1°) | +49: -3.71 / +52: -4.27 |
| 4. **width** | 80 (main lobe match) | 46: -4.26 / 60: -5.02 |
| 5. **plateau pos** | broadside (R52 探) | TBD |
| 6. **seed** | 0 (lucky GD init) | other seeds: -4~-7 dB |

任一維度偏離都顯著下降——+13.44 是極窄 attraction basin。

## 跨頻率 Peak 圖譜（截至 R51）

| Freq × n × width | Best inc | Suppression | Structure |
|------------------|----------|-------------|-----------|
| 5.6 GHz × 19 × 46 | +60° | +11.82 | knife-edge ±1° |
| **28 GHz × 13 × 80** | **+51°** | **+13.44 ★** | knife-edge ±1° |
| 60 GHz × 15 × 60 | +62° | +10.52 | multi-modal |

## Epistemic 進展（截至 R51）

| Round | 假說 | 結果 |
|-------|------|------|
| R12 | ±60° inc 最佳 | R47 否定 → 28 GHz best=+51° |
| R32 | +11.82 universal | broadside-specific |
| R35 | best GD → SA | 修為 SA-per-restart |
| R37 | +5.16 mean 真實 | +9.67 with new logic |
| R39/40 | aperture 主導 best target | 確認 |
| R41-46 | 各維度 sweet spot | 全部找到（n, width, inc） |
| R47 | inc=+60° universal | 否定 (28 GHz=+51°) |
| R48 | peak 是 ±5° broad | 否定 (knife-edge ±1°) |
| R49 | 5.6 GHz 有 hidden peak | 否定 (+60° 也是 knife-edge) |
| R50 | 60 GHz 也是 knife-edge | 否定 (multi-modal) |
| R51 | n=11/15 達 +14 | 否定 (n=13 sweet) |

## 累計（52 rounds, 85+ commits）

### 工具庫
17+ scripts:
- 3 design tools (with full 6 維度 control)
- 6 sweep tools
- 4+ benchmark tools
- 多個 diagnostics
- 3 層完整文檔 + 30 round summaries

### 紀錄歷程
```
v1                                          −4.08 dB
v6 generator best                           −0.46 dB
GD+SA reheat=2 (R25)                        +9.75 dB
SA-per-restart 5.6 GHz × 19 × +60° (R37)   +11.82 dB
SA-per-restart 28 GHz × 13 × +51° (R48)    +13.44 dB ★ 物理紀錄
```

從 v1 到當前 = **17.52 dB 改善**

## Open Questions

1. **+13.44 是否 broadside-only on 28 GHz**（R52 探）
2. 是否有更好頻率 × n × width × inc × plateau 組合 > +13.44
3. GPU-batched SA 加速
4. Array factor 數學能否預測 +13.44 上限
