# /loop Round 49–50 兩輪總結

> 接續 round 47-48 總結。Round 49-50 完成跨頻率 inc fine grid，確認 knife-edge
> peak structure 是否普遍存在。

## Round 49 — 5.6 GHz × 19 inc Fine Grid（確認 knife-edge）

5.6 GHz × 19 × broadside × width=46 × inc {±2°}:

| inc_θ | suppression |
|-------|-------------|
| +58° | +8.88 |
| +59° | +9.77 |
| **+60°** | **+11.82 ★** |
| +61° | +10.50 |
| +62° | +10.54 |

完整 inc structure:
```
+50: 8.88   +55: 8.71   +58: 8.88   +59: 9.77
+60: 11.82 ★  knife-edge peak
+61: 10.50  +62: 10.54  +65: 10.36  +70: 9.28
```

**結論**：
- 5.6 GHz peak 也是 knife-edge（跟 28 GHz × +51° 一致）
- ±1° 下降 1~2 dB
- 未破 +13.44 紀錄

## Round 50 — 60 GHz × 15 × width=60 inc Fine Grid（執行中）

7 inc values: +50, +55, +58, +59, +61, +62, +65°
（結果回來補表，看是否也是 knife-edge structure）

## 累計三頻率 Knife-Edge 對照

| Configuration | Peak inc | Suppression | Peak width |
|---------------|----------|-------------|------------|
| 5.6 GHz × 19 × width=46 | **+60°** | +11.82 | ±1° (R49) |
| **28 GHz × 13 × width=80** | **+51°** | **+13.44 ★** | ±1° (R48) |
| 60 GHz × 15 × width=60 | TBD | TBD | TBD (R50) |

## 物理規律總結（48 rounds 累積）

### 三大物理規律
1. **Sweet width ∝ 1/aperture**（main lobe 匹配）
   - 9.5λ → width=46
   - 7.5λ → width=60
   - 6.5λ → width=80
2. **Sweet inc 是 knife-edge sharp peak**（±1° 級）
   - 5.6 GHz × 19: +60°
   - 28 GHz × 13 × width=80: +51° (新紀錄)
3. **Triple Sharp Peak**——freq + n + width + inc + plateau pos + seed 6 維度
   都需匹配才達物理上限

### 紀錄歷程
```
v1                                          −4.08 dB
v6 generator best                           −0.46 dB
GD+SA reheat=2 (R25)                        +9.75 dB
SA-per-restart 5.6 GHz × 19 × +60° (R37)   +11.82 dB
SA-per-restart 28 GHz × 13 × +51° (R48)    +13.44 dB ★ 物理紀錄
```

**從 v1 到當前 = 17.52 dB 改善**

## Epistemic 進展鏈（截至 R49）

| Round | 假說 | 結果 |
|-------|------|------|
| R12 | ±60° inc 最佳 | R47 否定 → 28 GHz best=+50° |
| R32 | +11.82 universal | broadside-specific |
| R33 | specular-avoidance | chaotic |
| R35 | best GD → SA | 修為 SA-per-restart |
| R37 | mean +5.16 真實 | +9.67 with new logic |
| R39/40 | aperture 主導 best target | 確認 |
| R41 | 越大越好 aperture | 否定 |
| R42 | inc broad plateau | 否定 (sharp peak) |
| R43 | width 影響小 | 否定 (sharp peak) |
| R44 | 更多 restart 提升命中率 | 否定 (1/10) |
| R45 | width=46 universal | 否定 (60 GHz=60) |
| R46 | width=60 universal | 否定 (28 GHz=80) |
| R47 | inc=+60° universal | **否定 (28 GHz=+51° 達 +13.44)** |
| R48 | peak 是 ±5° broad | **否定 (knife-edge ±1°)** |
| R49 | 5.6 GHz 有 hidden peak | 否定 (+60° 也是 knife-edge) |

## 累計（50 rounds, 81+ commits）

### 工具庫
17+ scripts:
- 3 design tools (with SA-per-restart, --freq, --inc_theta)
- 6 sweep tools
- 4+ benchmark tools
- 多個 diagnostics
- 3 層完整文檔 + 29 round summaries

## Open Questions

1. **60 GHz 是否也 knife-edge**（R50 探，執行中）
2. **是否有更高 frequency × n × width × inc 組合 > +13.44**
3. GPU-batched SA 加速
4. Array factor 數學解析 +13.44 上限值
