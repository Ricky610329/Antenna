# /loop Round 47–48 兩輪總結

> 接續 round 45-46 總結。Round 47 大突破：+13.41 dB（28 GHz × 13 × width=80 ×
> inc=+50°）— 破之前 +11.82 dB 紀錄。

## Round 47 — 新最高紀錄 +13.41 dB

### 實驗（28 GHz × 13 × width=80 × broadside × 5 restart × SA-per-restart）

| inc_θ | suppression |
|-------|-------------|
| +30° | +9.30 |
| +45° | +9.74 |
| **+50°** | **+13.41 ★ NEW RECORD** |
| +55° | +9.43 |
| +60° (R46 baseline) | +10.53 |

### 重大發現
1. **+50° 是 sharp peak**——±5° 都顯著低
2. **28 GHz × 13 × width=80 best inc = +50°**（不是 +60°）
3. 物理紀錄轉移：5.6 GHz × 19 × +60° (+11.82) → **28 GHz × 13 × +50° (+13.41)**
4. 不同 (freq × n × width) 配置有自己的 sweet inc

從 v1 到當前 = **17.49 dB** 改善（之前 15.9 dB）

## Round 48 — +50° Peak Fine Grid（執行中）

試 +48 / +49 / +51 / +52° 看是否有更尖 peak。
（結果回來補表）

## 累計圖譜（48 rounds 統合）

### 各配置最佳結果

| Configuration | Best inc | Best width | Suppression |
|---------------|----------|-----------|-------------|
| 5.6 GHz × 19 (9.5λ) | +60° | 46 | +11.82 dB |
| **28 GHz × 13 (6.5λ)** | **+50°** | **80** | **+13.41 ★** |
| 60 GHz × 15 (7.5λ) | +60° | 60 | +10.14 |

**+13.41 是新最高紀錄**，在 28 GHz 而非 5.6 GHz！

### 物理規律總結

1. **Sweet width ∝ 1/aperture**（main lobe match）
2. **每個 (freq × n × width) 有自己的 sweet inc**——不是 universal +60°
3. **Sweet inc 是 sharp peak**——±5° 顯著下降
4. **+13.41 命中需要 6 維度匹配**：freq + n + width + plateau pos + inc + seed

### 工具庫累計
17+ scripts:
- 3 design tools (with SA-per-restart, --freq, --inc_theta)
- 6 sweep tools
- 4+ benchmark tools
- 多個 diagnostics
- 3 層完整文檔 + 28 round summaries

## 紀錄歷程

```
v1                                          −4.08 dB
v6 generator best                           −0.46 dB
GD multi-restart                            +1.82 ~ +9.51 dB
GD+SA reheat=2 (R25)                        +9.75 dB
SA-per-restart 5.6 GHz × 19 × +60° (R37)   +11.82 dB
SA-per-restart 28 GHz × 13 × +50° (R47)    +13.41 dB ★ NEW RECORD
```

**從 v1 到當前 = 17.49 dB 改善**

## Open Questions

1. **+50° peak 細網格是否更尖**（R48 探）
2. 5.6 GHz / 60 GHz 是否也有 inc=+50° 的隱藏 peak？
3. GPU-batched SA 加速
4. Array factor 數學能否預測 +13.41 上限值

## 對使用者最終建議（更新）

```bash
# 28 GHz 部署（新最佳）
python script/design_pattern_for_target.py \
  --element_num 13 \
  --freq 28e9 \
  --inc_theta 50 \
  --plateau_start 137 --plateau_w 80 \
  --steps 1500 --n_restarts 5 \
  --sa_steps 8000 --sa_reheat_cycles 2

# 期望 max +13.41 dB suppression
```

或 5.6 GHz 部署用 width=46, inc=+60°，max +11.82 dB。
