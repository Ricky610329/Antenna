# /loop Round 35–36 兩輪總結

> 接續 round 33-34 總結。Round 35-36 主軸：發現 design tools 的 SA-per-restart bug 並修正。

## Round 35 — Design Tool Algorithmic Bug

### 發現
舊邏輯：跑 N restarts 找 best GD → 對它做 SA。
**問題**：差 GD 的 attraction basin 在 SA 下能跳更遠；好 GD 反而早卡 local plateau。

### 5 restart 對照（5.6 GHz × 19×19 × inc=+30°）

| seed | GD | SA | gain |
|------|-----|-----|------|
| 0 | +6.99 | **+9.69 ★** | +2.70 |
| 1 | +4.75 | +7.98 | +3.23 |
| 2 | +1.67 | +6.60 | **+4.93** |
| 3 | +3.95 | +8.71 | +4.76 |
| 4 | **+7.17** | +8.60 | +1.43 ← best GD 卻最差 SA gain |

舊邏輯 best:「best GD (+7.17) → SA (+8.60) = +8.60」
新邏輯 best:「each SA → max = +9.69 (seed 0)」
**+1.09 dB improvement**

### 修正
`design_pattern_for_target.py`: 對每個 restart 立刻做 SA，取 best across。
- 成本: N × GD + N × SA（多 N-1 個 SA 時間）
- 效益: ~+1 dB suppression

## Round 36 — 破紀錄嘗試 + Batch Tool 修正

### Test +60° broadside 5 restart × SA-per-restart
| seed | GD | SA |
|------|-----|-----|
| 0 | **+11.82** | **+11.82** ← 物理上限 |
| 1 | +5.82 | +8.34 |
| 2 | +3.60 | +9.32 |
| 3 | +2.87 | +6.56 |
| 4 | +7.91 | +8.36 |

best = +11.82（seed 0 GD 命中物理上限）。SA 不能跨 wide gap。

### Design_batch.py 修正
跟 round 35 design_pattern_for_target 一致——design_one() 內部處理 SA-per-restart。

## 累計（36 rounds, 60+ commits）

### 紀錄歷程
```
v1                          −4.08 dB
v6 generator best           −0.46 dB
GD multi-restart            +1.82 ~ +9.51 dB
GD+SA reheat=2              mean +8.38, max +9.75
GD+SA SA-per-restart (R35)  +9.69 (inc=+30° broadside)
GD+SA SA-per-restart (R36)  +11.82 (inc=+60° broadside, 物理紀錄)
```

### Epistemic 進展（截至 R36）
- ~~±60° 最佳~~（R34 否定 → ±30° broadly best）
- ~~specular-avoidance 假說~~（R33 否定）
- ~~+11.82 是 lucky 命中~~（R31 修正：是 reproducible 物理上限）
- ~~+11.82 是 universal 最佳~~（R32 修正：broadside-only）
- **「best GD → SA」非最佳**（R35 修正：每 restart 各別 SA）
- **+11.82 與 +9 之間有 wide gap**（R29/R36 SA 不能跨）

### 工具庫累計
17 個 scripts:
- 3 design tools (with SA-per-restart fix)
- 6 sweep tools
- 4 benchmark tools (含 test_determinism)
- 多個 diagnostics
- 3 層完整文檔

## 對使用者最終工作流程

```bash
python script/design_pattern_for_target.py \
  --element_num 19 \
  --freq 5.6e9 \
  --inc_theta 60 \
  --plateau_start 154 --plateau_w 46 \
  --steps 1500 --n_restarts 5 \
  --sa_steps 8000 --sa_reheat_cycles 2 \
  --device cuda:0
```

5-7 分鐘 → expect mean +9 dB，max +11.82 dB（broadside），worst ~+6 dB。

## Open Questions

1. 對其他 freq × n × inc_θ 配置，新 SA-per-restart 邏輯能改善多少？
2. GPU-batched SA 加速
3. Array factor 數學能否預測 chaotic patterns
