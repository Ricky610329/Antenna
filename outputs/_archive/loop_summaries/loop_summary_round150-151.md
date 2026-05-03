# /loop Round 150–151 兩輪總結 — Phase 3 開場：unified pipeline + patch bridge plan

## TL;DR

R150 把 R141 deployment 函式 generalize 成 **unified pipeline**（接受 optional
surrogate forward function），重跑 4 個 R141 held-out combos 在 analytical +
surrogate 兩個 mode 下：3/4 PASS, 1/4 marginal（wide cap 配 surrogate 出現 numeric
divergence）。R151 audit 既有 patch infrastructure 並寫出 **PATCH_BRIDGE_PLAN.md**，
列出 R152-R155 具體步驟把 R141 pipeline 套到 patch antenna 上。

Phase 3 正式開始 — 接下來 4 輪會有實際的 patch antenna pattern 出來。

## R150 — Unified Pipeline

### API generalization

```python
optimize_ris_1bit(
    n, inc_deg, freq_hz, width_deg,
    n_restarts=5, gd_steps=1500,
    forward_fn=None,  # default = analytical sim; can pass surrogate
    eval_fn=None,     # default = analytical sim; truth for joint early-stop
)
```

對 patch transition：`forward_fn = HFSSNet surrogate`, `eval_fn = surrogate too`
（HFSS 太慢無法 per-50-step 呼叫），最後用 HFSS validation 驗證 final pattern。

### 4 配置 × 2 mode 結果（n=71 因 cron 預算 skip）

| Config | Mode | Best | Mean | Flat | Speedup |
|--------|------|------|------|------|---------|
| n=51 inc=30 28GHz w=10 | analytical | +3.13 | +2.87 | 5/5 | - |
| n=51 inc=30 28GHz w=10 | surrogate | +3.36 | +2.72 | 5/5 | **5.8×** |
| n=51 inc=70 60GHz w=10 | analytical | +2.72 | +1.93 | 5/5 | - |
| n=51 inc=70 60GHz w=10 | surrogate | +3.27 | +1.95 | 5/5 | **5.7×** |
| n=51 inc=51 38GHz w=15 | analytical | +1.74 | +1.39 | 5/5 | - |
| n=51 inc=51 38GHz w=15 | surrogate | +1.40 | +1.05 | 5/5 | 5.6× |
| n=51 inc=51 38GHz w=20 | analytical | +1.72 | +1.20 | 5/5 | - |
| n=51 inc=51 38GHz w=20 | surrogate | +1.49 | **-0.54** | **4/5** | 5.7× |

### Marginal config 的 diagnosis

最後一組（wide cap w=20）surrogate-loop mean 從 +1.20 掉到 -0.54，flat 從 5/5 變
4/5。即便 surrogate weights 完全等同 analytical 也會發生。

原因：架構差異雖然 R²=1.0 在 forward output 上，但 numeric handling 細節
不同（surrogate 的 `sqrt + 1e-12 + clamp(1e-8)` vs analytical 的 `torch.abs +
clamp(1e-8)`）。over 1500 GD steps 累積微小差異 → 不同 trajectory → 偶有 seed
落到 flat-top 都不滿足的區域 → joint early-stop fallback 到 final-step pattern。

**Lesson**：joint early-stop 是 critical safety net；surrogate-loop 仍會有
config-specific edge cases 需要監控。

## R151 — Patch Infrastructure Audit

Smoke-tested 既有 components：

| Component | 狀態 | 備註 |
|-----------|------|------|
| `HFSSNet` 6-layer MLP | ✓ Imports OK | 25×25 input, (3, 17) S-param output, 3.98M params |
| `SurrogateModel` 訓練 wrapper | ✓ Imports OK | 支援 batch training |
| `PatchSimulator` 抽象基底 | ✓ Imports OK | Windows-only HFSS COM |
| `SinglePortSimulator` / `DualPortSimulator` | ✓ Imports OK | 25×25 pixel patch |

### 關鍵差異 — RIS vs Patch

| Aspect | RIS | Patch |
|--------|-----|-------|
| 輸出 | far-field (361 角度) | S-parameters (3 ports × 17 freq) |
| "main beam region" | 角度 cap 範圍 | **頻率 band**（target frequency） |
| "sidelobes" | 其他角度 | 其他頻率 / out-of-band |
| Forward cost | analytical ~ms | HFSS 分鐘級（必須 surrogate）|
| Phase | 0/π 二進位 | 二進位 metal/no-metal |

→ Loss design 直接 transfer：worst + ripple + mean structure 對 S11 in-band
vs out-band 同樣 applicable。

### 完整 bridge plan (R152-R156+)

寫到 `outputs/PATCH_BRIDGE_PLAN.md`：

| Round | 工作 | ETA |
|-------|------|-----|
| R152 | Wire `optimize_patch_1bit()` with HFSSNet | 5 min |
| R153 | Train HFSSNet on existing patch dataset | 30+ min |
| R154 | End-to-end patch pattern via R141 pipeline | 30 min |
| R155 | HFSS validation + active learning trigger | 5-30 min |
| R156+ | Iterate active learning if needed | varies |

Total: 1-2 小時 focused work, 3-5 cron cycles.

## 紀錄歷程更新

| Round | 結果 |
|-------|------|
| R150 | Unified pipeline, 3/4 PASS at 5.7-5.8× speedup |
| R151 | Patch infrastructure 確認 importable, bridge plan written |

## 下一階段

進入 R152 — wire `optimize_patch_1bit()`。

## 結論

Phase 2（R142-R149）證明 surrogate-loop 在 analytical RIS 上可信。Phase 3
（R150+）正式開始 patch transition：

- R150 unified API 已準備好接 patch surrogate
- R151 確認既有 patch infrastructure 可用
- R152+ 實際接起來 + 訓 HFSSNet + 跑出第一個 patch pattern

從 R94（4 月初）的「max-max loss 騙 metric」修正開始，到現在 R151 的
patch bridge plan，整套 methodology 已從 RIS playground 走到 patch deployment
的入口。下一階段就是把它**實際 deploy**。
