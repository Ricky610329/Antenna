# R94-R156 與 Lab 既有 Pipeline 整合計劃

> 2026-05-03 — 取代之前 `PATCH_BRIDGE_PLAN.md` 的天真規劃。
>
> 這份文件 audit 過 lab 既有 codebase 後寫成，紀錄 R94-R156 工作如何
> **plug-in** lab 既有 amortized G(spec)→pattern + online learning pipeline，
> 而非平行重建。

---

## 真實狀況

Lab 已有完整 amortized pipeline：

```
antenna/training/trainer.py (478 行)
  ↳ Generator: BiasedGumbelSigmoidGEN (反 collapse 校準 + Gumbel-Sigmoid)
  ↳ Binary discretization: BinarySTE (forward 硬二值化, backward identity)
  ↳ Loss: custom_loss_tolerance (side_excess² + main_deficit²)
  ↳ Forward: simulator.simulate() (RIS analytical 或 HFSS COM)
  ↳ Surrogate: HFSSNet, online finetune via train_one_data
  ↳ Online dataset: accumulates (pattern, signal) pairs
  ↳ Tau annealing: AdaptiveCyclicalScheduler 4.0 → 0.1
  ↳ Rollback: early_stop("real_loss") + retrain surrogate full batch
```

入口 CLI:
```bash
python -m antenna train +experiment=train_ris_binary_pretrained
```

預跑結果在 `result/RIS-binary-pretrained-v1/` (有 100 個 generator
checkpoint, 但 log 只到 epoch 8 — 訓練可能 stuck 在 `饋入點座標越界` ERROR
或其他原因, 待 debug)。

---

## R94-R156 對應 Lab Pipeline 的 Plug-in 點

### Plug-in #1: Loss 補強（高優先 / 低風險）

Lab 現用 `custom_loss_tolerance`（`antenna/ris/__init__.py:95`）：

```python
side_loss = (prediction[mask_side] - sidelobe_threshold).clamp(min=0).pow(2).mean()
main_loss = (main_target - prediction[mask_main]).clamp(min=0).pow(2).mean()
return side_loss + main_weight * main_loss
```

**問題**: 只懲罰違反 spec, 達標就 0. Generator 學到「剛好 -20 dB」就停，
不會繼續往 -25/-30 dB 推。

**R94-R156 的補強**: R119 的 `mean(side)` term 會 **持續** 把整片 sidelobe
distribution 往下推, 不管有沒有達標.

**具體 patch**: 在 `custom_loss_tolerance` 加第三個可選參數 `area_weight`:
```python
def custom_loss_tolerance(
    prediction, target, sidelobe_threshold=-20.0,
    main_target=None, main_weight=1.0,
    area_weight=0.0,  # NEW: R119 mean(side) term
):
    # ... existing side_loss + main_loss ...
    if area_weight > 0:
        area_loss = area_weight * prediction[mask_side].mean()  # 越負越好
        return side_loss + main_weight * main_loss + area_loss
    return side_loss + main_weight * main_loss
```

新 yaml config:
```yaml
loss_params:
  sidelobe_threshold: -20.0
  main_target: 0.0
  main_weight: 1.0
  area_weight: 0.5   # R119: 持續下壓 sidelobe distribution
```

**驗證方式**: 跑兩個 run（with vs without area_weight），比 `pic/response_vs_target.png`
的 sidelobe 分佈寬度。

---

### Plug-in #2: Generator Pretraining 資料供應（中優先 / 中風險）

**問題**: Lab pipeline 從 random-init G 開始, 前 N 個 epoch G 出垃圾 pattern,
surrogate 也學不到東西, 整個收斂慢. `result/RIS-binary-pretrained-v1/online.dataset.log`
顯示 6 epoch 才加 3 筆有效資料.

**R94-R156 的補強**: 用 `optimize_ris_1bit()` 跑 ~50-200 個不同 spec, 出
gold-quality (spec, pattern) pairs, 直接 supervised pretrain G 一次, 然後再
丟進 lab pipeline 做 online refinement.

**Workflow**:
```python
# Step 1: 產 supervised pretraining dataset
from script.ris_core import optimize_ris_1bit, select_1bit_recipe

specs = sample_diverse_specs(n=200)  # 設計多樣化 spec
pairs = []
for spec in specs:
    result = optimize_ris_1bit(**spec)  # ~30 sec each
    pairs.append((spec, result["best"]["pattern"]))

# Step 2: Supervised pretrain G
G = BiasedGumbelSigmoidGEN()
for epoch in range(supervised_epochs):
    for spec, gold_pattern in pairs:
        soft_pred = G(encode_spec(spec))
        loss = BCE(soft_pred, gold_pattern)
        # ... backprop

# Step 3: 把 pretrained G 餵進 lab pipeline online learning loop
torch.save(G.state_dict(), "result/G_pretrained_from_per_task_GD/checkpoint.pth")
# 改 train_ris_binary_pretrained.yaml 的 generator.pretrained_path
```

**預期效果**: 大幅減少 lab pipeline cold-start 階段的 wasted HFSS calls.

**風險**: spec encoding 對齊 — `optimize_ris_1bit()` 接 `(n, inc, freq, width)`,
lab `G(target_curve)` 接的是 target response curve。需要 conversion layer.

---

### Plug-in #3: Validation Oracle（中優先 / 低風險）

**問題**: Lab pipeline 的 G 訓完, 怎麼知道 G(spec) 出的 pattern 是「夠好」
還是「只比 random 好一點」?

**R94-R156 的補強**: 給定任意 spec, `optimize_ris_1bit(spec)` 跑 30s 出
ground-truth optimum (或近似 optimum). 比對 G(spec) 的 metric vs 這個 oracle,
如果差距 < 1 dB 就視為 G 已 saturate.

**Acceptance criterion**:
```
G saturated if: 
  for spec in held_out_specs:
    G_pattern = quantize(G(encode(spec)))
    G_metric = sim(G_pattern).criterion()  # 用 lab 的 custom_loss_tolerance
    
    oracle_pattern = optimize_ris_1bit(**spec)["best"]["pattern"]
    oracle_metric = sim(oracle_pattern).criterion()
    
    assert oracle_metric - G_metric < 1.0  # G 達 oracle 的 ~90%
```

**Why oracle**: 沒有 oracle, 你不知道 G 還能不能再 train 進步。如果 oracle
跟 G 拉開 5 dB 表示還有空間; 如果只差 0.3 dB 表示 G 接近 sub-problem optimum,
要再進步只能換架構或加 regularizer.

---

### Plug-in #4: Surrogate Retrain Frequency 決策（低優先 / 證據型）

R148/R149 結論: surrogate weights 加 5-20% Gaussian noise, optimization 結果
mean worst **反而 improve** (joint early-stop 用 truth filter 掉 bad pattern).

**對 lab pipeline 的 implication**:
- Lab 現在每 epoch `train_one_data` 把 surrogate online finetune 一次
- 可能 over-fit 到剛跑出來的那一筆，gradient 變 biased
- 建議: 改成 **batch retrain every K epochs**, K=10-20 試試, 給 surrogate
  穩定點的 update signal

**這純粹是參數調整**, 不需要動 code, 改 yaml `surrogate.hfss_max_epoch` 配合
lab pipeline 的 trigger logic 即可.

---

### Plug-in #5: Multi-band Loss for Broadband Generator（patch 階段）

R154 證明 sum loss across freqs 可以做 broadband（10% rel BW universal PASS,
patch typical 5-10% BW 涵蓋）.

**對 patch G 的 implication**: lab pipeline 的 `target` 目前是單一頻率
response curve. 改成 multi-freq target list, loss 對每個 freq 算
`custom_loss_tolerance` 加總, 一個 G 可同時對多 freq 出 robust pattern.

**修改點**:
```python
# antenna/training/trainer.py 內 loss 計算改:
loss = sum(custom_loss_tolerance(response[f], target[f], **params)
           for f in target_freqs)
```

**Caveat**: R155 證明 32% rel BW 會 fail flat-top. 如果 patch 要 UWB,
需要 architectural 升級 (bigger pixel grid 或 multi-port).

---

## 不要做的事

- ❌ **不要寫 `optimize_patch_1bit()`**: patch HFSS scenario 不該走 per-task GD,
  lab pipeline 已 cover.
- ❌ **不要包 ris_core.py 成 production API**: 它是研究工具 + supervised data
  generator, 不是 inference path.
- ❌ **不要在 patch 階段重新做「4 軸 universal validation」**: lab amortized
  G 可同時對多 spec 推論, 不需要 axis-by-axis grid search.
- ❌ **不要把 R150-R156 的 PATCH_BRIDGE_PLAN.md 當有效 plan**: 那是 audit
  lab codebase 之前寫的天真版本, 已被本文取代.

---

## 接下來實際要做（Action Items）

### 立即（不需 HFSS）

1. **Run lab pipeline 確認 baseline**:
   ```bash
   python -m antenna train +experiment=train_ris_binary_pretrained
   ```
   觀察:
   - 100 epoch 跑得完嗎? 之前 `result/RIS-binary-pretrained-v1/log` 只到 epoch 8
   - `饋入點座標越界` ERROR 是 hard fail 還是 warning?
   - Final `real_loss` 收斂到多少? `pic/response_vs_target.png` 長怎樣?

2. **Debug 阻塞**: 如果 pipeline 沒跑通, 先 fix bug 再談 enhancement.

3. **建立 Validation Oracle baseline**: 跑 `optimize_ris_1bit()` 對 ~10 個
   diverse spec, 把 oracle metric 紀錄下來, 之後對比 G(spec) 用.

### 短中期（plug-in 補強）

4. **Plug-in #1**: 加 `area_weight` 到 `custom_loss_tolerance`, A/B test.
5. **Plug-in #2**: 用 `optimize_ris_1bit()` 跑 ~50 specs 出 supervised
   pretraining pairs, 試 warm-start G.

### 長期（patch transition）

6. **Patch surrogate train**: 用 PatchSimulator 跑 ~200 HFSS samples, 訓
   HFSSNet, 上 lab pipeline. 不需重新建 patch-specific optimizer.
7. **Multi-band G**: 修改 loss 加總 across freqs, 試 broadband generation.
8. **Active learning acquisition**: lab 目前是 random/passive, 可加
   disagreement-based acquisition function (G 跟 surrogate 預測差最大的 spec).

---

## Bottom Line

R94-R156 不是 lab 主路徑替代品, **是工具箱**：
- Loss 心得 → Plug-in #1 (`area_weight`)
- 優化能力 → Plug-in #2 (supervised pretraining data)
- 品質基準 → Plug-in #3 (validation oracle)
- Noise 證據 → Plug-in #4 (retrain frequency tuning)
- Multi-band loss → Plug-in #5 (broadband G)

接下來方向轉換: 把心力從「建 per-task GD methodology」改成「跑通 lab pipeline +
plug-in 補強 + 收集 patch HFSS data」. 不再做平行系統.
