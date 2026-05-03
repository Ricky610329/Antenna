# /loop Round 5–8 兩輪總結

> 接續 round 1-4 總結。期間 2026-04-29，目標：突破 RIS conditioning failure。

## 完整實驗對照（11 個 run + 1 工具）

| Run | 機制 | suppression mean | hamming% | 評語 |
|-----|------|------------------|----------|------|
| v1 | 純 binary + pretrained surrogate | −4.08 | n/a | baseline |
| v2 | + H/I/J 反 collapse 三 combo | −1.84 | n/a | first improvement |
| v3 | + 位元遷移（curriculum quantization）| −2.21 | n/a | 持平 |
| v4 phase1+postq | 結構化 surrogate + 後處理量化 | −1.63 | n/a | 略優 |
| v4 phase2 | + binary fine-tune | −1.32 | n/a | 略優 |
| v5 | plan D multi-target binary STE | −6.40 | n/a | 失敗 |
| **v6** | **plan D multi-target + cond_reg=1** | **−0.46** | **0.40%** | **史上最好** |
| v7 | cond_reg=5 (太強) | −2.69 | n/a | 過強反傷 |
| v8 | plan E direct RIS sim + Biased | −6.08 | 0.00% | 失敗 |
| v9 | plan E direct RIS sim + plain MLP | −4.23 | 0.00% | 證偽 noise 假說 |
| **direct GD per-target** | **連續 GD + 後量化** | **+2.76** | n/a | **接近物理極限** |
| direct GD upper bound | 連續無量化 | +4.77 | n/a | 物理極限 (best loss=22) |

## 關鍵診斷（按發生順序）

### Round 5 — 暴露 BinarySTE 缺陷
寫 `direct_pattern_search.py`，繞過 generator 直接 GD on logits：
- BinarySTE direct GD: −3.34 dB
- Continuous→hard direct GD: +3.05 dB
- **差 6.4 dB** → BinarySTE 訓練本質有問題

### Round 6 — 找到 conditioning failure 真因（自以為）
診斷原 trainer 每 epoch 餵固定 target → generator 從未看過多樣輸入。實作
`script/train_multi_target.py`（plan D），32 組 target pool 隨機抽。

### Round 7 — Plan D 表面成功實則失敗
- v6 cond_reg=1.0 達 **−0.46 dB**（史上最好）
- 但檢查 hamming distance：**~0%**（10 target 給幾乎相同 pattern）
- 「最好」只是巧合找到通用 fixed pattern，conditioning 沒突破

### Round 8 — Plan E + Plain MLP 雙重證偽
- v8 直接用可微 RIS sim（曾俊瑋 113 路線）：suppression −6.08, hamming 0%
- v9 Plain MLP 無 Gumbel noise：suppression −4.23, hamming 0%
- **不論 surrogate 還是 generator class，conditioning 都解不了**
- 真正根因：**generator 為 32 個 target 妥協出「平均最佳」pattern，這是
  one-shot generator vs per-target optimization 的本質差距**

## 真正的工程結論

`script/design_pattern_for_target.py`：
- 對單一 target 跑 2000 步 GD on logits（連續）+ 後處理量化
- 60 秒收斂到 **+2.76 dB suppression**（vs generator 最好 −0.46，**+3.22 dB**）
- 接近物理上限 +3.05 dB
- 直接輸出可部署的 `pattern_binary.npy`

**對使用者的真實 use case**（每次部署只服務一個固定 target），這個工具就是
最終解。Generator-based 路線只有在「需要對 dynamic target instant 反應」
（如賴昱鈞 113 RL 場景）才值得繼續。

## 程式碼累積（Round 5-8）

```
新增：
  direct_pattern_search.py        驗證物理極限
  post_quantize_eval.py           phase 1 連續權重 → 硬二值化評估
  train_multi_target.py           plan D（multi-target + cond_reg）
  train_direct_ris.py             plan E（無 surrogate、可微 RIS sim）
                                  + PlainMLPGen 對照
  design_pattern_for_target.py ★  單目標最佳化工具（最終建議用）
```

## Git 進度

Round 5-8 累積 6 個 commits（總 14 個 round 1-8 累積）：
- f77c964 direct_pattern_search 揭露 BinarySTE 缺陷
- 4d80bf6 post_quantize_eval
- 08ad306 plan D minimal trainer
- 150e131 plan D + cond_reg
- e46e659 plan E direct RIS sim
- (待 commit) per-target design tool

全部 push 到 `ricky/modernize`。

## 下一步建議

1. **若使用者 use case = 單一 target 部署**：用 `design_pattern_for_target.py`，
   案件結束。Generator-based 路線的 ROI 已經 negative。
2. **若使用者堅持 generator path**：需要拋棄 BiasedGumbelSigmoidGEN 改架構，例如：
   - 直接 hypernetwork（target → generator weights）
   - 或 retrieval-based（pre-compute lookup table，runtime 查最近）
3. **若想驗證物理上限**：跑多個不同 target 的 direct GD，看不同 (θ, φ) 配對下
   suppression 怎麼分布——這是真正的物理特性研究。
