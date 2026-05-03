# /loop Round 1–4 綜合總結

> 期間 2026-04-29，目標：RIS binary phase {0, π} pattern 生成，attack generator
> conditioning failure（對所有 target 給同一 pattern）。

## 工作脈絡

```
Round 1 (基礎建設)
  ├─ commit 1: BinarySTE + schema 欄位 (binary_mode, surrogate.pretrained_path/freeze)
  ├─ commit 2: trainer 整合 binary_mode + pretrained surrogate workflow + HFSSNet batch
  ├─ commit 3: BiasedGumbelSigmoidGEN 校準 + balance loss + inspect 雙 sigmoid bug 修
  └─ commit 4: 位元遷移（curriculum quantization）trainer + phase1/phase2 yaml

Round 2 (端到端 v3)
  ├─ commit 5: run_bit_migration.sh + generate_structured_patterns.py
  └─ 啟動 v3 訓練（背景，~1 hr）

Round 3 (修正 + 配套)
  └─ commit 6: pretrain_surrogate.py 加 --n_structured + 公式校正 (plane-wave reflectarray)

Round 4 (v3 結果分析 + v4 啟動)
  ├─ v3 結果: pattern on-rate 55%，suppression mean −2.21 dB
  ├─ 診斷：phase 1 min_loss 從 epoch 2 鎖死 494 → conditioning failure 還在
  └─ 啟動 v4 = 結構化 surrogate + 位元遷移 一條龍（背景）
```

## 數值對照

| Run | 機制 | Pattern on-rate | Suppression mean | Suppression max | 結論 |
|-----|------|----------------|------------------|-----------------|------|
| v1 | 純 binary + pretrained surrogate (random) | 42% | −4.08 dB | +0.34 dB | 沒 collapse 但無方向性 |
| v2 | + H/I/J 三反 collapse combo | 53% | −1.84 dB | +0.92 dB | **+2.24 dB**（H/I/J 各有貢獻）|
| v3 | + 位元遷移（連續→binary）| 55% | **−2.21 dB** | +3.20 dB | 與 v2 持平、未突破 conditioning |
| v4 | + 結構化 surrogate（執行中）| ? | ? | ? | 預期突破 conditioning|

## 關鍵發現（按重要性）

### 1. Inspect tool 有 critical bug，修了之後才看到真相
- 原本 `inspect_ris_run.py` 對 GumbelSigmoid 輸出又套一次 `sigmoid()`，把所有值推到 [0.5, 0.73]，hard threshold 永遠 100% on
- 結果是 v1 / v2 都被誤判為「全 1 collapse」，實際上 trainer 走的 BinarySTE path 早就是合理 binary pattern
- Round 1 commit 3 才發現並修

### 2. BiasedGumbelSigmoidGEN 的 BiScaleNorm 反而害事
- 原版有 `BiScaleNorm` 把 logits 壓到 [-1, 1]，再大的 bias init 都被吃掉 → 初始 on-rate 一面倒
- Round 1 commit 3 移除 BiScaleNorm + 加啟動校準（32 次隨機 forward 平移 bias 到 mean ≈ 0）→ 初始 on-rate 49.78–50.22%
- **注意**：這個改動跟 114 學年錢鵬予論文（在 patch 場景驗證 BiScaleNorm 有效）矛盾——但 binary RIS 場景不適用

### 3. 物理單位的 suppression 指標比 loss 更有意義
- 原 inspect 只看 max(resp)，不分 main / sidelobe 區
- Round 1 commit 3 加上 `main_peak / side_max / suppression_dB` 三項
- 這三個是**物理上「特定範圍高、其他抑制」的直接量化**

### 4. 位元遷移不是 conditioning failure 的解藥
- 原預期：先連續學會 conditional → 再量化只是「微調量化邊界」
- 實際：phase 1（連續）就學不會 conditional（min_loss epoch 2 卡死）→ phase 2 從一個已經失敗的權重出發，當然救不回來
- **真正的根因**：surrogate 的訓練資料分佈太窄（純隨機 binary，響應幾乎都是雜訊）→ surrogate 給不出「往定向 beam 走」的梯度

### 5. 結構化 pattern 確實比隨機定向
- Round 3 用 plane-wave reflectarray 公式生成 50 個結構化 pattern：peak−top20 sidelobe **= 4.00 dB**
- 50 個隨機 pattern 同指標：1.47 dB
- **2.7× 更強的定向性** → 給 surrogate 看這種 pattern 才有可能教會它「pattern → directional beam」

## 程式碼架構演進

```
新增類別 / 函式：
  BinarySTE                            (autograd)
  BiasedGumbelSigmoidGEN.__init__       重寫含校準
  trainer._maybe_binarize               STE 注入點
  trainer._load_pretrained_generator    位元遷移
  pretrain_surrogate.py                 surrogate 預訓練（含 --n_structured）
  generate_structured_patterns.py       線性相位梯度生成器
  run_bit_migration.sh, run_full_v4.sh  端到端 batch
  inspect: main/side/suppression 指標   物理量化

新 yaml：
  train_ris_binary.yaml
  train_ris_binary_pretrained.yaml
  train_ris_binary_v2.yaml              三 combo
  train_ris_phase1_continuous.yaml      位元遷移 phase 1
  train_ris_phase2_binary.yaml          位元遷移 phase 2

Schema 欄位新增：
  binary_mode: bool
  binary_balance_weight: float
  surrogate.pretrained_path: str | None
  surrogate.freeze: bool
  generator.pretrained_path: str | None
```

## 下一步（v4 結束後）

1. **如果 v4 突破 conditioning**：定下「結構化 surrogate + 位元遷移」為 v4 baseline，進入細部調參
2. **如果 v4 仍 fail**：問題在 generator 架構而非資料——考慮：
   - 把 generator 換成更直接的 conditional 結構（如 FiLM / hypernetwork）
   - 或回去找賴昱鈞 113 RL 路線（離散動作搜尋 vs 梯度下降）
   - 或評估蔡奇倫的 Smooth G-STE（雖然我之前認定它本質沒料）

## Git 進度

7 個 commits pushed 到 `ricky/modernize`：
- `c347fc3` BinarySTE
- `77e7b87` binary_mode + pretrained surrogate
- `38ad7d9` 反 collapse + inspect bug 修
- `35100f2` 位元遷移
- `8705cd5` 結構化 pattern 工具 + run_bit_migration.sh
- `d5e6669` pretrain_surrogate.py 加 --n_structured
- (待 commit) run_full_v4.sh
