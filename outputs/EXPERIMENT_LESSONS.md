# 156 Rounds of RIS / Patch Optimization — Lessons Learned

> 2026-05-03。涵蓋 R1 → R156，但**不是** round-by-round timeline
> （那是 `REPORT_R94_to_R156.md` 的事）。本文按主題組織，把整段研究的
> 心得 distill 成「下一個碰到類似問題的人應該先讀這個」的篇幅。
>
> Reading time: ~10 min。

---

## 0. 如果你只讀 5 句話

1. **Loss 設計 = use case 翻譯**。`max(main) − max(side)` 那種
   single-headline metric 一定會被 optimizer 騙成尖峰；要寫 `min(main) − max(side)`
   外加 ripple penalty 與 distribution penalty (`mean(side)`)，三段缺一不可。
2. **Trajectory selection 比 final-step 還重要**。Adam 跑 1500 步後的 final 往往
   不是最佳；用 **joint early-stop**（在 trajectory snapshots 裡挑「worst 最高 AND
   flat-top 通過」的那個）穩定漲 +2 dB。
3. **Surrogate cold-start 幾乎必失敗**。R142-R145 連續 4 輪用 random data 訓 CNN
   都失敗（R² ≤ 0）。**Warm-start from analytical sim weights** → R²=1.000000。
   未來 patch HFSS 訓 surrogate 也應先建 analytical baseline 做 warm-start。
4. **跨 axes universal 之前先把單一 config 跑死**。我們花了 R94→R121 才把單個
   config 解到 distribution 滿意，再跨 inc/freq/steering/width 推論 (R123-R134)。
   反過來會被 axis combinatorial 淹沒。
5. **Surrogate noise 不是缺點，是 regularization**。R148 weight noise 0/5/10/20%
   全 PASS，且 noise 越大 mean worst 越**好**（joint early-stop 用 truth filter）。
   不要追求 R²=1.0，追求 R²>0.8 + 有 truth eval 即可。

---

## 1. Loss 設計

### 1.1 Single-headline metric 一定會被騙

R57-R63 用 `loss = -(max(main) - max(side))`：

```
metric report:    worst suppression = +30.99 dB  ← 看起來超棒
真實 binary 量化: worst = -18.21 dB              ← main 是單一尖峰，量化後塌掉
```

**心得**：任何「max / argmax / 單一頻點 / 單一角度」型 loss 都會讓 optimizer
找尖峰、犧牲整片區域。Patch S11 也一樣 — 不要用 `min(S11)` 過 spec，要用
`max over band(S11) < threshold`。

### 1.2 三段式損失的物理意義

```python
loss = -(soft_min(main) - soft_max(side))    # R94: worst-case
     + rw  * (soft_max(main) - soft_min(main))  # R94: ripple penalty (flat-top)
     + lambda_m * side.mean()                   # R119: distribution penalty
```

| Term | 用途 | 沒有它會怎樣 |
|------|------|--------------|
| `soft_min(main) − soft_max(side)` | 強制 main 整片打贏 side 最高點 | optimizer 出尖峰 |
| `rw·(max(main)−min(main))` | 抑制 ripple，逼成平頂 | main 中央凹陷穿過 -3 dB |
| `λ·side.mean()` | 把整片 sidelobe distribution 拉低 | side_max 達標但平均仍很高，量化敏感 |

### 1.3 為什麼是 `mean(side)` 而不是 L2 / ReLU

R118 比較 4 種 sidelobe formulation。`mean` 直接把整片 distribution 左移，
不傷 worst-case 也不傷 flat-top。L2 與 ReLU 都有 trade-off。

**通則**：當 metric 是「整片」性質，用 `mean()` 推 distribution；單點 worst
用 `soft_max/soft_min` 配 logsumexp(β=20)。

### 1.4 Soft min/max 的 β 選擇

`(1/β) · logsumexp(β·x)` β=20 在 dB scale 下接近 hard max 但保留可微梯度。
β 太小（<5）→ 太鈍，optimizer 不收斂到極值；β 太大（>50）→ 退化成 max，
GD 訊號稀疏。

---

## 2. Optimizer 與 Trajectory Selection

### 2.1 Adam (lr=0.05) × 1500 GD steps × multi-restart

是 RIS playground 上的穩定組合。R137 確認 800 步就足夠收斂；多跑只是保險。
**多 restart (≥3 seeds)** 才看得到 distribution，single seed 永遠樂觀。

### 2.2 Final-step 是個陷阱

Trajectory 中段常有比 final-step 更好的 snapshot — 因為 Adam 後期會在窄區域
震盪，把已經很 flat 的 main 推成微 ripple。

### 2.3 Simple early-stop 的災難

R139 試「step 中 worst 最高的 snapshot」→ flat-top 從 5/5 暴跌到 1/5
（Config B/C）。原因：worst 最高的那一步往往就是「main 變尖峰」的那一步，
worst 飆但 flat 全失。

### 2.4 Joint early-stop 是正解

R140：**「在所有 flat-top 通過的 snapshot 裡，挑 worst 最高的」**。
等於 hard-constrained 最佳化：先過 flat-top → 再比 worst。

```python
flat_valid = [s for s in snapshots if flat_top_ok(s)]
return max(flat_valid, key=lambda s: s.worst) if flat_valid else snapshots[-1]
```

R141 用 joint-ES 把 6 held-out configs 從 5/6 推到 6/6 PASS。
**Lesson：trajectory selection 屬於 methodology 一部分，不是 post-hoc tweak**。

---

## 3. Surrogate Engineering

### 3.1 Cold-start CNN 在 analytical sim 上幾乎必失敗

R142-R145 連續 4 輪：

| 嘗試 | 結果 | 失敗模式 |
|------|------|---------|
| R142 標準 CNN, random data | R²≈0 | stuck on mean response |
| R143 physics-aware arch, random | R²=-0.74 | overfit + log gradient 病態 |
| R144 physics-aware, trajectory data | R²=-3.21 | dynamic range 太大反而更糟 |
| R145 warm-start 含 bug | R²=-0.97 | indexing 錯了 |

**原因**：analytical RIS sim = `|sum_k W_k · exp(j·phase_k)|`，數學結構非常
具體。隨機 init CNN 從 random init 找這個 manifold 是 cold-start hard 問題。

### 3.2 Warm-start from analytical 是 turning point (R146)

把 `sim.pre_calAF[0]` 的實虛部直接複製到 surrogate 的 linear layer weights：

```
untrained R²:               1.000000
mean abs err:               0.000002 dB
```

**Patch implication**：HFSS surrogate 訓練之前，先建一個 analytical bridge
（physics-based forward model）做 warm-start。即使 analytical 不準，
warm-started CNN 收斂遠快於 cold-start。

### 3.3 Trajectory data 不一定比 random data 好

R144 用 R141 optimization 軌跡 5000 snapshots 訓，結果**比 random 還差**。
原因：trajectory 中響應 dynamic range（worst 從 -30 dB 走到 +3 dB）太大，
loss 被極端值主導。

**心得**：data distribution 對齊 inference 是常識，但「distribution range」
也要對齊。Trajectory snapshots 要 normalize 或限制 range。

### 3.4 Surrogate noise 是 regularization (R148-R149)

| Weight noise | R² | Mean worst | Verdict |
|--------------|------|-----------|---------|
| 0% | 1.0000 | +0.68 | PASS |
| 10% | 0.9267 | +0.87 | **PASS（更好）** |
| 20% | 0.7845 | +0.94 | **PASS（更好）** |

Joint early-stop 用 **truth eval (analytical sim)** 篩 trajectory。
Surrogate noise 等於 exploration noise；壞 pattern 被 truth 篩掉，
留下 noise-helped escape 的好 pattern。

**Patch implication**：HFSS surrogate 典型 R²~0.85-0.95 在 envelope 內 OK。
不要追求 R²=1.0；追求 R²>0.8 + truth eval 在 critical path。

### 3.5 Continuous-aware vs binary-only surrogate

R146 用 `(1-2x)` 只能處理 binary。GD 在 continuous params 跑，要寫 cos/sin 版：

```python
phase = x * pi
cos_p, sin_p = cos(phase).T.flatten(), sin(phase).T.flatten()
F_real = real_lin(cos_p) - imag_lin(sin_p)   # complex × complex
F_imag = real_lin(sin_p) + imag_lin(cos_p)
amp = sqrt(F_real^2 + F_imag^2)
out = 20*log10(amp/max(amp))
```

對 binary 退化成原 form。**Lesson**：surrogate forward 必須對 continuous input
正確；否則 GD gradient 在 binary boundary 附近爆炸或 0。

---

## 4. Hardware Constraints

### 4.1 1-bit pivot (R128) 是必要的痛

R94-R127 都用 2-bit (4 phase levels) 拿 +3 dB worst。R128 切到 1-bit 真實
hardware spec → 大量 config 退化到 +1 dB 或更糟。**這不是 methodology 失敗，
是 reality check**。

**心得**：所有 paper-quality 結果先用 hardware-realistic 約束跑一次。
2-bit、3-bit 只用來 ablation 對比，不寫進 deployment recipe。

### 4.2 Aperture-vs-X trade-off 是物理定律

| Trade-off | 證據 |
|-----------|------|
| aperture vs steering | R125-R127: +45° 即使 continuous phase 也只到 +1.32 dB |
| aperture vs bandwidth | R155: n=51 在 32% rel BW flat-top 崩潰 |
| aperture vs multi-target | R102: T1+T2 多目標 ~5 dB cost |

**通則**：當 worst-case 卡在低 +dB 位置又跨多個 axes，先用 bigger n 試
（R127, R133 都是 n=51 → n=71 rescue）。Recipe tuning 救不了物理 limit。

### 4.3 Fab tolerance 要先量

R136：phase noise up to 5% 在 R141 recipe 上仍 PASS。**Methodology 應該包含
fab tolerance budget**，不要等到 tape-out 才發現 ±2% 製程偏差就崩。

---

## 5. Recipe Selector 設計

### 5.1 從 grid search 蒸餾，不要硬寫死

`select_1bit_recipe(n, inc, freq, width)` 是 4D 決策樹，每個 branch 都對應某個
round 的 grid search 證據（R119/R129/R131/R133）。**不要憑物理直覺寫 selector**
— 你的直覺對 1-bit 量化的非線性效應是錯的。

### 5.2 Boundary refinement 是常態

R134 first-cut selector 在 width=15° fail（在 R119/R129 邊界）。
R135 把 boundary 從 width>20° 收緊到 width>12° → R141 6/6 PASS。

**心得**：selector boundary 永遠要從 held-out config validation 校正。
Grid 不夠細的地方一定有 valley，selector 跨 valley 會 fail。

### 5.3 何時 grid-search、何時 extrapolate

- **Grid search**：低維 (≤2 axes)、單個 config 的 recipe (rw, λ) tuning
- **Extrapolate**：跨 aperture (n=51 → n=71) 的 selector branch，先
  extrapolate 再 sample 1-2 點 verify

R141 對 n=71 是 extrapolate（R129 wide recipe scaled），驗證後直接放進 selector，
省掉完整 n=71 grid。

---

## 6. Multi-frequency / Broadband

### 6.1 Joint > single (R154)

3 個 in-band freqs (36/38/40 GHz, ~10% BW) `loss = sum_freqs L(R119)`：
- single-freq @38 → 36GHz mean +0.80, 一個 seed fail
- multi-freq joint → 全 freq mean ~+2.0+，**在 38GHz 也比 single 好 +0.46 dB**

**Mechanism**：joint loss 是 implicit regularization，optimizer 被困在多頻共識區。

### 6.2 BW limit 是 aperture 問題 (R155)

| BW | flat | verdict |
|----|------|---------|
| ~10% | 3/3 | PASS |
| ~32% | 1-2/3 | FAIL flat-top |
| ~53% | 1-2/3 | FAIL |

物理：每個 binary pixel 同時 contribute 所有頻率，BW 越寬 constraints 越多
vs aperture DOF。10% BW 是標準 patch 規格 → methodology 直接 cover。30%+ UWB
需要 architectural rethink (bigger n、relax flat-top、或多層結構)。

### 6.3 Patch BW 預算表

| Patch BW spec | Action |
|--------------|--------|
| 5-10% | direct deploy, R141 recipe |
| 20-30% | bigger n (n=71+) |
| 30%+ | architectural — 重評估 spec |

---

## 7. Failure Modes Catalog

不要重蹈的 dead-ends：

| 編號 | 失敗 | 為什麼 | 別再做 |
|------|------|--------|--------|
| F1 | `loss = -(max(main) - max(side))` | optimizer 找尖峰，量化敏感 | 用 worst-case |
| F2 | Single seed report | 樂觀 outlier | 至少 5 seeds，看 mean 與 min |
| F3 | Final-step trajectory | 後期 Adam 把 flat 推成 ripple | joint early-stop |
| F4 | Simple early-stop (worst max) | flat-top 崩潰 | joint (worst AND flat) |
| F5 | Random-init CNN surrogate cold-start | manifold 找不到 | warm-start from analytical |
| F6 | Trajectory data 訓 surrogate | dynamic range 過大 | normalize 或 truncate range |
| F7 | 追求 R²=1.0 surrogate | 不必要 | R²>0.8 + truth eval 篩 |
| F8 | 2-bit/3-bit 報結果當 deployment | hardware 不允許 | 1-bit ablation 才寫 paper |
| F9 | Recipe 憑直覺 extrapolate | 1-bit 量化非線性 | grid search 過邊界 verify |
| F10 | Single freq 跑 broadband 用 case | off-band 崩 | multi-freq joint sum loss |
| F11 | width=15° 用 narrow recipe | boundary valley | selector boundary 永遠校正 |
| F12 | inc=0 + mmWave 用通用 recipe | grating-lobe 區域特性 | rescue branch (R131/R133) |

---

## 8. 對 Patch Antenna Transition 的具體 takeaway

我們花 156 rounds 在 RIS playground 把 methodology derisk 到位；下面是
**直接搬到 patch 的 mapping**：

| RIS playground | Patch (HFSS surrogate-in-the-loop) |
|---------------|------------------------------------|
| Far-field 1D response (361 angles) | S-parameters (3 ports × 17 freqs) |
| main beam angular cap | in-band frequency window |
| sidelobes | off-band response / unwanted ports |
| Analytical sim (~ms) | HFSS (~minutes) — surrogate **mandatory** |
| Truth = analytical | Truth = HFSS COM (sparingly) |
| Worst-case + ripple + mean | Same loss skeleton, redefine main/side |
| Joint early-stop (worst AND flat) | Same — joint (S11<-10dB AND S21<-15dB) |
| Recipe selector (n, inc, freq, width) | Recipe selector (port count, BW spec, …) |
| Warm-start from `pre_calAF` | Warm-start from physics model (cavity / TLM) |

**最該怕的 risk 已 derisk**：
- ✓ Loss + workflow + 思維 transferable (R147)
- ✓ Imperfect surrogate (R²~0.85) 仍能跑 deployment (R148-R149)
- ✓ Multi-freq broadband loss 結構直接對應 patch S-param spec (R154-R155)

**剩下的 risk**：HFSS data 收集（patch 側 data engineering，不是 methodology 問題）。

---

## 9. Methodology 的 framework-agnostic deliverables

可以直接搬到任何 binary spatial optimizer 的 building blocks：

```
1. Loss skeleton
   loss = -(soft_min(main) - soft_max(side))
        + rw * (soft_max(main) - soft_min(main))
        + λ * side.mean()

2. Optimizer recipe
   Adam(lr=0.05), 800-1500 GD steps, ≥3 multi-restart seeds

3. Trajectory selection
   joint early-stop:
     pick max(worst) AMONG snapshots where flat_top_ok(s)

4. Recipe selector skeleton
   if hardware_realistic_axis crossings:
       branch to rescue recipe
   else:
       baseline recipe

5. Surrogate workflow
   warm-start from physics-based forward model →
   fine-tune with optimization-distribution data →
   keep truth eval in joint early-stop

6. Multi-spec extension
   loss_multispec = sum_specs loss_singlespec(spec_i)
```

每一塊都有 R94-R156 的 round 證據；換 domain 時 swap forward function 與
main/side 定義即可。

---

## 10. 結語

「156 rounds 的 RIS optimization 教會我們什麼？」

答：**真實可部署的 methodology 不是 metric optimization 問題，是
「loss 設計 + trajectory selection + hardware-realistic 約束 + 跨 axes 驗證
+ surrogate noise tolerance」**的整合系統。任何單一 piece 漂亮（高 R²、高 dB
worst）都不代表 deployment 過關。

下個碰到類似問題的人，先 internalize §0 的 5 句話、把 §7 的 failure modes
catalog 印出來貼在桌上、然後再開始寫 loss。會省下大概 100 rounds。
