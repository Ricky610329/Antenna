---
name: project_litreview_direction
description: 文獻定論 — 輸 random 屬預期;治本=不確定性/信任域門控 surrogate + active learning + harvest warm-start
metadata: 
  node_type: memory
  type: project
  originSessionId: 46183b31-ca6a-4a82-9ee0-98dfff4174bc
---

文獻調查結論(2026-06-26,4 個 web agent;詳見 `docs/research_landscape.md`):

- **我們輸 random best-of-N 在文獻預期內、不是 bug**:nanophotonics 權威綜述(arXiv 2307.08618)明說「單一設計目標 + 同一模擬器」時傳統/全域優化通常勝、DL 常更差且總成本更高;理論上「資料驅動法用同一模擬器永遠贏不過迭代法」;random best-of-N 是公認強 baseline → 輸它代表 baseline 夠硬,誠實寫即可。
- **病根**:對 NN surrogate 做梯度下降會收斂到 surrogate 的「奇點/對抗樣本」(類 FGSM)——梯度把預算花在「SM 說好、真 EM 爛」的假洞;G 是 deep-image-prior 重參數化(Hoyer NeurIPS'19,與我們 G 同構),強表達放大過擬合假洞;MATCH-OPT(ICML'24):解品質上界=surrogate 梯度差,**梯度準比值準更關鍵**(我們 SM 只用 MSE 擬合值)。
- **真 SOTA = adjoint 拓樸優化**(每步 2 模擬拿全梯度),但要可微 EM/伴隨場 → **HFSS 黑箱拿不到、擱置**。GLOnet 贏是因為用真物理梯度、刻意不用 surrogate。

**Why(治本方向,落地優先序):**
1. **C=攻 SM 品質**:給 SM 裝**不確定性(deep ensembles>MC-dropout)+ 信任域門**,G 梯度只在可信區生效、出界強制送真 HFSS(= source-critic/trust-region,把學長 elite 重訓原則化)。這是文獻第一優先、治本。
2. **warm-start 非真冷啟動**:先用 harvest(24189/10023)離線把 SM 訓到對歷史分布可信,再進線上。
3. **B=生成先驗(VAE,別急 diffusion——24k 太少)** 中期加分、服務 C(縮搜尋空間,引導仍靠 SM);最直接天線對照=VAE+test-time-optimization(arXiv 2505.18188)。
**How to apply:** 新方法用「worst-margin(dB) vs HFSS-call 曲線、對比 random best-of-N」誠實對標;贏不過 random 不要宣稱進步。關聯 [[project_benchmark_vs_random]] [[project_generator_hyperfeature_pivot]] [[project_sm_training_redesign]] [[project_radiation_pattern]]。
