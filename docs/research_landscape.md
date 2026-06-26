# 文獻地景:NN-surrogate 反向設計 — 我們的定位、病根與路線

> 由 4 個文獻調查 agent(WebSearch/WebFetch)彙整(2026-06-26)。companion 文件:[`senior_method.md`](senior_method.md)(學長方法)、benchmark 結論見記憶 `project_benchmark_vs_random`。
> 可信度標:**高**=同行評審/逐句核實;**中**=單篇預印本/自評;**低**=僅二手摘要。

## 一句話定位
學長這套(生成器 G + **學來的 NN surrogate** 梯度引導 + STE 二值化 + 退火 + elite 重訓)= **2019 GLOnet 時代舊範式的變體**,而且選了文獻**點名有系統性缺陷**的子路線:「讓生成器在一個沒有不確定性/信任域約束的 NN surrogate 上自由跑梯度」。我們**輸給 random best-of-N**,正是文獻預測的典型症狀,**不是 bug**。

## 1. 我們的結果在文獻預期內(誠實定位)
nanophotonics 反向設計權威綜述(逐句核實,arXiv 2307.08618 / [PMC11501815](https://pmc.ncbi.nlm.nih.gov/articles/PMC11501815/)):
- 「**簡單的傳統方法常常勝過笨重的 GPU 黑箱優化**」;
- 「**若是單一設計目標,即使複雜場景,傳統全域優化通常更划算**」;
- 「**深度學習在很多情況反而更差,且加上資料生成+訓練的總成本更高**」;
- 「production-scale metasurface 設計全用傳統查表,DL 只在 toy problem 測試」。
- 理論:**資料驅動法若基於「產生訓練資料的同一模擬器」,永遠贏不過迭代法**。
- NAS 文獻([1902.07638](https://arxiv.org/abs/1902.07638)):**random best-of-N 是出名的強 baseline**(好解密度高的空間隨機天生強)。
→ 我們是「單一 spec + 同一 HFSS」,**輸 random 完全在預期內**,且代表 baseline 夠硬(避開 phantom progress)。可信度:**高**。

## 2. 病根(四份交叉印證的因果鏈)
1. **Surrogate adversarial 鑽洞(核心病)**:對 NN surrogate 輸入做梯度下降,會收斂到 surrogate 的「奇點/對抗樣本」(機制類似 FGSM)——梯度把預算花在「**SM 說很好、真 EM 很爛**」的假洞。(綜述逐句核實,PMC11501815)可信度:**高**。
2. **DIP 放大**:G 是 deep-image-prior 式神經重參數化(Hoyer NeurIPS'19,[1909.04240](https://arxiv.org/abs/1909.04240),與我們 G 同構;CNN 隱式正則→更平滑連通,實證 99/116 達最佳 vs 基線 66/116)。好處真實,但**強表達+易過擬合 → 過擬合假洞更兇**;DIP **必須 early-stopping**([2112.06074](https://arxiv.org/abs/2112.06074)),而「何時停止信任 SM」沒解。可信度:**高**。
3. **STE 梯度失配**:forward 硬二值/backward 軟代理會 bias 更新,是已知代價(退火緩解)。可信度:**高**。
4. **MATCH-OPT(ICML'24,[v235/hoang24a](https://proceedings.mlr.press/v235/hoang24a.html))**:用 surrogate 優化,解品質**上界 = surrogate 與真函數的「梯度差」最大值**——**「梯度對不對」比「值對不對」更關鍵**。我們 SM 只用 MSE 擬合響應值,沒管梯度。可信度:**高**。
→ 我們「best pattern 爛、輸 random」= 以上合症。

## 3. 什麼會贏(文獻共識)
- **真正 SOTA(單目標高自由度)= adjoint 拓樸優化**:每步 2 次模擬拿全場梯度、可上 10⁵–10⁶ 自由度。**前提:可微 EM 求解器/伴隨場**。**HFSS 黑箱沒有 → 對我們多半不可行**(除非換 solver)。可信度:**高**。
- **GLOnet([Nano Lett 2019](https://github.com/jonfanlab/GLOnet))贏 adjoint,是因為用「真物理梯度」、且刻意拿掉 surrogate**——不是「ML 打敗物理」。我們用學來的 surrogate 梯度正是它避開的。可信度:**高**。
- **無物理梯度時的主流答案 = trust-region / domain-confinement surrogate + active-learning infill + 不確定性門控**:
  - Source Critic Regularization(NeurIPS'24,[2402.06532](https://arxiv.org/abs/2402.06532)):用判別器把優化軌跡**約束在 surrogate 可信分布內** = 學長「elite 重訓」的原則化版。
  - 天線界 domain-confinement surrogate([Sci Rep 2025](https://www.nature.com/articles/s41598-025-91643-3))、變保真 active learning 約 **~200 次等效高保真 EM** 達標(省 40–90%,[PMC12214509](https://pmc.ncbi.nlm.nih.gov/articles/PMC12214509/))。
  - 不確定性估計:**deep ensembles > MC-dropout**(校準好)。可信度:**高/中**。
- **生成先驗的正確用法 = 學「合理流形」+ forward 代理/物理梯度引導(DPS / classifier guidance)**,不是純條件生成:
  - 最直接天線對照:**矩形貼片 VAE + test-time optimization**([2505.18188](https://arxiv.org/abs/2505.18188))——生成後用 forward 代理梯度精修 + 用代理**不確定度**挑候選,與我們 `G→SM→反傳` 幾乎同構;明說「高性能設計在資料集中稀少」。
  - diffusion 先驗**資料飢渴**(MetaGen 用 **360 萬**筆,[2506.21748](https://arxiv.org/html/2506.21748v1))→ 我們 **24k 太少** → 走「**弱先驗(VAE)+ 強代理引導**」。
  - 0.1% 達標的不平衡:百分位條件([2510.05160](https://arxiv.org/abs/2510.05160))/ 負樣本約束([2306.15166](https://arxiv.org/pdf/2306.15166))。可信度:**中**。

## 4. 對 B / C 的裁決 + 優先序(針對:HFSS 黑箱、無梯度、有 ~2 萬筆 harvest)
1. **C(攻 SM 品質 / active learning)= 文獻第一優先、治本**:
   - 給 SM 裝**不確定性 + 信任域門**:G 梯度只在 SM 可信區生效,**出界強制送真 HFSS、不信 SM**(= source-critic / trust-region,把 elite 重訓升級)。
   - 不確定性:**deep ensembles**(優於 MC-dropout;roadmap item4 原寫 MC-dropout 可改)。
   - 「何時停止信任 SM 梯度」= DIP early-stopping 對應物(SM vs 真 EM 在 elite 點落差當觸發)。
2. **warm-start 不是真冷啟動**:我們有 harvest 24189/10023 筆 → **先離線 warm-start SM** 再進線上(對應 radiation Stage 3 待辦),別假裝從零。
3. **B(生成先驗)= 中期、加分非主力**:harvest 學 latent 先驗(**VAE 即可,別急 diffusion——資料不夠**),把搜尋限在內插區;**B 服務 C**——先驗縮小搜尋空間,引導品質仍靠 SM。
4. **adjoint(路線 A)**:只有願意換可微 EM solver 才考慮;HFSS 黑箱下擱置。

## 5. 建議下一步(可排路線圖)
1. **治本實驗**:SM 加不確定性門(ensemble)+ 信任域;G 梯度出界 → 強制真 HFSS。**目標:贏過 random best-of-N**(現在的鐵門檻)。
2. **warm-start SM from harvest**(離線),再進線上。
3. **誠實對標**:G+SM 線上 vs trust-region surrogate + active-learning,**同 HFSS 預算比 worst-margin(dB) vs HFSS-call 曲線**(沿預算、不只終點)。若 G+SM 仍輸 → 主路徑換路線 B,G 降級為「候選產生器」而非「被 SM 梯度牽著走的主體」。

## 誠實聲明
- 「線上 co-training(迴圈內持續更新 surrogate)+ 生成器 + STE 二值像素」這個**精確三合一**未找到天線域直接對應論文(負面結論,非不存在)——既是風險也是潛在貢獻點。
- 量化數字(GA +94%、SB-SADEA 3–7×、GLOnet 勝 adjoint 等)多為單篇自評,可信度中,引用打折。
- 逐句核實:newcomer's guide、Neural-Adjoint benchmark、Hoyer reparam、Early-Stopping-DIP;其餘部分依二手摘要,已標可信度。
