# 二值化相關文獻表 — BiScaleNorm 的同類方法與梯度設計參考

> 建立：2026-06-24。**這是 study 筆記,不是定案、未改任何 code。** 任何移植到本專案前須先測 golden 並與使用者討論(尤其碰到 loss/正規化語意時)。
>
> 動機:`antenna/models/generators.py` 的 `BiScaleNorm` 把 G 輸出穩定到固定值域([−1,1]、保留零點),讓下游 `AntennaPattern.binarization()`(sigmoid + mean 門檻 + tau)有穩定、可比的語意。學長等於獨立重造了文獻裡兩條成熟路線的雛形——這份表把那些路線記下來,供之後挑選優化方向。

---

## 1. 一句話定位:本專案的二值化在文獻地圖上的位置

本專案的核心步驟 `G(spec) → logits → 二值化 → 0/1 像素圖 → SM/HFSS → loss → 反傳`,在學界其實是兩個獨立領域各自發展、且高度同構的問題:

| 領域 | 它怎麼稱呼這件事 | 對應本專案 |
| --- | --- | --- |
| **二值化神經網路 / 量化(BNN/Quantization,ML 界)** | 用 STE 讓 `sign`/`round` 可反傳;二值化前先「整形」權重/激活分布 | `pattern.py` 的 STE、`BiScaleNorm` |
| **密度法拓樸最佳化 / 反向設計(Topology Optimization,工程界)** | 連續 density∈[0,1] → 用平滑 **Heaviside 投影**(陡峭度 β)推成 0/1,β 連續化退火 | `binarization()` 的 sigmoid+tau、ACP 排 tau |

**關鍵洞見:** 你們的 `sigmoid(steepness·(x − mean))`,在拓樸最佳化界就是 **tanh 平滑 Heaviside 投影**(Wang/Lazarov/Sigmund 2011),`tau ↔ 1/β`、`mean 門檻 ↔ 投影閾值 η`。那邊有 20 年的收斂性、穩健性、灰度懲罰、最小尺寸控制的分析可借。而 `BiScaleNorm`「二值化前先把分布整形/縮放」這件事,在 BNN 界叫 **balance + scale before sign**(XNOR-Net 的 α、IR-Net 的 Libra-PB、ReActNet 的可學門檻)。

---

## 2. 文獻表

連結以實際查得的頁面為準(arXiv / 期刊 / GitHub)。「對本專案的意義」欄標 ★ 者最值得先看。

### A. 可微分二值化與梯度估計(STE 家族)— 對應 `pattern.py::binarization()`

| 論文(年/會議) | 核心一句話 | 對本專案的意義 | 連結 |
| --- | --- | --- | --- |
| Bengio, Léonard, Courville, *Estimating or Propagating Gradients Through Stochastic Neurons* (2013) | STE 起源:前向走硬門檻,反向當恆等函數(梯度直通) | 你們 `(hard−soft).detach()+soft` 的理論依據 | [arXiv:1308.3432](https://arxiv.org/abs/1308.3432) |
| Courbariaux, Bengio, David, *BinaryConnect* (NeurIPS 2015) | 訓練保留全精度權重、前向二值化;權重夾在 [−1,1] | 「全精度 logits + 二值前向」的原型,與你們一致 | [arXiv:1511.00363](https://arxiv.org/abs/1511.00363) |
| Hubara et al., *Binarized Neural Networks* (2016) | 權重與激活都 ±1;STE 帶飽和項 `1_{|x|<1}` 截掉大梯度 | ★ 飽和截斷 = 你們 `clamp(±10)` 的精神;BNN 也靠 BatchNorm 喬門檻 | [arXiv:1602.02830](https://arxiv.org/abs/1602.02830) |
| Yin et al., *Understanding Straight-Through Estimator...* (ICLR 2019) | 理論分析:為何「恆等」這種有偏估計仍能收斂、何時會壞 | ★ 想知道 STE 何時不穩、該配什麼,先讀這篇 | [arXiv:1903.05662](https://arxiv.org/abs/1903.05662) |
| Liao et al., *Real-time Scene Text Detection with Differentiable Binarization (DBNet)* (AAAI 2020) | 把二值化寫成可微近似函數,門檻變成「可學、逐位置自適應」 | 函式同名;示範「門檻不必固定,可讓網路自己學」 | [arXiv:1911.08947](https://arxiv.org/abs/1911.08947) |

### B. ★ 二值化前的「分布整形/縮放」— 對應 `BiScaleNorm`(最相關)

| 論文(年/會議) | 核心一句話 | 對本專案的意義 | 連結 |
| --- | --- | --- | --- |
| Rastegari et al., *XNOR-Net* (ECCV 2016) | 二值權重配一個縮放因子 `α = mean(|W|)`(L1),讓 ±1 逼近實值 | ★ 與 BiScaleNorm 同類(都在「縮放」),但用 **均值**而非 max/min → 對極值穩健、不會被單一像素主宰尺度 | [arXiv:1603.05279](https://arxiv.org/abs/1603.05279) |
| Qin et al., *IR-Net (Libra-PB + EDE)* (CVPR 2020) | 二值化前先「**平衡(零均值)+ 標準化(除標準差)**」,使二值權重資訊熵最大、sign 穩定;反向用 EDE 漸近 sign | ★★ 最貼近 BiScaleNorm 的目標(穩定分布再二值化),且講清楚「為何要 zero-mean + std」。用統計量而非極值 | [arXiv:1909.10788](https://arxiv.org/abs/1909.10788) |
| Liu et al., *ReActNet (RSign / RPReLU)* (ECCV 2020) | 把 sign 的門檻、PReLU 的位移都做成**可學參數**,顯式學分布的 reshape/shift | ★ 「門檻該放哪」交給學習,而非由資料 max/min 硬算;且用 PReLU(你們 GEN 也用) | [arXiv:2003.03488](https://arxiv.org/abs/2003.03488) |

### C. 溫度 / 陡峭度退火 — 對應 `tau` 與 ACP 排程

| 論文(年/會議) | 核心一句話 | 對本專案的意義 | 連結 |
| --- | --- | --- | --- |
| Jang, Gu, Poole, *Categorical Reparameterization with Gumbel-Softmax* (ICLR 2017) | 用溫度 τ 的 softmax 近似離散抽樣;τ→0 趨近 one-hot,訓練時退火 | ★ 你們 tau 的「軟→硬」退火思路的標準參照;附帶討論退火排程 | [arXiv:1611.01144](https://arxiv.org/abs/1611.01144) |
| Maddison, Mnih, Teh, *The Concrete Distribution* (ICLR 2017) | 同期姊妹作:連續鬆弛離散變數的機率框架 | 退火二值化的理論底子 | [arXiv:1611.00712](https://arxiv.org/abs/1611.00712) |
| Gong et al., *Differentiable Soft Quantization (DSQ)* (ICCV 2019) | 用 tanh 做可微軟量化,訓練中逐步逼近硬量化(內建退火) | tanh 版的 soft→hard,與 sigmoid+tau 同型,可比較 | [arXiv:1908.05033](https://arxiv.org/abs/1908.05033) |
| Choi et al., *PACT* (2018) | **可學的截斷上界** α(取代固定 clip),用 STE 學 | ★ 你們 `clamp(±10)` 是寫死的;PACT 示範把截斷邊界變可學 | [arXiv:1805.06085](https://arxiv.org/abs/1805.06085) |
| Esser et al., *Learned Step Size Quantization (LSQ)* (ICLR 2020) | 量化步長可學,且校正梯度尺度 | 處理「量化邊界的梯度尺度」,對應你們梯度集中/尺度問題 | [arXiv:1902.08153](https://arxiv.org/abs/1902.08153) |

### D. ★ 拓樸最佳化 / 反向設計(本問題的工程本家)

> 這一區最值得看。天線像素「金屬/不金屬」= 經典 density-based topology optimization,且他們的「平滑 Heaviside 投影 + β 連續化」就是你們 sigmoid+tau 的成熟版,有完整收斂/穩健/可製造性理論。

| 論文(年) | 核心一句話 | 對本專案的意義 | 連結 |
| --- | --- | --- | --- |
| Bendsøe & Sigmund, **SIMP** 法(1989 起;專書 2003) | 連續 density 配懲罰指數 p,讓中間灰度「不划算」→ 收斂到黑白 | ★ 「懲罰灰度逼出二值」的根:可考慮把 grayscale penalty 當正則,而非只靠 STE | [可微教學(FEniCS)](https://comet-fenics.readthedocs.io/en/latest/demo/topology_optimization/simp_topology_optimization.html) |
| Guest, Prévost, Belytschko, *Achieving minimum length scale... projection functions* (IJNME 2004) | 首創用平滑 Heaviside 投影把 density 推成 0/1,同時控制最小特徵尺寸 | Heaviside 投影源頭;「投影=可微二值化 + 尺寸控制」 | 期刊 IJNME 61(2):238–254(DOI 見內文) |
| **Wang, Lazarov, Sigmund**, *On projection methods, convergence and robust formulations...* (Struct Multidisc Optim 2011) | tanh 平滑 Heaviside `H(ρ,β,η)`,β 控陡峭度並**連續化(continuation)**;提出 robust(eroded/intermediate/dilated 三場)公式 | ★★ 與 `sigmoid(steepness·(x−mean))` 數學同型:`tau↔1/β`、`mean↔η`。robust 三場、β 排程值得借 | [ResearchGate](https://www.researchgate.net/publication/227263817_On_projection_methods_convergence_and_robust_formulations_in_topology_optimization) |
| Vercruysse et al., *Analytical level set fabrication constraints for inverse design* (Sci. Rep. 2019) | 把「最小間隙/曲率」等可製造性寫成可微懲罰項,與性能一起優化 | ★ 你們已有 TV/island/SC/gap loss;這篇是同方向的解析式可製造性約束參考 | [Nature Sci. Rep.](https://www.nature.com/articles/s41598-019-45026-0) |

### E. 應用域:像素天線 / EM 二值圖樣設計(背景脈絡)

| 論文(年) | 核心一句話 | 對本專案的意義 | 連結 |
| --- | --- | --- | --- |
| Guo et al., *Topology optimization design of antennas with complex radiation characteristics* (MOTL 2024) | 直接對天線做密度法拓樸最佳化(含方向圖需求) | ★ 與你們方向圖→loss 的方向最貼近的天線拓樸最佳化案例 | [Wiley MOTL](https://onlinelibrary.wiley.com/doi/10.1002/mop.33649) |
| *Versatile unsupervised design of antennas...* (Sci. Rep. 2024) | 用 ML + 彈性參數化設計天線 | ML 設計天線的近期同類工作,可對標 | [Nature Sci. Rep.](https://www.nature.com/articles/s41598-024-80319-z) |
| 像素天線 + BPSO / 特徵模態(多篇) | 傳統做法:基因/二元 PSO 搜尋像素開關 | 你們梯度法的「對照組」(離散搜尋 vs 可微) | (見 §3 搜尋關鍵字自行延伸) |

---

## 3. 如果只先讀三篇

1. **Wang, Lazarov, Sigmund 2011(Heaviside 投影)** — 你們 sigmoid+tau+mean 門檻的「正規成熟版」,且附收斂與穩健性分析。讀完會知道 β(=1/tau)該怎麼排、為何要 robust 三場。
2. **IR-Net / Libra-PB(Qin 2020)** — `BiScaleNorm`「先整形再二值化」的 ML 對應;它解釋為何用 **zero-mean + 標準差**(統計量)而非極值,正好點到 BiScaleNorm 的弱點。
3. **ReActNet / RSign+RPReLU(Liu 2020)** — 把「門檻/位移」做成**可學參數**,且用 PReLU(你們 GEN 也用),是「BiScaleNorm 之外門檻還能怎麼放」最直接的啟發。

---

## 4. BiScaleNorm vs 文獻做法:差異與可借鑑點

`BiScaleNorm` 的特徵:**逐樣本(per-pattern)、正負兩側分開、用 `max` / `|min|`(極值)當縮放分母、零參數、固定不可學**。

| 面向 | BiScaleNorm 現況 | 文獻常見做法 | 啟發 |
| --- | --- | --- | --- |
| 縮放尺度來源 | **極值** `max(v)` / `\|min(v)\|` | 統計量:`mean(\|·\|)`(XNOR-Net α)、`std`(IR-Net) | 極值會被單一像素主宰 → 梯度集中在 argmax/argmin、尺度逐步跳變;改用均值/標準差更穩 |
| 門檻位置 | 下游用 `mean` 當門檻(固定規則) | 可學門檻/位移(ReActNet RSign、DBNet、PACT) | 門檻可考慮做成可學或更穩健的統計量 |
| 對稱性 | 正負各自獨立縮放(雙側) | IR-Net 平衡到零均值後對稱標準化 | 雙側各除不同極值會讓正規化後均值≠0、分布形狀被不對稱拉扯 |
| 數值安全 | 無 eps;`max=0` 時 `torch.where` 兩支都求值 → 反向可能 NaN;`max` 很小 → 梯度爆炸 | 量化法多半夾值/加 eps/截梯度 | 加 eps 與 grad clip 是低風險防護(見 CLAUDE.md 待優化清單) |
| 退火 | tau 由 ACP **循環** | Heaviside β 多為**單調連續化**;Gumbel τ 單調退火 | 「循環 tau」是本專案較特別的選擇,值得對照單調 β 連續化看收斂差異 |

> 共通結論:文獻幾乎都用**統計量(mean/std/L1)或可學參數**當二值化的尺度與門檻,而非 `max`/`min` 極值。BiScaleNorm 用極值是它「想法對、但實作最脆弱」的地方——這與先前在梯度面觀察到的「梯度集中於極值像素、小分母爆炸、零分母 NaN」完全一致。

---

## 5. 備註(誠實聲明)

- 本表為文獻 study,**未修改任何程式碼**,golden 不受影響。
- 表中連結均來自實際網路查詢結果或標準 arXiv/期刊頁面;少數工程界期刊(Guest 2004 等)僅給文字引用 + DOI 提示,未附可能出錯的連結。
- 任何「把某方法移植進來」的決定(例如把 BiScaleNorm 的 max/min 改成 std、或加 grad clip / eps / 可學門檻),都屬於會動到數值或 loss 語意的改動,**須先測 `python -m pytest tests/ -q` 保 golden,並先與使用者討論再動**。
