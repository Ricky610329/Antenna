# 精讀筆記：2026 ISSCC 13.2 — 可控架構風格的 RFIC 端到端 AI 設計（Classical ↔ Non-Intuitive）

> 讀取方式說明：本 PDF 檔（`docs/reference/2026-ISSCC-...LNAs.pdf`，sha256 前綴 `f4f74f8a`）為 IEEE Xplore 下載的 3 頁 ISSCC digest。
> Metadata 標題／DOI 與內文一致，確認就是 13.2 這篇。頁碼採 digest 印刷頁：**p.230=正文首頁、p.231=正文次頁＋圖 13.2.1/13.2.2、p.232=圖 13.2.3–13.2.7**。
> ⚠ 讀檔過程踩到一個陷阱（已解決、非本論文問題）：共用 scratchpad 有其他 fork 正在處理 2024 MWSCAS 那篇，殘留同名 `page1.png` 覆蓋，第一次渲染誤看成別篇；改用唯一檔名 `fresh_p*.png` 重渲確認無誤。**全部三頁（含所有圖表）皆已讀，無「未讀」頁。**

---

## 1. 出處與一句話定位

- **Venue**：2026 IEEE International Solid-State Circuits Conference (ISSCC 2026), Session 13「Circuits for AI and AI for Circuits」, paper 13.2；發表 2026/02/17。DOI `10.1109/ISSCC49663.2026.11409170`。共 3 頁 digest（p.230–232）。
- **作者**：Jonathan Zhou\*¹, Emir Ali Karahan\*², Juho Park¹, Sherif Ghozzy¹, Kaushik Sengupta¹（¹Princeton University；²now with Marvell）。\*＝Equally Credited Authors。**注意：這是 Kaushik Sengupta 組**（與我們專案敘事背書的 Sengupta 三篇同一組；本篇引用其自家 [2][3][5]）。（p.230）
- **一句話**：提出一條**從 spec 到 layout 的統一演算法設計流**，專攻 mmWave/sub-THz **LNA**，用**強化學習（PPO）挑架構/拓樸/偏壓/介面阻抗**＋**逆向 EM 設計把目標 S 參數實體化成金屬幾何**；關鍵賣點是**「架構風格可控」**——同一組規格既能長成傳統傳輸線（classical）也能長成任意像素化（pixelated）結構，**性能等價**、差別只在人的可讀性/可除錯性。實測兩顆晶片（24–90GHz 可重構 diplexing LNA、80–160GHz dual-peaking LNA）皆達 SOTA。（p.230 abstract/vision）
- **對我們的核心關聯**：這篇把「**spec→幾何是一對多（underdetermined），所以『風格/可解釋性』是可以免費加的約束**」講成明確論點——正是我們「碎片雲冠軍」（人眼看不出邏輯但性能最好）的**外部理論背書**：pixelated 與 classical 性能無差、只差人眼是否讀得懂。

---

## 2. 方法管線（spec → GDS）

**輸入**＝規格（S 參數目標、NF、Gain、Pin/Psat 曲線）＋**架構風格選擇（classical t-line vs pixelated）**＋設計者約束（拓樸/面積/參數偏好、可放入人設計的 active/passive）。（p.230；圖 13.2.1）

**兩段式合成（pixelated 路徑）**：（p.230）
1. **RL 探索架構**：決定拓樸、級數（stage count）、元件尺寸（transistor sizing）、偏壓（biasing）、以及**各級介面阻抗（interface impedances）**，以滿足多目標規格。
2. **逆向 EM 設計**：把上一步指定的目標散射參數，用「所選風格」生成實體 EM 結構。

**兩種 pixelated 合成策略**（p.230）：
- (A) **forward AI model + metaheuristics** 產生 EM 結構（＝先有代理正向模型，再用啟發式搜尋反解幾何）；
- (B) 把**傳輸線參數直接塞進 RL**，做 circuit–EM 聯合最佳化。

**RL 引擎規格**（p.230 正文＋圖 13.2.2 網路方塊）：
- 形式化為 **Markov Decision Process**；policy network 把「設計狀態＋效能指標」映到「動作機率分佈」。State ＝ **Architecture + Circuits + EM Interface**（圖 13.2.2 標題）。LNA 以多個 cell 串接，級間有 `Z_out,n / Z_in,n` 介面，M 個 stage；EM 介面可選 Pixel 或 Classical。
- **Reward ＝ hard + soft targets**，對違反約束的偏差施罰。
- **演算法 ＝ PPO**（引用 Schulman 2017 [8]）。
- **Actor**：3 層 MLP、每層 128 神經元 → 圖示維度 `[52]→[128]→[128]→[128]→[32]`（輸入 52 維狀態、輸出 32 維動作分佈；動作含 ΔV_b1、Topology、ΔL_e1 等的機率分佈）。
- **Critic**：3 層 MLP、每層 256 神經元 → 圖示 `[52]→[256]→[256]→[256]→[1]`（輸出 1 維 value）。

**訓練規模數字（關鍵）**（p.230）：
- **~350,000 個 circuit–EM 範例**訓練；
- **~24 小時 on 192-core cluster**；
- 訓練資料**直接來自模型與「商用電路模擬器＋自製 EM 代理模型」互動**產生（≠ 350k 次真 EM！見 §5 解讀）；
- EM 代理模型的訓練資料，來自在**商用 EM 模擬器**跑「設計空間內隨機取樣」（**論文未揭露這個真 EM 樣本數**）。
- 訓練完成後：**output-stage layout 5–10 分鐘**合成；**完整 LNA（含輔助電路）數小時**內完成。
- circuit / EM 合成**分工（partitioning）**是端到端效率的關鍵。

**Pareto canvassing（圖 13.2.4，p.232）**：RL 可**快速掃出整個 PDK 的 Pareto front**。
- 3D bar：合成的一堆 LNA，軸 = Gain(dB) 5–35 × NF(dB) 4–9 × Power(mW) 0–25+，每根 bar 一顆合成 LNA。
- 「Pareto Fronts by Power Bin」散點（NF vs Gain），依 **PDC 五個功耗 bin** 上色並各畫一條 Pareto 前緣：**4.8–7.8 / 7.8–9.2 / 9.2–11.7 / 11.7–15.1 / 15.1–27.7 mW**。Gain 最高到 ~30dB、NF 4–9dB，數量目測約 150–200 顆合成點。
- 「Gain vs power」小圖：DC Power vs Gain 的**階梯狀 Pareto front**＋標「**Unachievable**」（斜線）不可達區。
- 三顆範例電路示意（含電晶體尺寸、偏壓、綠框內的**複數目標阻抗**）：`P_DC=5.4mW`（in 78+j16 / 53+j30、interstage 80−j18 / 28+j3、out 166+j83 / 46+j25，2μm×4 / 2.9μm×4）、`P_DC=11mW`、`P_DC=22.7mW`。綠框＝逆向 EM 要去實現的介面阻抗規格。

**與先前 inverse 工作的差異**：先前逆向工作**不允許選 EM 結構「type」（pixelated vs classical）**[4-6]；相較 [9]（Dall-EM，PCB 濾波器用 diffusion model），本篇**不訓練任何 diffusion model**，且是**on-chip mmWave/sub-THz passive 與電路合成在同一 tight loop**。（p.230）

---

## 3. 物理意義：「classical 與 pixelated 風格等價」在電磁上為何成立

**論文的實證**（圖 13.2.3，p.232，90nm SiGe BiCMOS）：拿同一組目標，逆向設計出的**像素化結構**與傳統**傳輸線結構**，在**全部散射參數（S11/S12/S22 幅值＋相位、30–100GHz）近乎重疊**。三組例子都成立。

**為什麼（用直白的話）**：
- 一個 matching network / passive，對外只透過**埠（ports）**與電路耦合。埠所「看到」的，只有**多埠 S 參數（等價地說：阻抗轉換行為）在頻帶上的整體散射**——**不是金屬長什麼樣子**。電路整合只在意埠行為。
- 「達成某組埠 S 參數」的金屬幾何**不唯一**。一片 2D 金屬圖樣的自由度（25×25 或 16×16 個 metal/no-metal 像素 → 指數級組合）**遠遠多於**約束數（頻帶上幾個複數 S 參數）。→ 解不是一個點，而是一整片**高維解流形（degenerate manifold）**。
- 因為 passive 是**線性、被動**，它被埠 S 參數**完全描述**；內部場怎麼繞、金屬怎麼碎，只要邊界（metal/no-metal 佈局）讓波在埠上散射結果相同，對外就一模一樣。傳統 t-line 只是這片解流形裡「光滑、可解析、人看得懂」的一個特解；碎片雲是同一流形裡「不直觀」的另一個特解。
- **深層洞見（給主線轉述）**：**spec→幾何是嚴重欠定（massively underdetermined）的**，所以你可以在「達標」之外，**免費疊加一個人類偏好的正規化項**——「長得像 classical」或「長得像 pixelated」——而**不犧牲性能**。風格是一個**自由變數**，不是性能取捨。這正是「架構風格可控」能成立的物理根據。

**依賴哪些物理性質**（成立的前提）：
1. **埠等價**：外部只在意埠 S 參數（電路整合的介面就是阻抗/散射）。
2. **設計空間夠大**：像素網格的組合數 ≫ 約束數，兩種「風格」的解都落在可行域內。
3. **線性/被動**：passive 可被 S 參數完全刻畫（非線性/主動元件不適用此等價論證）。
4. 頻帶有限：只要求「這段頻率」上匹配，未約束的自由度就成為可塑造風格的餘裕。

**對我們 25×25 patch 的直接映射**：我們的三標 spec（帶內 S11≤−10 / Gain≥+4、輻射 ±45°、可製造）約束的是**遠場輻射＋輸入匹配**——一樣是「埠/場的整體行為」，一樣**遠少於**625 個像素的自由度。所以我們的**碎片雲冠軍是這片欠定解流形上的合法非唯一解**：人眼讀不出設計邏輯，不代表它次優；反而是「沒有被人類 template 先驗綁死」才搜得到更好的角落。這篇是把這個直覺**在晶片上量測驗證過**的證據。

---

## 4. 關鍵數字表

### 4.1 訓練/合成成本（p.230）
| 項目 | 數字 |
|---|---|
| 訓練樣本 | ~350,000 circuit–EM 範例（來自模擬器＋EM 代理，非真 EM） |
| 訓練算力/時間 | 192-core cluster × ~24 小時，PPO |
| Actor / Critic | 3-layer MLP 128/層（`52-128-128-128-32`）/ 3-layer MLP 256/層（`52-256-256-256-1`） |
| EM 代理訓練資料 | 商用 EM 模擬器隨機取樣（**數量未揭露**） |
| 合成速度 | output-stage 5–10 分鐘；完整 LNA 數小時 |
| 展示頻段 | proof-of-concept LNA「30–160GHz」／abstract「24–150GHz」／圖 13.2.1「30–170GHz」（**多處口徑不一，見 §6**） |

### 4.2 晶片 1：可重構頻率-diplexing LNA（this work, 24–90GHz；圖 13.2.5/13.2.6/13.2.7）
| 指標 | 數值 |
|---|---|
| 製程 | 90nm SiGe BiCMOS（GlobalFoundries） |
| 架構 | Frequency-diplexing LNA，2-stage（Cascode + Common-emitter），**級配置由 AI 決定**；靠**逆向設計 3-port 結構**分頻 |
| 頻寬 | Combined 24–90GHz；Mode1（低頻）24–55GHz；Mode2（高頻）55–90GHz；可切三模（低/高/合併路徑） |
| Gain | Mode1 14–17dB；Mode2 14–16dB（abstract 記 peak 16dB） |
| NF | Mode1 3.8–5dB；Mode2 4–6.8dB（abstract/正文另記「3.8–5.7」「3.8–6.8」，口徑略異） |
| DC Power | Mode1 24mW；Mode2 32.4mW（正文：低頻模 ~24mW、高頻模 ~32mW） |
| Core Area | **0.70 mm²**（比對照組 0.06–0.37 明顯大——AI diplexing 架構付了面積代價） |

### 4.3 晶片 2：Dual-peaking LNA（this work, 80–160GHz；classical t-line；圖 13.2.5/13.2.6/13.2.7）
| 指標 | 數值 |
|---|---|
| 製程 | 90nm SiGe BiCMOS |
| 架構 | **3-stage Cascode cells + RL-optimized T-lines**（走 classical 風格路徑） |
| 頻寬/雙峰 | 75–95GHz 與 150GHz（peaks at 85 & 150GHz） |
| Gain | **30dB @85GHz** / **18dB @150–160GHz** |
| NF | **min 5.8dB @85GHz** |
| DC Power | 表：**63 mW**（⚠ 正文誤植「63mA」，單位不一致，見 §6） |
| Core Area | 0.46 mm² |
| 量測註記 | 150GHz 峰因「無 150GHz VNA」改用**小訊號源量測**（非完整 VNA S 參數，見 §6） |

### 4.4 對照組（圖 13.2.7，全為 Manual 設計，摘幾個對照）
- 表 1（vs 24–90GHz）：[10]Park ISSCC'22 27–38G/17.6dB/NF5.2–7.8/66mW/0.19；[11]Zhao JSSC'25 22.6–73.9G/15.2dB/NF4.06–4.94/17.5mW/0.06；[13]Yu JSSC'17 54.4–90G/17.7dB/NF5.4–7.4/19mW/0.37。
- 表 2（vs 80–160GHz）：[17]De Filippi JSSC'25 105–175G/23dB/NF5–6.5/60mW/0.109；[19]Moradinia MWTL'23 120–157.6G/26.5dB/NF7.2/20.6mW/0.69。
- **本工作賣點不是單項碾壓，而是「唯一 End-to-end AI flow（可選 classical/pixelated 風格）」＋ record BW（24–90 全 mmWave 帶）**；面積/功耗常大於對手。

---

## 5. 可移植配方（搬到我們的少樣本 per-task regime）

### ✅ 能搬（觀念層，直接強化我們敘事）
1. **「spec→幾何一對多、風格是免費約束」＝碎片雲冠軍的理論背書**（§3）。這篇在**晶片實測**上證明 pixelated≡classical 性能，只差人眼可讀性。→ 我們寫報告時可直接引為「非直觀像素解不劣於直觀解」的第三方證據，且**同一 Sengupta 組**（與我們既有背書鏈一致）。
2. **「canvassing the Pareto front of the PDK」的哲學**（p.230，「often an overlooked benefit」）。我們的**批次 HFSS 驗證線**本質就是在掃 achievable front（worst-margin × 面積/可製造）。可把我們的批次驗證重新框成「用有限預算掃出可行前緣」，而非只找單一冠軍。
3. **Reward = hard + soft targets、罰違反約束**：對應我們三標的 `worst_margin` 打分（帶內 S11/Gain＝hard、輻射/可製造＝可調權重）。設計語言一致。
4. **兩段式分工（RL 定架構/目標阻抗 → 逆向 EM 實現幾何）對應我們的 SM＋搜尋**：他們的「EM surrogate（CNN pattern→S 參數）」＝我們的 **SM**；他們的「inverse/metaheuristic 反解像素」＝我們的 **generator/搜尋**。**架構鏡像成立**。
5. **關鍵成本澄清（別被 350k 嚇到）**：那 350k **是代理/模擬器呼叫，不是 350k 次真 EM**。真 EM 只花在「訓練 EM 代理」（數量未公開）。→ 與我們**用 SM 代 HFSS**是同一招；我們少樣本（累計 ~2–3k 真值）對得上「真 EM 只用來養代理」的邏輯，不是他們攤提數十萬真模擬。這點要在轉述時講清楚，否則會誤以為規模不可比。

### ❌ 不能搬 / 規模不匹配
1. **離線大訓練攤提**：192-core×24h、350k 代理範例的 offline PPO，前提是「訓一次、之後分鐘級合成」。我們是**online per-task**，沒有這種一次性大攤提，也不做 train-once-deploy。
2. **RL-over-architecture（MDP、逐級建 LNA、選拓樸/偏壓/介面阻抗）**：那套是為「多級主動電路」設計的重機具。**我們的問題是單一 pattern（一片被動幾何），沒有多級電路要排**——RL 排架構的部分不適用；**只有「逆向 EM 把目標響應變成幾何」這一半對得上我們**。
3. **風格 toggle（classical t-line 參數化直接進 RL）**：我們目前只做 pixelated，不需要 t-line 分支。概念上若製造端要更「規整」，可考慮加「平滑/方塊化」風格正規化，但非當前需求。
4. **無 online / active learning**：他們是純離線訓練後部署；我們的價值主張正是 online per-task + active。這是**我們相對他們的差異化**（不是缺點），別去模仿他們的離線範式。

### 一句話配方
> 「借他們的**觀念**（一對多⇒風格免費、Pareto canvassing、hard+soft reward、SM＝EM surrogate 的鏡像），別借他們的**規模與 RL 重機具**。」

---

## 6. 侷限與審稿人視角

**他們的 human-in-the-loop 精確界線（重要，與我們敘事對照）**：
- **事前約束**：設計者輸入拓樸/面積/參數偏好、風格選擇（classical vs pixelated）、可放入人設計的 active/passive（圖 13.2.1「Designer Inputs」）。
- **事後挑選**：面對「一對多」的多個達標解，設計者依**主觀準則**（explainability, simplicity, area, sensitivity, debugging, interfacing）挑一個（p.230 明文）。
- **界線**：**只有「先約束、後挑選」，沒有 loop 中途的互動迭代**——人不會 turn-by-turn 去操控 RL 的搜尋軌跡。是「constrain then select」，不是「converse and refine」。
- **對我們的意義**：這正好標出**我們可以更進一步的空間**——我們的 agent 編排理論上能做「中途互動導引」（agent＋human-in-the-loop 共同調度），這篇沒做。寫報告要誠實：**這篇的 human-in-the-loop 比我們主張的『共同優化』更淺**，別把它當成互動式的先例；反過來，它是「事前/事後」型 human-in-the-loop 的乾淨範例。

**其他侷限（審稿人會盯的點）**：
1. **「風格等價、性能零代價」只有一組對照例（圖 13.2.3）**，沒有統計研究、沒有 ablation 說明等價「多常成立/何時失效」。單例證明。
2. **代理準確度未獨立公證**：整套依賴 EM 代理，但論文除了圖 13.2.3 那組比對，未系統性驗證代理 vs 真 EM 的誤差分佈；「GDS-ready 端到端模擬性能競爭 SOTA」是**模擬值**，最終晶片實測才是真憑據（僅兩顆）。
3. **量測基數小**：只有 2 顆晶片；且 dual-band 150GHz 峰**因無 150GHz VNA 改用小訊號源量測**（非完整 VNA S 參數）——18dB@150–160 的可信度打折。
4. **口徑不一致**：
   - Dual-peaking DC power：正文「63mA」vs 表「63mW」（**單位錯植**）。
   - 展示頻段：abstract「24–150」、正文「30–160」、圖 13.2.1「30–170」，高端 150/160/170 三種說法；低端 20/24/30 混用（Fig 13.2.5/6 標「20–90」但表標「24–90」）。
   - Mode 頻寬/NF 數字在正文與表間略有出入（NF 3.8–5 / 3.8–5.7 / 3.8–6.8 都出現過）。
5. **面積代價**：可重構 diplexing LNA 0.70mm²，遠大於對照 0.06–0.37mm²——AI 架構換來 BW/功能，付了面積。誠實看待「AI 一定更好」的敘事。
6. **「人做不出的新架構」是定性主張**：frequency-diplexing 2-stage「由 AI 決定」很有說服力，但「a human couldn't」屬修辭，非可證偽的量化。

**總評**：紮實的 proof-of-concept demo（兩顆 SOTA 晶片＋清楚的「風格可控」論點），但**不是通用性的證明**；規模數字（350k/192-core）要正確解讀為「代理攤提」而非「真 EM 攤提」。對我們最有價值的是**§3 物理論點**與**§6 human-in-the-loop 界線**，而非其 RL/離線工程細節。

---

## 7. 金句引用（原文＋頁碼）

> **[一對多 / 取捨由人決定 — 核心金句]** (p.230)
> "Mapping from an RFIC to its specifications is unique, whereas the reverse mapping—from specifications to a realizable RFIC (architecture, topology, circuit, EM, and parameters)—is inherently one-to-many. Multiple designs can satisfy the same specifications, with final choices often shaped by subjective considerations such as explainability, simplicity, area, sensitivity, debugging, and interfacing with ancillary circuits. Thus, controlling the evolution of architecture is essential for the broader adoption of AI-enabled RFIC design."

> **[風格等價、性能零代價]** (p.230)
> "It is important to note that the architectural choice between classical and non-intuitive does not come at a cost [in] performance. As can be seen in Fig. 13.2.3, different EM structures (both classical and non-classical) can achieve similar EM performance that can be guided through an algorithm."

> **[圖 13.2.3 caption — 全 S 參數近乎相同]** (p.232)
> "The same EM performance can be realized by either inverse designed pixelated structures or more 'classical' transmission-line based structures as demonstrated by the example in 90nm SiGE BiCMOS showing near-identical performance across all scattering parameters."

> **[Pareto canvassing — 常被忽略的好處]** (p.230)
> "AI not only allows synthesis of particular circuits, but also allows canvassing of possible designs in a rapid fashion, allowing us to evaluate the Pareto front achievable in a given process design kit. This is often an overlooked benefit."

> **[設計者偏好 / 可解釋 / 可除錯 = 賣點]** (p.230)
> "this framework supports controlled architectural evolution, enabling designer preference, interpretability, and seamless integration with surrounding circuits—facilitating debugging, fault isolation, and broader usability."

> **[先前逆向工作不允許選「風格」]** (p.230)
> "Through this, the designer can enforce their own constraints on topology, architecture, and interpretability, where prior inverse works did not allow for EM structure 'type' selection between pixelated or classical [4-6]."

> **[350k / 24h / 192-core — 訓練規模原文]** (p.230)
> "Training with ~350,000 circuit–EM examples requires ~24 hours on a 192-core cluster using proximal policy optimization (PPO) [8] with a three-layer multilayer perceptron (MLP) actor of 128 neurons per layer and a three-layer MLP critic of 256 neurons per layer. The training data is generated directly from the model's interactions with a commercial circuit simulator and a custom EM surrogate model..."
