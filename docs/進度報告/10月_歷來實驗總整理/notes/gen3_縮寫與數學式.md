# 第三代 吳維文：縮寫全稱與方法數學式，第五版素材

> 整理日期 2026-10-24。用途是「歷來實驗總整理」第五版第三代各頁補縮寫全稱，並新增方法的數學式。依據是 `docs/唯一真實.md` §1.4 的 10-24 條：縮寫第一次出現補全稱、各代方法配一式一行的簡單數學式、式子要和碩論或程式一致。
> 範圍：第四版第 12–16 頁的正文、圖上文字、時間軸泡泡、備忘稿，以及第 32 頁名詞表中屬於第三代的列。第 2、3 頁的第三代列沒有縮寫。
> 素材與頁碼寫法：
> - 碩論定稿：ailab NAS `電子論文.pdf`，複本在 scratchpad `math_gen3/thesis_final.pdf`，與 `thesis_gen3/src/thesis.pdf` 的 MD5 相同。寫成「p.20／PDF 30」，前者是正文頁碼，後者是 PDF 頁。
> - 口試簡報 `Oral.pptx` 寫成「口試 s62」；月報與實驗室簡報沿用 `gen3_碩論對照.md` 的簡稱，例如「202512 s2」「G11 s6」，頁號依 scratchpad `gen3/txt/` 的抽字結果。
> - 程式只拿來對照，式子以碩論為準；對照的是 repo `antenna/losses.py`、`antenna/optim/scheduler.py`。

## 0 先講結論

- **縮寫：** 主表 26 項，都在第四版第三代頁面或名詞表出現。碩論查得到全稱或命名的 12 項；只有月報或實驗室簡報有全稱的 3 項，ACS、IS、GC；全稱屬通用名稱、碩論沒寫的 9 項；查無全稱的 2 項，RO4003C 與 K 接頭，兩者都是型號名。另列圖上英文術語與第四版沒出現、但第五版放式子後會出現的詞。
- **數學式：** 選 5 式，依一次迭代的資料流排：ACP 排程、二值化溫度 τ、只罰不達標的響應 loss、SC loss、DLF 篩選。饋入連通率列為備選。5 式中 4 式等同碩論原式，只把幾條式子併成一行；響應 loss 與 DLF 標了簡化表示。
- **放法：** 另開一頁「第三代方法的數學式」接在第 12 頁後面，一張 5 列表格。第 12 頁右欄不放式子，只把三項機制的小標補中文全稱；英文全稱放新頁與名詞表。
- **要先修的四處：** 第四版備忘稿的「ACS」是我們自己縮的，月報沒有這個縮寫；第 12 頁圖上寫 Adaptive Cyclical Scheduler，碩論正文定名 Adaptive Cyclical Policy；「Base」在碩論三個實驗各指不同設定；第 12 頁圖上的 Island Suppression 指 SC loss，不是月報的 IS loss。細節見 §2.7。

---

## 1 縮寫表

欄位說明：「出處」寫最早給出全稱的位置；「通用名稱」表示碩論與月報都沒有展開、這裡用業界通用的全稱；「查無全稱」表示是型號或名稱，不是縮寫展開。「第四版位置」寫它在第四版出現的頁與位置。

### 1.1 三項機制與生成架構

| 縮寫 | 英文全稱 | 中文譯名 | 白話一句 | 出處 | 第四版位置 |
|---|---|---|---|---|---|
| GEN | Gradient Estimation Network | 梯度估計網路 | 第一代起的代理模型，學 HFSS 的響應，好把 loss 的梯度傳回生成器 | 碩論 p.1／PDF 11、p.6／PDF 16，角色說明 p.39 | p12 描述句與圖上 |
| ACP | Adaptive Cyclical Policy | 自適應循環策略 | 學習率與二值化溫度一起週期升降，停滯就提前重啟 | 摘要 p.ii、p.1／PDF 11、§3.2 p.20／PDF 30 | p12 右欄、p13 泡泡、p15 表、p32 |
| ACS | Adaptive Cyclical Scheduler | 自適應循環排程器 | ACP 定名前的名稱，同一個東西 | 202512 s2–3、G08 s19；碩論圖 3-11 圖上也寫 Scheduler，p.38／PDF 48 | p12 圖上全名、p13 備忘稿「當時叫 ACS」 |
| DLF | Dynamic Loss Filter | 動態損失過濾 | 每次重訓代理模型前，只挑優於歷史平均 loss 的樣本 | 摘要 p.ii、§3.5 p.35／PDF 45 | p12 右欄與圖上、p13 泡泡、p15 表、p32 |
| SC loss | Spectral Connectivity Loss | 圖譜連通度損失 | 把像素當成圖，圖越容易被切開 loss 越大 | 摘要 p.ii、§3.3 p.28／PDF 38 | p12 右欄與圖上、p13 泡泡、p15 圖與表、p32 |
| Linear、Reset、Peak | 三種重啟策略名，不是縮寫 | 線性回升、從頭暖身、直接跳峰值 | 停滯重啟時學習率從哪裡接回；碩論採 Linear | 表 3-1 p.24／PDF 34 | p13 備忘稿 |
| CNN | Convolutional Neural Network | 卷積神經網路 | 試過當生成器，小樣本下難穩定訓練 | p.5／PDF 15、p.37／PDF 47 | p13 備忘稿 |
| GAN | Generative Adversarial Network | 生成對抗網路 | 同上，試過但沒採用 | p.5／PDF 15、p.37／PDF 47 | p13 備忘稿 |
| AI | Artificial Intelligence | 人工智慧 | 第四版用在月報檔名「AI 模型設定」 | p.1／PDF 11 | p13 備忘稿 |

### 1.2 孤島抑制與連通指標

| 縮寫 | 英文全稱 | 中文譯名 | 白話一句 | 出處 | 第四版位置 |
|---|---|---|---|---|---|
| TV | Total Variation Loss | 總變差損失 | 懲罰相鄰像素的差異，只看緊鄰一格 | 碩論 p.12／PDF 22、式 3；中文名見 G11 s5 備忘稿 | p13 泡泡、p15 圖上 |
| IS | Island Suppression Loss | 孤島抑制損失 | 比較每格與周圍一圈平均的差異 | 202601 s3、G11 s6；碩論改名為 KNS | p13 泡泡、p32 |
| KNS | k-Neighborhood Smoothing Loss | k 鄰域平滑損失 | 即月報的 IS，用 k×k 平均圖比較 | 碩論 p.29／PDF 39、式 11 | p13 備忘稿 |
| GC | Gap Closing Loss | 閉運算 loss，第四版泡泡用語 | 把先膨脹再侵蝕的閉運算寫進 loss | G11 s7、202602 AI 設定 s1、s5；碩論只有後處理的形態學閉運算 Morphological Closing，p.12 | p13 泡泡、p32 |
| R_feed | Feed Reachability | 饋入埠連通性，第四版稱饋入連通率 | 與饋入點連成一塊的金屬占全部金屬的比率 | p.30／PDF 40、式 26 p.51／PDF 61 | p15 圖上 R_feed |

### 1.3 比較基準、起點與規格

| 縮寫 | 英文全稱 | 中文譯名 | 白話一句 | 出處 | 第四版位置 |
|---|---|---|---|---|---|
| Base | Baseline 的簡稱，碩論只寫 Base | 比較基準 | 碩論三個實驗各指不同設定：表 4-2 是固定學習率，圖 4-4 是寫入即保留，表 4-3 是 ACP 與 DLF 加熱啟動、不加孤島抑制 | p.53、p.55、p.61／PDF 63、65、71 | p15 圖上與表 |
| Base-1、Base-2 | 碩論命名，不是縮寫 | 理論貼片起點一、二 | 理論尺寸取整到 0.2 mm 時，捨去得 Base-1、進位得 Base-2 | p.33／PDF 43 | p13 備忘稿 |
| RO4003C | 查無全稱，基板型號 | 高頻電路基板 | 碩論只寫型號與 εr 3.55、板厚 0.508 mm | 表 3-2 p.33／PDF 43 | p13 備忘稿 |
| n257 | 5G NR 頻段代號；NR 為 New Radio，通用名稱 | n257 頻段 | 26.5–29.5 GHz 的毫米波頻段 | p.50／PDF 60 寫「5G NR 標準中的 n257 頻段」，沒有展開 NR | p12 備忘稿 |
| HFSS | High Frequency Structure Simulator，通用名稱 | 高頻結構模擬器 | Ansys 的全波電磁模擬軟體，四代都用它驗證 | 碩論只寫 Ansys HFSS，p.3／PDF 13、p.48／PDF 58 | p12、p13、p32 |
| EM | Electromagnetic，通用名稱 | 電磁 | 圖上 Full-wave EM Simulator 即全波電磁模擬 | 碩論以「全波電磁模擬 Full-wave EM Simulation」對譯，p.39／PDF 49 | p12 圖上 |
| S11、S21、S22 | S 為 Scattering，通用名稱 | S 參數；S11 反射係數、S21 傳輸係數 | S11 越低代表越少能量被反射回饋入端 | p.3／PDF 13、p.44／PDF 54 | p12 備忘稿、p16 圖上 Reflection coefficient |
| dB、dBi | decibel；decibel relative to isotropic，通用名稱 | 分貝；相對等向天線的分貝 | dBi 是天線增益的常用單位 | 碩論增益寫 dB；dBi 見 202603 Final s3、s21 | p16 備忘稿 |

### 1.4 實作量測

| 縮寫 | 英文全稱 | 中文譯名 | 白話一句 | 出處 | 第四版位置 |
|---|---|---|---|---|---|
| ENIG | Electroless Nickel Immersion Gold，通用名稱 | 化學鎳浸金 | PCB 表面處理的化金層，月報寫 Ni/Au 化金層 | 202603 Final s3 | p16 備忘稿 |
| K 接頭 | 查無全稱，K 是接頭型號名 | K 型同軸接頭 | 毫米波量測常用的同軸接頭，模擬時一併建模 | 202603 Final s3「HFSS 納入 K Connector 模擬」 | p16 備忘稿 |
| ASE-CCU | Advanced Semiconductor Engineering、National Chung Cheng University，通用名稱 | 日月光、中正大學 | 兩方的聯合審查會議 | 月報檔名 `ASE CCU 20260528` 等 | p16 備忘稿 |
| Sim.、Meas.、Freq. | Simulation、Measurement、Frequency | 模擬、量測、頻率 | 樣品 1 的 S11 圖例與座標軸 | 202603 Final s21 原圖 | p16 圖上 |

### 1.5 圖上的英文術語，不是縮寫

第五版若要中文化或在備忘稿對照，可用這份：

- **第 12 頁圖，碩論圖 3-11：** Target Response 目標響應、Generator 生成器、Buffer 緩衝區、Full-wave EM Simulator 全波電磁模擬器、Experience Replay 經驗回放、Predicted Result 預測結果、Loss 損失、Island Suppression 孤島抑制。Experience Replay 的英文見演算法 3 標題 p.36；Island Suppression 在這張圖指 SC loss。
- **第 14 頁形狀庫，202511 月報原圖：** H-Line、V-Line 是水平線、垂直線；Diag 是斜線；Thin、Bold 是細、粗；Rot 是旋轉角，Rotation 的通用簡寫；Dot、Circle 是點、圓；+、X 是正十字、斜十字；T、L、U 是字母形。
- **第 13、14 頁備忘稿：** Gumbel-Softmax 與 Gumbel-Sinkhorn 是兩種可微的離散挑選方法，Gumbel 是人名，不是縮寫；前者從多個形狀挑一個，後者把挑選做成排列，約束較嚴，202511 s12–13。Sigmoid 是 S 形函數，第三代用它做可微二值化，見 §2 式 2。

### 1.6 第四版沒出現、第五版可能用到

| 縮寫或符號 | 英文全稱 | 中文譯名 | 出處 | 說明 |
|---|---|---|---|---|
| τ | Temperature parameter | 二值化溫度 | p.20／PDF 30、式 4 | 第四版只寫中文「二值化溫度」；放式子後會出現 |
| λ_2 | Fiedler value、Algebraic Connectivity | 代數連通度 | p.13／PDF 23、p.31／PDF 41 | SC loss 式子裡的符號 |
| STE | Straight-Through Estimator | 直通估計器 | p.42／PDF 52 | 前向四捨五入、反傳沿連續值走 |
| MLP | Multi-Layer Perceptron | 多層感知機 | p.19／PDF 29、p.38／PDF 48 | 第四版 p13 備忘稿只寫中文 |
| CVAE | Conditional Variational Autoencoder | 條件變分自動編碼器 | p.37／PDF 47 | 第四版 p13 寫「條件變分自編碼器」，碩論用語是「自動編碼器」 |
| DMCVAE | Dual Mirror CVAE | 雙鏡射條件變分自動編碼器 | 202512 s12 | 碩論沒有；中文譯名是本整理暫譯 |
| CyclicLR、ReduceLROnPlateau、Cyclical Mutation | 兩個是 PyTorch 排程器名，第三個是錢鵬予位元翻轉的對照實作 | 週期學習率、停滯降學習率、週期變異 | p.55／PDF 65、表 4-2 | 只在碩論 ACP 對照實驗出現 |
| RL | Return Loss | 回波損耗 | 202603 Final s2–3 | 第四版 p16 已寫中文 |
| WCNC | IEEE Wireless Communications and Networking Conference，通用名稱 | 無線通訊與網路研討會 | P01、P07 | 2025/08 規劃投稿，推定未錄取；第四版沒有提，建議不加 |
| TAAI | Conference on Technologies and Applications of Artificial Intelligence，通用名稱 | 人工智慧技術與應用研討會 | G09 | 參會心得，海報作者是顏俊儀、主題不是天線，見 `gen3_嘗試歷程.md`；建議不加 |

---

## 2 數學式

五個式子依一次迭代的資料流排列，對應碩論演算法 4 p.41／PDF 51：先由 ACP 更新學習率與溫度，第 4 行；生成器輸出經溫度二值化，第 7–8 行；算響應 loss 與 SC loss 並反傳，第 12–15 行；HFSS 結果進緩衝區，DLF 篩出重訓資料，第 19–21 行。

本檔 14 個 mathtext 字串都已用 matplotlib 3.11.1 逐一解析並渲染，0 失敗；`mathtext.fontset` 設 `cm` 或預設 `dejavusans` 都能過，總覽圖在 scratchpad `math_gen3/doc_math_all.png`，檢查腳本 `math_gen3/check_doc_math.py`。分數建議用 `\dfrac`，投影片上字比較大。

### 式 1　ACP 自適應循環策略

**投影片一行：**

```
$\eta=\eta_{min}+\Delta\eta\cdot\Phi(T_{cur}),\quad \tau=\tau_{min}+\Delta\tau\cdot\Phi(T_{cur})$
```

**重啟規則，可放第二行或只放備忘稿：**

```
$c_p\geq P\ \Rightarrow\ T_i\leftarrow\max(\gamma_{decay}\,T_i,\ T_0/2)$
```

**進度係數 Φ，放備忘稿：**

```
$\Phi=\dfrac{T_{cur}}{T_{warm}}\ \ (T_{cur}<T_{warm}),\quad \Phi=\dfrac{1}{2}\left[1+\cos\left(\pi\,\dfrac{T_{cur}-T_{warm}}{T_i-T_{warm}}\right)\right]\ \ (T_{cur}\geq T_{warm})$
```

- **符號：**
  - η：生成器的學習率；Δη＝η_max − η_min，程式設定由 1×10⁻⁶ 到 5×10⁻³。
  - τ：二值化溫度，用法見式 2；Δτ＝τ_max − τ_min，程式設定由 0.1 到 4.0。
  - Φ：週期內的進度係數，暖身段由 0 線性升到 1，之後依餘弦降回 0。
  - T_cur、T_i、T_warm：目前在週期的第幾步、目前週期長度、暖身步數；T_warm＝T_i 乘暖身比例，程式設定週期 T_0＝100、暖身比例 0.2。
  - c_p、P：連續幾步沒刷新歷史最低 loss，與耐心值 25；監看的是每步 HFSS 的真實 loss。
  - γ_decay：週期縮減因子 0.7；T_0/2 是週期長度的下限。
- **白話：** 學習率和溫度綁在一起，每個週期先升溫探索、再降溫收斂；連續 25 步沒刷新最低 loss，就提前重啟並把週期縮短。
- **出處：** 式 5–10、表 3-1、演算法 1，碩論 p.23–27／PDF 33–37；圖 3-2、圖 3-5。超參數見口試 s62 附錄，註明取自程式，與 `antenna/optim/scheduler.py` 的 `AdaptiveCyclicalScheduler` 同名。
- **與碩論的關係：** 排程那行等同式 8–10，Δη、Δτ 是本整理的簡寫；重啟那行等同式 7。省略了四件事：
  1. 改善判定，式 5–6：本步 loss 低於歷史最低就更新最低值並把 c_p 歸零，否則 c_p 加 1。
  2. 重啟後從哪一步接回。碩論採 Linear，從目前學習率在暖身線上的對應位置平滑爬回峰值：`$T_{cur}\leftarrow\lfloor\rho\,T_{warm}\rfloor,\ \ \rho=\dfrac{\eta_{now}-\eta_{min}}{\eta_{max}-\eta_{min}}$`；Reset 是從頭暖身，Peak 是直接跳到峰值，表 3-1。
  3. 週期跑完時 T_i 乘上 T_mult，程式設定 1，週期不增長，演算法 1 第 20–21 行。
  4. 演算法 4 第 22–24 行另有「長期停滯時把生成器退回歷史最佳」的機制，不屬於 ACP 本身。

### 式 2　二值化溫度 τ

**投影片一行：**

```
$P_{soft}=\sigma\left(\dfrac{\tilde{P}-k}{\tau}\right)$
```

- **符號：**
  - P̃：生成器輸出的 25×25 連續值圖。
  - k：二值化門檻。碩論 §3.2.2 p.21 寫取像素均值，演算法 4 的註解寫固定門檻，兩處說法不同，式子本身不受影響。
  - τ：二值化溫度，由式 1 的 ACP 排程。
  - σ：sigmoid 函數，輸出落在 0 到 1。
  - P_soft：軟性圖形，送進代理模型；再四捨五入成 0 或 1 的硬圖形送 HFSS。
- **白話：** 溫度高時輸出平滑、灰階多，方便探索；溫度低時幾乎只剩 0 和 1 的硬邊界，接近可製造的圖形。
- **出處：** 式 4，碩論 p.21／PDF 31；三種溫度的效果見圖 3-3 p.22；硬二值化見表 3-3 p.42／PDF 52 與演算法 4 第 8 行。
- **與碩論的關係：** 等同式 4。省略了硬二值化的直通估計器：`$P_{hard}=P_{soft}+\mathrm{sg}[\,\mathrm{round}(P_{soft})-P_{soft}\,]$`，sg 是停止梯度，前向等於四捨五入，反傳沿 P_soft 走；之後再疊上饋入固定金屬遮罩，成為送 HFSS 與 SC loss 的完整圖形。

### 式 3　只罰不達標的響應 loss

**投影片一行：**

```
$\mathcal{L}_{EM}=\ell\,(\hat{S}_{11}[\,\hat{S}_{11}>-10\,],\,-10)+\ell\,(\hat{G}[\,\hat{G}<4\,],\,4)$
```

**總 loss，可並列：**

```
$\mathcal{L}_{total}=\mathcal{L}_{EM}+w_{SC}\cdot\mathcal{L}_{SC}$
```

- **符號：**
  - Ŝ_11、Ĝ：代理模型對 26.5–29.5 GHz 各評分頻點預測的反射係數與固定方向增益，單位 dB。
  - 方括號：只取出不達標的頻點，也就是 S11 高於 −10 dB、增益低於 4 dB 的那幾點。
  - ℓ：Smooth L1，差距小於 1 時取平方的一半，大於 1 時取絕對值減 0.5。
  - w_SC：SC loss 的權重，碩論的平衡設定是 5×10⁻⁴。
- **白話：** 已經達標的頻點不扣分，只把不達標的頻點往規格線拉；帶內全部達標時 loss 等於 0，這就是第 15 頁表上「響應 loss 0.00」的意思。
- **出處：** 式 16–18 與圖 3-14，碩論 p.43–44／PDF 53–54；w_SC 見 p.63／PDF 73。
- **與碩論的關係：** 簡化表示，差在四點：
  1. 碩論寫成符號 y*_min、y*_max，這裡直接代入單埠規格 −10 dB 與 4 dB。
  2. 碩論分寫 L_s11 與 L_gain 兩式，單埠沒有寫出相加式；這裡寫成相加，與雙埠式 23 逐項相加的寫法一致。
  3. ℓ 的形狀碩論只引文獻 [33]，沒有寫出；這裡用 PyTorch 預設的 β＝1，與程式 `antenna/losses.py` 的 `custom_loss_minmax` 一致，程式對不達標的頻點取平均。
  4. 碩論 p.50 另列帶外規格，但 loss 只算帶內，圖 3-14 的綠色區。
- **補一句：** 同一個式子套在代理模型的預測上，用來反傳給生成器；套在 HFSS 結果上就是真實 loss，ACP 監看與 DLF 篩選用的是後者，演算法 4 第 18、25 行。

碩論原式，放備忘稿用：

```
$\mathcal{L}_{s_{11}}=\mathcal{L}_{smooth}(\hat{y}[\hat{y}>y^{*}_{min}],\,y^{*}_{min})$
$\mathcal{L}_{gain}=\mathcal{L}_{smooth}(\hat{y}[\hat{y}<y^{*}_{max}],\,y^{*}_{max})$
```

### 式 4　SC loss 圖譜連通度損失

**投影片一行：**

```
$\mathcal{L}_{SC}=\dfrac{1}{\lambda_2(L)},\quad L=D-A,\quad A_{ij}=p_i\,p_j+\varepsilon$
```

- **符號：**
  - p_i：第 i 格是金屬的程度，0 到 1；輸入是疊上饋入固定金屬後的完整圖形，共 625 格。
  - A_ij：只有上下左右相鄰的兩格之間有邊；兩格都是金屬時約為 1，任一格是介質時只剩 ε。
  - ε：基礎連通值 10⁻²，讓孤立區域也收得到梯度。
  - D：度矩陣，對角線是每一格所有邊權重的和。
  - L：圖拉普拉斯矩陣；λ_2(L) 是它的第二小特徵值，即 Fiedler 值，又稱代數連通度。
- **白話：** 把 625 個像素看成一張圖，圖越容易被切成兩塊，λ_2 越小、loss 越大；整張圖一起算，所以離饋入點很遠的大斷裂也罰得到，這是 TV、KNS 這類只看鄰近像素的方法做不到的。
- **出處：** 式 13–15 與演算法 2，碩論 p.31–32／PDF 41–42；輸入圖形見表 3-3 p.42；25×25 算一次約 0.03 秒，p.32。程式 `antenna/losses.py` 的 `SpectralConnectivityLoss` 與此相同。
- **與碩論的關係：** 等同式 13–15，三式併成一行。兩點補充放備忘稿：加入 ε 後整張網格都有邊，λ_2 只當相對的連通程度，不是「有沒有斷開」的判定，p.13、p.31；輸入經直通估計器，前向時 p_i 已是 0 或 1，梯度回到連續的 P_soft，p.42。

### 式 5　DLF 動態損失過濾

**投影片一行：**

```
$B_{train}=\{\,(P_i,y_i,\mathcal{L}_i)\in B\ :\ \mathcal{L}_i\leq\bar{\mathcal{L}}_t\,\},\quad \bar{\mathcal{L}}_t=\mathrm{mean}(\mathcal{H})$
```

- **符號：**
  - B：經驗回放緩衝區，每一筆 HFSS 結果都無條件存進去。
  - P_i、y_i、𝓛_i：第 i 筆的金屬圖形、HFSS 響應與真實 loss。
  - 𝓗：到目前為止所有真實 loss 的紀錄。
  - 𝓛̄_t：紀錄的累計平均，當作本輪門檻；碩論記為 λ_t。
  - B_train：本輪拿來重訓代理模型的樣本子集。
- **白話：** 每次重訓代理模型前，用「到目前為止的平均 loss」把整個緩衝區重新篩一遍，只複習比平均好的樣本；平均會隨訓練下降，門檻自動變嚴。
- **出處：** 演算法 3、圖 3-9、圖 3-10，碩論 p.35–37／PDF 45–47；1000 次訓練下最小 loss 約 3.6 對 1.71，p.54／PDF 64。
- **與碩論的關係：** 等同演算法 3 第 1–4 行；門檻符號由 λ_t 改成 𝓛̄_t 是簡化表示，避免和式 4 的 λ_2 撞名。省略了對照組：「寫入即保留」只在樣本寫入當下比一次門檻，之後永不重篩，p.35。

### 2.6 備選與未選

**饋入連通率，備選。** 第 15 頁表與圖上都有這個指標，若新頁想放 6 列可以加：

```
$R_{feed}=\dfrac{|C_{feed}|}{|M|}$
```

- M 是所有金屬像素，C_feed 是從饋入點只走上下左右能到的那塊金屬；對角相接不算連通。雙埠要求所有饋入點在同一塊上，碩論只定義這種情況的比率，程式在饋入點分屬不同塊時記為 0。
- 出處：式 26 p.51／PDF 61，四連通定義式 12 p.30／PDF 40；程式 `antenna/losses.py` 第 229–265 行。
- 沒選進主表的理由：它是評估指標、不參與訓練，碩論 p.52 明寫它不可微、訓練靠 SC loss；名詞表 p32 已有一句定義。

**未選：**

- **GEN 的整體目標，式 1 p.7／PDF 17：** `$\theta^{*}=\arg\min_{\theta}\,\mathcal{L}_{EM}(\hat{f}(G_{\theta}(R^{*})),\,R^{*})$`。這是陳廷茂建立的線上學習骨架，放第一代較合適，第三代沿用。
- **TV loss，式 3 p.12／PDF 22：** 第三代只拿它當對照組。另外碩論式 3 寫的是相鄰像素的絕對差，同頁內文卻說「梯度平方差」，程式 `tv_loss` 用的是平方差，三處不一致，不建議上投影片。
- **KNS loss，式 11 p.29／PDF 39：** 同屬對照組，局部算子，第四版沒有展開。

### 2.7 碩論與素材之間要留意的不一致

1. **ACP 的英文全稱：** 碩論正文定名 Adaptive Cyclical Policy；第 12 頁用的碩論圖 3-11 圖上寫 Adaptive Cyclical Scheduler，月報 2025/12 起也用 Scheduler。全稱以正文 Policy 為準，第 12 頁備忘稿補一句「圖上的 Scheduler 是定名前的寫法」。
2. **ACS：** 月報寫全名 Adaptive Cyclical Scheduler，實驗名稱縮成 ACScheduler，找不到「ACS」這個縮寫。第四版 p13 備忘稿的「當時叫 ACS」建議改成「當時稱 Adaptive Cyclical Scheduler」。
3. **Base 一詞三義：** 表 4-2 是固定學習率，圖 4-4 是寫入即保留，表 4-3 是 ACP 與 DLF 加熱啟動、不加孤島抑制。第 15 頁表已寫「固定學習率，不加機制」與「ACP 與 DLF，左圖 Base」，處理正確；新頁若提到 Base 也要寫明是哪一個。
4. **Island Suppression：** 第 12 頁圖上的「Island Suppression (SC Loss)」是孤島抑制這一類 loss 的總稱，這裡指 SC loss；月報的 IS loss 是另一個局部 loss，碩論改名 KNS。
5. **二值化門檻 k：** 碩論 §3.2.2 寫像素均值，演算法 4 註解寫固定門檻。式 2 照寫 k，不寫取值。
6. **代理模型重訓時機：** 演算法 4 寫每一步都用 B_train 重訓；口試 s62 附錄的備忘稿寫程式只在停滯觸發時才用菁英子集重訓。式 5 以碩論為準，備忘稿不寫重訓頻率。
7. **代理模型架構：** 碩論 p.38 寫生成器與代理模型都是多層感知機；口試 s62 附錄備忘稿寫代理模型是 U-Net 回歸器。第五版若寫架構，以碩論為準，或避開不寫。

---

## 3 建議放在投影片哪裡

### 3.1 另開一頁「第三代方法的數學式」，接在第 12 頁後

建議新增一頁，成為新的第 13 頁，全份由 33 頁變 34 頁。不放第 12 頁右欄的理由：第 12 頁左側已有碩論圖 3-11 的大圖，右欄有四條說明，再加五個式子與符號會多出至少十行，違反 §1.5「一頁內容不能多」與「一頁一張大圖或一張表」。

- **標題：** 第三代方法的數學式
- **描述一句：** ACP 同步排程學習率與二值化溫度，loss 只罰不達標並加上連通項，DLF 決定代理模型複習哪些樣本。
- **主體：** 一張 5 列、3 欄的表，欄位是「名稱與英文全稱」「一行式子」「一句白話」。
  - 第 1 列　ACP，Adaptive Cyclical Policy｜式 1 的排程那行｜學習率與溫度先升後降，停滯 25 步就重啟
  - 第 2 列　二值化溫度 τ，Temperature｜式 2｜溫度越低，輸出越接近 0 或 1
  - 第 3 列　響應 loss｜式 3 的 𝓛_EM 那行｜達標的頻點不扣分，全部達標時為 0
  - 第 4 列　SC loss，Spectral Connectivity Loss｜式 4｜圖越容易被切開，loss 越大
  - 第 5 列　DLF，Dynamic Loss Filter｜式 5｜只用優於歷史平均的樣本重訓代理模型
- **式子的做法：** python-pptx 不支援原生公式，建議用 matplotlib mathtext 把每個式子渲染成透明背景 PNG，`mathtext.fontset='cm'`，字高約 20 pt，貼進表格第二欄；或整張表一起畫成一張圖。第 1 列最長，約占表寬六成，其餘都短。
- **備忘稿：** 每列的符號定義、出處與「與碩論的關係」照 §2 抄；總 loss 式、ACP 重啟式與 Φ、直通估計器、碩論原式 17–18 也放這裡。

### 3.2 其他頁補全稱

- **第 12 頁：**
  - 描述句的「GEN 即代理模型」改成「圖中 GEN 是梯度估計網路，即代理模型」；英文 Gradient Estimation Network 放備忘稿或名詞表。
  - 右欄三個小標補中文全稱：「ACP 自適應循環策略」「DLF 動態損失過濾」「SC loss 圖譜連通度損失」，每個後面接原本那一句。英文全稱放新頁第一欄與名詞表，第 12 頁不再加長。
  - 備忘稿補一句：圖上 Adaptive Cyclical Scheduler 即 ACP 定名前的寫法；Island Suppression 這裡指 SC loss；HFSS 全稱 High Frequency Structure Simulator。
- **第 13 頁時間軸：** 泡泡有 8 字上限，維持縮寫；全稱寫進備忘稿各行第一次出現處：TV Total Variation、IS Island Suppression、GC Gap Closing、SC Spectral Connectivity、KNS k-Neighborhood Smoothing；「當時叫 ACS」照 §2.7 第 2 點改；未畫成泡泡那行補 CNN、GAN 全稱，「條件變分自編碼器」改成碩論用語「條件變分自動編碼器」。
- **第 14 頁：** 備忘稿補 Gumbel 是人名、不是縮寫；圖上英文標籤可依 §1.5 加一行中文對照。
- **第 15 頁：** 備忘稿補「圖上 R_feed 即 Feed Reachability，饋入連通率」「TV loss 為 Total Variation、SC loss 為 Spectral Connectivity」「左圖 Base 指 ACP 與 DLF 加理論貼片起點、不加孤島抑制」。
- **第 16 頁：** 備忘稿補 ENIG 化學鎳浸金、K 接頭為型號名、ASE-CCU 即日月光與中正大學的聯合審查、dBi 單位。
- **第 32 頁名詞表：** 左欄「ACP」「DLF」改成「ACP 自適應循環策略」「DLF 動態損失過濾」；「孤島抑制 loss」那列的白話改成「SC 圖譜連通度，另試過 TV、IS、GC」。表格放得下時另加一列「SC loss｜圖越難切開，loss 越小」。英文全稱若要放，建議另加一欄或放備忘稿，避免白話欄變長。
