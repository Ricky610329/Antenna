# 第一、二代：縮寫全稱與方法數學式

> 整理日期 2026-10-24。用途：日月光「歷來實驗總整理」第五版，依 `docs/唯一真實.md` §1.4 10-24 條「縮寫補全稱、方法配簡單數學式」，補第一代陳廷茂、第二代錢鵬予兩段。
> 依據：兩本碩論定稿 PDF，與 ailab NAS `A:\歷屆交接與電子論文\` 原檔 MD5 一致，複製到 scratchpad `math_gen12\src\` 後閱讀，NAS 原檔未動。陳廷茂〈基於梯度估計網路設計貼片天線金屬圖形之研究〉70 頁，2024/07/29 匯出；錢鵬予〈基於二階段梯度探索之貼片天線金屬圖形生成方法〉59 頁，2025/07/04 匯出。
> 頁碼寫法：「p.」是論文印的頁碼。陳廷茂 PDF 頁＝論文頁＋10，錢鵬予 PDF 頁＝論文頁＋9，與 `gen1_碩論對照.md`、`gen2_碩論對照.md` 相同。
> 範圍：第四版第 7–11 頁的正文、時間軸泡泡、圖上文字與備忘稿，加上第 2 頁四代一覽的第一、二代列，以及第 32 頁名詞表屬於這兩代的詞。第 5 頁天線規格雖不在範圍內，S11、dBi、n257 第一次出現在那裡，一併處理。

## 0 結論

- **縮寫：** 範圍內共 35 條，另有 7 條在投影片只寫中文、要補英文時可用。35 條中 15 條碩論寫了全稱；9 條碩論有用到、但沒寫全稱或只寫中文；11 條碩論完全沒提到。後兩類共 20 條，全稱標「通用名稱」，見 §1.4。
- **數學式：** 第一代 4 式、第二代 4 式，全部回到碩論式號或演算法行號核對過，也都用 matplotlib mathtext 實際渲染通過。碩論只用文字描述、沒有式號的，標「簡化表示」並寫明省略了什麼。
- **建議：** 第 7 頁之後、第 10 頁之後各新增一頁「第 N 代方法的數學式」，每頁 3 式，做成三列表格：名稱、式子、一句白話；符號定義放備忘稿。整份由 33 頁變 35 頁。縮寫全稱寫在第一次出現的那一句，詳見 §3。

## 1 縮寫表

### 1.1 方法與工具

| 縮寫 | 英文全稱 | 中文譯名 | 一句白話 | 出處 | 第四版出現處 |
|---|---|---|---|---|---|
| HFSS | High-Frequency Structure Simulator | 高頻結構模擬器 | Ansys 的三維全波電磁模擬軟體，四代都用它算響應 | 陳廷茂 p.8；p.16 寫 Ansys HFSS | p2、p7、p8、p11 正文與泡泡，p32 名詞表 |
| MPGN | Metal Pattern Generation Network | 金屬圖形生成網路 | 第一代的生成網路，輸入期望響應、輸出 0/1 金屬圖形 | 陳廷茂 p.14 | p7 正文與圖 |
| GEN | Gradient Estimation Network | 梯度估計網路 | 第一代的代理模型，學會 HFSS 的輸入輸出關係，只負責把梯度傳回 MPGN | 陳廷茂 p.15；錢鵬予 p.8 | p7 正文與圖寫全稱，p11 備忘稿寫 GEN |
| STE | Straight-Through Estimator | 直通估計，第二代稱直通估計器 | 反向傳播時把二值化那一步當成不存在，梯度直接穿過 | 陳廷茂 p.21；錢鵬予 p.21 | p8 備忘稿只寫中文「直通估計」；新增數學式頁會用到 |
| BiScaleNorm | 碩論自創名稱，沒有另寫展開的英文全稱 | 雙尺度正規化 | 正值除以最大值、負值除以最小值的絕對值，把輸出壓回有梯度的範圍 | 錢鵬予 p.16、p.22 | p2、p10、p11 泡泡，p32 名詞表 |
| EMA | Exponential Moving Average | 指數移動平均，碩論寫「指數平均」 | 新值與舊值按比例混合，讓 BiScaleNorm 的極值平滑變化 | 錢鵬予 p.23 | 只會出現在新增的第二代數學式頁 |
| Smooth L1 | Smooth L1 Loss | 平滑 L1 損失 | 誤差小時像平方、誤差大時像絕對值的 loss；L1 即絕對誤差 | 錢鵬予 p.20；L1 為通用名稱 | p11 泡泡與備忘稿 |
| MSE | Mean Square Error | 均方誤差 | 誤差平方的平均 | 陳廷茂 p.24 | p7 備忘稿寫中文，p8、p11 備忘稿寫 MSE |
| MAE | Mean Absolute Error | 平均絕對誤差 | 誤差絕對值的平均 | 通用名稱；碩論沒有提到 | p8 備忘稿 |

### 1.2 神經網路元件與優化器

| 縮寫 | 英文全稱 | 中文譯名 | 一句白話 | 出處 | 第四版出現處 |
|---|---|---|---|---|---|
| ReLU | Rectified Linear Unit | 線性整流單元 | 負值輸出 0、正值原樣輸出的激活函數 | 陳廷茂 p.22 | p8 備忘稿 |
| LeakyReLU | Leaky Rectified Linear Unit | 洩漏線性整流單元 | 負值乘上一個小常數、不直接歸零 | 碩論 p.22 有定義式、沒寫全稱，全稱為通用名稱 | p8 備忘稿 |
| PReLU | Parametric ReLU | 參數化線性整流單元 | 負值那一側的斜率可以學 | 陳廷茂 p.22 英文全稱；中文為通用譯名 | p8 備忘稿 |
| Tanh | Hyperbolic Tangent function | 雙曲正切函數 | 把輸入壓到 −1 到 1 之間 | 陳廷茂 p.22 | p8 備忘稿 |
| BN | Batch Normalization | 批次正規化 | 依一批資料的平均與變異數，把每層輸出標準化 | 陳廷茂 p.27 | p8 備忘稿已寫全稱 Batch Normalization |
| LayerNorm | Layer Normalization | 層正規化 | 依同一筆資料在該層的平均與變異數標準化 | 錢鵬予 p.22，文獻 Ba 等 2016 | p11 備忘稿 |
| CNN | Convolutional Neural Network | 卷積神經網路 | 用小卷積核掃過圖形抽特徵的網路 | 陳廷茂 p.10；錢鵬予 p.6 | p8、p11 備忘稿 |
| LSTM | Long Short-Term Memory | 長短期記憶網路 | 處理序列資料的循環網路 | 通用名稱 | p11 備忘稿 |
| DyT | Dynamic Tanh | 動態雙曲正切 | 用可學縮放的 tanh 取代正規化層 | 通用名稱，Zhu 等 2025〈Transformers without Normalization〉 | p11 備忘稿 |
| Adam | Adaptive Moment Estimation | 自適應矩估計 | 依梯度的一階、二階動量自動調整每個參數的步長 | 錢鵬予 p.11 只寫「自適應優化策略」，英文全稱為通用名稱，Kingma 與 Ba 2015 | p11 備忘稿 |
| SGD | Stochastic Gradient Descent | 隨機梯度下降 | 每次用一小批資料的梯度更新參數 | 錢鵬予 p.12 寫中文；英文見其參考文獻 [25] 題名 | p11 備忘稿 |
| RMSProp | Root Mean Square Propagation | 均方根傳播 | 用梯度平方的移動平均縮放步長 | 通用名稱，Hinton 2012 課程講義 | p11 備忘稿 |
| NAG | Nesterov Accelerated Gradient | Nesterov 加速梯度 | 先沿動量方向看一步，再在那裡算梯度 | 通用名稱 | p11 備忘稿 |
| AdaBelief | 不是縮寫，出自論文〈AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients〉 | 無通用譯名 | 依梯度與其預測值的落差調整步長的 Adam 變體 | 通用名稱，Zhuang 等 2020 | p11 備忘稿 |
| AdaBound | 不是縮寫，出自論文〈Adaptive Gradient Methods with Dynamic Bound of Learning Rate〉 | 無通用譯名 | 給 Adam 的步長加上下界，訓練後期逐漸變成 SGD | 通用名稱，Luo 等 2019 | p11 備忘稿 |
| Ranger | 不是縮寫，RAdam 與 Lookahead 合成的優化器名稱 | 無通用譯名 | 兩種穩定訓練的優化技巧疊在一起 | 通用名稱，Wright 2019 | p11 備忘稿 |
| ApproxSign | approximate sign function | 近似符號函數 | 用分段二次函數逼近 Sign，讓梯度不再處處為 0 | 錢鵬予 p.21 寫「近似 Sign 函數」，引用 Bi-Real Net；ApproxSign 是該文獻的命名 | p11 備忘稿 |

### 1.3 強化學習、頻段與射頻符號、其他

| 縮寫 | 英文全稱 | 中文譯名 | 一句白話 | 出處 | 第四版出現處 |
|---|---|---|---|---|---|
| DDPG | Deep Deterministic Policy Gradient | 深度確定性策略梯度 | 輸出連續動作的強化學習演算法 | 陳廷茂 p.11–12 | p8 備忘稿 |
| DQN、Dueling DQN | Deep Q-Network、Dueling Deep Q-Network | 深度 Q 網路、對決式深度 Q 網路 | 學每個動作值多少分的網路；對決式把狀態價值與動作優勢分開估 | 陳廷茂 p.13 表中出現 DQN、沒寫全稱，全稱為通用名稱，Mnih 等 2015、Wang 等 2016 | p8 備忘稿 |
| PPO | Proximal Policy Optimization | 近端策略優化 | 每次只讓策略小幅更新的強化學習演算法 | 通用名稱，Schulman 等 2017；碩論沒有提到 | p8 備忘稿 |
| n257 | 不是縮寫，是 3GPP 定義的 5G 頻段編號 | n257 頻段 | 26.5–29.5 GHz 的毫米波頻段，第一、二代天線都以它為目標 | 陳廷茂 p.1；錢鵬予 p.33 | p2、p5、p9 正文 |
| n260 | 不是縮寫，是 3GPP 定義的 5G 頻段編號 | n260 頻段 | 37–40 GHz 的毫米波頻段，第二代碩論附錄拿來試換任務 | 錢鵬予 p.47 只寫 n260，頻率範圍依 3GPP 定義 | p2 正文、p11 泡泡 |
| S11、S21、S22 | Scattering parameters | 散射參數，簡稱 S 參數 | S11 是埠 1 的反射，S21 是埠 1 到埠 2 的穿透，S22 是埠 2 的反射 | 通用名稱；錢鵬予 p.48 附錄圖 3 用到三者，第一代碩論只寫「反射係數」 | p5、p9 正文，p11 備忘稿 |
| dBi | decibels relative to an isotropic radiator | 相對等向性天線的分貝 | 天線增益的單位，以理想的全向天線為 0 dBi | 通用名稱；陳廷茂 p.25 起當單位使用，沒寫全稱 | p5、p9 正文 |
| GPU | Graphics Processing Unit | 圖形處理器 | 平行運算晶片；Ansys 回覆頻域求解的貼片天線用它幫助有限 | 通用名稱 | p8 備忘稿 |
| RIS | Reconfigurable Intelligent Surface | 可重構智慧表面，投影片稱電磁面鏡 | 可調整反射相位的表面；第二代拿它的相位分布任務當不用跑 HFSS 的試驗台 | 通用名稱；碩論沒有提到 | p11 備忘稿 |

### 1.4 查不到全稱、標通用名稱的

- **碩論寫了全稱，15 條：** HFSS、MPGN、GEN、STE、EMA、Smooth L1、MSE、ReLU、PReLU、Tanh、BN、LayerNorm、CNN、DDPG、n257。
- **碩論有用到、但沒寫全稱或只寫中文，9 條：** BiScaleNorm 只有中文「雙尺度正規化」、沒有展開的英文，不另造；Adam 只寫「自適應優化策略」；SGD 只寫中文，英文取自它的參考文獻題名；LeakyReLU 有定義式、沒寫全稱；DQN 只在文獻表中出現；ApproxSign 只寫「近似 Sign 函數」；n260 只寫名稱、沒寫頻率範圍；S11、S21、S22 只用在附錄圖說；dBi 只當單位用。
- **碩論完全沒提到，11 條：** MAE、LSTM、DyT、RMSProp、NAG、AdaBelief、AdaBound、Ranger、PPO、RIS、GPU。這些出自週報、程式或簡報。
- **共同原則：** 後兩類的全稱都用公認的標準名稱或原始文獻題名，沒有自己編。L1 併在 Smooth L1 那一列，屬通用名稱。
- **不是縮寫：** AdaBelief、AdaBound、Ranger 是名稱；n257、n260 是頻段編號。表中照實寫明。

### 1.5 投影片只寫中文、要補英文時可用

| 縮寫 | 英文全稱 | 中文 | 出處 | 第四版出現處 |
|---|---|---|---|---|
| RL | Reinforcement Learning | 強化學習 | 陳廷茂 p.1 | p8 泡泡與備忘稿 |
| IL | Imitation Learning | 模仿學習 | 陳廷茂 p.11 | p8 備忘稿 |
| MLP | Multi-Layer Perceptron | 多層感知機 | 錢鵬予 p.18，寫明生成器與代理模型都是 MLP | p8、p11 備忘稿寫「全連接網路」 |
| FC | Fully Connected Layers | 全連接層 | 錢鵬予 p.26 | p8、p11 備忘稿 |
| SWA | Stochastic Weight Averaging | 隨機權重平均 | 錢鵬予 p.14 全稱、p.15 縮寫，文獻 Izmailov 等 | p11 備忘稿寫「權重平均」 |
| SGDR | Stochastic Gradient Descent with Warm Restarts | 熱重啟 | 錢鵬予 p.14，文獻 [25] Loshchilov 與 Hutter | p11 備忘稿寫「熱重啟」 |
| GA | Genetic Algorithm | 基因演算法 | 錢鵬予 p.6；p.24 寫明變異機制借自基因演算法 | 若第二代數學式頁提到翻轉的來由才會用到 |

### 1.6 沒列入的英文詞

- **單位：** GHz、mm、dB 是單位，不列。
- **人名、公司與程式庫：** Ansys、PyTorch、numpy 不列。
- **學長原圖的英文標籤：** Custom Normalize、Mutation、Selection Mode、Sign 函數探討、Random Sampling、Opening & Closing、GEN Issue，第 11 頁備忘稿已逐一對應中文，不重列。
- **近似二值化的另兩種：** Swish-like、Smooth Thresholding 是學長簡報的用語，不是縮寫，也找不到公認全稱，維持原文。
- **角度符號：** 第 8 頁備忘稿的 Phi、θ 是球座標角度，不是縮寫。

## 2 數學式

式子一律是 matplotlib mathtext 能渲染的寫法，已逐式實際渲染檢查，沒有用 `\begin{cases}`、`\text` 這類 mathtext 不支援的語法。每式的「白話」可直接當投影片上的一句；「符號」放備忘稿。

### 2.1 第一代

**式 1-1　二值化與直通估計**

```
$s=\mathrm{Sign}(z)\in\{-1,1\},\quad b=\frac{1+s}{2}\in\{0,1\},\qquad \frac{\partial\mathcal{L}}{\partial z}=\frac{\partial\mathcal{L}}{\partial s}$
```

- **符號：**
  - z：MPGN 最後一層、二值化之前的連續輸出，每格一個值。
  - Sign：符號函數，z ≥ 0 得 1，z < 0 得 −1。
  - s：二值化後的 ±1。
  - b：位移到 0 或 1 的金屬圖形，1 放金屬。
  - 𝓛：loss。最後一式的意思是反向傳播時把 Sign 的導數當成 1，梯度直接穿過，也就是 STE。
- **白話：** 前向時硬切成有金屬或沒金屬，反向時假裝這一刀不存在，梯度照樣傳回生成網路。
- **出處：** Sign 是式 (5)，p.21；輸出位移到 0 或 1 見 §3.1，p.16；STE 見 §3.3.1，p.21–22，引用 Bengio 等 2013。
- **簡化表示：** 碩論只給 Sign 的式 (5)，位移與 STE 都是文字描述、沒有式號。b = (1+s)/2 是我們把「位移至 0 或 1」寫成式子；STE 那一式依 p.22「反向傳播時直接對原始的連續參數進行更新」寫成。

**式 1-2　只罰未達標頻點的 loss**

n257 天線版：

```
$\mathcal{L}=\frac{1}{N}\sum_{n=1}^{N}\max(R_n-C_R,\,0)^2+\frac{1}{N}\sum_{n=1}^{N}\min(G_n-C_G,\,0)^2$
```

濾波天線版：

```
$\mathcal{L}=\sum_{y\in\{R,G\}}\frac{1}{N}\sum_{n=1}^{N}\left[\max(y_n-U_n,\,0)^2+\min(y_n-L_n,\,0)^2\right]$
```

- **符號：**
  - N：評分頻點數，每 0.5 GHz 一點；n257 版 7 點，濾波版 24–32 GHz 共 17 點。
  - R_n：第 n 個頻點的反射係數 S11，單位 dB，取 GEN 的預測值。
  - G_n：第 n 個頻點的增益，單位 dBi，同樣取 GEN 的預測值。
  - C_R：反射係數門檻 −10 dB。C_G：增益門檻 4 dBi。
  - U_n、L_n：濾波版該頻點的上界與下界遮罩，反射係數與增益各一組，見圖 3-13。
  - max(·, 0)：S11 已低於 −10 dB 的頻點貢獻 0；min(·, 0)：增益已高於 4 dBi 的頻點貢獻 0。
- **白話：** 已經達標的頻點不扣分，只有沒達標的頻點按差距平方計分。
- **出處：** n257 版是式 (9)、(10)、(11)，§3.4，p.25–26；濾波版是式 (12)–(16)，p.26–27，遮罩見圖 3-13。
- **與碩論的關係：** n257 版只是把式 (9)(10)(11) 合寫成一行，內容相同；濾波版把式 (12)–(15) 四項重排成一行，內容相同。碩論式子裡反射係數與增益共用 Mask_U、Mask_L 兩個符號，實際是兩組不同遮罩，這裡寫成「各一組」。

**式 1-3　代理模型把梯度傳回生成網路**

```
$\frac{\partial\mathcal{L}}{\partial\theta}=\frac{\partial\mathcal{L}}{\partial\hat{y}}\,\frac{\partial\hat{y}}{\partial x}\,\frac{\partial x}{\partial\theta},\qquad \hat{y}=\mathrm{GEN}_{\phi}(x),\ \ x=\mathrm{MPGN}_{\theta}(s^{*})$
```

- **符號：**
  - θ：MPGN 的權重，這一步要更新的對象。
  - φ：GEN 的權重，這一步凍結不動。
  - s*：期望響應，也就是 MPGN 的固定輸入；雙層板是 7 點反射係數串 7 點增益，共 14 個數。
  - x：MPGN 輸出的 0/1 金屬圖形。
  - ŷ：GEN 預測的反射係數與增益。
  - ∂ŷ/∂x：HFSS 給不出、改由 GEN 提供的那一段梯度。
  - ∂x/∂θ：MPGN 內部的梯度，二值化那一步靠式 1-1 的 STE 穿過。
- **白話：** HFSS 只給響應、不給梯度，就用一個學會 HFSS 行為的網路頂替中間那一段，讓整條鏈可以微分。
- **出處：** §3.1，p.16–17，圖 3-1；§3.3.2，p.23–24。第二代碩論演算法 1 第 10 行把同一件事寫成 𝓛(GEN(G(S)), S)，錢鵬予 p.28。
- **簡化表示：** 第一代碩論以文字與流程圖描述這條鏈，沒有寫成式子。省略二值化的位移，以及四層板 MPGN 分成兩個輸出分支。

**式 1-4　代理模型的線上訓練**

```
$\mathcal{D}\leftarrow\mathcal{D}\cup\{(x_t,\,y_t)\},\qquad \mathrm{MSE}_{\mathcal{D}}(\phi)=\mathrm{mean}\,(\mathrm{GEN}_{\phi}(x)-y)^2<10^{-4}$
```

- **符號：**
  - t：第 t 次迭代，每次只跑一次 HFSS。
  - x_t：這次生成的金屬圖形。y_t：HFSS 模擬出的反射係數與增益。
  - 𝒟：累積的資料，對應圖 3-1 的圖形暫存與結果暫存兩個 buffer。
  - MSE：GEN 預測與 HFSS 結果的均方誤差，在 𝒟 上平均。
  - 10⁻⁴：GEN 的訓練停止門檻，碩論寫 0.0001。
- **白話：** 每次迭代把新的模擬結果存進資料池，GEN 在累積的資料上訓練到誤差夠小，才拿來傳梯度。
- **出處：** MSE 與 0.0001 見 §3.3.2，p.24；每次存下圖形與響應、以線上學習訓練 GEN 見 §3.1，p.16；圖 3-1，p.17。
- **簡化表示：** 碩論以文字描述、沒有式號。碩論沒有明寫每次是否用全部累積資料重訓，這裡只寫成「在累積資料上訓練」，與 `gen1_碩論對照.md` C7 的判定一致。

### 2.2 第二代

**式 2-1　原始設計為什麼會卡住：STE 加梯度裁減**

```
$s=\mathrm{Sign}(x),\qquad \frac{\partial\mathcal{L}}{\partial x}=\frac{\partial\mathcal{L}}{\partial s}\cdot\mathbf{1}[\,|x|\leq 1\,]$
```

- **符號：**
  - x：生成器輸出端、二值化之前的值。
  - s：二值化後的 ±1。
  - 𝟏[·]：指示函數，條件成立為 1，否則為 0。
- **白話：** 原始設計只讓落在 −1 到 1 之間的值傳梯度，數值一旦跑出這個區間，梯度就變成 0，訓練停住。
- **出處：** §3.3.1，p.21–22，圖 3-4(b)。
- **簡化表示：** 碩論以文字與圖描述、沒有式號。這裡說的「原始設計」是第二代碩論對第一代程式的描述；第一代碩論本身只寫 STE，沒有寫裁減。

**式 2-2　BiScaleNorm**

```
$\hat{x}=\frac{x}{M}\ \ (x>0),\qquad \hat{x}=\frac{x}{|m|}\ \ (x<0)$
```

```
$M\leftarrow\alpha M+(1-\alpha)\max(x),\qquad m\leftarrow\alpha m+(1-\alpha)\min(x)$
```

- **符號：**
  - x：生成器某一層的輸出。
  - max(x)、min(x)：這一層輸出的最大值與最小值。
  - M、m：兩個極值的指數移動平均 EMA。
  - α：平滑係數，碩論預設 0.9。
  - x̂：正規化後的輸出，大約落在 −1 到 1。
- **白話：** 正負兩側各自除以自己的極值，把數值壓回式 2-1 會傳梯度的區間；兩側縮放比例不同，所以它同時有激活函數的效果。
- **出處：** 式 (16)–(18)，§3.3.1，p.23；正負兩側的梯度 1/max(x)、1/|min(x)| 見同頁文字；生成器 5 層全連接層的激活函數都換成它，p.26。
- **與碩論的關係：** 與式 (16)–(18) 一致。碩論式 (18) 左邊仍寫 x，這裡寫成 x̂ 以區分前後；x = 0 的情況碩論沒有定義。

**式 2-3　上下界 loss 加 Smooth L1**

投影片用的一行：

```
$\mathcal{L}=\sum_{j=1}^{4}\mathbf{1}[\hat{S}_j\notin\Omega_j]\cdot\mathrm{SmoothL1}(\hat{S}_j-S_j)$
```

碩論原式，以帶內增益為例，其餘三項同形：

```
$\mathcal{L}_{G,n257}=\mathbf{1}[\hat{S}_{G,n257}<S_{G,n257}]\cdot\mathrm{SmoothL1}(\hat{S}_{G,n257}-S_{G,n257})$
```

```
$\mathcal{L}_{total}=\mathcal{L}_{G,n257}+\mathcal{L}_{G,other}+\mathcal{L}_{R,n257}+\mathcal{L}_{R,other}$
```

- **符號：**
  - j：四項，帶內增益、帶外增益、帶內反射係數、帶外反射係數；帶內指 n257，帶外指 24–32 GHz 中 n257 以外的頻點。
  - Ŝ_j：代理模型預測的響應。
  - S_j：理想響應，即碩論圖 3-3 的目標曲線。
  - Ω_j：該項允許的範圍。帶內增益要不低於 S，帶外增益要不高於 S，帶內反射係數要不高於 S，帶外反射係數要不低於 S。
  - 𝟏[·]：指示函數，沒達標時為 1。
  - SmoothL1：平滑 L1 損失。
- **白話：** 和第一代一樣只罰沒達標的頻點，差別是把平方換成 Smooth L1，誤差大時不會被平方放大。
- **出處：** 式 (11)–(15)，§3.2，p.19–20，圖 3-3；四項的方向依式 (11)–(14) 的不等號。
- **簡化表示：** 一行式是把式 (11)–(14) 四項合寫。碩論沒有寫出 Smooth L1 本身的定義，也沒有寫各頻點是加總還是平均，這裡同樣不寫。S_j 不要填成規格文字的數值，圖 3-3 的目標曲線與 p.33 的規格文字不完全相同。

**式 2-4　二階段生成：翻轉與回滾**

翻轉：

```
$\hat{y}_i\leftarrow 1-\hat{y}_i,\quad i\in I_k,\quad |I_k|=k=\mu\cdot\mathrm{len}(\hat{y})$
```

回滾與觸發：

```
$y^{*}\leftarrow\hat{y}\ \ \mathrm{if}\ \ \mathcal{L}(\hat{y})<\mathcal{L}_{\min};\qquad \hat{y}\leftarrow\mathrm{mutate}(y^{*},\mu)\ \ \mathrm{when}\ \ t\ \mathrm{mod}\ c=0,\quad \mu\leftarrow\lambda\mu$
```

- **符號：**
  - ŷ：生成器輸出的 0/1 金屬圖形，25×25 共 625 格。
  - μ：變異率，起始 0.05。len(ŷ)：像素總數。
  - k：這次要翻轉的格數，第一次約 31 格。I_k：隨機挑出的 k 個位置。
  - y*：歷史最佳圖；第二階段開始時設為第一階段的最佳圖。
  - 𝓛_min：歷史最小 loss，演算法中的 minloss。𝓛(ŷ)：這次圖形的 loss，演算法 2 第 16 行算出的值。
  - t：迭代次數。c：變異週期，400 次。
  - λ：衰減參數，每次變異後把 μ 縮小，碩論沒給數值。
- **白話：** 每 400 次迭代從歷史最佳圖隨機翻轉少量像素，逼生成器跳到附近的新圖；翻轉後沒變好，下一次仍從歷史最佳圖出發，這就是回滾。
- **出處：** 翻轉是式 (19)–(21)，§3.3.2，p.25；觸發、衰減與回滾是演算法 2 第 2、4–7、19–21 行，p.29，文字見 §3.4.2，p.28–30；μ = 0.05、c = 400 見 §4.2.2，p.36。
- **簡化表示：** 翻轉一行與式 (19)–(21) 一致。回滾一行是演算法 2 的濃縮，省略了每次變異後生成器先強制擬合變異圖的步驟，即演算法 2 第 6 行；該行碩論寫成 min_G 𝓛(G(Ŝ), ŷ)，但 §3.3.2 寫明生成器是固定輸入，若要寫進投影片，輸入宜寫成期望響應 S，寫之前先跟協調者確認。

## 3 建議放在投影片哪裡

### 3.1 新增兩頁數學式頁

- **第一代：** 第 7 頁是整頁流程大圖，沒有空間；第 9 頁是結果表。建議在第 7 頁之後新增「第一代方法的數學式」。
  - 描述句：「二值化、只罰未達標頻點的 loss 與代理模型傳梯度三條式子，串成可以反向傳播的流程。」
  - 三列表格，欄位為名稱、式子、白話，依序放式 1-3、式 1-2、式 1-1。先講整條鏈，再講 loss 與二值化。
  - 式 1-4 線上訓練與四式的符號定義放備忘稿；濾波版 loss 只放備忘稿。
- **第二代：** 第 10 頁已是左圖右文，右欄兩個小標加六條，沒有空間。建議在第 10 頁之後新增「第二代方法的數學式」。
  - 描述句：「BiScaleNorm 把各層輸出壓回有梯度的區間，loss 改成只罰超出上下界並用 Smooth L1，二階段生成以翻轉像素加回滾跳出局部解。」
  - 三列表格，依序放式 2-2、式 2-3 的一行版、式 2-4 的翻轉與回滾兩行。式 2-1 不單獨成列，改成 BiScaleNorm 那一列的白話：「原始設計只在 −1 到 1 內傳梯度，BiScaleNorm 把數值壓回這個區間」；式本身與符號放備忘稿。
- **頁數：** 33 頁變 35 頁，第一代 4 頁、第二代 3 頁。每頁一張表、三列，符合 §1.5 一頁一張表與不擁擠。
- **做法：** 式子建議用 matplotlib mathtext 輸出成透明底 PNG 貼進表格欄，字級與正文一致；python-pptx 沒有好用的公式物件，圖片也方便 Ricky 自己換。
- **替代方案：** 若頁數要壓，第二代可以不加頁，把第 10 頁右欄改成式 2-2 與式 2-4 兩式，各配一句白話，loss 式放備忘稿。第一代沒有這個選項。
- **措辭：** 各代 loss 定義不同，數學式頁不放 loss 數值、不跨代比較，與現有備忘稿一致。

### 3.2 縮寫全稱補在第一次出現的地方

依 §1.1「少括號」，全稱寫成子句，不用括號。以下是建議寫法，第四版頁碼。

| 頁 | 縮寫 | 建議寫法 |
|---|---|---|
| p2 | HFSS | 「以全波電磁模擬軟體 HFSS，即 High-Frequency Structure Simulator，驗證是否通過規格。」這是全份第一次出現 |
| p2 | BiScaleNorm | 第二代列寫「BiScaleNorm 雙尺度正規化穩定二值化訓練」 |
| p5 | n257、S11、dBi | 不在本次範圍，建議只在備忘稿補：n257 是 3GPP 定義的 26.5–29.5 GHz 毫米波頻段；S11 是反射係數；dBi 是相對等向性天線的增益單位 |
| p7 | MPGN、GEN | 描述句末改為「圖中 MPGN 即 Metal Pattern Generation Network，是生成網路；Gradient Estimation Network 簡稱 GEN，是代理模型。」 |
| 新增第一代數學式頁 | STE、MSE | 白話欄寫「直通估計 STE，即 Straight-Through Estimator」；MSE 只在備忘稿出現時寫「均方誤差 MSE」 |
| p10 | BiScaleNorm | p2 已補全稱，此頁可不再補；若 p2 沒補，小標改「BiScaleNorm 雙尺度正規化：穩定二值化訓練」 |
| 新增第二代數學式頁 | EMA、Smooth L1 | 備忘稿寫「M、m 是極值的指數移動平均 EMA」；表格白話寫「Smooth L1 即平滑 L1 損失」 |
| p11 泡泡 | Smooth L1、n260 | 泡泡字數有上限，不改圖；備忘稿該行補「Smooth L1 即平滑 L1 損失」「n260 是 3GPP 定義的 37–40 GHz 頻段」 |
| p8 備忘稿 | DDPG、DQN、PPO、MAE、CNN、PReLU、Tanh、LeakyReLU、ReLU、GPU | 各在第一次出現的那一行補一句，例如「DDPG 即深度確定性策略梯度、Dueling DQN 即對決式深度 Q 網路、PPO 即近端策略優化」 |
| p11 備忘稿 | GEN、RIS、CNN、LSTM、Adam、AdaBelief、NAG、RMSProp、AdaBound、SGD、MSE、L1、ApproxSign、LayerNorm、DyT、Ranger、S11、S21、S22 | 同上；GEN 已在原圖對照句說明是代理模型，補「全稱 Gradient Estimation Network」即可 |

### 3.3 附錄名詞表

- 第 32 頁兩欄各 13 列已滿，不宜再加。
- 建議新增一頁「附錄：縮寫全稱」，只收投影片正文或新增數學式頁上看得到的縮寫：HFSS、MPGN、GEN、STE、MSE、BiScaleNorm、EMA、Smooth L1、n257、n260、S11、S21、S22、dBi，第一、二代約 14 列，兩欄排。
- 只在備忘稿出現的縮寫，例如強化學習與優化器那一串，不進附錄，在備忘稿第一次出現時補全稱即可。
- 第三、四代的縮寫若也要進同一頁，由協調者合併；本檔只負責前兩代。
