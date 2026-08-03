# 可微模擬器(diffsim)實作指導書 — 交接包

> **給另一個 session 的實作指導**(2026-08-02 討論定稿;本 session 只寫方案不動工)。
> 讀完本檔應該可以直接開工,不需要回讀討論記錄。討論脈絡:docs/discuss/scratch.md「diffsim」條。

---

## ⚠ 執行結果（2026-08-03 收尾；本節後補，以下原文保留不動）

**本指導書的鏈已全部走完，結論是：diffsim 在實用指標上沒有價值，路線收線。**

| 階段 | 結果 |
|---|---|
| 階段 0 幾何 | ✅ `.sab` 二進位直解，零 HFSS 清掉「待確認」 |
| L1 腔模型 → GATE 1 | ✅ 過（判讀被自己的稽核修正兩次） |
| L2 MoM → GATE 2 | ❌ 未過（物理修正後仍未過） |
| L3 精確核 → G-L3a | ❌ 當時未過；**修掉饋電模型後其實過得了**（[analysis-09](log/analysis-09-diffsim-l3.md) §6） |
| 殘差 head ＋最終 SM 比較 | ✅ 完成 —— **物理當特徵也沒價值**（[analysis-10](log/analysis-10-sm-vs-physics.md) §10） |

**★ 最重要的一課不在物理，在指標**：全程用層內 rank ρ 當 gate，
但這條線若要有用，用途是**選批**——而兩者在這問題上**幾乎正交**。
最強的證據是縱向的：修一個**真的**物理 bug 讓 ρ 改善 **+124%**，
而選批的 `P(勝隨機)` 從 18% → 17%，**完全沒動**（analysis-10 §7.2）。

**留下來的資產**（都獨立於上面成立）：
1. `script/sm_selection_audit.py` —— 把 SM 的準度翻譯成選批能力的稽核工具
2. **主 SM 在完全未見過的域選批比隨機還差**（P=21–26%）⇒ 開新域別用它挑候選
3. `_load_clean_stores()` 自動納入 `dedust_auto*`/`dedust_c*` ⇒ SM 訓練集是 **587 店**，
   任何「SM vs 其他」比較沒扣這層都會**系統性高估 SM**
4. 「驗收指標要用**使用方式**的指標」＋「**全域均勻**的改善 ρ 可代理／**結構性**的必須量 top-K」
   → `docs/discuss/decisions.md`
5. `script/diffsim/l3.py` —— 四管道獨立驗證過的精確分層 Green's function
6. 三個真 bug 的修復＋回歸測試：MKL 多執行緒複數 LU／饋線死端 stub／L1 二值梯度 NaN
7. [proposal-mesh-convergence](discuss/proposal-mesh-convergence.md) —— HFSS 網格收斂實驗規格（待排）

**★★★★ 根因已確認**（analysis-10 §37，2026-08-03）：**埠模型** ——
delta-gap 直接打在貼片邊上，沒有饋線。

最乾淨的證據是**無因次**的（不需要 `Z_c`、不需要擬合）：同一個模態振幅下的饋入電流比
`I_feed/I_max`，delta-gap 打邊是理論的 **2.22×**、接 9mm 饋線後是 **1.30×**
—— 兩者相除 **1.71 = 埠代價的硬下界**，而剩下的 1.30 ≈ **1/η = 1.33**
⇒ **接上饋線後殘餘落差全部由表面波解釋，物理上沒有缺口。**
完整的帳：`Balanis 529Ω × η 0.75 ÷ 2.05 = 195Ω`，實測 195.4Ω。

⚠ **HFSS 那邊的埠在 22.5mm 饋線的遠端**（`single_port.py:374-407`，Lumped Port @x=27.5mm、
`DoDeembed=True`）—— 貼片是被**真實 TEM 行進波**餵的。去嵌入只搬參考面（相位），
不會把它變回集總源 ⇒ **兩邊的埠物理從一開始就不同，這是結構性的、不是數值的。**

❌ **撤回：舊的「`1/√N` 離散化天花板」結論是統計假象**（原文留在 analysis-10 §19）。
那組「7→23 格比值 0.192→0.406」是**固定 dx、改貼片物理尺寸**，量到的是埠因子隨貼片大小變。
真網格收斂（固定 3.4mm 貼片、dx 0.2→0.068mm、未知數 ×8.7）比值只 **0.369 → 0.387，完全平的**。

⭐ **為什麼這個修法應該同時修好量級與排序**：§22（全域常數 k）與 §25（用 `Re/η` 修）
都栽在「比值不是常數」（0.19–0.41，spread **1.46×**）。埠修好後 spread 降到 **1.2×**
（11/17/23 格 = 0.74/0.91/0.88）—— 這是第一個「修完之後 `k` 幾乎變成常數」的介入。

**其餘已排除的候選**（附量化，免得再猜）：點對點動態核上限 ~25%（`Im(G)` 在 r=0 解析）、
有限地平面對 `Zin` 是振盪型 <10–15%、εr=1 時比值仍只有 0.715（缺口不是介質）。

❌ **已試過並失敗**：解析的加權基底（`f_j = w_j·rooftop_j`）——
那在 Galerkin 下是**恆等變換**（`J = W(WZW)⁻¹WV = I`），改振幅不改變函數空間。

**★ 方法論教訓**（analysis-09 §6.3）：**排除法只能排除你想到的。**
我曾列四條「獨立證據」宣稱排除所有可能、結論是「離散化天花板」——
四條到今天都仍成立，**結論卻錯了**，因為第五條（饋電模型）我根本沒列進候選，
而且它當時**已經有人在查**。⇒ 建模假設的完整清單在 analysis-09 §8。

---

## 0. 定位(三句話)

1. **要輪廓,不要復現**:目標是排名器+梯度產生器,不是 100% 重現 HFSS。真相仍由 HFSS 公證,
   凍結尺/公證文化原樣沿用——diffsim 的角色與現在的 SM 同構,差別是外推力來自物理結構而非訓練集覆蓋。
2. **鏈可微=節點可置換**(Ricky 定調):資料量已到 3 萬筆規模,物理做不出來的節點可以用
   learned module 置換、離散處用 STE 解。物理節優先,學習節記帳(見 §4)。
3. **L3(正統全推導)是退路不是禁區**:不用逼到正統,但卡住時它存在(見 §3)。

## 1. 已考證的幾何事實

來源 = `antenna/patch/patch_simulator/single_port.py`(行號對應現檔)。這是唯一真相,
指導書如與 code 不一致以 code 為準。

| 項 | 值 | 來源行 |
|---|---|---|
| 像素網格 | **25×25 = 625**,像素 0.2mm,貼片畫布 5×5mm,起點 (0,0) | :150-184 |
| 貼片高度 | z = 0.508mm(基板上表面),銅厚 0.035mm | :300, :306 |
| 基板 | Rogers RO4003(εr≈3.55, tanδ≈0.0027),厚 **0.508mm** | :214-234 |
| 像素接縫 | 每盒 +0.01mm 重疊(保連通;平面片近似可忽略) | :301-305 |
| 饋電 | **同層微帶饋線 edge-feed**(feed_line 與貼片 Unite 成一體);Lumped Port 50Ω,積分線 (27.5mm, 2.5mm) z:0→0.508mm;DoDeembed=True | :358-400 |
| 垂直結構 | **零 via**——唯一垂直物是 port 激勵面 → 平面電流片=**精確幾何**,不是近似 | 全檔 |
| 地平面 | 有限(ApplyInfiniteGP=False),開放輻射邊界 OpFreq 28GHz | :402-413 |
| 頻帶 | 24–32GHz,17 點(0.5GHz 步);自適應細化 @28GHz,MaxDeltaS 0.02 | :415-469 |
| 輸出 | S11(dB) 17 點 + boresight RealizedGainTotal(dB) 17 點 | :500-597 |

~~**待確認**:板總尺寸、feed_line 寬度/走線幾何、GND 大小~~ → **2026-08-02 用 SAB 二進位解析清零**(`geom.sab_probe`,結果在 `geom.py` 開頭的表)。
⚠ 原本這裡寫「L2 只把饋線當固定『常開』格,長度誤差被 de-embed+校準吃掉」——**那是錯的**,
見上面的根因段與 analysis-10 §37:饋線定義的是**埠的場結構**,不是只有長度/相位。
L2 現在有 `feed_len` 把線建進格網(`SOLVERS['l3fl']`)。

## 2. 輸出契約(先定介面,一切可置換)

```python
diffsim(pattern) -> {'S11': (17,), 'Gain': (17,)}   # dB;頻點 np.linspace(24, 32, 17)
# pattern: (625,) float ∈ [0,1](排名時餵二值;優化時鬆弛)
```

- 跟資料集樣本 y 的格式一致(前 17 = S11,後 17 = Gain)→ `worst_margin`/oob/rad 評分
  **直接沿用 `script.dedust` 同一把尺,下游零改動**。
- 全鏈 torch complex,端到端可微;二值化留在鏈外(STE 進場見 §4)。

## 3. 簡化階梯

> 核心簡化槓桿只有一個:**stackup 永遠不變**(基板/地/饋電/網格全樣本共用,變的只有 625 個
> 像素的開關)。所以所有難算的 EM(分層 Green's function、Sommerfeld 積分、饋電細節)都是
> 常數——一次算掉,或乾脆從資料擬合掉。

### L1:可微廣義腔模型(1–2 天,單樣本 ms 級)——先跑這個

- 數學:patch 腔模型推廣到任意像素形狀 = 金屬區域上的 2D Helmholtz 特徵問題
  (PMC 側壁 → Neumann 邊界;εr、h 已知)。輸入阻抗用模態展開:
  `Zin(ω) ≈ jωμh · Σₙ ψₙ(feed)² / (kₙ² − k²(1 − j/Q))`,S11 = (Zin−50)/(Zin+50)。
- 實作:625 格稀疏 Laplacian(以像素 density 加權=天然連續鬆弛),`torch.lobpcg` 取前 ~15 模。
  特徵值/向量的梯度:lobpcg 可反傳,或隱函數定理手寫。feed 位置=饋線接入點最近像素
  (y=2.5mm 邊中點,x 看 §1 待確認,先用邊中點不會差到哪)。
- Q:先用單一經驗常數,或當一個可擬參數(第一個「學習節」)。
- 抓得到:諧振頻率位置、模態與饋點的耦合強弱 → **S11 輪廓**。
  抓不到:輻射 Q 精度、金屬島間空氣耦合、饋線本體效應。
- 島嶼:非連通成分不接饋點,ψₙ(feed)=0 自然去耦,不用特別處理;負片/slot 域的模式反轉
  (挖空區當共振域)L1 先不做,只求正片域 ρ。

### L2:譜域 MoM,核用擬合(1–2 週)——甜蜜點

- 均勻網格 rooftop 基底:25×25 → 未知數 ≈ 2·25·24 ≈ **1200,Z 矩陣稠密直解就好**
  (1200³ 複數 GPU ms 級、CPU 也扛得住;25×25 根本不需要 FFT/迭代法,50×50 才需要)。
- 平移不變性:`Z[m,n] = K(r_m − r_n; ω)`——K 是(徑向距離 × 分量)的小表。三個來源:
  - **(b) 參數化 K(徑向樣條/小表)+ 用可微鏈在幾百筆 HFSS 樣本上端到端擬合**(推薦起手)。
    優雅之處:求解器本身可微,擬核=對解算器反傳,不用另寫擬合器。
  - (a) 解析分層 Green(Michalski–Mosig MPIE + Sommerfeld 積分表格化)= L3 退路,正統但工程重。
  - (c) 整個 K 換 learned 卷積 = Model 置換的極端(§4)。
- 饋線:**建進格網**(`feed_len` 列 × `LINE_COLS` 欄,恆金屬),delta-gap 移到線的遠端,
  `S11` 用**駐波法**萃取(= HFSS wave port);不需要 `Z_c`、不需要 de-embed 距離。
  ~~常開 cells + delta-gap 打貼片邊;de-embed 差異交給校準~~ ❌ 2026-08-03 撤(§37)。
- 遠場:電流片的遠場積分(解析、便宜)→ Gain;Realized = Gain × (1−|S11|²)。
- 頻率:17 點各解一次(batch 維度平行);之後可加有理擬合省點。
- 校準:獨立校準切片(從全史另抽,與驗證集不相交)上做每頻點仿射(a·x+b)吃掉系統偏移
  ——合法,不碰排名;凍結尺永不進校準(§5 資料分割鐵則)。

### L3:正統推導 = 退路

不是禁區(Ricky 2026-08-02):(b) 擬不穩就退回 (a) 正統 MPIE。關鍵字:Michalski & Mosig
mixed-potential integral equation;分層介質 Green's function 表格化。stackup 固定,一次性成本。

## 4. Model 置換與 STE

- **可置換節點表**:核 K → learned 卷積;Q/損耗 → learned 標量頭;末端殘差修正頭
  (31k 樣本訓,把 diffsim 輸出修向 HFSS)= physics-anchored SM。
- 原則:置換單位=鏈中一節、tensor 介面不變;**哪一節是學的要記帳**(可追溯哪部分是物理、
  哪部分是資料)。
- 二值化:優化用 STE 或 sigmoid 溫度退火+投影排程(光子學逆設計成熟套路)。
- ⚠ **吸收體陷阱**(文獻已知雷):中間態電導率=有損片,吸功率讓 S11 好看但不輻射——
  優化目標必須綁輻射效率項,且最終候選一律硬二值送 HFSS 公證。純排名(輸入二值)不觸雷。

## 5. 驗收協議(同一把尺,資料裁判)

- 驗證集:凍結尺 30(`dedust_r50b1b_frozen`,**唯一副本勿動**)+ 全史分層抽
  qual/near/neg/senior/bridge 各 20–30 筆。
- 指標:① **rank ρ(Spearman)diffsim-wm vs HFSS-wm = 主 KPI**(與 SM 前瞻 ρ 同口徑,
  analyze batch 的算法);② 仿射校準後 S11 每頻點 MAE;③ **負片域 ρ**——SM 的死穴、
  diffsim 的存在理由,單獨報。
- **Gates(發車前寫死)**:L1 ρ≥0.4 → L2 值得做;L2 裸 ρ≥0.6 → 繼續投;
  L2+殘差頭應顯著超過純 SM,否則誠實收檔記「不成立」。
- **Gate=單向門**:ρ 不到就停+誠實記錄,**禁止對著驗證集反覆調參調到過**(Goodhart 護欄)。
- **資料分割鐵則**:擬合核/仿射校準/殘差頭用的 HFSS 樣本與驗證集**必須不相交**;
  **凍結尺 30 只做最終報數,永不進擬合**(§3 L2 的「校準」改在獨立校準切片上做)。
- 對照數字:SM 域內前瞻 ρ 各批波動見 round 檔/`docs/kpi.csv`;OOD 凍結尺誤差 `docs/kpi_ood.csv`。

## 6. 資料資源

- ~31k 筆真值:NAS `T:\碩一_鄒穎麒's\antenna\dataset\dedust_*`(SampleStore 一筆一檔 `.pt`,
  `(x, y)`:x=625 二值,y=34=S11 17+Gain 17);讀法 `antenna/utils/store.py`。
- rad pattern 子集:含 rad 的 store(SinglePortRadSimulator 萃取);凍結尺 30 筆含 rad。
- 評分函式:`from script.dedust import ...`(sel_score/worst_margin/rad_window_margin)。
- 全史索引快取:討論 session scratchpad 的 `res_index.json`/`pt_index.json` 模式可抄
  (id → store/margin,掃一次 NAS 自建也行)。

## 7. 工程紀律

- 環境 conda `ant`;先 `torch.cuda.is_available()` 查 GPU(沒有也行,25×25 CPU 扛得住)。
- 位置:`script/diffsim.py` 或 `script/diffsim/` 起步——**不碰 `antenna/` 核心**,
  層級單向不變式照舊;`python -m pytest tests/ -q` 全綠收尾。
- 記錄:開工時 ONGOING 🔜 升 🔵;零 HFSS 的驗證走 `analysis-NN` 檔,gates 發車前寫死;
  凍結尺讀數若進 kpi 檔,另立欄位別跟 SM 混。
- 參考關鍵字(查文獻用):pixelated antenna adjoint topology optimization、SIMP for
  conductors、rooftop basis / spectral domain MoM、MPIE Michalski-Mosig、
  photonic inverse design adjoint(ceviche / Meep,先例最成熟的鄰域)。

## 8. 階段計畫

| 階段 | 工作 | 出口 |
|---|---|---|
| 0(半天) | 幾何確認:.sab 開一次量 feed_line/板尺寸(或 only_create_project) | §1 待確認清零 |
| 1(1–2 天) | L1 腔模型 spike → 分層驗證集 ρ | gate:ρ≥0.4 |
| 2(1–2 週) | L2 擬合核 MoM;卡了退 (a) 正統/升 (c) learned K | gate:裸 ρ≥0.6 |
| 3(視結果) | 殘差頭+Gain/rad 支線+STE 梯度優化器(梯度爬山 vs 像素翻轉;山頭枚舉) | 對 SM 的顯著優勢 |
