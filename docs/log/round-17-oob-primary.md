# Round 17 — 帶外主目標：低側裙擺攻堅

- **狀態**: proposed  <!-- proposed | running | analyzing | concluded | archived -->
- **提出 / 開跑 / 結論**: 2026-07-09 / — / —
- **一句話問題**: 低側裙擺（24–25.5 GHz，全家族帶外地板 ≈9.2）是體質還是可壓？
- **一句話結論 (TL;DR)**: 待分析
- **指向**: `script/dedust.py` select-r17（`_surgery` 算子）· 結果夾 §3 · discuss/scratch「帶外拆側發現」· decisions「目標層級定案」

> 本檔只放**連結指向**其他層,不複製內容。

## 1. 假設 (Propose)
- **問題 / 假設**：Ricky 定調「帶外當主目標探索幾輪」。拆側數據（scratch 2026-07-09）已證：
  低側全家族鎖死（gain_lo 3.3–4.0 / s11_lo −5.3±0.4 / rolloff_lo≈0.3，單側貢獻 ≈9.2），
  高側旋鈕已找到（中央塊，rolloff_hi 3→7）但被 `oob_bad=max(兩側)` 遮住。
  **假設**：低側裙擺＝碎片雲的水平諧振尾巴 → 切斷水平電流路徑（vslot/colcut）可把它上推出 24–25.5。
- **為何現在做**：R16 進行中已回數據給出拆側破案；帶外自 R11 追蹤以來從未當過主目標
  （字典序第三位幾乎不 bind、R15 GA 權重僅 0.02）——這是第一輪帶外主目標批。
- **判準（發車前寫死）**：
  - **T 臂主判準**：任一 surgery 使 `oob_gain_max_lo ≤ 3.2`（基線 3.6–4.0，Δ≥0.5）且 wm ≥ −1
    ⇒「低側可動」成立 → 下輪劑量精修；若全臂 lo 不動（±0.3）⇒「低側＝家族體質」定案
    → 判準拆側提案（低側=體質常數/高側=可優化量）升 decisions 討論（改判準需 Ricky 拍板）。
  - **C 臂**：任一雙中央塊三標過且 `oob_bad < 10.14`（現行三標內帶外紀錄 a022）＝帶外紀錄更新；
    副產品＝中央塊可加性 Δoob(雙塊) ≈? ΔA+ΔB。
  - **N 臂**：a024 三次中位 wm ≥ +0.30 ⇒ 換王（i02 +0.29 → a024）；a017 兩次中位 wm ≥ 0
    ⇒ 三標過＋oob 9.86＝帶外紀錄直接改寫。
- **依據**：decisions「目標層級定案」（oob_bad 定義與 tiebreak 地位不動,本輪只是把它當研究對象）·
  scratch「帶外拆側發現」· R14 消融（翼買帶內付帶外）· r15v 理論模板（0 翼端點帶外 4.1）

## 2. 實驗設計 (Design)
| 臂 | kind | 筆數 | 內容 | 對照 baseline |
|---|---|---|---|---|
| T 帶外手術 | surgery | 20 | `_surgery`：vslot（切水平電流,主攻低側）×5、colcut（雲寬縮減）×3、hslot（切垂直電流,對照）×6、rowcut（雲高縮減,對照）×3、mslot（主件陷波槽）×3；錨點 c25/x00 | 錨點自身（c25 oob 11.76 / x00 10.74;lo 側 ≈9.2） |
| C 雙中央塊 | dblcentral | 9 | c25/x00 中央帶（col11,w3）疊兩塊,候選對寫死 | addmap 單塊 Δ（可加性檢驗） |
| N 公證 | notarize | 6 | a024×3（挑戰 i02）＋a017×2（卡線 −0.01）＋a022×1 | 鐵則:紀錄級一律重複量測 |
- **判準（用哪把尺）**：`worst_margin` 同一把尺＋`oob_metrics` 拆側欄（lo/hi/rolloff）為本輪主讀數。
- **HFSS 預算**：35 筆 ≈1.5–2hr。批次原則 ≥30 ✓（hslot/rowcut 即冗測對照）。
- **設計筆記**：原案「翼平移/剖半/寄生條」在實際幾何上全數不可行——**家族實形＝上半碎片雲**
  （雲頂到 row0、中央帶在 i02 已滿），無平移空間、剖不斷、無寄生條空位 → 改切削式手術（slot＝
  貼片天線經典陷波手法）。此觀察本身即成果：champions 的「雙翼」意象要修正為「雙雲」。

## 3. 執行紀錄 (Run)
```
# 開發機(已做): python -m script.dedust select-r17   → dedust_r17_input 35 筆
#               python -m script.dedust check-dup --input dedust_r17_input   → 重複 0 ✓
# 正式機(誰先空誰跑;37 r16b 收完即接):
#   python -m script.dedust run --input dedust_r17_input --store dedust_r17
# 任一機看進度: python -m script.dedust report --input dedust_r17_input --store dedust_r17
```
| 臂 | 機器 | 狀態 / 進度 | 結果夾 (NAS) |
|---|---|---|---|
| T+C+N | 待機器（37/218 誰先空） | 待發 | `dataset/dedust_r17` |
- **事件 / crash / 全域變更**：—

## 4. 分析 (Analyze)
（待收檔）

## 5. 結論 (Conclude)
（待分析）

## 6. 後續決策 (Next)
- 低側可動 ⇒ 下輪劑量精修（贏家手術 × k 掃描）；低側體質 ⇒ 判準拆側討論＋帶外線收斂到高側精修。
- a024 換王與否 → champions.md 更新。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r17`
- ONGOING 動作: （收檔時補）
