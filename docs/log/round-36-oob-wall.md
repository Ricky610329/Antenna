# Round 36 — oob 9.0 抗線攻堅：低 oob 錨嫁接 × 批 50 新規模 × CNN 單 rank

- **狀態**: running（2026-07-23 凌晨開輪;自主續輪宣告制）
- **提出 / 開跑 / 結論**: 2026-07-23 / 2026-07-23 / —
- **一句話問題**: dual 高原證明「wm 過線側壓 oob」卡死 9.08-9.14——**反向嫁接**（selfgen 低 oob 錨
  oob<9.0∧rad≥0,爬 wm 最後 0.02-0.13）能不能從沒人爬過的另一側破 9.0 紀錄？同時:tier 再平衡
  第一格落地（批 75→50）+CNN 排序主鍵首輪。
- **指向**: [round-35](round-35-fast-cadence.md)（兩堵牆/轉正判準/再平衡二連讀）· decisions
  「Tier 再平衡規則」· chains/c3g*.jsonl

## 1. 假設 (Propose)
- **證據**：①dual 三鏈（c1d4/5/6）150+px 高原=wm 過線群聚 oob 9.08-9.14,d=1 動不了 oob 量級;
  ②selfgen 池８顆近線材料**全 rad≥0**（嫁接錨候選,依 dual score:a218_00010 −0.02〔wm+0.13/oob9.05〕>
  a218_00016 −0.04 > a37_00138 −0.05〔wm+0.10/**oob8.99 已內側**〕）;③嫁接側從未被爬過（毫米距離
  同量級,鄰域全新）;④tier0:tier1=2.37×/2.03× 二連讀。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **主指標**：可用帶外紀錄（wm≥0.15∧rad≥0∧oob<9.0;門檻引 `docs/records.json`）——嫁接鏈或
    批線任一命中=紀錄候選→**公證鐵則**（/notarize;select-repeat ×2→3/3 一致才記帳）。
  - **嫁接鏈 c3g**（goal dual+--expert 標配;錨序=a218_00010→dry 換 a218_00016→a37_00138）：
    單錨連兩包 dry=換下一顆;**三顆全 dry=嫁接判死**（回報,R37 換機制——d=2 跳步另案）。
    鏈上限 ≤2 條並行不變;錨發鏈前驗 rad≥0（已驗,八顆全過）。
  - **批 50 規模**（tier 再平衡第一格）：三批三標率 ≥R35 的 0.75×（=6.0%;規模減半統計容忍）∧
    批週期 ≤R35 中位的 75%——過=常駐;三標率 <4.5%（0.55×）=規模傷產出,回報並回 75。
  - **CNN 單 rank**（O 臂 --cnn-solo 預設開）：O 臂三標率對 R33-35 雙 rank 帳（基線 62% R33b2）
    ——掉過半（<31%,三批合併讀）=回雙 rank;「全鏈換錨/zoo 預設」範圍**留 Ricky 裁決**（凍結尺衝突）。
  - **asym 進鍵判定**（R35 遺留）：b1 判讀時彙總 R35 三批 asym×realized——G 臂內 |ρ|≥0.3 →
    R37 起 free 預過濾鍵;<0.3 記錄續。denovo asym=None 缺口本輪補（生成端算）。
  - rad-key 退鍵（R36 移除;復鍵帳跨批記錄,前瞻 ≥0.4 連兩批復鍵）;struct-pen 4.0 常駐;
    誤差錨外掛常駐;**批數 ≤3**;五軸面板;修訂留註記。
- **配額（每批 50=2 夾）**：G 18（free 12/oobp 6——free 減半=外推區止損）／L 8／M 5／O 3／I 8／
  D 3／W 3／C 2。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| 嫁接鏈 c3g | selfgen 低 oob 錨×dual×expert | 命中紀錄→公證;三錨 dry=判死 |
| 批 50 | 配額全臂等比縮,free 額外減半 | 三標率 ≥6.0%∧週期 ≤75% |
| CNN 單 rank | --cnn-solo（select-r36 預設） | O 三標 ≥31%（三批合併） |
| asym 判鍵 | b1 判讀彙總 R35 資料 | |ρ|≥0.3 進鍵 |

## 3. 執行紀錄 (Run)
```
# 嫁接鏈（開發機;錨已驗 rad;expert 標配）:
python -m script.dedust chain --name c3g1 --anchor a218_00010 --source-input dedust_auto218_input --anchor-score -0.02 --goal dual --expert
# 批線發車（v54 全訓出爐後;seed 70+N;--rad-key 不帶=退鍵）:
python -m script.sm_invert gen --sm sm_reanchor54.pth --rad-head rad_head54.pth --out-dir tmp/invert_stage_r36bN --n-free 12 --n-surg 0 --n-champ 0 --n-oob 6 --seed <70+N>
python -m script.dedust select-r36 --batch N --sm sm_reanchor54.pth --gstage tmp/invert_stage_r36bN --rad-head rad_head54.pth --novelty
python -m script.dedust check-dup --input dedust_r36bNa_input   # a/b 分開;exit 1 不發車
python -m script.dedust jobs-add --input dedust_r36bNa_input --store dedust_r36bNa --prio 3   # ×2
python -m script.dedust watch --stores dedust_r36bNa,dedust_r36bNb   # 背景掛
```
| 批 | 狀態 |
|---|---|
| — | （開輪;v54 訓練中,出爐即發 b1） |
| c3g1 p01 | ★ 04:4x **嫁接首包命中紀錄候選**（單次）:c3g1p01_24 **wm+0.200∧rad+0.430∧oob 8.970**（<紀錄 9.0;dual score +0.009 首正）——公證 r36n1 ×2 已發（prio 2）;鄰域肥（後四名 wm 0.14-0.21/oob 9.20-9.31）;鏈換錨續爬 p02 |
| n1 公證 | ★★ 05:1x **3/3 bit 級一致→紀錄易主 9.0→8.97**（wm+0.200/rad+0.430/oob8.970 三次全同;records/champions/memory 已換帳;對比圖目檢過;L4 乾涸歸零） |
| 1（r36b1{a,b}） | ✅ 06:37 收（05:26 發車;週期 71 分,零 error。合格 3/50=6%〔壓判準線,三批合併讀〕:l36b1_007 wm+0.16/oob9.53+M 2 筆;影子 CNN 四批連勝〔2.73 vs 4.29/ρ+0.704 vs −0.086〕;G free 100% adv 第四批〔12 席同滅=結構性鐵證〕;⚠ CNN 單 rank 實際 b2 起生效〔select 條件 batch≥2 遺留,O 臂 b1=MLP 舊鍵——三批合併讀改記 b2-b3〕;M 臂 n=5 前瞻略過;v55 輕量重錨中） |
| 3（r36b3{a,b}）末批 | 🔵 11:59 發車（v56 全訓〔凍結 1.289 平穩/shadow56 尺1 2.288〕;--dyn-simcap 0.08〔b2 多樣性警報回應〕;diagb 記錄鍵首批;查重 0×2;無鏈插隊=全速。tier 0 中午收案:c3g2 dry2〔8.99〕第三錨戰略不發;**c4lo 一包定讞=王系內左側 d=1 斜率實測零**〔best 3.69 vs 錨 3.68〕→R37 換系統基石） |
| 2（r36b2{a,b}） | ✅ 09:18 收（07:3x 發車;週期 ~105 分,49/50〔缺 1=g36b2_016 oobp 橋接毒樣本三連敗放棄,誠實記〕。合格 4:**O 臂 CNN 單 rank 首批即中 o36b2_002 wm+0.36∧oob11.63**+K/I 各 1;影子對決**首敗**〔MLP 1.64/ρ+0.767 雙贏,n=7 小樣本追加帳——轉正判準已成立不翻案〕;G free 100% adv ×5;**多樣性警報首響**〔歷史最近鄰 12<15〕→b3 帶 --dyn-simcap 0.08;帕累托 +1;v55 輕量凍結 1.231 逼近 v50 低點〔零代價三連證〕） |

## 4. 分析 (Analyze)
- ★★ **可用帶外紀錄易主 9.0 → 8.97**（2026-07-23 05:1x 公證 R36n1 **3/3 bit 級一致**:wm+0.200/
  rad+0.430/oob 8.970 三次全同）——**c3g1p01_24**,嫁接鏈首包命中（開輪 70 分鐘）。對比圖
  [newking](assets/round-36/newking_usable_oob_897.png)（響應與前任幾乎重合,rad 窗內合規）。
  - **機制驗證**:主假設「反向嫁接」一包過——selfgen 錨 a218_00010（wm+0.13/oob9.05）d=1 一步
    wm +0.07∧oob −0.08 同向改善;高原側三鏈 150px 磨不出的,肥沃側 25px 即中。
  - **血統/表型**:血統 selfgen（非王系親代鏈=第二血脈實體化）;⚠表型 dyn_struct=王朝結構
    （「好錨天然王朝」四度驗證——結構收斂≠血統壟斷,多樣性帳看血統）。
  - R30「9.0=真天花板」定案範圍修正:該窮舉僅覆蓋高側構造法前緣——嫁接=第三路,天花板 8.97 續探
    （鏈已換錨 +0.009 續爬）。

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- **★ 左右側拆帳制上線（07-23 午,Ricky 定軸,decisions 專節）**：8.97 被判同型邊際推進——
  usable_lo/hi 獨立紀錄（基線 +3.68/−5.92）;c4lo 左側壓制鏈首航（錨 c1d6p02_10,goal lo）;
  **R37=左側戰役輪**（lo 軸判別器+SM 配比議題接棒）;usable_oob 總帳型推進降級。
- d=2 跳步（chain --d 支援,另案小改）=嫁接判死時的備援;域專家三顆/軸判別器（R36+ 原排程）;
  校正表 per-臂完整版。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r36b*`;公證 `r36n*`;鏈帳 docs/chains/c3g*.jsonl。
