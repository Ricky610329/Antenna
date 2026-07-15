# Round 29 — G 臂主力輪：SM 梯度反傳生成 × 四帶 trust-region × 承重圖知情約束

- **狀態**: running（2026-07-15 開輪;Ricky 2026-07-14 拍板「R28 收完就照這樣開 G 臂,這部份多跑一點。
  其他的啟發式或者是探索的方案 可以再降低一點比例。」;宣告制）
- **提出 / 開跑 / 結論**: 2026-07-14 / 2026-07-15 / —
- **一句話問題**: 把 pattern 當變數、凍結 SM+rad 頭做梯度反傳（sm_invert）——SM「認為」能三達標的
  pattern,HFSS 量回來有多少是真的？哪個距離帶開始 SM 過度自信（adversarial 率）？梯度生成能不能
  同時解 R28 判死的「三軸耦合」（多目標 loss=wm+rad+oob 一起壓）？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: [round-28](round-28-inblock-rad-surgery.md)（承重圖=知情約束來源;塊內手術判死→G 臂接棒）·
  工具=`script/sm_invert.py`（gen 四帶）＋`select-r29 --gstage` · decisions「反自餵雙軸」·
  terrain 定案（SM 局部可信）

## 1. 假設 (Propose)
- **證據**：①sm_invert 首跑（R28 前置）=近帶誠實/遠帶自信膨脹,half d60 pred wm+0.81/rad+2.94=
  頂級候選;②R28 承重圖=命脈/自由塊知情約束（凍命脈→梯度只動自由區）;③R28 判死「塊內手術修 rad」
  →多目標梯度=唯一活路;④Ricky 確認機制:「用期望響應微調 pattern,隨機 init 也可以→多樣性更多」。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **主判準（G 臂存活線）**：G 臂整體「pred 達標→realized 三標」轉換 ≥1 筆/批 或 各帶 realized
    wm 中位 > M 臂中位——兩者全不過連兩批 → G 臂降回工具位（gallery 用）,帶帳本回報。
  - **帶別 adversarial 率**（本輪核心讀數,每批記帳）：各帶 |pred_wm − realized_wm| 中位＋
    「pred 達標而 realized 崩」比例——畫出 SM 可信半徑的**實測地圖**（champ 25 ≺ surg/oobp 60 ≺ free 自由）。
  - **oobp 帶=低側資料泵**：realized 低側對比 >2dB 比例 vs 全語料基線 1.2%（反自餵②的資料面驗證）。
  - **紀錄門檻**：引 `docs/records.json`（wm 0.50/inband 0.61/usable_oob 9.0）;紀錄級一律公證。
  - **rad 頭**：b3 +0.124<0.3——R29b1 保留 --rad-key,**再 <0.3=連兩批→b2 退鍵**。
  - **可用帶外**：R28 零推進連 3 批——R29=分布級介入;**R29 兩批仍零推進→窮舉公證證天花板**（升級回報）。
  - **批數 ≤3**（硬上限）;D 12 續 min sel 帳;I 14 續 ikpi 帳;判準修訂只能在結果回來前＋留註記。
- **配額（每批 150,Ricky 拍板結構）**：**G 76**／M 14（前瞻母體）／O 12／I 14／D 12／W 10／C 6／S 6／F 0。
  G 內部 mix：free 28（random-init 自由帶=多樣性主力）/surg 24（half 手術帶 d≤60+命脈凍結）/
  champ 12（冠軍中帶 d≤25）/oobp 12（超規格帶外期望 oob≤6,w_oob×5=低側資料泵）。

## 2. 實驗設計 (Design)
| 臂 | 配額 | 生成 | 判準 |
|---|---|---|---|
| G-free | 28 | random init,無距離罰 | 多樣性主力;adversarial 率上界（SM 最自信=最危險區） |
| G-surg | 24 | t07h/p00h 錨 d≤60,**凍命脈塊**（t07h {4,7}/p00h {3,6},R28 §4 真值） | rad 多目標=接棒 R28;pred rad +2 內誠實（次可加折扣） |
| G-champ | 12 | s28b3_005/m23b4_030 錨 d≤25 | 近帶=SM 最可信;紀錄鄰域推進 |
| G-oobp | 12 | uoob/half 混錨 d≤60,oob 目標 ≤6 | 低側資料泵;帶外期望超規格 |
| M/O/I/D/W/C/S | 74 | 沿 r22mix（root-cap 0.6/simcap 0.12/novelty） | 對照母體+梯子房租+資訊帳 |
- 工具鏈：`sm_invert gen`（staging,jitter 0.8 多樣化,--seed 批次遞增）→ `select-r29 --gstage`
  （hist 查重+SM 統一打分+編 id `g29bN_*`）→ check-dup → jobs-add。
- 判讀：analyze batch＋**帶別 adversarial 表**（pred vs realized 按 band 拆,收檔手動）＋ikpi。

## 3. 執行紀錄 (Run)
```
# 發車（開發機,conda ant;每批照 /batch-cycle）:
python -m script.sm_invert gen --sm sm_reanchor<vN>.pth --out-dir tmp/invert_stage_r29bN --seed <N>
python -m script.dedust select-r29 --batch N --sm sm_reanchor<vN>.pth --gstage tmp/invert_stage_r29bN --rad-key --novelty
python -m script.dedust check-dup --input dedust_r29bNa_input   # a..f,exit 1=停
python -m script.dedust jobs-add --input dedust_r29bNa_input --store dedust_r29bNa --prio 3   # ×6
# 收檔判讀: analyze batch --round 29 --batch N ＋ 帶別 adversarial 表 ＋ analyze ikpi --round 29 --batch N --pre v<發車> --post v<重錨>
```
| 批 | 狀態 |
|---|---|
| 1（r29b1{a-f}） | 🔵 11:2x 發車（v33;gen 76 筆足額 {free28,surg24,champ12,oobp12};錨點 714;查重 0×6;--rad-key --novelty;三機體制——216 復役審查過〔r28g1c 23ok/1err=4%〕） |

## 4. 分析 (Analyze)
（待收檔）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- lo-active 配額 ~50%（decisions 反自餵②）——oobp 帶承載,b1 讀數後調。
- sm_denovo 對決設計檢討（四連敗;閒時）。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r29b*`;公證 `r29n*`;填空池 `r29g*`。
