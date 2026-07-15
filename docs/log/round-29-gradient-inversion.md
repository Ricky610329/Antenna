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
  - **修訂註記（2026-07-15 17:0x,b2 已發車、結果未回）**：Ricky 拍板**戰略換軸**（詳 decisions
    「戰略換軸」節）——①Round KPI 換五軸面板（SM 準度主 KPI/覆蓋/整體水位/變現/健康）,
    工具已落地（sm_reanchor held-out→docs/kpi.csv;analyze batch 覆蓋+水位/前緣段）;
    ②錨點池王朝抽樣 0.7→**0.4**（--dyn-frac,**b3 起**——b2 已生成不受影響）;
    ③G 臂長駐主力——§1「連兩批不過→降回工具位」的降級判決**由本戰略覆蓋**（Ricky:
    「G 臂要繼續做…整體的想法應該是資料的多樣性,然後慢慢強化 SM 的準度」）,
    存活線改為觀測指標（照報不判死）。
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
| ★ 216 連環故障**定罪** | 14:4x Ricky 本機檢查:**C 槽 0 GB=磁碟滿**——真兇=`_dedust_<store>` 工作目錄無清理機制（78 job 暫存吃光系統碟;0x80070223=HFSS 寫暫存失敗;重開機=pagefile 收縮假好轉）。「216 單機痼疾」推翻=全機隊制度傷（218 首現同錯=也快滿）;治本=run() 跑完自動刪工作目錄＋三台手動清存量;b1 由 37/218 續跑（a-e ✔ 全零 error） |
| 1 收檔 | ✅ 15:34（150/150 全批零 error;判讀見 §4;重錨 v34——**ikpi grad +4.50 史上最高**;維護日:三台清磁碟+新版 worker〔跑完即刪+啟動清掃+probe 探針〕,37/218 探針驗收全綠） |
| 2（r29b2{a-f}）判決批 | 🔵 16:2x 發車（v34;gen {free24,surg16,**champ24 加倍**,oobp12};錨點 722;查重 0×6〔f 單獨補跑,六連跑 10m timeout 老問題〕;--rad-key --novelty;**adversarial training 閉環量測**=同攻擊打 v34,adv 率降幅=SM 補洞速度） |

## 4. 分析 (Analyze)
**b1（2026-07-15 15:34 收檔,150/150 全批零 error）——G 臂首航：SM 被梯度打穿,champ 帶唯一活口**：
| 帶 | n | pred_wm 中位 | real_wm 中位 | |Δwm| 中位 | adv 率(pred≥0→real<−1) | 超 M(−0.41)? |
|---|---|---|---|---|---|---|
| free | 28 | +0.72 | −9.81 | 10.66 | 100% | ✗ |
| surg | 24 | +0.74 | −11.49 | 12.20 | 100% | ✗ |
| champ | 12 | +0.04 | **−5.62** | **5.63** | 90% | ✗ |
| oobp | 12 | +0.56 | −12.27 | 12.62 | 100% | ✗ |
1. **主判準①②雙不過**（三標轉換 0/全帶輸 M 臂中位）——**b2=判決批**（連兩批不過→G 降回工具位）。
2. **adversarial 定性**：|pred−real| 10-12 dB=梯度專鑽 SM 軟肋（terrain 隨機方向 3% 可信≠梯度方向）;
   SM 可信半徑實測=**champ d≤25 半可信（|Δ|5.6）、d≥60 全崩**。
3. ★ **champ 帶活口**：g29b1_062_champ_king pred +0.08→**real +0.43**（低估!）——新王 d≤25 鄰域
   梯度找到真解（rad −1.88 未三標;oob 20.6）;近錨可信量化成立→b2 champ 加倍。
4. **低側泵部分成立**：oobp 帶 realized lo峰中位 **−4.25**（free +0.01/champ −0.12）——wm 崩但
   低側期望真實傳導,資料泵機制驗證過。
5. **反自餵收成**：76 筆=SM 最大誤差點的黃金訓練資料（主動學習極品）→ v34;b2 同攻擊重打
   =adversarial training 循環的首次閉環量測（adv 率變化=SM 補洞速度）。
6. 常規臂：I 29% 三標（i29b1_011 +0.47 近王）;C 33%;無紀錄候選;可用帶外零推進**連 4 批**
   （§1 回報線:R29 兩批仍零→窮舉公證,b2 判）;rad 頭前瞻 +0.341 續鍵（連兩批退鍵解除——帳
   +0.161/+0.405/+0.124/+0.341 蹺蹺板但兩讀 ≥0.3）。

**b2（2026-07-15 20:20 收檔,150/150 全批零 error;三機全天無事故）——adversarial training 閉環
首讀:防禦成功＋G 臂主判準①達成**：
| 帶 | b1→b2 |Δwm| 中位 | b1→b2 pred_wm 中位 | b1→b2 adv 率 |
|---|---|---|---|
| champ | 5.63→**4.00** | +0.04→−0.15 | 90%→**75%** |
| surg | 12.20→9.31 | +0.74→**−1.40**（吹牛樣本 24→**0** 筆） | 100%→n/a |
| oobp | 12.62→10.47 | +0.56→−1.40 | 100%→n/a |
| free | 10.66→10.66 | +0.72→+0.37 | 100%→100% |
1. ★ **G 臂首個 realized 三標**：g29b2_049_champ_exking（wm+0.16/rad+0.20,前任王 d≤25 梯度解）
   ——**主判準①「轉換 ≥1/批」達成**,G 臂自掙存活權（判準②中位<M 仍 ✗;戰略覆蓋下轉觀測）。
2. **v34 學習形態=「遠域學會誠實」非「遠域變準」**：surg/oobp 帶 pred 中位轉負、吹牛樣本歸零
   ——adversarial 漏洞被堵;free 帶 |Δ| 不降=遠域隨機零免疫（76 筆補不了 625 維,局部性確認）。
3. ★★ **又一件 margin 王挑戰**：o29b2_011_o26b2_007_o2 wm **+0.56（單次,三標,oob 12.29）**
   >王 0.50——O 臂 o26b2 系冷支;公證 r29n1a 已發車（20:3x,prio 2）。
4. SM 整體變準旁證：M 臂中位轉正（+0.01）＋前瞻 oob ρ +0.653（p=.011）;wm ρ +0.490;
   rad 頭 +0.247<0.3（b1 +0.341≥0.3=不連續,b3 保留 --rad-key）。
5. 新儀表首讀（KPI②③）：近王 3%/血統 13%/新血 24%;**批內 NN 32<40→恆溫器⑥首觸發**
   （回應=b3 反塌縮 init 已上線,配額不動）;水位 P90 +0.16/作戰區 43/150;**前緣增量 +7**
   （含 2 筆 G-free!）;誤差錨 top=G 樣本 |Δ| 18-30（已入 error_anchors.json,b3 吸收）。
6. 低側泵轉弱（oobp lo 中位 b1 −4.25→b2 −0.17）——v34 悲觀化後梯度不再深入低側自信區,觀察。

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- lo-active 配額 ~50%（decisions 反自餵②）——oobp 帶承載,b1 讀數後調。
- sm_denovo 對決設計檢討（四連敗;閒時）。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r29b*`;公證 `r29n*`;填空池 `r29g*`。
