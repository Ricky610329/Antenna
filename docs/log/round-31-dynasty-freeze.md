# Round 31 — 王系凍結輪：變現機制轉移中繼帶 × std 校準 × 同期 rad 頭首航

- **狀態**: running（2026-07-16 午開輪;宣告制;Ricky 叮嚀「王系列不要再增加了——低側壓不下去
  是王系的問題」直接觸發）
- **提出 / 開跑 / 結論**: 2026-07-16 / 2026-07-16 / —
- **一句話問題**: champ 帶在王鄰域建成的可信半徑（adv 47%、|Δ| 1.08）——把錨全面換成中繼帶家族
  （t03/n09 系,低側活）之後,同樣的「近錨梯度變現」能不能在**低側家族**重現？王系增量歸零後,
  低側資料佔比上升能不能實質改變 SM 的低側導航？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: [round-30](round-30-lowside-beachhead.md)（中繼帶/天花板定案）· decisions「SM 利用率
  雙升級」「SM 準度目標線」· memory feedback-value-axis-oob（王系凍結叮嚀）· kpi.csv

## 1. 假設 (Propose)
- **證據**：①champ 帶王鄰域 adv 47%=可信半徑機制成立;②中繼帶 b2 兩筆命中（lo −4.2∧wm +0.11∧
  oob 7.77）但 b3 複現不穩=需要更密集的火力;③王系 lo 全群 +3.6~+4.7=結構性壓不了低側
  （Ricky 定調）;④v39 起三件套+ensemble+同期 rad_head39（化石問題終結）。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **主判準（bridge 變現）**：champ-bridge 帶（中繼四錨 d≤25 梯度）realized「lo ≤−2 ∧ wm ≥0」
    ≥1 筆/批——中繼帶的 champ 級變現;連兩批 0=中繼帶 d≤25 也不可信,回報+轉新系列（升級規則沿用）。
  - **std 校準（pred_std 首讀,②的進鍵判準）**：全批 std 三分桶的 |pred_wm−real| 中位單調遞增
    → 校準成立→b2 起 std 進鍵（低 std 高分=變現/高 std=探索分流）;不單調=只記錄續觀察。
  - **rad_head39 首航**：M 臂 rad 頭前瞻 ρ 與化石帳（−0.02/+0.365/+0.552 蹺蹺板）對照——
    穩定 ≥0.4 連兩批=同期頭勝化石,rad 進 pred_sel 常駐。
  - **王系凍結審計**：近王+王血統根繼續壓（champ 王錨已移除+dyn-frac 0.4→**0.2**）;
    d_dyn 中位不回落。
  - 紀錄門檻引 records.json;紀錄級一律公證;批數 ≤3;五軸面板每批必報;修訂留註記。
- **配額（每批 150）**：G 64（free 28/champ-**bridge** 24〔中繼四錨〕/oobp 12〔中繼三錨〕;
  surg 併 bridge）／L 20（中繼帶鄰域,r_feed 鍵）／M 14／O 8／I 12／D 16／W 10／C 4／S 2。
- **工具首航疊加**：gen --oversample 3（漏斗）+select 候選池 ×2+pred_std 記錄+rad_head39+
  ensemble sm_ens39_{1,2}——「SM 利用率元年」批。

## 2. 實驗設計 (Design)
| 臂 | 配額 | 生成 | 判準 |
|---|---|---|---|
| G-bridge | 24 | 中繼四錨（y28b1_035/y28b2_010/n27b1_017/f3_011）d≤25 梯度 | lo≤−2∧wm≥0 ≥1/批 |
| G-free | 28 | init 三分+抖動,oversample 3× | 多樣性主力+adv 率追蹤 |
| G-oobp | 12 | 中繼三錨,oob 目標 ≤6 | 低側泵 |
| L | 20 | 中繼六錨鄰域 d1-40,r_feed 鍵 | lo≤−2∧wm≥−2 ≥2/批（續帳） |
| 常規 | 66 | dyn-frac 0.2（王系凍結）+候選池 ×2 | 對照+資訊帳 |
- 判讀：analyze batch（五軸）＋std 三分桶校準表＋bridge 帶 lo×wm 散點（收檔手動）。

## 3. 執行紀錄 (Run)
```
# 發車（開發機,conda ant;每批照 /batch-cycle）:
python -m script.sm_invert gen --sm sm_reanchor<vN>.pth --rad-head rad_head<vN>.pth --out-dir tmp/invert_stage_r31bN --seed <20+N>
python -m script.dedust select-r31 --batch N --sm sm_reanchor<vN>.pth --gstage tmp/invert_stage_r31bN --rad-key --novelty
python -m script.dedust check-dup --input dedust_r31bNa_input   # a..f 分開跑
python -m script.dedust jobs-add --input dedust_r31bNa_input --store dedust_r31bNa --prio 3   # ×6
```
| 批 | 狀態 |
|---|---|
| 1（r31b1{a-f}） | 🔵 13:0x 發車（v39〔凍結基準 1,772 筆建立/近1.12/rad_head39 訓練 ρ+0.32⚠<0.4,真判定=b1 M 臂前瞻/ens39×2〕;**gen 漏斗首航 228→76**（每帶 3× 挑 top）;pred_std 記錄生效;誤差錨 +8;dyn-frac 0.2;錨點 792;查重 0×6。⚠ 誠實註記:gen 漏傳帶配額（bridge 12/surg 24,設計 24/0——half 錨非王系方向未偏,主判準口徑不變,b2 修正） |
| 1 收檔 | ✅ 15:39（149/150;判讀見 §4——★★std 校準大過→進鍵/★rad_head39 +0.684 新高/bridge 0 命中/L 里程碑 wm+0.07∧lo−5.02;重錨 v40〔⚠凍結交集 416/1772=切分位移 bug,已修 ff29454,v41 回滿〕） |
| 2（r31b2{a-f}） | 🔵 16:3x 發車（v40+rad_head40+ens40;**std 進鍵首航**〔變現臂 LCB=pred−std〕;帶配額修正 bridge 24/surg 0;誤差錨 +8;錨點 812;查重 0×6;**bridge 判死線批**〔再 0 命中=轉新系列〕;diag 探針 12 筆同佇列 prio 5） |
| 2 收檔 | ✅ 20:06（150/150 零 error;判讀見 §4 b2——bridge 判死/★L 4 命中大過/★rad 鍵常駐/★detach 彩蛋 +0.53;重錨 v41〔凍結尺假縮水 353=雜湊口徑 bug,63a8939 修,v42 真回滿 1772〕） |
| 3（r31b3{a-f}）末批 | 🔵 21:0x 發車（v41+rad_head41+ens41;**bridge 24 席判死轉移**=G52〔free36/oobp16〕+L**32**〔中繼六錨加倍,主攻同框複現+rad 閘〕;誤差錨 +8;錨點 817;查重 0×6） |

## 4. 分析 (Analyze)
**b1（2026-07-16 15:39 收檔,149/150〔b1c 殘 1 error=bridge 樣本〕）——三判決:std 校準大過+
rad_head39 新高+bridge 首敗但 L 出里程碑**：
1. ★★ **std 校準完美單調**（n=133 三分桶 |pred−real| 中位:低 **0.94**/中 4.00/高 **10.05**——
   差 10 倍,SM 信心度=真）→ **std 進鍵**（b2 起:變現臂 LCB=pred−std 折價;探索臂不動）
   ——多樣性×SM 雙向閉環合攏（「SM 更主導」階段 1 達成）。
2. ★ **rad_head39 前瞻 +0.684（p=.007）史上最高**（化石帳最高 +0.62;訓練 held-out ρ0.32 的
   警訊=考卷變難非頭變差,前瞻實測定案）——制度合訓首戰勝,連兩批判準 1/2。
3. 主判準:bridge 帶 **0 命中**（n=11;梯度拉 lo 有效〔−2.8~−5.7〕但 wm 全崩 −3.6~−6.4——
   中繼錨 d≤25 密度不足以支撐梯度;b2 再 0=判死轉新系列）;**L 臂 1 命中（判準 ≥2 未達但
   出里程碑）**:l31b1_017_lb_y10n09 **wm +0.07∧lo −5.02**=史上最佳同框（超 R30b2 的 −4.21;
   rad −2.38 仍是閘;oob_bad 32.8 高側爆=非可用解,價值=同框證明+SM 教材）。
4. 王系凍結審計 ✓:近王 3%/血統根 9%/d_dyn 中位 261;恆溫 ⚠ 邊緣（38/36,雙升中）;
   前緣 +1;I 臂 4 三標效率王;公證候選 0。

**b2（2026-07-16 20:06 收檔,150/150 零 error）——bridge 判死+L 大過+rad 鍵常駐+detach 彩蛋**：
1. **bridge 判死**（連兩批 0 命中,n=24 足額;best 進步 −3.59→−0.70 但不達標）——中繼錨 d≤25
   梯度變現路線死;席位 b3 轉 L/free;R32 照升級規則轉新系列。
2. ★ **L 臂 4 命中（判準 ≥2 大過;帳 2/0/1/4）**:l31b2_005_lb_n09 **wm +0.15∧lo −4.50**
   （同框紀錄推進,wm 首達 buffer 級;rad −2.6=唯一殘閘）;l31b2_018 lo **−8.10**∧wm −1.02。
   **鄰域變異＞梯度**（L 勝 bridge）——中繼帶的可走性靠翻 bit 不靠反傳。
3. ★ **rad 鍵常駐達成**：rad_head40 前瞻 +0.495（連兩批 ≥0.4:0.684/0.495）——同期頭制度確立。
4. ★ **detach 彩蛋**：o31b2_000_x30d_02_deta **wm+0.53 三標**（本批最高,距王 0.03）——錨=對角
   探針 detach 樣本（進錨點池首批即產出）;對角修法第三重驗證（因果+可製造+變現）。
5. std 進鍵首航:O 臂合格率未顯著變（樣本小,續觀察）;恆溫 ⚠（37/27）;近王 **1%**/血統 7%;
   前緣 +5;公證候選 0（+0.53<0.56）。

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 對角探針 12 筆（dedust_r30diag_input 備妥）——佇列空檔搭車。
- 簇地圖（多樣性×SM 終極整合,R32 級候選）;sm_denovo 對決檢討（閒時）。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r31b*`;公證 `r31n*`。
