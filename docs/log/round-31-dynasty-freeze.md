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
| 1（r31b1{a-f}） | 🔵 準備中（v39 重錨中=三件套+ensemble 首航;gen 待 v39） |

## 4. 分析 (Analyze)
（待收檔）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 對角探針 12 筆（dedust_r30diag_input 備妥）——佇列空檔搭車。
- 簇地圖（多樣性×SM 終極整合,R32 級候選）;sm_denovo 對決檢討（閒時）。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r31b*`;公證 `r31n*`。
