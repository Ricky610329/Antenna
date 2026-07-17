# Round 33 — 反王朝結構輪：表型黑名單 × rad 閘攻堅 × CNN 混合鍵 × B 泵續投

- **狀態**: running（2026-07-17 晚開輪;Ricky「輪次持續開不要停」+「只要避免王系列的底下一大塊、
  上面兩個中等大小的,其他都值得嘗試（不考慮小碎塊）」;跨週末輪——NAS 07-18 17:00 關機/07-19 停電,
  b1 今晚發車,餘批週一恢復後續）
- **提出 / 開跑 / 結論**: 2026-07-17 / 2026-07-17 / —
- **一句話問題**: 全史八成困王朝結構（表型驗證 2026-07-17）——把「底1大+上2中」黑名單化
  （生成端+select 軟過濾,錨定臂豁免）之後,結構多樣性/PCA 佔據能不能實質改變？同時:
  rad 閘攻堅（同框系 rad 全負六批,錨換「lo 壓∧rad 半好」交集帶）+CNN 混合鍵記錄+B 泵續投。
- **一句話結論 (TL;DR)**: 待跑
- **指向**: [round-32](round-32-strait-crossover.md)（表型驗證/影子三讀/B 泵首航/同框 wm 首過 0）·
  decisions「王朝重定義」「影子 CNN 對決」· rad 地形圖（半島中上段觀察）· ONGOING 週末排程

## 1. 假設 (Propose)
- **證據**：①表型可分性驗證:王朝家族結構判 100%/功能判 lo>0 96%;「其他」族群 81% 也是王朝結構
  =血統凍結擋不住結構收斂;②rad 地形圖:rad 過標=平原軸（黑圈遍布）,同框系困境=地理性
  （中繼尖端=rad 紅區,半島中上段有綠點+rad 0 圈）;③交集帶掃描:lo≤−2∧rad≥−1 全史僅 1 筆
  （g29b1_031)=錨組核心;④影子 ρ 三連勝=CNN 排序器;⑤B 泵 b3:血統 B>A+SM 高估 ~1dB。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **主判準（反王朝結構）**：批內王朝結構佔比（select 印「結構判」行+收檔 realized 口徑）
    顯著低於全史基線 81%——目標 **<40%（減半）**;恆溫雙口徑不惡化（批內 NN≥30/歷史 NN≥15）。
    連兩批 <40% ∧ 探索臂產出質量不塌（D/W/free 三標率不歸零）=過濾常駐;佔比降不下來=過濾強度
    （罰 +2.0）上調再試。
  - **rad 閘（L-RADGATE 六錨）**：realized「同框（wm≥−2∧lo≤−2）∧ **rad≥−1**」≥1 筆/批
    ——六批 rad 全負的突破線;連兩批 0=交集帶不可鄰域填充,回報+換法（往半島中上段定向變異）。
  - **B 泵續投**：同 R32b3 判準（wm≥0.15∧rad≥0∧oob_bad<9.0 ≥1=紀錄候選）;審計=pred−real
    偏差中位 vs b3 的 ~1dB（v45 吃進 b3 24 筆教材後應收斂）。
  - **CNN 混合鍵（記錄版,照 std 進鍵先例）**：pred_wm_cnn 全批記 manifest;判讀審計=
    「若用 CNN 排序會選到誰」假設檢定（CNN rank vs MLP rank 的 realized ρ 對比）——CNN 顯著優
    → b2 起排序進鍵。
  - 影子對決四讀續帳（轉正判準不變）;rad 頭復鍵帳 0/2 續記;紀錄門檻引 records.json;
    紀錄級一律公證;批數 ≤3（跨週末）;五軸面板;修訂留註記。
- **配額（每批 150）**：G 60（free 24/oobp 12/**B 泵 24**=selfgen 三錨集中）／**L 24**（RADGATE 六錨）
  ／M 14／O 8／**I 16**（+4,連三批穩定產線）／**D 16**（+4,反王朝自由帶）／W 8／C 4。

## 2. 實驗設計 (Design)
| 臂 | 配額 | 生成 | 判準 |
|---|---|---|---|
| 表型過濾 | 全域 | gen free init+產出雙過濾;select core/cold/D/W 罰 +2.0（錨定臂豁免） | 批內佔比 <40% |
| L-RADGATE | 24 | 六錨（g29b1_031/b28b3_004/x30d_10/l31b3_003/f2_015/f2_029）鄰域 d1-40,r_feed 鍵 | 同框∧rad≥−1 ≥1/批 |
| G-B泵 | 24 | --champ-anchors selfgen 三錨（bp_a37a/bp_a37b/bp_a216） | 同 R32b3+高估審計 |
| CNN 鍵 | 記錄 | select 全候選 pred_wm_cnn（sm_shadow45） | 排序假設檢定 |
- 工具:select-r33（dyn_struct 軟過濾+RADGATE+CNN 記錄,2026-07-17 實作）;gen free 雙過濾。

## 3. 執行紀錄 (Run)
```
# 發車（開發機,conda ant;每批照 /batch-cycle）:
python -m script.sm_invert gen --sm sm_reanchor<vN>.pth --rad-head rad_head<vN>.pth --out-dir tmp/invert_stage_r33bN --n-free 24 --n-surg 0 --n-champ 24 --n-oob 12 --seed <40+N> --champ-anchors "a37_00279:bp_a37a,a37_00281:bp_a37b,a216_00006:bp_a216"
python -m script.dedust select-r33 --batch N --sm sm_reanchor<vN>.pth --gstage tmp/invert_stage_r33bN --rad-head rad_head<vN>.pth --novelty
python -m script.dedust check-dup --input dedust_r33bNa_input   # a..f 分開跑
python -m script.dedust jobs-add --input dedust_r33bNa_input --store dedust_r33bNa --prio 3   # ×6
```
| 批 | 狀態 |
|---|---|
| 1（r33b1{a-f}） | 🔵 2026-07-17 20:34 發車（v45 三件套〔MLP 凍結 1.26/影子凍結 2.65,遠帶 CNN 連四版反超 2.48<2.88〕;**表型過濾首航**〔gen free 雙過濾;select 候選池王朝表型 55%<基線 81%,罰 +2.0〕;**CNN 記錄鍵首航**〔sm_shadow45→pred_wm_cnn〕;L-RADGATE 六錨/B 泵三錨;誤差錨 +8;錨點 847;查重 0×6） |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 週末斷點:b1 今晚收;若明午前判讀完可趕 b2（17:00 NAS 關機前收）;餘批週一。
- selfgen 端結構過濾（dedust worker）=正式機下次 pull 生效（週一重啟時自動帶上）。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r33b*`;公證 `r33n*`。
