# Round 38 — 影子二號輪：臂A 制度內對決 × lo 判別器 × tri 三鏈

- **狀態**: running（2026-07-23 深夜開輪;自主續輪宣告制;Ricky「都排,有空都跑跑看」授權架構線）
- **提出 / 開跑 / 結論**: 2026-07-23 / 2026-07-23 / —
- **一句話問題**: analysis-06 的架構翻案（臂A 三尺全勝）能不能通過**制度內盲測**（批線前瞻,
  非離線 held-out）接掌回歸主鍵？同時:lo 判別器上崗（記錄鍵）、tri 三鏈續攻左側合格解、
  selfgen 換種生效。
- **指向**: [round-37](round-37-left-colony.md)（殖民開工三選三/tri 前緣 −0.31）·
  [analysis-06](analysis-06-arch-bakeoff.md)（架構翻案）· decisions「Tier 再平衡」（三連 ≥2×）

## 1. 假設 (Propose)
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **影子二號（主軸）**：臂A 架構（ResBlock CNN+鏡射增強）進 sm_reanchor 制度段訓 `sm_two<N>`
    ——**b1 跑批間實作,b2 起 analyze batch 三模盲測**（MLP/影子 CNN/二號）;
    判準=**連兩批三尺（誤差中位/前瞻 ρ/adv 率）全勝 MLP → 接回歸主鍵**（照影子 CNN 轉正先例;
    凍結尺併讀）;輸=降級 ens 成員。
  - **lo 判別器**：臂B 架構同段訓 lo 頭 → select 記錄鍵 `pred_lo`（照 asym/std 先例記錄版,
    R39 判進鍵;判準=批前瞻 lo ρ≥0.5 連兩批）。
  - **tri 三鏈**（tiers 三連 ≥2×→鏈位 2→3;批 50 維持觀察）:c5tri2（−0.40）/c6tri4（−0.50）/
    **c7tri**（備池④ c2radp12_12,−0.55）;里程碑=左側合格解（wm≥0.15∧rad≥0∧lo≤−2,公證+推播）;
    usable_lo 0.5/格;dry 2 收鏈 sideways/換備池慣例續。
  - **selfgen 換種生效**（R37 欠帳）:機器 pull+worker 重啟後,追蹤 selfgen 新增樣本王朝結構佔比
    （目標 <40%;基線 ~80%）。
  - v60=全訓+response 模式（轉正後首個全訓版;偶數版含 ens/shadow）;批 50×3;多樣性恆溫
    （歷史 NN<10 → D/W+2 或 simcap 降）;殖民 KPI 口徑續;批 ≤3;公證鐵則;修訂註記。
- **配額（批 50=2 夾;同 R37 含 D/W+2 常態化）**：G 12/L 12（新大陸半 ref 半 rej 常駐）/I 8/M 5/
  O 3/K 2/D 6/W 6=54。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| 影子二號 | 制度段訓 sm_two+b2 起三模盲測 | 連兩批三尺全勝 MLP→接主鍵 |
| lo 判別器 | 同段訓+select 記 pred_lo | 前瞻 lo ρ≥0.5 連兩批→R39 進鍵 |
| tri 三鏈 | c5tri2/c6tri4/c7tri | 左側合格解=里程碑 |
| selfgen 換種 | 機器 pull 後生效 | 新樣本王朝佔比 <40% |

## 3. 執行紀錄 (Run)
```
# v60 全訓（response 模式;偶數版全配件）:
python -m script.sm_reanchor train --add "dedust_r37b3a,dedust_r37b3b" --out sm_reanchor60.pth --ds-mode response
# 第三鏈:
python -m script.dedust chain --name c7tri --anchor c2radp12_12 --source-input dedust_c2rad_p12_input --anchor-score -0.55 --goal tri
# 批線（v60 出爐後;seed 90+N;select-r37 參數沿用=round 號用 --round?（select-r37 寫死 37——R38 沿用
#   select-r37 parser+手動 round 覆蓋不可行,R38 select 用 select-r37 --batch N（id 前綴仍 37?
#   ——不行,round 號要 38:實作極小 select-r38 parser（clone,round=38,D/W 預設 6））:
python -m script.dedust select-r38 --batch N --sm sm_reanchor60.pth --gstage tmp/invert_stage_r38bN --rad-head rad_head60.pth --novelty
```
| 批 | 狀態 |
|---|---|
| — | 開輪紀事:v60 全訓落地（response 全訓首版）;**selfgen 換種生效**（07-24 02:1x Ricky 三台 pull+重啟——tier2 改產左側家族資料,王朝種子留 20%）;train-two（sm_two60+lo 判別器）補訓中;tri 三鏈過夜爬（c5tri3 wm+0.11 錨/c6tri4/c7tri 換錨） |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- GNN 第三視角（凍結候選）;獨立艙（凍結）;d=2 備援;批 50→25 降格判定（R38 收輪讀 tiers+多樣性）。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r38b*`;公證 `r38n*`;鏈帳 docs/chains/c5tri2/c6tri4/c7tri.jsonl。
