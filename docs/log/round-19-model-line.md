# Round 19 — 模型線第一批：王結構變異資料收集（SM v5 前置）

- **狀態**: running（2026-07-10 兩機發車;a 夾曾被 218 誤跑但零結果落地,無污染）
- **提出 / 開跑 / 結論**: 2026-07-09 / 2026-07-10 / —
- **一句話問題**: 密集的組件級變異真值（王鄰域 800 筆）能否把 SM 的新區域排序救回來（R15 分布外 ρ+0.03 的病）？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: 觸發＝R17/R18 判死低側（decisions 2026-07-09）· `select-r19data` · scratch「模型線接棒」塊

## 1. 假設 (Propose)
- Ricky 方向（2026-07-09）：「大量基於現有王結構做組件級隨機 variation，37/218 各收 400 組不重複，
  為訓練輪作準備」＋「不一樣的 pixel 要有合理的分布」。
- 假設：SM 在新分布上排序崩（R15 教訓）是**覆蓋問題**——王鄰域的算子生成分布一旦有密集真值，
  v5 重錨後排序可用 → GA over 組件-算子空間才有可靠的 fitness。
- **判準（發車前寫死）**：
  - 資料批本身無假設判準（收集批）；品質門檻已在生成端強制＝擾動幅度五帶配額
    （1-3px 15%／4-10px 30%／11-25px 30%／26-60px 18%／61-120px 7%，對錨點 diff_px）、
    零重複（批內＋27 夾歷史）、可製造（全件 ≥4px 零粉塵）、金屬量 323-504。
  - **SM v5 門檻（收檔後）**：vargen held-out 上 wm 排序 ρ ≥ 0.5（p<.01）且 oob 排序顯著
    → GA 發車；不達 → 誠實記錄＝「覆蓋不是瓶頸」，模型線降級討論。
  - **搭載公證（r19a）**：cc_c25_r6s2_r9s3 兩次中位 wm ≥ +0.30 ⇒ margin 王挑戰（vs a024 +0.35）;
    cc_x00_r5s2_r8s3 三標過且 oob < 10 ⇒ 帶外平衡榜首;cc_c25_r6s2_r9s2 rad ≥ +0.56 ⇒ rad 王易主。

## 2. 實驗設計 (Design)
| 夾 | 機器 | 筆數 | 內容 |
|---|---|---|---|
| dedust_r19a_input | 37 | 400 vargen＋6 notarize | 交錯分夾（兩機分布一致）；notarize＝R17 三筆單次紀錄級 ×2 |
| dedust_r19b_input | 218 | 400 vargen | 同分布 |
- 生成：九王錨點加權（a024 .20/c25 .15/x00 .15/c21 .12/a15 .10/i02 .10/g16 .08/c18 .06/g14 .04）
  × 算子鏈 1-3 個（addblock .28/surgery .20/flips .17/rmblock .10/resize .10/wingtrim .08/realloc .07）
  × seed 20260709 全決定性。實際錨點/算子/幅度分布見 manifest（`ops`/`diff_px` 欄）。
- 標籤好壞全譜＝訓練訊號（負樣本一樣有用），非優化批。
- HFSS 預算：每機 400 筆 ≈ 13-15hr（過夜＋半天）。

## 3. 執行紀錄 (Run)
```
# 開發機(已做): python -m script.dedust select-r19data --n 400   → 雙夾 806 筆
#               check-dup 兩夾各 400 筆,重複 0 ✓
# 37 : python -m script.dedust run --input dedust_r19a_input --store dedust_r19a
# 218: python -m script.dedust run --input dedust_r19b_input --store dedust_r19b
```
| 夾 | 機器 | 狀態 | 結果夾 |
|---|---|---|---|
| r19a | 37 | 🔵 running（07-10） | `dataset/dedust_r19a` |
| r19b | 218 | 🔵 running（07-10） | `dataset/dedust_r19b` |

## 4. 分析 (Analyze)
（待收檔；收檔後流程＝sm_reanchor CLEAN_STORES 補 R17/R18/R19 → v5 訓練 → held-out 驗排序 → §4 貼數字）

## 5. 結論 (Conclude)
（待分析）

## 6. 後續決策 (Next)
- v5 過門檻 → GA over 組件-算子空間（R20）；cc 公證結果 → champions.md。

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r19a`、`dataset/dedust_r19b` · ONGOING 動作:（收檔時補）
