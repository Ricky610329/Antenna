# Round 51 — 橋接與進鍵輪:正負中間帶產線 × lo 進鍵首航 × 尺擴容

- **狀態**: running（2026-08-02 12:0x 開輪;自主續輪宣告制;R50 收輪接棒）
- **提出 / 開跑 / 結論**: 2026-08-02 / 2026-08-02 / —
- **一句話問題**: 橋接資料（正負中間帶,Ricky 08-01 提案）能不能讓 two 的冷啟動曲線變單調/加速——並補辦 lo 進鍵首航與儀器修繕。
- **指向**: [round-50](round-50-morphology.md)（儀器元年/曲線非單調）· decisions「型態體系軸/雙頭制」·
  scratch「橋接臂候選/池粒度教訓/z50b1_017」

## 1. 假設 (Propose)
- **判準（發車前寫死;Ricky 可隨時否決）**：
  1. **橋接產線**:2.5k=25 池×**100 席**（粒度教訓落實;三式 bri_dil/ero/mix,母本=r48/r49 正片輸入夾;
     prio 4 入 neg_stores=two 課程教材）;KPI=two OOD 尺走勢（橋接入鍋前後對比);三標免疫照 decisions。
  2. **批線 ≤3 批**（select-r51=r50 配置:正 30/負 20 分層 6 臂/學長 10 消耗制;seed 20260809;v99+ 配套顯式當版）。
  3. **凍結尺擴容 15→30**:從已測負片池（b9-b17）抽 15 筆補入 `_frozen`（同代多池、id 序偶數位口徑;
     搬檔+除帳同 b1 協議）;判準=擴容後曲線批間波動 <±0.5=「尺噪音」假說證實。
  4. **lo 進鍵首航（四度不順延,主動觸發）**:b1 前跑補池 `select-r21harvest --tag r51g1 --lo 6 --o 0 --wild 0 --shards 3`;
     存活判準沿 R49 §1④（兩批內 L 臂 ≥1 筆 lo≤−1,全空退鍵）。
  5. **錨銀行只記不開鏈**:學長近標 6 個體（F18644/F21881/F24038/F6161/F5477/F17279）+rad 天賦 3
     （F6161/F15032/F9609）列 R52 擇錨名單;本輪鏈位留白。
  6. 紀錄級一律公證;3 批必收輪;KPI② 分池補修=本輪工程債。
- **修訂紀律**:發車後修訂只能在對應結果回來前+日期註記。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| 橋接 2.5k | 25×100 小池 prio 4 | two OOD 曲線走勢對比 |
| 批線 | =r50 配置 ×3 批 | 五軸;I 臂五連觀察 |
| 尺擴容 | 15→30 | 批間波動 <±0.5=噪音假說證實 |
| lo 首航 | 補池 --lo 6 | 兩批 ≥1 筆 lo≤−1 |

## 3. 執行紀錄 (Run)
```
# v99(b3 正片+學長入主鍋;b3 負片→neg_stores):
python -m script.sm_reanchor train --add "dedust_r50b3a,dedust_r50b3c" --epochs 30 --out sm_reanchor99.pth
python -m script.sm_reanchor train-two --epochs 30 --out sm_reanchor99.pth
# lo 進鍵首航: python -m script.dedust select-r21harvest --tag r51g1 --lo 6 --o 0 --wild 0 --shards 3
# b1: staging seed 511 → select-r51 --batch 1 → select-neg --round 51 --batch 1 --n 20 --stratify --arms <6 臂>
#     → select-senior --round 51 --batch 1 → check-dup ×3 → prio 3 → watch
# 橋接產線: select-bridge --round 51 --batch 30~54 --n 100(check-dup+prio 4+neg_stores,逐池)
```
| 批/包 | 狀態 |
|---|---|
| 尺擴容 | ✅(08-02 12:2x,§1③):b9-b17 均勻抽 15 搬 `_frozen`(名單續記 FROZEN_LIST.txt);**凍結尺 n=30**;v99 起讀數=新尺口徑(與舊 15 尺讀數不可直接比,曲線圖分段標註) |
| 發車 | v99 鏈跑動中(train→two→lo 首航→b1 三夾→橋接 25 池→watch) |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- R52:錨銀行擇錨攻堅(§1⑤ 名單);雙頭制門檻檢查(負片 5k);凍結尺擴容驗證後續。

## 7. 歸檔指向 (Archive)
- 結果夾 `dedust_r51b*`;橋接池 `dedust_r51b3xb`;kpi_ood.csv 續帳。
