# Round 10 — 精修 × 物理歸因（三標全過攻堅 + 設計規律 Stage B）

- **狀態**: running（2026-07-06 午後，兩批同時發車）
- **提出 / 開跑 / 結論**: 2026-07-06 / 2026-07-06 / —
- **一句話問題**: s05（差 Gain 0.29＋rad 0.91）/ g24（差 wm 1.85）能否被精修到**三標全過**？pattern 的空間重要度（哪塊承重）真值長怎樣、能否萃取成 generator 可載入的規則？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: 工具 `script/dedust.py`（select-refine1／select-occlude）＋`script/sm_reanchor.py` · 種子與判準源頭 [round-09](round-09-pool-revalidation.md) §5 · 規劃討論 `docs/discuss/scratch.md`「R10 候選：設計規律目錄」塊

> 戰略背景（R6-R9 的合流）：批次假設迴圈的單位 HFSS 效率已勝過我們線上與學長線上；本 round 把三格
> 同時往前推——**產品**（三標全過第一筆）/**能力**（重錨 SM 當導航儀）/**知識**（物理歸因 → 規則 → generator 先驗）。

## 1. 假設 (Propose)

- **承接**: R9 給了種子（s05 −0.29 缺 Gain/rad；g24 rad ✓ 缺 wm）與缺陷定位（s05 的 Gain 低點在 26.5-27 帶緣、rad 缺口在 phi90 θ≈−45~−20 凹陷，見 `assets/round-09/champions_curves.png`）；SM 重錨（Stage A）已完成、導航儀合格。
- **假設**: ① s05/g24 的鄰域（保對稱／小步）含三標更好的點；② 5×5 遮蔽掃描能給出可操作的空間規則（承重區/死區）,且與 SM 歸因對質後可信；③ 重錨 SM 的候選排序在乾淨區絕對值也可用（R9 G 臂只驗了排序）。
- **判準（發車前寫死）**:
  - 產品：任一筆 **wm≥0 且 rad≥0**（三標全過）＝本 round 完勝；退一步 wm> −0.29 或「wm≥−1 且 rad≥0」＝前緣推進。
  - 知識：遮蔽 Δ 圖上「承重塊」（|Δwm|>1）的空間分布是否集中/可命名；X 臂（對稱化救援推廣）預測「變差」的可證偽測試。
  - 能力：phase-2 收檔後,重錨 SM 對 phase-1/2 新點的預測誤差（真 held-out,分布外的分布外）。

## 2. 實驗設計 (Design)

| 批 | 機器 | 內容 | 筆數 | 狀態 |
|---|---|---|---|---|
| **O 遮蔽掃描** | 218 | s05/g24 各 24 個 5×5 區塊逐一清空（手術式,feed 保留） | 48 ≈2.4hr | 🔵 running |
| **ref1 精修盲階段** | 37 | W s05 保對稱鄰域 k{2,4,8}×6 ／ X 對稱化救援推廣（g15/g24/e73/e39,預測=變差）／ Y g24 小步鄰域 | 18+4+18=40 ≈2hr | 🔵 running |
| **Stage A SM 重錨** | 開發機 | 266 筆乾淨真值×8＋harvest 重放 2000 | 零 HFSS | ✅ held-out 3.20→**1.41**（合格） |
| **ref2 精修知情階段** | 37（今晚過夜） | 遮蔽圖定向編輯＋重錨 SM 從大候選池挑 top-K | 待生（收 O+ref1 後） | 🔜 |

- 全部決定性；margin/rad 同一把尺；噪聲地板 ≈0（R9 附錄公證）→ 每個 Δ 都是真效果。

## 3. 執行紀錄 (Run)

```
# 218:  python -m script.dedust run --input dedust_occl_input --store dedust_occl
# 37:   python -m script.dedust run --input dedust_ref1_input --store dedust_ref1
# 開發機: python -m script.sm_reanchor train && python -m script.sm_reanchor eval   # ✅ 已完成
```
- 事件: 2026-07-06 午後兩批發車（監看掛開發機,停滯 40 分警報）。sm_reanchor 修了兩個初始化缺口
  （`setDefaultCoordinate`／`AntennaResponse.use(spec)`——腳本獨立跑訓練管線時的必要開場,已記進腳本）。

## 4. 分析 (Analyze)

（收檔後：遮蔽 Δ 熱圖 ×2、ref1 三臂表、X 臂預測對答案；ref2 判讀另補）

## 5. 結論 (Conclude)

- 待。

## 6. 後續決策 (Next)

- ref2 收檔 → 若三標全過：進入「可製造冠軍」驗證（重複公證＋學長樹外部檢查）＋規則寫入 generator（R11 工程）；
  若未過：遮蔽圖＋兩階段殘差 → 規則清單迭代,DIP 生成式權重上升。

## 7. 歸檔指向 (Archive)

- configs/README 列: 無（零新 config；dedust.py 擴充 select-refine1/select-occlude、新 script/sm_reanchor.py）
- 結果夾: `DATASET_PATH/dedust_occl/`、`dedust_ref1/`（NAS）；SM 權重 `sm_reanchor.pth`
- memory: [[project_sm_training_redesign]]（重錨=週期 harvest 重錨落地）
- ONGOING 動作: 已加 🔵；收檔移 ✅＋README 索引
