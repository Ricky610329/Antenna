# Round 7 — 除塵驗證（de-dust）：達標 pattern 的粉塵是不是 load-bearing？

- **狀態**: proposed（工具/選集/預篩備妥，待 .37 機器停妥即發）
- **提出 / 開跑 / 結論**: 2026-07-03 / — / —
- **一句話問題**: harvest 池達標 pattern 拔掉 1-3px 粉塵後，S11/Gain margin 撐不撐得住？順帶：它們的 radiation ±45° 覆蓋過不過？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: 工具 `script/dedust.py` · 輸入 `DATASET_PATH/dedust_r7_input/`（NAS）· 結果 `DATASET_PATH/dedust_r7/` · 討論源頭 `docs/discuss/scratch.md`「戰略討論」塊（2026-07-03）· decisions「可製造性定義」· memory [[project_benchmark_vs_random]]

> 本檔只放**連結指向**，不複製內容。本 round 燒 HFSS（15 筆）＝佔 round 編號；上游離線分析（margin×結構交叉、碎片尺寸分布）見 scratch 塊，工具化部分已併入 `script/dedust.py select`。

## 1. 假設 (Propose)

- **問題 / 假設**: lab 真目標＝S11＋Gain＋rad ±45° 覆蓋＋**可製造**（裁切製程：允許不連通、不要很多 1×1）四約束。離線交叉分析實錘：池內 18 筆達標 pattern 全是「3-8 塊大銅片＋10-18 顆 1-3px 粉塵」（Hamming≤100 下實為 **2 個家族**）、整池零筆「無 1px 碎片」達標——乾淨子空間學長沒踩過。**假設：粉塵不是 load-bearing**（1-3px 孤島電尺寸小、耦合弱；analysis-01 說細碎度是雙邊最強負因子）→ 拔掉後 margin 大致撐住 → 可製造達標 pattern 直接到手。
- **為何現在做**: R6 說「達標 pattern 已在池內」但那 18 筆不可製造——這是「歷史資料直接用」路線的最後一哩；且每次 solve 順帶方向圖＝rad ±45° 首次驗證＋Stage-3 `harvest_single_rad` 冷啟動資料（一魚三吃）。R5 D 臂提早收（墊底 -7.66@34、D-only 隔離問題 R3/R4 已答過）讓出 .37。
- **預期結果與判準**:
  - (a) 存在「d1/d3 版 wm≥0 且 rad_margin≥0」→ **lab 終極目標到手**（零搜尋、純資料分析）；
  - (b) 除塵版掉 ≤~1dB → 粉塵貢獻小 → 從除塵版 warm-start 局部精修（analysis-01：局部半徑數十翻轉可導、配方已知）；
  - (c) 掉很多（>2dB）→ 粉塵在這些設計上是共振的一部分 → 「乾淨可製造」需要在乾淨子空間內重新搜尋（DIP sigmoid 天生住那裡）。
  - 副判準：原版重驗 vs 池 margin（舊 HFSS response 重算）之差＝**R6 oracle 的真偽檢驗**；rad ±45°/floor 3dB 覆蓋首次有真數字。
- **依據**: [round-06](round-06-offline-expected-best.md)（分布≫策略、oracle +0.38）· [analysis-01](analysis-01-pattern-anatomy.md)（結構配方/細碎負因子）· scratch 2026-07-03 戰略討論塊（碎片尺寸分布數字）· decisions「可製造性定義」（2026-07-03）

## 2. 實驗設計 (Design)

非訓練 round：15 筆固定 pattern 各跑一次 HFSS（`SinglePortRadSimulator`，同 solve 收 S11/Gain＋phi0/phi90 方向圖）。

| 組 | 內容 | 筆數 |
|---|---|---|
| 家族代表 | F0（池 +0.38）/ F1（池 +0.07）＝達標 18 筆的 2 個家族 best | 2×(orig+d1+d3)=6 |
| 近標補充 | near1 -0.01 / near2 -0.14 / near3 -0.27（與家族 Hamming>100、主件 173-243px 更整塊） | 3×(orig+d1+d3)=9 |

- 變體：`d1`＝拔 <2px（只拔 1px 粉塵）、`d3`＝拔 <4px（1-3px 全拔，**全碎片≥4px＝可製造形**）；feed 組永不拔。
- **判準尺**: margin＝`antenna.losses.worst_margin`＋現行 targets（與 R6/analysis-01 同尺）；rad＝`rad_window_margin`（±45° 窗內 min(gain)−(G0−3dB)，正=過）。
- **HFSS 預算**: 15 筆（估 1.5-4 hr，機器 .37）。**可中斷續跑**（results.json 增量落盤、跑過的跳過）。
- SM 預篩（`sm_harvest.pth`，零 HFSS）已跑：預測除塵掉 0.2-1.6dB（Gain 側）——但 SM 對這批絕對偏差 ~1.3dB（p00_orig 池真值 +0.38 被預測 -0.94），只當方向訊號。

## 3. 執行紀錄 (Run)

```
# 正式機 .37（repo 根目錄；conda env 有 HFSS COM 的那套）
python -m script.dedust run            # 預設 config=configs/single_r5_explore.yaml、HFSS 工作目錄 _dedust_r7
# 之後任一機
python -m script.dedust report         # 匯總表（貼 §4）
```
- 輸入（已備妥 2026-07-03）：`DATASET_PATH/dedust_r7_input/`（15 個 .pt＋manifest.json 含池 margin/碎片統計/SM 預篩）。
- 結果：`DATASET_PATH/dedust_r7/`（SampleStore＝(pattern,真響應)＋`rad/*.pt` 方向圖原始資料＋results.json）。
- 事件：—

## 4. 分析 (Analyze)

（跑完把 `python -m script.dedust report` 的表貼這裡）

- 待答清單：① d1/d3 vs orig 的 Δwm（主假設）② orig 重驗 vs 池 margin（oracle 真偽）③ rad_margin 分布（±45° 這關到底嚴不嚴）④ SM 預篩 vs HFSS（SM 對乾淨區的校準）。

## 5. 結論 (Conclude)

- **學到什麼**: 待
- **決策**: 待（依 §1 判準 (a)/(b)/(c) 分岔）
- **促成 / 排除哪個候選**: 待（(b)/(c) → 「除塵版 warm-start 局部精修」round；rad 資料 → Stage-3 pretrain 解鎖）

## 6. 後續決策 (Next)

- (a) 成 → 收尾：對外交付 pattern＋轉「池頂端 warm-start」常規化；(b)/(c) → 開 warm-start 精修 round（候選生成端加「無粉塵修復」＝ `strip_small` 進管線）。
- rad 資料無論如何入袋 → `harvest_single_rad` 合併、rad head pretrain（Stage 3）解鎖。

## 7. 歸檔指向 (Archive)

- configs/README 列: 無（零新 config；工具腳本 `script/dedust.py`，測試 `tests/test_dedust.py`）
- 結果夾: `DATASET_PATH/dedust_r7/`（NAS）
- memory: [[project_benchmark_vs_random]] · [[project_radiation_pattern]]（Stage-3 資料來源）
- 設計文件: decisions「可製造性定義」（2026-07-03）
- ONGOING 動作: 發車時 🔵 標 running、收檔移 ✅
