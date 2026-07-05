# Round 9 — 池頂端重驗＋乾淨前緣探索（過夜批次，162 筆）

- **狀態**: proposed（輸入 162 筆已備妥 NAS，37 可發車；使用者 ~10hr 無人值守窗）
- **提出 / 開跑 / 結論**: 2026-07-05 / — / —
- **一句話問題**: 池內帳面達標的 pattern 在現行 HFSS 設定下還活著幾筆？「池值→現行值」校正曲線長怎樣？＋跨家族乾淨投影鄰域裡有沒有更好的可製造 pattern？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: 工具 `script/dedust.py`（select-r9/run/report，`--input dedust_r9_input --store dedust_r9`）· 輸入 `DATASET_PATH/dedust_r9_input/`（162 筆＋manifest 含 SM 預篩與 anchor_pool_wm）· 前作 [round-08](round-08-clean-mapping.md)／[round-08-report](round-08-report.md)（漂移警訊出處）· R6 oracle [round-06](round-06-offline-expected-best.md)

> 本檔只放**連結指向**。R9 是 R8「意外收穫」的直接後續：池值系統性樂觀 → R6 的 oracle（+0.38 達標）與整個 24k 池資料的可信度都要重刻。

## 1. 假設 (Propose)

- **承接**: R8 A 臂實錘「池記錄值 → 現行 HFSS 重跑」14/15 向下、中位 −0.52、最大 −1.80；配噪聲地板 0.00（b00_ref ≡ R7 p03_d3）→ 漂移是**系統性設定差**（學長當年 vs 現行），不是隨機噪聲。
- **假設**: ① 池內 18 筆帳面達標（wm≥0）打完折後**大部分不再達標**（悲觀估全滅）；② 漂移量可用「池值→現行值」的單調校正曲線描述（讓 24k 池資料折價後續用）；③ R7 曾重驗 p00 得 +0.44 達標——若本輪 t00（同為池頂）也活，表示池頂不是全滅、精修仍有種子。
- **預期結果與判準**（發車前寫死）:
  - **主判準**: T 臂 18 筆中現行設定 wm≥0 的存活數 `k`。`k=0` → R6 oracle 修正為「未知」，精修天花板下修、DIP 生成式權重上升；`k≥1` → 現行設定已知解存在，精修 round 種子確定。
  - **副判準**: 30 筆 (池值, 現行值) 散點的校正關係——線性/單調偏移？帶寬多少？供 24k 池資料折價使用。
  - **附帶**: 30 條 rad 曲線入袋（Stage-3 累計 ~142）；T 臂 pattern 全是碎片雲（n_comp 21-35、粉塵 5-16 顆）——順看達標與碎片度在現行設定下的關係。
- **依據**: [round-08-report](round-08-report.md)「意外收穫」節 · [round-07](round-07-dedust.md)（p00 重驗 +0.44 的單點先例）· [round-06](round-06-offline-expected-best.md) 侷限⑤

## 2. 實驗設計 (Design)

**重驗組**（R8 池值漂移警訊的直接後續）：

| 臂 | 內容 | 筆數 | 買什麼 |
|---|---|---|---|
| T | 池內帳面 wm≥0 **全數**（降冪 t00-t17） | 18 | oracle 裁決：現行設定的已知解存在性 |
| N | 近標帶 [-1, 0) 共 133 筆 rank 分層（`spread_idx`） | 12 | 校正曲線 0 附近加密 |
| M | 深帶 [-3, -1) rank 分層 | 12 | 校正曲線延伸到搜尋工作區 |

**探索組**（找有效 pattern；錨點＝**池 top-300 greedy 家族聚類的跨家族代表前 6 名** F0-F5——
普查實錘 R8 乾淨前緣只是邊緣家族（上下兩分型 F13,a02 之後連 top-300 都排不進）、top 家族是全面散布
碎片雲等多種形態，見 `assets/round-09/pool_families.png`；`perturb_repair` 內建除塵 → 探的是各家族
**乾淨投影**的鄰域）：

| 臂 | 內容 | 筆數 | 買什麼 |
|---|---|---|---|
| E | 每錨點 k=0（純除塵=家族乾淨投影真值）+ k∈{4,8,16,32}×3 seed | 78 | 跨家族局部地形＋撞可製造新 best |
| G | 960 候選（錨點×k≤48×40 seed）SM 排序＋互異(Hamming>30)取前 32 | 32 | 批次版 SM-guided 搜尋（SM 池內誤差 1.5-2.4 可用） |
| S | 錨點前 5 × {全鏡射, 10-5-10 部分對稱} | 10 | 「把對稱做對」候選初測 |

- margin 同一把尺；manifest 帶 anchor_pool_wm/SM 預篩（重錨前基線再+162 點）。
- **HFSS 預算**: 162 筆 ≈ **8.1 hr** @3分/筆（R7/R8 實測；worst ~9.5hr,含 COM error 重試裕度)。可中斷續跑。

## 3. 執行紀錄 (Run)

```
# 正式機 .37（先 git pull）
python -m script.dedust run --input dedust_r9_input --store dedust_r9
# 任一機看進度/收檔
python -m script.dedust report --input dedust_r9_input --store dedust_r9
```
- 事件: —

## 4. 分析 (Analyze)

（`report` 表貼此；散點=池值 vs 現行值＋y=x 參照線；T 臂存活數與存活者清單）

## 5. 結論 (Conclude)

- 待。分岔預告：T 臂 `k=0` → 精修天花板下修、DIP 權重上升；`k≥1` → 存活者＝精修種子。
  探索組判讀：E 臂 k=0（六家族乾淨投影真值）誰最高＝可製造 warm-start 排行；E/G 任一筆 > −1.80
  （R8 乾淨前緣真值）＝「跨家族探索有效」；G 均值 vs E 均值＝SM 導引 vs 盲抽的增值；S 對稱 vs 錨點
  乾淨投影＝對稱先驗初判。

## 6. 後續決策 (Next)

- 解鎖鏈：R9 (oracle 裁決) × SM 重錨 (C 臂判準後半，離線待跑) → 精修 vs DIP 分岔定案。
- 校正曲線落地後，R6 的圖（oracle 線/池抽樣線）加折價註記或重刻。

## 7. 歸檔指向 (Archive)

- configs/README 列: 無（零新 config；`script/dedust.py` 擴充 select-r9＋`spread_idx` 測試）
- 結果夾: `DATASET_PATH/dedust_r9/`（NAS）
- memory: [[project_benchmark_vs_random]]
- ONGOING 動作: 發車時 🔵、收檔移 ✅（與 R8 一起補 README 索引）
