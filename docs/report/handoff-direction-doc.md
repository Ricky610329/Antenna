# 交接包 — 「研究方向文件」寫作 session（2026-07-28）

> 給另一個 session 用：Ricky 要總結全案、撰寫**研究大致方向的架構文件給老師**（8 月目標）。
> 原則＝**指向不複製**：本檔只給敘事骨架、真相源指標與規範；數字與細節去源頭讀。
> 由批次線 session 產出；協作規範（§5）為提案，Ricky 拍板後生效。

## 0. 接手閱讀順序

1. root `CLAUDE.md`＋harness memory（自動載入，兩 session 共用）
2. `docs/log/MILESTONES.md`（里程碑索引）→ `docs/log/README.md`（round 索引）
3. `docs/discuss/decisions.md`（原則定案）→ `docs/discuss/scratch.md`（末段近期脈絡，含 2026-07-28 高原判定）
4. `docs/records.json`（紀錄真相源）＋`configs/ONGOING.md`（live 現況）
5. 前例報告：`docs/report/progress-r1-r18.md`（格式參考）＋桌面「進度報告_v2.md」（**對外用語表在此**）

## 1. 研究弧線（敘事骨架——這份是本 session 腦中的綜合，寫作時的七幕結構）

1. **承接與重建**（R1–R14）：audit 學長 codebase → 核心精簡解耦（G⇄SM＋HFSS online learning）、golden 安全網、config 驅動；真 HFSS 驗證可跑。誠實基線：學習式搜尋輸 random best-of-N。
2. **方法轉向**（R7 起）：批次 HFSS 驗證線成主力（`dedust.py`）；同一把尺 `worst_margin`；文獻定調＝輸 random 屬預期，治本方向＝不確定性門控＋active learning（`docs/research_landscape.md`）。敘事定調（07-08）：**線上學習＝工具，agent＋human-in-the-loop 共同優化**。
3. **工程化／弱模型化**（R23 前後）：批次迴圈 skill 家族、records.json 真相源、公證鐵則（3/3）、判準發車前寫死、每輪 ≤3 批；方法原則定案於 decisions.md（帶外主鍵、軸相關枯竭、效率評、自主續輪宣告制）。
4. **戰略換軸＝資料飛輪**（07-15，R31）：多樣性→SM 準度→變現；五軸 KPI；王系凍結反馬太。
5. **左側戰役**（R37–，07-24 達陣）：左右側拆帳、左側（usable_lo）＝主戰場；**c8trip03_01＝左側合格解首例**（36hr 戰役）。
6. **結構理解期**（R41–R44）：金屬組分析（組＝變異單元→生成單元）、組義字典（王朝三位一體出現率 92%）、對角＝左側門票（方向性定案）、**可製造化三連負**（事後清潔死路→生成端唯一路）、刀鋒解、鏡射手性。組文法 v1 未勝 old＝誠實負結果。
7. **接力期**（R45–46，現在進行）：深水右爬接力線——探索臂當礦場（深左原礦）＋組級鏈當精煉廠（爬 wm）；**g 線 +5.32（wm −8.04→−2.72），全史首個「隨機起點→苗子帶」血統**；苗子斷層被鏈跨過。高原判定三條件進行中（scratch 07-28 條）。

**招牌成果候選**（寫作時挑選）：①線上系統真 HFSS 驗證可跑 ②批次線方法論（公證／判準寫死／機器真相源）③左側合格首例 ④隨機→苗子帶接力血統 ⑤組文法＋組義字典（結構性理解）⑥誠實負結果集（可製造化三連負、SM 過濾稀薄區有害、組文法 v1）。

## 2. 數字真相源（引用數字只准來自這裡）

| 要什麼 | 去哪讀 |
|---|---|
| 紀錄王榜＋門檻 | `docs/records.json`（散文版 champions；以 json 為準） |
| 各 round 數字 | `docs/log/round-NN-*.md` §3/§4（未公證一律標「單次」） |
| 鏈線帳 | `docs/chains/*.jsonl` |
| KPI 時序 | `docs/kpi.csv`／`kpi_two.csv`／`kpi_shadow.csv` |
| 資料量 | 2026-07-28 快照：全史已測 ~24.7k 筆；SM 乾淨真值 23,061（train 17,031／held-out 6,030）——重算用 `python -m script.status` |
| 現成圖 | `script/figs/`（data_map／sm_capability／pareto_front／report_*.py 產線）；成品在 `docs/log/assets/`＋`docs/report/assets/` |

## 3. 對外 framing 規則（已定調，別重新發明）

- 目標定義：per-task「找一個最佳 pattern」；「任意 spec」＝設計規範，**非** target response。
- 學長對比：學長招牌＝t07_top（合規 +0.35 但含粉塵不可製造）；F2＝我們的對稱練習母本（非學長最好）；design_priors「F2 −6.44」是誤植。框架＝**學長證可合規，我們補可製造＋系統化**。
- 慣用 framing：飽和拚初始化／最後一哩路／款式變體（見進度報告_v2 用語表）。
- 誠實紀律：開頭必有 Scope 與 Limitations（模擬≠量測）；不把工具說成 production system；數字對 round 檔核對；模型數字標版本（vNN）。
- 報告產線與硬驗收（PNG 目檢／PDF 逐頁）＝`docs/report/CLAUDE.md`。

## 3.5 現況定位更新（2026-07-29,fanout 審視後）

- 「我們在哪」的誠實答案修訂：**不是高原期,是「單線深入＋儀器換代」期**——高原條件① 證據=g 線 n=1
  （d 線撞的是另一道 −7 牆,R46 誤植已 ★修正）;條件③ 操作化後帳面成立但 v85/v86 凍結尺在改善（破壞中）;
  c47d2（rad 正起點）在爬。兩輪 fanout 審視存檔=`docs/discuss/audit-fanout-2026-07-29.md`+`audit-round2-2026-07-29.md`
  （含翻案:SM 排序在 d=1 鏈鄰域無價值——鏈有效靠 25 席全測;寫作引用「SM 引導搜尋」相關敘述時注意此修正）。
- 探索方向優先序已定（decisions 同名節）:生成端知識化→定向嫁接→組圖表示。

## 4. 未定案清單（寫作時**別寫死**的東西）

- c45g4 續爬中——「隨機→合格全程」是否達成未知；高原判定三條件（scratch 07-28）未定。
- 「深 lo 解本質不可製造」＝**推論待驗**（decisions 有標），寫作只能當 hypothesis。
- 獨立艙（血統獨立測試）＝設計凍結未跑；GNN bakeoff 未觸發（pot ~30k 線）；鏡射＝rad 修復旋鈕＝候選未實作。
- 時間軸（8 月架構文件／10 月收斂）Ricky 自管，文件裡的時程由 Ricky 定。

## 5. 協作規範（兩 session 並行；提案，Ricky 拍板後生效）

- **單一寫者邊界**：批次 session（本線）持續自主跑 R46+，獨佔 `configs/ONGOING.md`、`docs/log/round-*`、`docs/records.json`、`docs/kpi*.csv`、`configs/clean_stores.txt`、`script/`。**寫作 session 一律不改這些檔**。
- 寫作 session 的地盤＝`docs/report/` 新 stem（＋`assets/`）；只新增檔案、不改既有檔；commit 前綴 `docs(report):`；commit 前先 `git pull --rebase`。
- 要新數字：跑唯讀命令（`script.status`／`script.analyze`／figs 腳本）可以；**不跑** select／jobs-add／sm_reanchor／chain（會踩批次線佇列與模型）。
- 兩邊都在 GAN branch；發現對方剛 commit 就先 rebase 再動，衝突時以批次 session 的帳為準。
