# 確定結論／決策（decisions · 確定層）

> 兩層「討論記憶」的**確定層**：經確認的結論、定案、方向。Claude 新增條目時**會主動告知你**。
> 半熟點子在 [scratch.md](scratch.md)（隨意層，Claude 不主動報）。研究實驗的正式記錄仍走 `docs/log/`——
> 本檔只放「討論中定案」的精簡條目（指向，不複製 docs/log 的內容）。新的記在最上面（新→舊）。

---

## round 編號只給「燒 HFSS」的實驗（2026-07-03）
- **定案**（Ricky）：round 編號保留給**實際使用 HFSS 的實驗**；純離線分析不佔 round。
- **實作**：離線分析改用 `docs/log/analysis-NN-<slug>.md` 系列（首篇 analysis-01 pattern 解剖），
  在 docs/log/README.md 另立索引小節。round-06（離線期望基準）已歸檔、不回改——慣例向後生效。

## 滑動視窗 SM 訓練量（2026-07-03，Round 5 主角）
- **由來（R4 實錘×2）**：① 深度欠訓——每輪訓完 elite 的訓練 loss 仍 7.7–10.6（學長壓到 0.1，差兩個數量級），
  連訓練集都沒擬合；② adaptive 探測自鎖——target 停 3–5、曲線 80–100% 平（快照擠低處→差異<雜訊→沒有往上的證據）。
- **兩端都被自家資料反對**：1–5 ep 欠訓（本實錘）／壓到 0.1 過擬合（R1 dlf_fit 最差＋Ricky 經驗）→ 目標＝
  從未測過的中間帶（fit_loss ~1–3）。
- **機制（Ricky 設計）**：`mode: adaptive_window`——每輪 elite 訓滿視窗頂 hi、沿 log2 階梯 [hi/16…hi] 快照
  member0；held-out argmin **區位**決定滑動：連續 patience(3) 次落「**上二階**」（hi 或 hi/2，**不必貼頂**——
  最佳點上方保留至少兩階冗餘，Ricky 2026-07-03 修正）→ hi×2、落「最低一階」→ hi÷2、中段 → 不動。起點 64、
  **上限 1024**（Ricky 定——爬到頂≈學長「破千」量級）、下限 8；×2 滑動使階梯 key 跨視窗重疊 4/5 →
  bucket EMA 沿用；held-out 鐵律不變。
- **ensemble 5→3（R5，Ricky 定）**：省 SM 訓練成本（×0.6）；代價＝不確定性估計（trust/κ 的成員分歧）略粗。
  （R4 期的「保持 5」決策是當時脈絡；R5 起以 3 為準。）⚠ R5 vs R4 因此是**兩個實質變更**，歸因寫整包。
- **已知代價（誠實）**：live SM 過衝最佳點（等衡 hi≈4–8×argmin，偏成長）；成本單位＝1 epoch＝elite_n 步：
  R4 實測 elite ~90–120 → 起點 64 ≈ 2.2 萬步/輪（幾十秒）、**爬到頂 1024 ≈ 35 萬步/輪（可能與 HFSS 同量級）**
  ——正式機務必盯 `time` 欄，失控把天花板降 256/512（一行 config）。順手 `replay_size 256→512`。
- 實作：`WindowSMTrainController`（training.py，繼承 Adaptive 共用 probe_stats）；configs `single_r5_*`；
  取代 scratch 裡的 fixed-K 案與 probe-round 案（M 點平均若雜訊仍主導再上，見 round-05 §6）。

## 曲線 x 軸綁「真實 HFSS 模擬次數」（2026-07-02）
- **定案**（Ricky）：loss/worst_margin 曲線要跟**實際模擬次數**綁定，不用 epoch——epoch 含 cache 命中/skip，
  跟模擬預算對不上、判讀會困惑。
- **實作**：新欄 `hfss_calls`（累計真實模擬；cache/skip 不加；dense 每 epoch 落、斷點續跑續舊值，舊 run 用
  sm_gap fresh 標記數回填 seed）；`run_curve`（benchmark/round_report 共用）x 軸改用它、cache 命中收斂成一點、
  舊 run 回退 epoch；TB index 群組同步。`--at` 語意隨之變成「HFSS-call 預算」（跨 run 對標更公平）。
- **限制**：跑一半的 run 要等下次重啟（表頭遷移）才開始有此欄；之前的列由 fallback 撐。

## 自適應 SM 訓練量：機制 + 旋鈕定案（2026-07-02）
承下「SM 線上訓練強度」的方向，機制與旋鈕定了（**設計方向定案，實作時可微調**）；工作筆記見 [scratch.md](scratch.md)。

**機制**：用 held-out 的 fresh HFSS 點量 SM 泛化，自調訓練量，取代固定 epoch / 逼 min_loss。
- 一輪 SM 訓練沿途存數個權重快照（採樣不同訓練量）→ 用「這一輪產生 pattern、剛過 HFSS 的 held-out 點」評快照
  → 「泛化 vs 訓練量」曲線的**形狀**給「加 / 減 / (學不動)」。
- **held-out 鐵律**：只能評「產生該點的那段（先前）訓練」的快照；**絕不可**評「正在對該點訓練」的快照
  （否則洩題 → 假結論『永遠多訓』）。
- **K=1 + EMA**：每輪一個 held-out 點就夠（配對比較抵消點難度），噪聲用 EMA 吸收、致動器慢走。

**旋鈕定案**：
- **ensemble 保持 5**（探測已與 ensemble 脫鉤，不必為省空間縮）；**訓練量探測只用 member0**（各成員同架構
  同資料、只差 init 擾動 → member0 的曲線代表全體）→ 快照只存 member0、~5 份/輪，與 ensemble 大小無關；ensemble(5) 負責 uncertainty。
- **判準＝配對 + 每桶 EMA + argmin**：同一 held-out 點上比各快照相對誤差（配對、抵消點難度）；各快照誤差
  按訓練量分桶、跨輪 EMA → 取平滑曲線 argmin 當目標訓練量（EMA 目標、慢走）。**護欄**：argmin 落快照範圍最外側
  → 用斜率方向把探測範圍外移/擴，別被固定範圍卡住。
- **快照 = ~5 份/輪、偏早密（log-ish 間距，knee 通常在早期）**。**存哪（實作定案，比原磁碟版更簡）：留記憶體**
  （`prev_snapshots` dict，每輪覆蓋上一輪）→ 零磁碟、零清理、resume 最多少一次 observation。member0 state_dict
  ~16MB×5≈80MB steady，可接受。（2026-07-02 實作時微調，取代原「存 `checkpoint/probe_snapshots/` + 滾動清理」。）
- **實作落點**：`sm_train.mode: adaptive`（新 mode）；`adaptive` 區段旋鈕 {snapshots, epoch_min, epoch_max, ema}；
  `AdaptiveSMTrainController` 在 `training.py`；快照/評估在 `surrogates.py`（`train_by_datas(snapshot_epochs=)` +
  `eval_snapshot`）；只探 member0。新追蹤訊號：`sm_train_epochs`/`probe_argmin`/`probe_min_err`/`probe_max_err`/`elite_n`。

**發車前健檢補記（2026-07-02，R4 派工前）**：
- **⚠ 歸因誠實**：`adaptive` 相對 `dlf` 其實換了**兩個機制**——除了 elite 訓練量 1→自適應，**也拿掉了對最新點
  的 `train_one_data` 激進單筆擬合**（單筆擬合＝已確認反模式；最新點反正在 elite 裡照訓）。R4 vs R3 的差異
  歸因要講「SM 更新規則整包換」，不能只說「訓練量自適應」。
- **死鎖修復（模擬實證）**：原控制器早期雜訊把 target 壓到 ≤5 後永遠救不回（整輪默默退化回 dlf）。成因×2：
  整數 target 上做 EMA、1.3× 加碼被 round() 吃掉（低訓練量吸收態）；argmin 在歷史所有桶選、範圍外的過期低值
  永遠投票。修法：**target 內部存 float**（對外才取整）＋ **argmin 只在本輪有觀測到的桶裡選**（bucket EMA 只
  平滑值）。修後三情境驗證：死鎖恢復、sweet spot 收斂、純雜訊保守退低（＝dlf 現狀＝安全失效；監測法：
  `probe_min_err≈probe_max_err`＝曲線平＝探測沒資訊）。
- **斷點續跑**：控制器狀態只在記憶體 → 重啟歸零重學；`seed_target()` 用 metrics.csv 最後一筆 `sm_train_epochs`
  續 target（bucket 重學即可）。HFSS 當機重啟頻繁，沒這個 500-epoch run 的自適應會一直被打斷。

## SM 線上訓練強度（2026-07-02）
- **每輪只訓 1 epoch（現行 `mode:dlf`）偏少** ✔ — SM 欠訓、系統性低估 HFSS ~3–4.5，是 R3 三臂 plateau
  的共同上游根因（gap 高 → `trust_t` 被鎖在 0.05、永不進入利用）。佐證＝R3 健檢。
- **「硬逼到 `min_loss=0.1`」實測效果也不好** ✔（Ricky 經驗）— 過度擬合 buffer＝記憶非泛化、又慢。
  → **方向不是「訓更多／逼到 0.1」，而是「訓到泛化最好」**；具體走「跨輪自適應訓練強度」（發展中，見 scratch）。
- **rollback / keep-best 用處不大** ✔ — 先前已討論 + 程式 2026-06-28 已移除 rollback（原實作還有 off-by-one
  等 bug、實際近 no-op）。→ **不重拾舊 rollback**；好解回收若真要做，另想機制、不走那條老路。
