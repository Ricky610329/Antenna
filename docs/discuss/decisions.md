# 確定結論／決策（decisions · 確定層）

> 兩層「討論記憶」的**確定層**：經確認的結論、定案、方向。Claude 新增條目時**會主動告知你**。
> 半熟點子在 [scratch.md](scratch.md)（隨意層，Claude 不主動報）。研究實驗的正式記錄仍走 `docs/log/`——
> 本檔只放「討論中定案」的精簡條目（指向，不複製 docs/log 的內容）。新的記在最上面（新→舊）。

---

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
  `eval_snapshot`）；只探 member0。新追蹤訊號：`sm_train_epochs`/`probe_argmin`/`probe_min_err`/`probe_max_err`。

## SM 線上訓練強度（2026-07-02）
- **每輪只訓 1 epoch（現行 `mode:dlf`）偏少** ✔ — SM 欠訓、系統性低估 HFSS ~3–4.5，是 R3 三臂 plateau
  的共同上游根因（gap 高 → `trust_t` 被鎖在 0.05、永不進入利用）。佐證＝R3 健檢。
- **「硬逼到 `min_loss=0.1`」實測效果也不好** ✔（Ricky 經驗）— 過度擬合 buffer＝記憶非泛化、又慢。
  → **方向不是「訓更多／逼到 0.1」，而是「訓到泛化最好」**；具體走「跨輪自適應訓練強度」（發展中，見 scratch）。
- **rollback / keep-best 用處不大** ✔ — 先前已討論 + 程式 2026-06-28 已移除 rollback（原實作還有 off-by-one
  等 bug、實際近 no-op）。→ **不重拾舊 rollback**；好解回收若真要做，另想機制、不走那條老路。
