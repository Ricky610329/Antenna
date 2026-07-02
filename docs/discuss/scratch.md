# 隨意討論記錄（scratch · 隨意層）

> 兩層「討論記憶」的**隨意層**：半熟點子、觀察、對話片段。Claude 隨手記、**不主動報**；
> 想法成熟或經你確認後，才升級到 [decisions.md](decisions.md)（確定層，Claude 新增會主動說）。
> 這層刻意低門檻：可雜、可錯、可刪、可跳躍。**不是研究日誌**（假設→實驗→結論走 `docs/log/`）；
> 這裡只是「讓討論本身有記憶」的便條牆。新的記在最上面（新→舊）。

---

## 2026-07-02

### 💡 想法：跨輪自適應 SM 訓練強度（用下一輪 HFSS 校準上一輪「該訓多少」）
**痛點（兩個極端都不對）**
- 現行 `mode:dlf` 每輪 elite 只訓 **1 epoch** → SM 欠訓、系統性低估 HFSS ~3–4.5（R3 三臂共同 plateau 根因）。
- 但**硬逼到 `min_loss=0.1`**（dlf_fit/train_one_data 那套）**實測效果也沒比較好**（Ricky 經驗）→ 過度擬合
  buffer/elite ＝ 記憶而非泛化，且慢。→ 要的是「剛剛好」的訓練強度，不是「更多」或「逼到底」。

**點子（Ricky）**：這輪先訓練，然後**用下一輪的 HFSS 模擬結果，回頭調整上一輪 SM 該訓的 epoch 數**。

**我的展開（待驗證）**
- 這回饋訊號其實**已經在**：`sm_gap`＝SM「訓練前」對「新點」的預測誤差＝held-out 泛化誤差，每個
  fresh-HFSS epoch 都落 csv。它就是「下一次 HFSS 驗證了上一輪 SM 訓得好不好」——正是 Ricky 說的東西。
- 所以這 idea ≈ 把 SM 每輪訓練強度做成「**以 sm_gap 為回饋的閉迴路控制**」，取代固定 epoch / 固定 min_loss：
  - sm_gap 在降 → 訓練有幫到泛化 → 可再多訓一點。
  - sm_gap 卡住，或 `sm_fit_loss` 已 << sm_gap（擬合 buffer 很好但新點還是差）→ 過擬合徵兆 → 別再多訓/換訓法。
- 實作草案（未定，任選）：
  1. `train_by_datas` 加「以近期 `sm_gap` EMA 為準的 early-stop」，而非只看 train loss 到 min_loss。
  2. 讓 `sm_elite_epochs` 隨 sm_gap EMA 動（gap 大→多訓幾輪、gap 小→少訓）。
  3. 從 buffer 切一小塊當 validation，per-round early-stop 在 val gap 上（真·泛化早停）。
- **關聯**：跟 `TrustController` 同一哲學（都拿 gap 當回饋控制），只是致動器不同——那個控探索/tau/κ，
  這個控「SM 訓練量」。也許可共用同一條 gap EMA。
- **待議 / 疑慮**：
  - 回饋天然「慢一拍」（用下一輪修這一輪）；長期靠 EMA 平滑應可接受，但初期抖動要防。
  - gap 高到底是「訓太少」還是「這區 SM 本質學不動（該換架構/該多探）」？兩者處置相反，得能區分。
  - 跟 rollback 無關（rollback 已判定用處不大，見 decisions）。

### 其他碎念
- R3 三臂 factorial 目前被「SM 欠訓」這個共同上游瓶頸汙染，還不能乾淨判 E/D/E+D（見健檢；正式結論待 docs/log/）。
- D/E+D 的 sigmoid 連通先驗確實把 r_feed 拉到 0.9+（假設成立），但沒轉成更好 loss → 瓶頸不在連通。
