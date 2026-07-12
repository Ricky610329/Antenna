# docs/log/ 撰寫規範（研究日誌）

> 這裡是 **append-only 時間軸**：一個「假設→實驗→結論」＝一個 round＝一檔。
> 索引在 [README.md](README.md)；大流程（四層別搞混）見 root CLAUDE.md「研究日誌」節。

## 檔名與生命週期

- 檔名：`round-NN-<slug>.md`（用 `_TEMPLATE.md` 起）／`analysis-NN-<slug>.md`（零 HFSS 的資料分析）／
  `round-NN-report.md`（判讀完整版，僅大 round 需要）。NN 全域嚴格遞增、單一編號宇宙（Ricky 2026-07-12）。
- 狀態：`proposed → running → concluded → archived`。**§1-§3 發車前填完**（假設/判準/指令），
  §4-§5 收檔時填，§7 是歸檔動作清單（README 索引 +1、ONGOING 移出 🔵、champions/memory 若有）。
- **過厚警訊**（R22 教訓,2026-07-12）：§4 超過 ~3 批、或內容漂到 §1 假設之外＝該收檔開新輪——
  輪的邊界是「假設被回答」，不是「管線停下來」；換王/公證是事件不是假設，記在當下運行的輪即可。
- 開新 round 檔時**順手掃一遍 `../discuss/scratch.md`**（熟的升 ONGOING 候選、死的標 ❌）。

## 三條硬紀律

1. **判準發車前寫死**。發車後要修訂：只能在任何結果回來之前，且必須留修訂註記
   （日期＋理由＋「發生在結果前」聲明）——範例見 round-17 §1 的帶外紀錄 9.04 修訂。
2. **append-only**：收檔後不回改結論。發現錯誤用「★ 修正（日期）」補註原地說明，
   保留原文可追溯——範例見 round-10 的 w17 +0.48 修正。
3. **單次 vs 公證**：未公證數字一律標「單次」、不進結論粗體；紀錄級宣稱（換王/紀錄/榜首）
   **公證後才能寫**（鐵則；假象案例：w17 +0.48、b20 +0.32）。

## 格式約定

- README 索引行：`| NN | 主題 | 狀態(日期) | 結論一句(粗體關鍵詞) | [round-NN](檔名) |`——收檔時 +1 行。
- 圖落 `assets/round-NN/`，一律由可重跑的 script 產生（`script/round_report.py` 或 figs 腳本），不手貼截圖。
- **紀錄易主（換王/破紀錄,公證後）→ 渲染新舊對比圖進 round 檔**：
  `python -m script.figs.champ_compare --new <id> --old <id> --out docs/log/assets/round-NN/newking_*.png`
  （自動定位 pattern/響應/rad;產出後照例 Read 目檢再嵌入）。
- round 檔**只連結、不複製**其他層內容（configs/README、docs 設計文件、結果夾）。
- 時間戳先跑 `date` 再寫（跨午夜 session 教訓，2026-07-10）。
- 批次實驗的機器指令寫進 §3（含重啟指令）——斷電/接手時這是唯一真相。
