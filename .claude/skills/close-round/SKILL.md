---
name: close-round
description: 收檔一個 research round——照固定清單完成判讀、歸檔、索引、commit（用法：/close-round NN 或 /close-round NN <補充說明>）
---

# 收檔 Round $ARGUMENTS

照下列清單**逐步執行並逐項回報勾稽**。撰寫紀律以 `docs/log/CLAUDE.md` 為準（判準寫死、
append-only、單次 vs 公證）。若任一步發現異常（判準未預註冊、數字對不上、批未跑完），
**停下來回報**，不要硬收。

## 清單

0. **輪邊界確認**：本輪 §1 假設已被回答？（過厚警訊:§4 超 ~3 批或主題漂移＝早該收——
   換王/公證是事件不是假設,不構成續輪理由）。接棒輪走 `/new-round`。
1. **查時間**：`date "+%Y-%m-%d %H:%M"`——所有時間戳用查到的值。
2. **確認批次真的收完**：對 round 檔 §3 列的每個 store 查 `results.json` 完成數 vs manifest 總數、
   error 數（error>0 先問要不要補跑）。公證臂逐筆核對重複一致性。
3. **拉數據**：批次線＝`python -m script.analyze batch --round NN --batch <各批>`（判讀一鍵化）
   ＋ **/gain-check**（三層帳,一行摘要貼 §4）;線上線＝`python -m script.round_report --round NN --runs …`
   （圖落 `docs/log/assets/round-NN/`）。
3.5 **資料健檢**：`python -m script.analyze data`——總帳/完整性/查重洩漏警報;
   L0 行（唯一樣本數）貼 §7,洩漏警報非零先查再收。
4. **§4 分析**：對照 §1 **發車前寫死的判準**逐條判定（過/不過都要寫）;未公證數字一律標「單次」;
   意外發現另立小節。
5. **§5 結論／§6 後續**：學到什麼、促成/否決哪個候選;新待辦升 ONGOING 🔜（含觸發條件）。
6. **§7 歸檔＋狀態**：round 檔狀態改 `concluded`、補結論日期與 TL;DR 一句話。
7. **README 索引 +1 行**：`docs/log/README.md` 時間軸表,格式
   `| NN | 主題 | ✅ archived（日期） | 結論一句(粗體關鍵詞) | [round-NN](檔名) |`;更新「最後更新」日期。
8. **ONGOING 搬家**：該 round 移出 🔵（標題劃線＋✅ 一句指標）;若有接棒 round 補新 🔵。
9. **掃 scratch**：`docs/discuss/scratch.md` 全檔——熟的條目升 ONGOING 🔜、死的標 ❌＋為什麼;
   本 round 產生的定案若屬「原則級」→ 追加 `decisions.md`（新增要主動告知使用者）。
10. **冠軍名鑑**：若換王/新紀錄（**必須公證過**,判定流程=/notarize）→ **先改 `docs/records.json`**
    （機器真相源）再更新 `docs/champions.md` 散文（頭銜表/榜/血統鏈）;對應圖重跑 figs 並目檢。
11. **memory**：若有戰略級變化（換王/方向轉向/新鐵則）→ 更新對應 memory 檔＋MEMORY.md 索引行。
12. **commit**：訊息格式 `docs(log): RNN 收檔 — <一句結論>`;**不 push**（除非使用者要求）。

## 收尾回報

以表格回報 12 步的執行狀態（✓/跳過＋原因），最後給「本 round 一句話結論」與下一步建議。
