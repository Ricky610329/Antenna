---
name: project_discussion_memory
description: "專案內「討論記憶」兩層：docs/discuss/scratch.md(隨意,不主動報) + decisions.md(確定,主動說)；接手先讀；含流動/生命週期三機制"
metadata: 
  node_type: memory
  type: project
  originSessionId: f1b86474-0089-4665-ad56-66605eb3e8b9
---

使用者要求「讓討論本身有記憶、在專案內」，2026-07-02 建立兩層（已註冊進 CLAUDE.md 慣例區）：
- `docs/discuss/scratch.md`（**隨意層**）：半熟點子/觀察，Claude **隨手記、不主動跟使用者報**。
- `docs/discuss/decisions.md`（**確定層**）：定案結論/方向，Claude **新增時要主動告知使用者**。

三機制（2026-07-02 評估後補、使用者核可，已入 CLAUDE.md）：
- **流動**：scratch（半熟）→ 熟了升 `configs/ONGOING.md` 🔜 候選區（what＋why＋**觸發條件**）→ 觸發即開 round。
  decisions 只放原則/定案、不放待辦。
- **生命週期**：升級/否決的 scratch 條目不刪、標狀態（✅ 已升級→去處 / ❌ 否決＋為什麼）留思路軌跡。
- **回顧節奏**：每次開新 round 檔時順手掃一遍 scratch（熟的升、死的標）——掛 round 邊界、不另設排程。

與 `docs/log/`（正式研究日誌，見 [[project_research_log]]）分開——這兩層是低門檻討論便條、只指向不複製。
**接手一個 session 先讀這兩檔**，才知道最近討論到哪、有哪些定案。
