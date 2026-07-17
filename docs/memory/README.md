# docs/memory/ — AI 協作記憶鏡像（不依賴 harness 的導出版）

> **這是什麼**：Claude Code 專案記憶（`~/.claude/projects/<本專案>/memory/`）的**鏡像快照**。
> 原始 memory 只活在 Claude Code harness 裡（自動載入）；這份鏡像讓「任何環境」——新機器、
> 別的 AI 工具、或直接人讀——都能接手完整的協作脈絡。**鏡像日期：2026-07-17**。

## 新環境接手的閱讀順序

1. 根 `CLAUDE.md`（專案憲法：北極星/護欄/慣例）＋各子目錄局部 CLAUDE.md
2. 本夾 [INDEX.md](INDEX.md)（每則記憶一行摘要）→ 挑相關的細讀
3. `docs/discuss/decisions.md`（原則定案層）＋ `configs/ONGOING.md`（live 操作板）
4. `docs/log/README.md`（研究時間軸索引）→ 最新 round 檔
5. 批次線接手：invoke `/takeover`（`.claude/skills/` 有全套 skill 文本，也是純 markdown 可直讀）

## 檔案分類（檔名前綴）

- `feedback_*`：Ricky 給過的工作方式回饋（含 Why/How to apply）——**協作紀律，優先讀**
- `project_*`：專案狀態認知（策略/冠軍/資料/容錯/報告…）——時效性各異，以 `docs/records.json`
  與 round 檔為準（記憶可能落後）
- `user_*` / `reference_*`：環境與外部資源指標

## 與其他文件層的關係（指向不複製）

repo 文件層（CLAUDE.md/decisions/log/ONGOING）是**真相源**；memory 是跨 session 的
「認知快取」——重疊處以 repo 為準。記憶檔內的 `[[wiki-link]]` 指向其他記憶檔的 `name:`。

## 同步方式（更新鏡像）

```bash
# Git Bash（開發機）;重大 memory 變更後或收輪時順手跑
cp ~/.claude/projects/C--Users-Ricky-Documents-GitHub-Antenna/memory/*.md docs/memory/ \
  && mv -f docs/memory/MEMORY.md docs/memory/INDEX.md && git add docs/memory
```
（INDEX.md=原 MEMORY.md 改名，避免與 harness 的載入慣例混淆。）
