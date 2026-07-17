---
name: project-batch-runbook
description: 批次迴圈已弱模型化(2026-07-12):主入口=/batch-cycle;判讀=analyze batch;門檻源=docs/records.json;重錨=sm_reanchor train --add;偵測=dedust watch
metadata: 
  node_type: memory
  type: project
  originSessionId: 514acb31-4aa0-43ec-a3b3-98cdf8e2a623
---

批次迴圈全流程已固化（Ricky 2026-07-12「讓比較弱的模型都可以做好」）——**接手任何一批收檔,
直接 invoke `/batch-cycle <round> <batch>`,不要靠對話記憶手寫 judge script**。

- 判讀一鍵：`python -m script.analyze batch --round R --batch N`（臂別/可用帶外/前瞻/紀錄候選＋
  現成公證指令/「→ 行動」摘要——照「→」行執行）。
- 紀錄門檻機器真相源：`docs/records.json`（換王先改它,champions.md 散文跟上;/notarize 管流程）。
- 重錨一鍵：`sm_reanchor train --add "六夾" --out sm_reanchorNN.pth`（清單=configs/clean_stores.txt,
  不再改原始碼）。
- 收檔偵測：`Monitor(command='python -m script.dedust watch --stores ...')`,不手寫 bash。
- skill 全家：/batch-cycle（主）/notarize /new-round /close-round /gain-check /stall-protocol。
- worker 全自動（watchdog/補測/tier-2 讓位/--selfgen 自產）——HFSS 制度上不停,agent 只顧帳本。
