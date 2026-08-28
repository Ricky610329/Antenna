---
name: project-batch-runbook
description: 批次迴圈已弱模型化(2026-07-12):主入口=/batch-cycle;判讀=analyze batch;門檻源=docs/records.json;重錨=sm_reanchor train --add;偵測=dedust watch
metadata: 
  node_type: memory
  type: project
  originSessionId: 514acb31-4aa0-43ec-a3b3-98cdf8e2a623
  modified: 2026-07-22T19:10:58.879Z
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

**R35 收輪更新（2026-07-23）**：批次規模 **R36 起 50/批**（tier 再平衡 2.37× 二連讀,decisions
「Tier 再平衡規則」——收輪必讀 analyze tiers 比值,≥2× 連兩輪降一格,tier1 地板 25 不歸零）;
輕量重錨隔批制常駐（奇數版 --no-ens --no-shadow,審計證零代價）;影子 CNN 連兩批三尺全贏=
轉正判準成立（**保守解=排序主鍵,「全鏈換錨」與凍結尺證據衝突留 Ricky 裁決**）;dual/wm 爬山鏈
標配 --expert（best 口徑 2/2）;rad-key 已退鍵;長駐 daemon 修 bug 後**必須重啟全部在跑進程**
（c2rad 假性收鏈教訓）。
