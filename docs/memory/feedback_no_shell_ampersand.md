---
name: feedback-no-shell-ampersand
description: Bash 工具內禁用 shell & 背景化——一律 run_in_background（2026-07-23 一天犯五次:孤兒進程/git 被連帶背景化/收不到通知）
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 514acb31-4aa0-43ec-a3b3-98cdf8e2a623
  modified: 2026-07-23T08:38:06.773Z
---

**規則**：要背景執行＝Bash 工具的 `run_in_background: true`。**永不**在命令尾加 `&`。

**Why**：`&` 背景化的進程 ①收不到完成通知（迴圈斷訊）②可能隨 shell 退出被殺（孤兒/半成品草稿夾）
③`cmd1 && cmd2 &` 會把**整條鏈**背景化（連 git commit 都進背景,落地狀態不明）。
2026-07-23 單日犯五次：gen/c1d2/watch/c3g2/c6tri2——每次都要殺進程+清草稿+重啟收拾。

**How to apply**：長跑命令（daemon/訓練/watch/發車鏈）→ 獨立一個 Bash 呼叫＋run_in_background。
快命令（git/查詢）→ 前景跑完。兩者永不混在同一條加 `&`。

再犯紀錄:2026-08-04 06:5x(judgement launched with `> /dev/null 2>&1 &`)——孤兒即殺(CommandLine 過濾精準獵殺,PowerShell 工具非 Bash;bash 會把 `$_` 展開成垃圾)。教訓不變:管道尾巴想都不要想加 &。
