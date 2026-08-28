---
name: feedback_shared_machine
description: 共機協調公約在 tmp/SESSION_COORDINATION.md（gitignore，git 看不到）；GPU 讓批次線、禁全域殺 python、TaskStop 停長時間 job 有誤殺嫌疑
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 450f84f5-a75c-40de-8a2b-ce9d67a54688
  modified: 2026-08-02T15:03:50.549Z
---

這台開發機常常**同時有兩個 session**（批次線 ⇄ 其他）。協調公約寫在
`tmp/SESSION_COORDINATION.md`——**`tmp/` 被 gitignore，git log 完全看不到**，
接手時要主動去讀，不然會以為沒有這個約定。

**Why**：2026-08-02 晚上批次線 session 的背景訓練鏈**連續兩次被外部終止**（21:0x、22:4x），
排查時才建立這份公約。

**How to apply**：
- **禁止**按名字殺進程（`taskkill /IM python.exe`、`Stop-Process -Name python*`、`pkill python`）
  ——另一個 session 的重錨訓練也是 `python.exe`，會陪葬。要殺只殺自己記下 PID 的。
- ⚠ **Claude Code 的 `TaskStop` 工具也要當心**：我用它停自己的背景 job，時間點
  （~22:4x）與對方報的第二次中止**吻合**，無法排除它的範圍是 process group。
  → 長時間背景 job 改成啟動時記 PID、需要停時 `Stop-Process -Id <PID>`。
- **GPU（RTX 2070 SUPER 8GB）批次線優先**：重錨每批佔 40–60 分、一天多次。
  離線分析預設 **CPU**；真的要 GPU 先 `nvidia-smi` 查有沒有別人的 python 在上面。
  ⚠ 我踩過：`head.py` 寫死 `cuda if torch.cuda.is_available()`，未查就佔用六次。
  **寫訓練 code 時 device 預設就給 cpu，別自動選 cuda。**
- 地盤：`configs/`・`docs/log/round-*`・`docs/records.json` 是批次線的；
  共用檔（ONGOING/scratch）只 append 自己的段落。
- **commit 只 add 指定路徑，不要 `git add -A`**——會把對方未提交的
  `kpi*.csv`／`clean_stores.txt` 掃進自己的 commit（我犯過，用 soft reset + unstage 還原）。

相關：[[feedback_check_clock]]（同一晚我也又犯了沒查時鐘就寫時間戳）。
