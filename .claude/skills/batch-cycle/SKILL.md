---
name: batch-cycle
description: 批次迴圈主 runbook——一批收檔後的完整清單（判讀→公證→重錨→發車→補池→掛偵測→記帳）。收檔通知到就 invoke；用法：/batch-cycle <round> <batch>（如 /batch-cycle 23 3）
---

# 批次迴圈 $ARGUMENTS

> 這是批次線的**主入口**。設計給任何模型照做：判斷都在工具輸出的「→ 行動」行裡，你的工作是
> **照抄執行＋在分支點選對路**。任一步異常（exit 1、數字對不上、工具報 ⚠）＝**停下回報使用者，不硬走**。
> 環境：開發機跑本清單全部指令（conda env `ant`）；正式機只跑 worker（已常駐，不用碰）。

## 清單（依序執行）

### ① 判讀
```
python -m script.analyze batch --round <R> --batch <N>
```
- 輸出「⚠ 未收全」→ 停，查 `python -m script.dedust jobs-ls`，等收全或問使用者。
- 記下輸出尾端的「→ 行動」①~⑤，後續照它走。

### ② 紀錄候選（若「→ 行動①」> 0 件）
- 照工具印出的 `select-repeat` ＋ `jobs-add` 指令**原樣執行**（公證批 prio 2，機器會優先跑）。
- 掛偵測等它收：`Monitor(command='python -m script.dedust watch --stores dedust_r<R>n<X>...')`。
- 收檔後 → **invoke `/notarize`** 做判定與記帳（換王/假象都在那裡處理）。

### ③ 重錨（每批必做）
- 照「→ 行動④」的指令，把 `sm_reanchorNN.pth` 的 NN 換成**現有最大版號＋1**
  （查現版：`ls T:\...\dataset\sm_reanchor*.pth` 或看 round 檔 §3 最後一批用的版號）：
```
python -m script.sm_reanchor train --add "<六夾逗號清單>" --out sm_reanchor<NN+1>.pth
python -m script.sm_reanchor train-two --out sm_reanchor<NN+1>.pth   # 影子家族(two/lohead)——缺此步 select 靜默停鍵
```
- 兩步都要跑（audit 2026-07-29:漏 train-two → 下批 select 按版號配對找不到 two/lohead,
  pred_wm_two/pred_lo 靜默停用、O 臂 rank 退回）;資料量大時全程可 >1hr,用 `run_in_background`。
  公證店（rNNnX）與收完的填空池也一併 `--add`。

### ④ 發下一批
- 指令模板在**該 round 檔 §3 的 code block**（唯一真相；旗標開關也看 §1 判準）；
  **--rad-head 必顯式帶當版**（parser default 是硬編舊版,會版本錯配;audit 2026-07-29）；
  select-rNN 若有 G 臂（--g>0）先跑 `sm_invert gen` staging（R46b3 教訓,round 檔 §3 有模板）：
```
python -m script.dedust select-r<R> --batch <N+1> --sm sm_reanchor<NN+1>.pth --rad-head rad_head<NN+1>.pth [--rad-key]
```
- 旗標分支：「→ 行動③」說退鍵 → 去掉 `--rad-key`（--rad-head 保留，pred_rad 續記前瞻）。
- **check-dup ×每夾，exit 1 ＝停**（絕不帶重複發車）：
```
python -m script.dedust check-dup --input dedust_r<R>b<N+1>a_input   # a..f 各跑一次
python -m script.dedust jobs-add --input dedust_r<R>b<N+1>a_input --store dedust_r<R>b<N+1>a --prio 3
```

### ⑤ 池存量（佇列永不見底）
```
python -m script.dedust jobs-ls
```
- 未跑完的填空池（rNNg*）剩餘 < 48 筆 → 補一池：
```
python -m script.dedust select-r21harvest --batch 5 --tag r<R>g<下一號> --seed <今天日期+序> --sm sm_reanchor<NN+1>.pth --n 72 --o 0 --wild 0 --lo 0 --shards 3
```
  → check-dup ×3 → jobs-add ×3 **--prio 9**。（佇列徹底見底時 worker 會自產，這只是第一道墊。）

### ⑥ 掛收檔偵測
```
Monitor(command='/c/Users/Ricky/miniforge3/envs/ant/python.exe -m script.dedust watch --stores dedust_r<R>b<N+1>a,...,f', persistent=true)
```
⚠ **必須用 ant env 完整路徑**——Monitor 的 shell 裡裸 `python`＝base miniforge，
會撞專案自訂 Path 類炸掉（2026-07-12 實測陷阱）。

### ⑦ 記帳（每批必做）
1. round 檔 §3 表加一行（批號/發車時間/查重 0）；§4 貼判讀重點（表格照 analyze batch 輸出精簡）。
2. **L0 必報**：判讀輸出開頭的 L0 行（全史真值/探索類佔比）抄進本批總結——常升目標（Ricky 定調）。
3. `git add` 相關檔 → commit（格式 `docs(log): rNN bN 判讀 — <一句>`）→ push。
4. 推播規則：**紀錄級事件才 PushNotification**（換王/破紀錄/停批），例行收檔不推。

## 分支表（遇到就走這裡）

| 狀況 | 動作 |
|---|---|
| jobs_state 出現 `.fail` | 停。看 fail 內容；HFSS 壞死→請使用者重開機器；修復後刪 `.fail`+`.claim` 重派 |
| analyze batch 報「未收全」且 jobs-ls 顯示卡住 >45 分 | stale 接管會自動發生；再卡→回報使用者 |
| 可用帶外**連三批**零推進（對照 round 檔 §1） | **invoke `/stall-protocol`**，不自動續產 |
| 判準寫死的「回報討論」節點觸發 | 停，帶 analyze batch＋gain 輸出回報使用者 |
| **第 3 批判讀完（每輪硬上限 3 批,Ricky 2026-07-13）**／或假設已被回答 | **invoke `/close-round <R>`** 結輪 → `/new-round` 開下一輪（不拖第 4 批） |
| 學費臂（D）滿 5 批 | 帶趨勢數據回報使用者裁決，不自動處決（漸進式成長條款） |

## 本清單不管的事
- worker／HFSS 容錯（watchdog、批尾補測、tier-2 讓位、自產）——全自動，出事走上表 `.fail` 分支。
- 輪結算歸檔細節 → `/close-round`；公證判定 → `/notarize`；開新輪 → `/new-round`。
