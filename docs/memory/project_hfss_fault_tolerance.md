---
name: project_hfss_fault_tolerance
description: HFSS 容錯是兩層；單筆模擬失敗不再帶走整個 run（2026-06-25 Unite COM crash 後補強）
metadata: 
  node_type: memory
  type: project
  originSessionId: a893e592-6080-4b99-ba60-102253c3a9db
---

線上學習對 HFSS 的容錯現在是**兩層**，不是一層：

1. **pattern.py `simulate()`（既有，377-384）**：`self._simulator(...)` 失敗 → `restart(kill=True)` 砍掉重啟模擬器、**重跑一次**。專治「暫時性 COM 抖動」。所以每個失敗 epoch 的 `simulator.__call__` 會被呼叫兩次（原始 + 重跑）。
2. **training.py `run_training()`（2026-06-25 新增）**：當第 1 層重跑「仍失敗」（典型＝確定性的病態幾何讓 `oEditor.Unite` 丟 COM 例外，single_port.py:343），例外落到主迴圈 `output_element.simulate()` 的 `try/except`：
   - **skip 這一筆**：log warning（epoch / pattern hash / 錯誤）、未到門檻時 `simulator.end(save_project=False)` 收掉半成品專案（避免孤兒專案外洩）。
   - **G 仍對 SM 走一步**：`sim_loss` 用 carry-forward 佔位（只餵 ACP + metrics），不寫 pattern 快取、不更新 SM。讓 G 漂離壞 pattern，避免「同一張病態圖反覆重燒」（失敗 pattern 不進去重快取，否則會無限重試）。
   - **連敗升級**：連續 `max_consecutive_skips`（預設 5）次 → `simulator.reopen()` 重生 COM session；reopen 後再連敗到門檻 → `raise`（判系統性故障，交全域 excepthook 寄信中斷），避免靜默空轉。

事故起點：正式機跑 `pixel_single_sc_rad_boundary_mirror_harvest`，第 1 層重跑也救不了 deterministic Unite 失敗 → 整個 run 死 + 寄信 + 人工重啟載回 15 epoch。這就是 CLAUDE.md 標「先不急、穩定時再說」的 HFSS 容錯被觸發的時刻。

回歸測試 `tests/test_sim_failure_skip.py`：注入會丟例外的 mock sim，驗證 skip 續跑 + 連敗 reopen→中斷。**根因（病態幾何本身）未動**——只補了「不讓一筆壞幾何帶走整個 run」。relates to [[feedback_discuss_before_loss_change]]、[[project_sm_training_redesign]]。

**第 3 層補強(2026-06-30,machine 37 死掉後)**:上面的「連敗 → `reopen()`」本身會摔死。事故鏈:reopen = kill ansysedt → `open()` → `GetAppDesktop`,但**剛 kill 完、新 ansysedt 的 COM/RPC server 還沒起來**(要數秒)→ `com_error -2147023174 'RPC 伺服器無法使用'`;而 `open()` 是單發無重試(`reopen` 裡的 `# sleep(7)` 被註解掉)、且 training.py 那行 `reopen()` 沒包 try → 例外逃到 excepthook、寄信、整個 run 死。修兩處(commit 65d5369):
- **`patch_simulator.open(attempts=6, wait=8)`**:改成「try 連線 → 失敗就 `kill()` 殘行程 + `sleep(wait)` + 重試」最多 attempts 次,真的連不上才往外拋。直接吸收「新 ansysedt RPC server 未就緒」這個 transient(本次根因)。reopen() 不用改(它呼叫的 open 自帶韌性)。
- **training.py 連敗到頂的 `reopen()` 包 try**:重生真的失敗才升級成系統性故障 `RuntimeError` 優雅中斷,不讓 raw com_error 半路逃逸。
回歸測試 `tests/test_simulator_open_retry.py`(mock `_dispatch`/`sleep`/`kill`,不需真 HFSS):驗前 2 次失敗→第 3 次成功(kill 2 次)、一直失敗→試滿 attempts 才拋。⚠ **真實復原行為待 machine 37(真 HFSS)驗證**(使用者「下次再驗」);AEDT 本身會崩(商用軟體長跑常態)治不了,此修讓「崩→重生」有韌性、run 自動復原。
