# tests/ 規範

- **一定從 repo 根跑**：`python -m pytest tests/ -q`（否則 sys.path 不對）。
- **golden 保真＝重構安全網**：雙容差——本機絕對 1e-4／CI 相對 1%。golden 重生流程與
  何時允許重生見 `docs/development.md`；golden 漂移＝紅燈，先找原因不是先重生。
- **每修一個 bug 補一條回歸測試**（測試名對應 bug 情境，如
  `test_liveness_advance_needs_fresh_heartbeat`）。
- **pytest warnings 清到零**當收尾標準。
- ⚠ worktree 陷阱：在 git worktree 裡跑會解析到主 repo 的 `antenna/`
  （`pythonpath=["."]` 以 rootdir 為準）——worktree 內驗證要小心結果指向誰。
