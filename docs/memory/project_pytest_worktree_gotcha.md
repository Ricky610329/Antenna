---
name: project_pytest_worktree_gotcha
description: 在 git worktree 裡跑 pytest 會因 pyproject.toml 的 pythonpath=["."] 而解析到**主 repo** 的 antenna/，導致 agent 以為在測自己的改動其實沒有；2026-04-22 Unit 17 發現
type: project
originSessionId: 9041afb1-fa0d-4f5f-87f7-0c929bd35f02
---
**問題**：`pyproject.toml` 設定 `[tool.pytest.ini_options] pythonpath = ["."]`。當 agent 在 `.claude/worktrees/agent-xxx/` 內跑 `pytest` 時，pytest rootdir 解析經常解析到**主 repo**（不是 worktree），於是 `import antenna` 匯入的是主 repo 的 antenna/，不是 worktree 裡本 agent 的修改版。

**Why**：這會讓 agent 回報的「pytest 通過」其實是在驗證主 repo 的程式碼，不是他們自己的 edits；是個沉默失敗點。

**How to apply**：
1. 之後若在 worktree 跑 pytest，顯式指定 pythonpath：`pytest -o "pythonpath=<worktree-abs-path>" tests/...`
2. 或 `cd` 到 worktree 後檢查 `python -c "import antenna; print(antenna.__file__)"` 確認 import 解析到對的路徑
3. 長期解法：考慮改 `pyproject.toml` 的 pytest 設定或改用 `pytest --rootdir=.` 來鎖定當下工作目錄
4. 2026-04-22 的 review 工作：Units 1-18 的測試通過訊號可信度受此影響；但由於每個 unit 改動都很小且多半是清理，主 repo 版本通常也會通過，結果影響可能不大。合併回 modernize 前建議手動跑一次完整 pytest。
