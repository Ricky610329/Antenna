---
name: project_fork
description: Antenna 專案的使用者 fork repo URL（Ricky610329/Antenna），供之後推送 review 分支使用
type: project
originSessionId: 9041afb1-fa0d-4f5f-87f7-0c929bd35f02
---
Antenna 專案上游 remote 是 `timmy90928/Antenna`，使用者已 fork 一份到自己帳號：

**Fork URL**：https://github.com/Ricky610329/Antenna

**Why**：使用者偏好把 review 工作在自己 fork 上進行，不直接推到上游。

**How to apply**：當 agent 要推送 `review/*` 分支時，加 remote（例如 `git remote add ricky https://github.com/Ricky610329/Antenna.git`）然後 `git push -u ricky <branch>`，或直接修改 `origin` URL。2026-04-22 的 review 工作使用者選擇**本地 commit、不 push**，未來如要開 PR 再手動推到 ricky remote。
