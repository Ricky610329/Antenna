---
name: user_python_env
description: 使用者使用 miniforge 管理 Python 環境（不是 Anaconda/Miniconda/venv）
type: user
originSessionId: 9041afb1-fa0d-4f5f-87f7-0c929bd35f02
---
使用者的 Python 環境管理工具是 **miniforge**（conda-forge 的輕量發行版，預設通道為 conda-forge）。

- **實際的 conda env 名稱為 `ant`**（不是 CLAUDE.md 裡寫的 `antenna`；Unit 9 agent 2026-04-22 確認）
- 啟用指令：`conda activate ant`
- 若要在 shell 中呼叫 conda/mamba，應假設已安裝 miniforge，而不是 Anaconda
- 子 agent 執行測試（pytest 等）前應先啟用 `ant` conda env
- 安裝套件建議優先使用 `mamba` 或 `conda install -c conda-forge`（miniforge 預設通道），必要時才用 `pip`
- CLAUDE.md 在「環境建置」章節寫的是 `antenna`（documentation drift），可順手修正
