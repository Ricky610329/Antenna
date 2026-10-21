# Round 92 — 非 a0 家族普查（兩埠斷開的歷史設計重量 11–40 GHz）

- **狀態**: running（`r81v6x01`）
- **提出 / 開跑 / 結論**: 2026-10-22 / 2026-10-22 / —
- **一句話問題**: v6 在 a0 家族撞結構牆；學長歷史設計中是否有兩埠斷開、又天生帶低頻抑制的其他家族，可作 v6 的新起點？
- **一句話結論 (TL;DR)**: 待分析
- **指向**: [round-91](round-91-v6-small-steps.md) · ONGOING 候選「非 a0 家族探索輪」· `script/r92_explore_seeds.py`

## 1. 假設 (Propose)

- 前置事實（建批時統計）：歷史 21,251 筆中**兩埠斷開僅 665 筆（3%）**；其通帶最佳最差點 −5.8 dB（margin −2.83）。round-81 所記「通帶可達 −1.08 dB」的歷史設計**全為兩埠相連**（直流直通，低頻抑制不可能）。a0 很可能已是兩埠斷開設計中通帶最好的家族。
- 普查目的：確認斷開的遠親家族在 11–19 GHz 的低頻行為；若有低阻 ≥ −5（v6 尺）且通帶不太差者，可作新起點。

## 2. 實驗設計 (Design)

- x01：兩埠斷開、與 a0（mg15_005）差 ≥ 150 格的歷史設計中，取通帶 margin 前 100 名，maxmin Hamming 分散挑 24 筆（與 a0 差 275–314 格；歷史通帶 margin −12.9～−2.8）。v6 尺、prio −3、一批。
- **判準（發車前寫死）**：有 ≥ 1 筆 v6 低阻 margin ≥ −3（與 a0 相當或更好）且通帶 margin ≥ −4 → 成立，開該家族的爬升輪；否則不成立——結論為「a0 是可得的最佳斷開家族，v6 需規格討論或全新結構（非歷史設計）」。

## 3. 執行紀錄 (Run)

- `python -m script.r92_explore_seeds --n 24 --pool 100 --out <dir>`
- `python -m script.r81_curve_sm enqueue --input <dir> --name r81v6x01 --prio -3 --shards 3 --config configs/dual_r81_v6.yaml`

## 4. 分析 (Analyze)

## 5. 結論 (Conclude)

## 6. 後續決策 (Next)

## 7. 歸檔指向 (Archive)
