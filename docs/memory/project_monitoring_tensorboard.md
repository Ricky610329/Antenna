---
name: project-monitoring-tensorboard
description: 訓練監控已改 TensorBoard (antenna/monitor.py)，取代 app.py 自渲染；app.py 只服務舊實驗與 dataset 瀏覽
metadata: 
  node_type: memory
  type: project
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
---

2026-06-10（commit `d323712`）：監控重設計完成。

- **新實驗**：`TrainingMonitor`（`antenna/monitor.py`）掛在 train.py 的 `on_epoch` hook；寫 `<結果夾>/tb/`。多機共享 NAS → 任一台 `tensorboard --logdir "T:\...\result"` 並排看全部實驗。訓練結束另存 `summary.png` 進結果夾。
- **架構決策**：實驗室「多機各跑一實驗 + NAS 共享」與 TB 的資料夾即匯流排設計同構，不自建 dashboard（學長的 app.py 等於手刻 TB）。
- **app.py 角色**：只服務舊實驗（歷史 `pic/`）與 dataset 瀏覽頁，自然退役、不刪。
- tensorboard 未安裝 → 監控降級警告、不擋訓練。正式機需 `pip install tensorboard` 一次。
- 未移植：SM 離線預訓練收斂曲線（罕用）。

相關：[[project-data-dual-track]]、[[feedback-prefer-simplicity]]
