---
name: project_design_audit_2026_07_01
description: 2026-07-01 全系統設計+bug 稽核結論：無 high-sev bug；4 個低風險修復已進 GAN；記下「刻意非bug」清單避免重複誤報
metadata: 
  node_type: memory
  type: project
  originSessionId: f1b86474-0089-4665-ad56-66605eb3e8b9
---

2026-07-01 對 GAN 分支做「改善設計＋抓 bug」全系統稽核（精讀核心 + 3 個 Explore agent + golden 基準 255→260 全綠）。

**結論：無 high 級正確性 bug**，核心防護網完善。落地 4 個低風險 commit（皆 golden 零漂移）：
A2/A3 config/邊界 fail-fast（targets 必填子鍵、warmup_ratio∈[0,1)、worst_margin 切片界線）；
A1 RunState 稀疏診斷欄（sm_gap/sm_fit_*）在 cached/skip epoch 留空不帶 stale 前值（save_row 加 _touched 集合）；
A4 移除 interval_loss 的 @overload；B1 run_training 抽 `_update_surrogate`/`_radiation_online_step` 純函式。

**刻意設計、非 bug（別再誤報）**：
- `SpectralConnectivityLoss.eigvals[1]` 不會越界——`num_nodes=height*width=625` 固定，L 恆 625×625（不是金屬像素數）。
- `custom_loss_minmax` 達標回 `loss_zero`（無梯度）＝單邊 hinge「夠好就不推」的規格意圖，非梯度斷裂。
- `merge()` base_pattern 在 CPU 初始化非效能問題——本專案 `config.device="cpu"`（開發+正式機皆是）。
- `FeedReachability` 4-連通 vs docstring「8-連通」＝已自承的文件不符；R_feed 是監控指標非訓練 loss。

**明確否決的建議**：把 `Path` 搬離 `utils/utils.py`（有 `__reduce__`、被烘進舊 checkpoint，[[project_data_dual_track]] 相關，CLAUDE.md 硬規則不可搬）；run_training 引入策略類階層（違反反 over-design）。

尚存但未動（out of scope）：`losses.py:313` FeedReachability.plot 的 `rate` unused-var（pre-existing；CI pyflakes 只擋 undefined name）。
