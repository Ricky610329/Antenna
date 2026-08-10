# 文件導覽（docs/）

天線反向設計專案的說明文件。**從這裡開始**，依需求往下找。

## 我想要…

| 需求 | 看這份 |
| --- | --- |
| **快速跑起來**（環境 / 切 branch / 執行 / TensorBoard 監看） | [`quickstart.md`](quickstart.md) |
| **寫/改實驗 config**（YAML 欄位、port、模型載入、結果夾結構、變體對照） | [`training.md`](training.md) |
| **理解系統怎麼運作**（三角色閉迴路、論文機制、模組地圖） | [`architecture.html`](architecture.html) |
| **改程式碼**（跑測試、golden 維護、加新架構/實驗、資料層、branch 慣例） | [`development.md`](development.md) |
| **找二值化/梯度的優化方向**（BiScaleNorm 同類方法、STE/tau/Heaviside 投影文獻表） | [`binarization_literature.md`](binarization_literature.md) |
| **看總進度（R1-R18 演進版）**（每輪最佳 gallery＋血統鏈＋帶外定案） | [`report/progress-r1-r18.md`](report/progress-r1-r18.md)（PDF 同資料夾;重建 `build_pdf.py <stem>`;撰寫規範 [`report/CLAUDE.md`](report/CLAUDE.md)） |
| 看 R1-R10 / R11-R14 分卷報告 | [`report/progress-r1-r10.md`](report/progress-r1-r10.md)／[`report/progress-r11-r14.md`](report/progress-r11-r14.md) |
| **對 pattern 做消融**（三類算子／量測紀律／歸因框架／14 條設計規則／可複製 checklist） | [`report/ablation-methodology.md`](report/ablation-methodology.md)（R54-R56 收攏） |
| **查研究時間軸**（每個 round 的假設→實驗→結論） | [`log/README.md`](log/README.md)（索引;撰寫規範 [`log/CLAUDE.md`](log/CLAUDE.md)） |
| **查現任冠軍與配方** | [`champions.md`](champions.md)（名鑑）＋[`design_priors.md`](design_priors.md)(設計規則) |
| **接續討論**（半熟點子/定案結論兩層） | [`discuss/scratch.md`](discuss/scratch.md)／[`discuss/decisions.md`](discuss/decisions.md) |
| **查外部文獻背書**（Sengupta 組三篇等） | [`reference/README.md`](reference/README.md) |
| **碩論大綱 v2**（agent+HITL 敘事、P0-P2 寫作順序、R4 三件事寫法） | [`thesis_outline.md`](thesis_outline.md) |
| 授權 | [`Licence IP.md`](Licence%20IP.md) |

## 建議閱讀順序

1. **新手上手**：`quickstart.md` → 跑一個 `configs/*.yaml`。
2. **要改實驗**：`training.md`（config 結構 + 變體對照表）。
3. **要改架構/讀懂原理**：`architecture.html`（術語已對齊論文，含重構歷程摘要）。
4. **要動手改 code**：`development.md`（測試、golden、zoo 擴充、branch 慣例）。

## 一句話地圖

> 入口 **`train.py`** ＋ 外部 **`configs/*.yaml`** 驅動 → 核心迴圈在 **`antenna/training.py`**（`run_training`）。
> 已**取代**舊的 `train_single.py` / `train_dual.py`（改用 `python train.py configs/xxx.yaml`）。

## 備註

- **`Paper.pdf`**（吳維文碩論）放在本資料夾但 **gitignored**（檔案大、版權），只在本機；文件中的論文術語對照以它為準。
- 文件若與程式碼不一致，**以程式碼與測試（`tests/`）為準**，並回報修正。
