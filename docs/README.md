# 文件導覽（docs/）

天線反向設計專案的說明文件。**從這裡開始**，依需求往下找。

## 我想要…

| 需求 | 看這份 |
| --- | --- |
| **快速跑起來**（環境 / 切 branch / 執行指令） | [`quickstart.md`](quickstart.md) |
| **寫/改實驗 config**（YAML 欄位、port、模型載入、變體對照） | [`training.md`](training.md) |
| **理解系統怎麼運作**（三角色閉迴路、論文機制、模組地圖） | [`architecture.html`](architecture.html) |
| **改程式碼**（跑測試、golden 維護、加新架構/實驗、branch 慣例） | [`development.md`](development.md) |
| **了解這次重構的設計決策**（整合類別 + ACP + 回滾分離） | [`refactor-proposal.html`](refactor-proposal.html)（已實作，存為設計紀錄） |
| 授權 | [`Licence IP.md`](Licence%20IP.md) |

## 建議閱讀順序

1. **新手上手**：`quickstart.md` → 跑一個 `configs/*.yaml`。
2. **要改實驗**：`training.md`（config 結構 + 變體對照表）。
3. **要改架構/讀懂原理**：`architecture.html`（術語已對齊論文）→ 需要時回看 `refactor-proposal.html` 的設計脈絡。
4. **要動手改 code**：`development.md`（測試、golden、registry 擴充、branch 慣例）。

## 一句話地圖

> 入口 **`train.py`** ＋ 外部 **`configs/*.yaml`** 驅動 → 核心迴圈在 **`antenna/training.py`**（`run_training`）。
> 已**取代**舊的 `train_single.py` / `train_dual.py`（改用 `python train.py configs/xxx.yaml`）。

## 備註

- **`Paper.pdf`**（吳維文碩論）放在本資料夾但 **gitignored**（檔案大、版權），只在本機；文件中的論文術語對照以它為準。
- 文件若與程式碼不一致，**以程式碼與測試（`tests/`）為準**，並回報修正。
