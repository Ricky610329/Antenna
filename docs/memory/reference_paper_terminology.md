---
name: reference_paper_terminology
description: 學長論文(docs/Paper.pdf)術語↔程式碼對應，含 GEN/ACP 的命名陷阱
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
  modified: 2026-07-28T15:15:50.333Z
---

`docs/Paper.pdf` = 吳維文碩論《自適應循環策略與圖譜連通度損失函數於貼片天線金屬圖形生成之研究》(中正大學, 劉立頌指導, 82 頁)。**這個 codebase 就是它的實作**。純文字曾抽到 `tmp/paper_text.txt`。PDF 可直接讀：`fitz.open(...)[i].get_text()`（**stdout 要 reconfigure utf-8，否則看起來像亂碼**；環境無 poppler，Read 的 pages 參數不能用）。實驗室血脈四代：陳廷茂(線上學習 GEN 框架) → 錢鵬予(BiScaleNorm+二階段變異) → 吳維文(ACP/SC Loss/DLF) → 我們。

**論文術語 ↔ 程式碼（兩個命名陷阱務必記住）：**
- **ACP = Adaptive Cyclical Policy** = `AdaptiveCyclicalScheduler`（**我們的路徑＝`antenna/optim/scheduler.py:19`**；學長版在他 repo 的 `functions.py`）。**只含「主動式高原偵測 + 自適應重啟 + lr/tau 雙耦合退火」三個排程子機制，不含 rollback**。
- **論文的 GEN = Gradient Estimation Network = 代理模型 SM**（我們＝`MLPSurrogate`/`HFSSNet`，`antenna/models/surrogates.py`；學長＝`OldSM`，`smodels.py`），**不是生成器**！「梯度估計」指它當不可微 HFSS 的可微替身、提供梯度。
- **程式碼的 `SigmoidGenerator`（2026-06-11 由 `SigmoidGEN` 改名，根除與論文 GEN 的撞名）= 生成器 G**（`antenna/models/generators.py:47`），名字裡的 GEN 只是 Generator 縮寫，**與論文 GEN 同名異義、角色相反**。

**rollback 與 ACP 是兩套獨立機制**（學長版：rollback 用 `TEMP.early_stop('real_loss', patience=10)`；ACP 內部另有一套只重啟 lr/tau 的 patience）。⚠ 論文 Algorithm 4 的敘述把 rollback 掛在「ACP 偵測到停滯」名下、模糊了這條界線——**以 code 為準，不以論文敘述為準**。→ **我們已於 2026-06-28 移除 rollback**（三條理由在 `antenna/training.py:813`：貪婪規則卡第一個山頭／退回舊 G 配當下變動的 SM 本質矛盾／原實作 off-by-one＋覆蓋最佳檔＝實際 ≈ no-op）。

**DLF（★2026-07-28 校正：舊記的「疑似不一致(待查)」已解）**：論文 DLF ＝「樣本無條件全收進 buffer，每次 SM 重訓時用**當下累計全域平均 λ_t 重新過濾整個 buffer**、只取 elite 子集訓」。**我們已移植為 `sm_train.mode: dlf_fit`**（`antenna/training.py:485`）。模式全集＝`SM_MODES`：single（學長原始單筆過擬合，**預設＋golden 基準**）／replay／dlf（elite 只訓 1 epoch＝under-trained 版）／**dlf_fit（＝論文原版）**／refit／adaptive／adaptive_window。論文點名批判的「**寫入即保留**」對照基準（loss<平均才寫入、寫入後永久保留 → 偽收斂）留在 `antenna/legacy/data.py:212 dynamic_loss_filter`，只服務舊 `.dataset`。

**其他對應**：SC Loss=`SpectralConnectivityLoss`（`antenna/losses.py:62`，Fiedler 值倒數 1/λ₂，論文主方法）；`island_suppression_loss`≈論文 KNS Loss；`total_variation_loss`/`GapClosingLoss` 是論文的對照基準/未正式採用；`FeedReachability`(R_feed)=評估指標非 loss；Warm Start=疊 lower/upper 饋電塊+物理初始解（我們另有 `harvest` 離線預訓＝放大版）。**論文沒有、我們新增的**：`worst_margin`（跨源同尺，`antenna/losses.py:661`）、`beam_coverage_loss`／`rad_window_margin`（±45°/3dB——正是他 §5.2 未來展望第 2 項）。

**規格對照**：論文 §4.1.2 n257（28GHz）帶內 S11 ≤ −10、Gain ≥ +4；**帶外 S11 > −2.5、Gain < −10——帶外本來就在他規格裡**，不是我們新增的判準（新增的只有 rad）。25×25 像素、0.2mm/px、5×5mm 貼片、RO4003C。⚠ 論文敘述與其 code 的帶外設定不一致（`docs/senior_method.md` 有記），引用時擇一並註明。
