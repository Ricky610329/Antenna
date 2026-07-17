---
name: reference_paper_terminology
description: 學長論文(docs/Paper.pdf)術語↔程式碼對應，含 GEN/ACP 的命名陷阱
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2adb79f3-22b2-4395-bf20-4dc71c4ffd49
---

`docs/Paper.pdf` = 吳維文碩論《自適應循環策略與圖譜連通度損失函數於貼片天線金屬圖形生成之研究》(中正大學, 劉立頌指導)。**這個 codebase 就是它的實作**。純文字曾抽到 `tmp/paper_text.txt`(82頁)。

**論文術語 ↔ 程式碼（兩個命名陷阱務必記住）：**
- **ACP = Adaptive Cyclical Policy** = `AdaptiveCyclicalScheduler`(antenna/functions.py)。**只含「主動式高原偵測 + 自適應重啟 + lr/tau 雙耦合退火」三個排程子機制，不含 rollback**。
- **論文的 GEN = Gradient Estimation Network = 代理模型 SM**(`MLPSurrogate (原名 OldSM)`/`HFSSNet`, smodels.py)，**不是生成器**！「梯度估計」指它當不可微 HFSS 的可微替身、提供梯度。
- **程式碼的 `SigmoidGenerator (2026-06-11 由 SigmoidGEN 改名，根除與論文 GEN 的撞名)` = 生成器 G**(models.py)，名字裡的 GEN 只是 Generator 縮寫，**與論文 GEN 同名異義、角色相反**。

**rollback 與 ACP 是兩套獨立機制**：rollback(載回歷史最佳 epoch + 重訓 SM)寫在 train 主迴圈(train_single.py:~338-350)，用獨立的 `TEMP.early_stop('real_loss', config['patience']=10)`；ACP 內部另有一套 patience(只重啟 lr/tau)。→ 重構時「排程」與「回滾」應拆成兩模組（[[project_R94_R156_scope]] 之外的設計依據）。

**其他對應**：SC Loss=`SpectralConnectivityLoss`(Fiedler 值 1/λ2，論文主方法)；`island_suppression_loss`≈論文 KNS Loss；`total_variation_loss`/`GapClosingLoss` 是論文的對照基準/未正式採用；`FeedReachability`(R_feed)=評估指標非 loss；Warm Start=疊 lower/upper 饋電塊+物理初始解。

**疑似不一致(待查)**：論文 DLF(動態損失過濾)是「每輪用累計平均重新過濾整個 buffer」；但 `train_single.py:388-389` 只做「real_loss<平均才寫入」(論文點名批判的對照基準)。train_dual 的 rollback 有 `online_dataset.filter(upper=avg)` 較接近 DLF。
