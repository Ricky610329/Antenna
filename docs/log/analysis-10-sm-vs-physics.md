# analysis-10｜物理模擬器 vs 現役 SM：在**三方都沒見過**的樣本上比 rank

- 狀態：running
- 開檔：2026-08-03 04:2x
- 零 HFSS。**Ricky 04:1x 授權對 `sm_reanchor*.pth` 開唯讀口**（只 `torch.load` 推論，
  不寫任何 NAS 檔、不佔 GPU、不碰 jobs/records/kpi）。
- 前情＝[analysis-09](analysis-09-diffsim-l3.md) §5.6：這是那份「誠實清單」上
  **未做／未答**的第一項——「diffsim 值不值得接進管線」在此之前**無法回答**。

---

## 0. 為什麼另開一檔

analysis-09 已收檔（MoM 路線收線）。本檔問的是**另一個問題**：
diffsim 的物理鏈（唯一有實用價值的產物＝L1）**跟現役資料驅動 SM 比，站在哪裡**。
這不是 diffsim 內部的比較，會牽動「要不要接進管線」的決策，照規範開新檔。

---

## 1. ★ 比較場的設計（本檔最重要的部分，比結果重要）

### 1.1 一個會讓所有比較失效的陷阱

**SM 的訓練集不是 `configs/clean_stores.txt` 那 513 行。**
`sm_reanchor._load_clean_stores()` 除了讀該檔，還**自動納入**
`dedust_auto*`（自產 tier-2）與 `dedust_c*`（鏈店）—— 實際 **587 店**。
（依據：`script/sm_reanchor.py:89-99` 的註解，2026-07-13 selfgen 修＋2026-07-24 鏈店修。）

照真正的 `CLEAN_STORES` 重算重疊：

| split | 層 | n | **SM 見過** | 都沒見過 |
|---|---|---|---|---|
| dev | clean | 150 | **145（97%）** | 5 |
| dev | neg | 150 | 0 | **150** |
| fit | clean | 29,062 | 28,245（97%） | **817** |
| fit | neg | 2,206 | 0 | **2,206** |
| val | clean | 30 | 28 | 2 |
| val | neg | 30 | 0 | 30 |

⇒ **clean 層有 97% 是 SM 訓練時見過的。在 clean 層比較 SM 與物理模型根本不公平**，
而這正是 analysis-08/09 一路在用的主戰場。**若不查這一步，整個比較會得出反過來的結論。**

### 1.2 三個比較場

| 場 | n | 誰沒見過 | 地位 |
|---|---|---|---|
| **neg_OOS** | 1,200（抽自 2,206） | mlp/cnn **0% 見過**；diffsim 零訓練 | **主場，唯一乾淨** |
| **clean_OOS** | 817（全部） | 同上 | 次場（分布仍與訓練集相近） |
| clean_INSAMPLE | 600 | **SM 見過** | 只當上界參考，**不當結論** |

⚠ `two`（cnn2）吃 clean+neg 全集 ⇒ **它在 neg 是 in-sample**，單獨標記，不與 mlp/cnn 同列判讀。

diffsim 側：L3 **零參數**（到處乾淨）；L1 的 4 個離散參數（`er/q/gap/diag`）在 **dev** 上選過
⇒ 在 `fit` 上評是 out-of-sample。**兩者在本比較裡都不佔訓練優勢。**

### 1.3 紀律

- 只讀，不寫 NAS；`SURROGATES[...]` 的工作目錄指向**本機** `tmp/`（原用法指向 NAS 的 `../tmp`）。
- 強制 CPU（`tmp/SESSION_COORDINATION.md` §2：GPU 讓批次線）。
- 主判準用**配對 bootstrap over 樣本**（不是 seed）——analysis-08 的教訓。

---

## 2. ★ 發車前預測（2026-08-03 04:2x 寫下，**腳本已在跑但我尚未看任何結果**）

寫下來才可證偽。commit 時間可對照 `scratchpad/sm_vs_phys.out` 的完成時間。

| 場 | 我預測誰贏 | 理由 |
|---|---|---|
| **neg_OOS** | **物理（L3）贏 mlp/cnn** | 主 SM 只吃正片，負片是純 OOD；L3 在 dev 的 neg 是 +0.563（純物理零擬合） |
| **clean_OOS** | **SM 贏 L1** | 雖然這 817 筆沒見過，但**分布**與訓練集高度相似（同產線同臂），SM 學到的是分布內的規律 |
| clean_INSAMPLE | SM 壓倒性贏 | 定義上如此，無資訊 |
| **two 在 neg** | 應該遠好於 mlp/cnn | 它 100% 見過，是 in-sample 上界 |

**若 neg_OOS 上 SM 也贏過物理**，那 analysis-08 §1「物理鏈可以當排名器」的價值主張
就要大幅下修——物理模型連在自己該有優勢的 OOD 域都輸，就只剩「可微」這個賣點，
而 analysis-09 §3.2 已證明可微性對 ∂L/∂ρ 沒有影響（核在梯度鏈上是常數）。

---

## 3. 結果

（待填）

## 4. 結論

（待填）
