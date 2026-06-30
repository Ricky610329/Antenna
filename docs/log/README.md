# 研究主線時間軸（decision log）

> 每個 round = 一個假設的完整生命週期(propose → run → analyze → conclude → archive)。
> 一個 round 一檔(`round-NN-<slug>.md`,用 `_TEMPLATE.md`),這裡只放時間軸索引。
>
> - **設計理由 / 為什麼**:看 `../*.md`(exploration_roadmap / research_landscape / guided_search_design / senior_method)。
> - **config 全集**:看 `../../configs/README.md`。
> - **現在在跑什麼 / 候選池**:看 `../../configs/ONGOING.md`(live 操作板)。
> - **跨 session 接手**:先讀本檔(走到第幾 round)+ ONGOING.md(手邊在跑什麼)。

最後更新:2026-06-29

## 時間軸
| Round | 主題 | 狀態 | 結論(一句) | 檔 |
|---|---|---|---|---|
| 01 | SM 訓練量 A/B(dlf/dlf_fit/refit) | ✅ archived | **訓練量非 bottleneck**;dlf≈refit>dlf_fit、皆差 spec ~4dB、未收斂 | [round-01](round-01-sm-training-ab.md) |
| 02 | ensemble + trust(文獻治本) | 🔵 running(2026-06-30 發,跑到 500) | — | [round-02](round-02-ensemble-trust.md) |
| 03 | DIP / generator 帶回來 | 🔜 已排(R2 後) | — | (待建) |

## 研究脈絡（一句話串起來）
generator G = 單 pattern 超特徵 → 轉 **generator-free SM-guided 搜尋**(輸 random)→ **Round 1** 測「是不是 SM 訓練不足」→ **否**(訓飽反而過擬合)→ 病灶是 SM-guided 搜尋本身 → **Round 2** 上文獻治本(ensemble + trust)→ 若仍卡,**Round 3** 把 generator(DIP 連通先驗)帶回來配治本。
