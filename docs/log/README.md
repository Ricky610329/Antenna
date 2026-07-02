# 研究主線時間軸（decision log）

> 每個 round = 一個假設的完整生命週期(propose → run → analyze → conclude → archive)。
> 一個 round 一檔(`round-NN-<slug>.md`,用 `_TEMPLATE.md`),這裡只放時間軸索引。
>
> - **設計理由 / 為什麼**:看 `../*.md`(exploration_roadmap / research_landscape / guided_search_design / senior_method)。
> - **config 全集**:看 `../../configs/README.md`。
> - **現在在跑什麼 / 候選池**:看 `../../configs/ONGOING.md`(live 操作板)。
> - **跨 session 接手**:先讀本檔(走到第幾 round)+ ONGOING.md(手邊在跑什麼)。

最後更新:2026-07-02

## 時間軸
| Round | 主題 | 狀態 | 結論(一句) | 檔 |
|---|---|---|---|---|
| 01 | SM 訓練量 A/B(dlf/dlf_fit/refit) | ✅ archived | **訓練量非 bottleneck**;dlf≈refit>dlf_fit、皆差 spec ~4dB、未收斂 | [round-01](round-01-sm-training-ab.md) |
| 02 | ensemble + trust(文獻治本) | ✅ archived(2026-07-01 停,②~417ep) | **治本微幅、未決定性**(②③微贏~0.3-0.5dB、①輸、皆未收斂;trust_t 卡低) | [round-02](round-02-ensemble-trust.md) |
| 03 | 探索 × DIP（factorial E/D/E+D） | ✅ archived（2026-07-02 停,E@189/D@101/E+D@132） | **E(lr↑)最佳 -3.63@89（¼ epoch 追平②）**;DIP 連通成功但停滯;三臂被 SM 欠訓汙染 → R4 修瓶頸重跑 | [round-03](round-03-explore-dip.md) |
| 04 | 自適應 SM 訓練量（修 R3 SM 欠訓瓶頸） | 🔵 proposed（2026-07-02） | — | [round-04](round-04-adaptive-sm.md) |

## 研究脈絡（一句話串起來）
generator G = 單 pattern 超特徵 → 轉 **generator-free SM-guided 搜尋**(輸 random)→ **Round 1** 測「是不是 SM 訓練不足」→ **否**(訓飽反而過擬合)→ 病灶是 SM-guided 搜尋本身 → **Round 2** 上文獻治本(ensemble + trust)→ 治本微幅未決定性、且實測搜尋「凍住」(每 epoch 才翻 ~6 像素)→ **Round 3** factorial 測「探索(lr↑)× DIP(generator 帶回來連通先驗)」的效果與加乘 → 健檢發現三臂共同瓶頸＝**SM 欠訓**(dlf 每輪只訓 1 epoch → 樂觀、trust 鎖 0.05 不利用) → **Round 4** 上**自適應 SM 訓練量**(held-out fresh 點自調每輪重訓 epoch 數)修這個瓶頸、重跑 E/D/E+D 去 confound。
