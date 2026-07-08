# Round 13 — 組數階梯系統對比（3/4/5/6 塊）

- **狀態**: running（2026-07-08 發車@218;blocks 批 63 新筆+併 ref3 已跑組數數據）
- **提出 / 開跑 / 結論**: 2026-07-08 / 2026-07-08 / —
- **一句話問題**: 組數（3/4/5/6 塊）對三標+選擇性的邊際報酬？c25（5 塊）奪王是偶然還是「多塊系統性更好」？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: 起因=R11 ref3 C 臂(Ricky 組數階梯)出新王 c25;工具 select-blocks(add_block+SM篩,歷史排除);
  前作 [round-12](round-12-consolidate-diversify.md);算子 [[project_generator_hyperfeature_pivot]]

## 1. 假設 (Propose)
- 背景：R11 c25（5 塊翼對）以 wm +0.22/rad +0.34 奪王,首個離開 3 塊構型的冠軍;R12 family2 否決第二家族
  → 結構突破的唯一活路是「組數」而非「換家族」。
- 假設：組數是可調的設計軸;多一塊=多一共振器,對 rad/選擇性有系統性增益,但帶內 wm 有邊際遞減/風險。
- 判準（發車前寫死）：各組數 best wm/rad/oob 曲線;若 4/5 塊系統性 rad>3 塊且 wm 不崩 → 組數升為設計先驗、
  進 generator;6 塊若無增益 → 邊際報酬遞減點定位在 5 塊。

## 2. 實驗設計 (Design)
| 批 | 機器 | 內容 | 筆數 | 狀態 |
|---|---|---|---|---|
| blocks | 218 | c21/a15 錨點 × {3 baseline+缺陷,4,5,6 塊} add_block 掃位+SM篩;歷史已跑自動排除 | 63 ≈3.1hr | 🔵 |
| （併入） | — | ref3 C 臂已跑的 4/5/6 塊（41 筆）分析時合併,不重跑 | — | ✅ |

## 3. 執行紀錄 (Run)
```
# 218: python -m script.dedust run --input dedust_blocks_input --store dedust_blocks
```

## 4. 分析 (Analyze)
（待收檔;分析＝blocks 新批 + ref3 C 臂 組數別 best 曲線）

## 5. 結論 (Conclude)
- 待。

## 6. 後續決策 (Next)
- 組數增益成立 → 寫進 generator 結構先驗;定位邊際報酬遞減點。

## 7. 歸檔指向 (Archive)
- 結果夾: `dedust_blocks/`＋`dedust_ref3/`（C 臂組數）;memory [[project_w17_champion]]
