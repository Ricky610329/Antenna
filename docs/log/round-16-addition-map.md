# Round 16 — 添加收益圖（治「移除成本圖 ≠ 添加收益圖」）

- **狀態**: running（2026-07-09 發車@218;37 並行跑 R15 收尾批）
- **提出 / 開跑 / 結論**: 2026-07-09 / 2026-07-09 / —
- **一句話問題**: 「在哪加塊會變好」的真值圖長怎樣？跟遮蔽圖（移除成本）相關嗎？R15 贏家是靠哪塊贏的？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: 起因=R15 知情臂失靈機制(b)（先驗跨算子誤用）;工具 select-addmap;前作 [round-15](round-15-pushbutton-vs-toolbox.md)

## 1. 假設 (Propose)
- R15 教訓：遮蔽圖量的是「移除既有金屬」的成本,知情臂拿它當「空位加金屬」的先驗→輸給盲搜。
- 假設：添加收益有自己的空間結構（贏家的塊集中在 rows 4-11 中帶）;量出真值圖=修復知識遷移邊界。
- 判準（寫死）：A 臂=x00 單塊(2×2)全掃的 Δwm/Δrad/Δoob 位置圖,與遮蔽圖做空間相關（顯著同構/異構都是答案）;
  B 臂=贏家（g14/i02/g16）逐塊移除,各塊貢獻>0.1 即功能塊。

## 2. 實驗設計 (Design)
| 臂 | 內容 | 筆數 |
|---|---|---|
| A addmap | x00 全合法位單塊 2×2（歷史已跑排除,與既有 R13/R15 單塊數據分析時合併） | ~8 |
| B 塊歸因 | g14/i02/g16 加料組件逐組移除（鏡射夥伴併組） | ~8 |

## 3. 執行紀錄 (Run)
```
# 218: python -m script.dedust run --input dedust_addmap_input --store dedust_addmap
# (37 並行: python -m script.dedust run --input dedust_r15v_input --store dedust_r15v — R15 收尾,記於 round-15 §3)
```

## 4. 分析 (Analyze)
（待收檔;A 臂分析=本批+R13 blocks+R15 全部單塊數據合併成完整添加收益圖）

## 5. 結論 (Conclude)
- 待。

## 7. 歸檔指向 (Archive)
- 結果夾: `dedust_addmap/`、`dedust_r15v/`
