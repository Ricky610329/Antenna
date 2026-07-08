# Round 14 — 組件級軸：消融（有無）× 尺寸（大小）

- **狀態**: running（2026-07-08 雙批發車:ablate@37 + resize@218）
- **提出 / 開跑 / 結論**: 2026-07-08 / 2026-07-08 / —
- **一句話問題**: 組件是不是可量化的功能單元？各組件貢獻多少（消融）？尺寸響應曲線長怎樣（resize）？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: 方向定調=Ricky「像素級改動→組件級大小調整」（scratch 2026-07-08）;
  算子 select-ablate/resize_component;前作 [round-13](round-13-block-ladder.md)（組數軸）

## 1. 假設 (Propose)
- 背景：R13 證明組數（塊的「有無」的一種）是真設計軸;analysis-01 說像素級擾動多在雜訊內;
  Ricky 定調:之後不測像素級,改測組件級。
- 假設：①組件是功能單元,消融（去翼）的 Δ 可量化且大於雜訊（SM 預覽 c21 去兩翼 −0.59→−1.83）;
  ②組件尺寸有平滑響應曲線（main/wings grow/shrink ±1±2 → wm/rad/oob 可預測地動）→ 若成立,
  resize_component 取代 perturb_repair 成為 R15+ 主力算子、generator 先驗改組件參數化。
- 判準（發車前寫死）：消融=各翼貢獻 |Δwm|>0.1（雜訊地板≈0,任何差都真,0.1=工程顯著）;
  resize=尺寸響應是否單調/平滑（vs 隨機跳動）——平滑=組件級軸成立。

## 2. 實驗設計 (Design)
| 批 | 機器 | 內容 | 筆數 | 狀態 |
|---|---|---|---|---|
| ablate | 37 | c21/a15 × (full 公證×2+去兩翼+主件+單翼組合) | 10 ≈0.5hr | 🔵 |
| resize | 218 | x00/c21/a15/c25 × {main,wings} × ±1,±2 圈（形態學,無效自動跳） | 13 ≈0.7hr | 🔵 |

註:resize 13/32=有效變體少,本身是資訊——小翼(≤9px)縮 1-2 圈即死=尺寸下限;c25 加塊對 shrink 全滅。

## 3. 執行紀錄 (Run)
```
# 37:  python -m script.dedust run --input dedust_ablate_input --store dedust_ablate
# 218: python -m script.dedust run --input dedust_resize_input --store dedust_resize
```

## 4. 分析 (Analyze)
（待收檔）

## 5. 結論 (Conclude)
- 待。

## 6. 後續決策 (Next)
- 兩者皆成立 → R15 組件參數化探索（resize+add_block 當主力算子,per-component 網格）;generator 先驗改組件級。

## 7. 歸檔指向 (Archive)
- 結果夾: `dedust_ablate/`、`dedust_resize/`;memory [[project_generator_hyperfeature_pivot]]
