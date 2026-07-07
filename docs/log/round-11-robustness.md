# Round 11 — 冠軍公差穩健化 × 規則普適性

- **狀態**: running（2026-07-07 兩批發車;ref3 待兩批收檔後過夜）
- **提出 / 開跑 / 結論**: 2026-07-07 / 2026-07-07 / —
- **一句話問題**: 冠軍的 margin 經得起幾 px 蝕刻誤差？承重圖規律跨家族（c21/a15）是否重現？
  能否把「公差穩健的高 margin」精修出來？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: 工具 select-tolerance/select-occlude/select-refine*（dedust.py）· 冠軍名鑑 `docs/champions.md` ·
  先驗目錄 `docs/design_priors.md` · 前作 [round-10](round-10-refine-attribution.md)

## 1. 假設 (Propose)
- 承接：R10 冠軍 margin 最大 +0.20＝「薄冠軍」;製造公差（蝕刻誤差）可能一步吃掉。
- 假設：①erode/dilate 1px 與邊緣缺陷對 margin 的衝擊可量化,且冠軍間有差異（可選穩健者）；
  ②s05/g24 的承重圖規律（底排承重/頂中 rad 旋鈕/張力定位）在 c21/a15 重現＝規則升級跨家族；
  ③以「穩健 margin」為目標的 ref3 能找到對公差不敏感的過線點。
- 判準（發車前寫死）：tol=各冠軍 erode/dilate/缺陷後的 margin 分布（穩健王=最小跌幅）;
  occl2=承重圖與 s05/g24 的空間相關性;ref3=任一筆「原樣＋erode1＋dilate1 三態皆三標」＝穩健冠軍;
  選擇字典序（decisions 2026-07-07 定案）：①硬約束（三標+穩健）→②帶內 min-margin↑→③帶外惡度↓（加分,永不換帶內）。

## 2. 實驗設計 (Design)
| 批 | 機器 | 內容 | 筆數 | 狀態 |
|---|---|---|---|---|
| occl2 | 37 | c21/a15 各 24 塊遮蔽 | 48 | ✅ 48/48（2026-07-07 收） |
| tol | 218 | c21/w17/a15 × (erode1+dilate1+邊緣缺陷 k{1,2,4}×6) | 60 ≈3hr | 🔵 |
| ref3 | 37 | 過夜:A 穩健盲掃48（occl2 低成本區）+B 字典序 SM 導引64（sm_reanchor3,含帶外）+C add_block 組數階梯47（翼對17/上3下1×16/2+2×10/6塊×4;Ricky 指示加大） | 159 ≈8hr | 🔵（2026-07-07 晚發車） |

## 3. 執行紀錄 (Run)
```
# 37:  python -m script.dedust run --input dedust_occl2_input --store dedust_occl2
# 218: python -m script.dedust run --input dedust_tol_input --store dedust_tol
# 37(ref3,過夜): python -m script.dedust run --input dedust_ref3_input --store dedust_ref3
```
- 事件: —

## 4. 分析 (Analyze)

### occl2（2026-07-07 收,48/48）——假設② 承重圖規律跨家族 → **過關**
- 底排承重全重現:下半遮蔽代價中位 c21 −5.69/a15 −5.74 vs 上半 −1.67/−1.78（s05 −5.46/−2.71、g24 −3.27/−0.88 同構）;
  底角 (3,0)/(3,4) 全員災難級（−11~−19）。
- 跨家族空間相關:c21/a15 vs s05 ρ=+0.53/+0.57（p<0.01）;a15 vs g24（F3 族）ρ=+0.72;c21 vs a15 ρ=+0.96（同高原）。
- 低成本區（ref3 編輯座標）:頂部 (0,2)(0,3)(1,1)(1,4)（Δwm −0.1~−1.3）;rad 旋鈕 (2,1)(4,1)(4,3)
  （Δrad +1.0~+1.7 但 wm 重傷）——wm-rad 張力空間定位重現。
- tol 早期（11 筆）:整面 erode/dilate 1px 對 c21 致命（−11.9/−16.8）;局部邊緣缺陷 k1/k2 過 7/9（最壞 −0.11）。

## 5. 結論 (Conclude)
- 待。

## 6. 後續決策 (Next)
- 穩健冠軍出爐 → 規則→generator 載入（R12 前置工程）＋實作量測候選名單。

## 7. 歸檔指向 (Archive)
- 結果夾: `dedust_occl2/`、`dedust_tol/`、（ref3 待定）;memory [[project_w17_champion]]
