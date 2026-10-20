# Round 87 — 爬山打分：平均 vs 上下文接龍模型單獨

- **狀態**: concluded（2026-10-21）
- **提出 / 開跑 / 結論**: 2026-10-21 / 2026-10-21 / 2026-10-21
- **一句話問題**: R86 前段準度顯示 CNN 變化量模型在好設計之間已無排序力（ρ −0.13），上下文接龍模型 +0.55、平均 +0.45；爬山若只用上下文模型打分，能否比平均挑得更好？
- **一句話結論 (TL;DR)**: **成立（差距小）**——上下文模型單獨爬山（tclimb）中位 −0.38、最佳 −0.25、> −1 占 100%，平均（eclimb）−0.40／−0.29／93%（v6a 尺各 15 筆單次）。另：exploit 改平均後中位 −0.42（R86 CNN −1.55）；explore 改平均後 −1.56 且多大失手，查明主因是大步變異（見 §4）。
- **指向**: [round-86](round-86-model-ensemble.md) §4 · `script/r81_curve_sm.py --climb-main/--climb-alt/--model-arms`

## 1. 假設 (Propose)

- 平均打分的增益幾乎全來自上下文模型；在 v6a 前段（WM −0.3～−1），CNN 的排序反而稀釋訊號。上下文模型單獨打分應在爬山臂有更高中位。

## 2. 實驗設計 (Design)

| 臂 | 名額 | 打分 |
|---|---|---|
| eclimb（主，基準） | 爬山 5 | (CNN + 上下文) ÷ 2 − 0.5·CNN std |
| **tclimb（測試）** | 爬山 5 | 上下文 − 0.5·CNN std |
| eexploit | 7 | 平均分 LCB（R86 預登記的後續，非本輪測試對象） |
| eexplore | 4 | 平均分 UCB（同上） |
| random | 3 | 均勻 |

- 兩爬山臂各自照親代分名額、同親代內交替取（`pick_climb_pair`）。
- **判準（發車前寫死）**：b107–b109 三批合計，tclimb 實測中位 ≥ eclimb 中位，且最佳 ≥ eclimb 最佳 − 0.1 dB → 成立（爬山改上下文單獨）；中位比 eclimb 差 0.3 dB 以上 → 撤回；其間 → 回報。以 v6a 尺計（中途若公證達標切回 v6，改以 v6 尺並註記）。
- eexploit／eexplore 只做描述（與 R86 的 dexploit／dexplore 比，非同批，不下因果結論）。
- 硬上限 3 批。

## 3. 執行紀錄 (Run)

- `python -m script.r81_curve_sm build --batch N --n 24 --climb-n 10 --delta-climb 10 --climb-main ens --climb-alt ctx --climb-alt-n 5 --model-arms ens --delta-exploit --delta-explore --ctx-shadow --climb-top 10 --max-flips 12 --exploit-lcb 0.5 --parent-score violation --p-cross 0.3 --disconnect --push 0.5 --reseam 0 --config configs/dual_r81_v6a.yaml --out <dir>`
- 入佇列：`python -m script.r81_curve_sm enqueue --input <dir> --name r81v6bN --prio -2 --shards 3 --config configs/dual_r81_v6a.yaml`

## 4. 分析 (Analyze)

### 2026-10-21 07:26：判讀（b107–b109，v6a 尺，72 筆）

| 臂 | n | 最佳 | 中位 | 平均 | > −0.4 | > −1 |
|---|---|---|---|---|---|---|
| **tclimb（上下文單獨）** | 15 | **−0.25** | **−0.38** | −0.39 | 60% | 100% |
| eclimb（平均） | 15 | −0.29 | −0.40 | −0.45 | 47% | 93% |
| eexploit | 21 | −0.33 | −0.42 | −0.67 | 48% | 81% |
| eexplore | 12 | −0.36 | −1.56 | −3.08 | 8% | 50% |
| random | 9 | −0.82 | −3.83 | −4.53 | 0% | 22% |

- 逐批中位 tclimb／eclimb：b107 −0.39／−0.40、b108 −0.38／−0.40、b109 −0.37／−0.35（二勝一負，差距 ≤ 0.05）。
- 判準：中位 −0.38 ≥ −0.40 ✓；最佳 −0.25 ≥ −0.29 − 0.1 ✓ → **成立**，但差距小、近乎持平。
- 準度（72 筆）：ρ CNN +0.40、上下文 +0.48、平均 +0.44；MAE 1.08／1.00／0.94。**前段（> −1.5，57 筆）三者皆近零**（+0.05／+0.16／+0.08）——好設計之間的差異已小於模型解析度。
- **explore 變差的主因是大步變異，不是打分**（b96–b110 共 345 筆子代對親代）：離親代 1–2 格中位 Δ −0.03、改善率 35%；3–4 格 −0.13、39%；5–6 格 −1.90、8%；≥ 10 格 −6 以下。親代 > −0.6 時翻 ≥ 5 格 **29 筆 0 筆改善**。「推移交界」變異在 explore／random 中位翻 6–8 格、Δ −3～−7。

## 5. 結論 (Conclude)

- **決策**：爬山 10 名額全改上下文單獨（`--climb-main ctx`）；exploit 留平均；explore 改回 CNN（`--explore-model cnn`）。b110 為 R87 設定延續批、b111 為新設定首批（皆不計判讀）。下一步 R88：利用／探索臂的小步限制同批對照。


## 6. 後續決策 (Next)

## 7. 歸檔指向 (Archive)
