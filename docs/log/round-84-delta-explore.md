# Round 84 — explore 臂改由變化量模型 UCB 排序

- **狀態**: running（掛在 R81 v6 批次內，b95 起）
- **提出 / 開跑 / 結論**: 2026-10-20 / 2026-10-20 / —
- **一句話問題**: 原 SM 的 explore 臂（UCB）在 b89–b92 只有 24% > −5；改用變化量模型的 UCB（均值＋std），探索名額能否同時保有探索性與可用率？
- **一句話結論 (TL;DR)**: 待分析
- **指向**: [round-83](round-83-delta-exploit.md) §5 · `script/r81_curve_sm.py --delta-explore` · [ONGOING](../../configs/ONGOING.md)

## 1. 假設 (Propose)

- R83 成立後原 SM 只剩 explore 臂在用；它選樣當下 ρ −0.63～+0.10。explore 改用變化量模型 UCB（地板＝有效均值的中位，nan 不選），預期可用率上升且仍保有 std 帶來的探索。
- 同批清理：`--reseam 0`（b89–b92 reseam 0% > −5，屬無產出設定清理，非新變因）。

## 2. 實驗設計 (Design)

| 臂 | 改動 | 對照 |
|---|---|---|
| dexplore（原 explore 名額 4） | 變化量模型 UCB 排序 | 歷史 explore（b89–b92 原 SM：17 筆，> −5 24%、中位 −7.12） |
| 其餘 | dclimb 10、dexploit 7、random 3、`--ctx-shadow` 不變 | — |

- **判準（發車前寫死）**：b95–b97 三批合計，dexplore「> −5」≥ 40% → 成立（定案）；< 20% → 撤回改回原 SM；其間 → 回報。另記錄 dexplore 與 dexploit 的平均 Hamming 距離（父代到候選），確認探索性沒有消失。
- **硬上限**：3 批（b95–b97）。紀錄級設計照 R81 公證鐵則。

## 3. 執行紀錄 (Run)

- 建批：`python -m script.r81_curve_sm build --batch N --n 24 --climb-n 10 --delta-climb 10 --delta-exploit --delta-explore --ctx-shadow --climb-top 10 --max-flips 12 --exploit-lcb 0.5 --parent-score violation --p-cross 0.3 --disconnect --push 0.5 --reseam 0 --config configs/dual_r81_v6.yaml --out <dir>`
  入佇列：`python -m script.r81_curve_sm enqueue --input <dir> --name r81v6bNN --prio -2 --shards 3 --config configs/dual_r81_v6.yaml`

## 4. 分析 (Analyze)

## 5. 結論 (Conclude)

## 6. 後續決策 (Next)

## 7. 歸檔指向 (Archive)
