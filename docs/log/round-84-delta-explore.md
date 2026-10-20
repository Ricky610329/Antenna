# Round 84 — explore 臂改由變化量模型 UCB 排序

- **狀態**: concluded（2026-10-20）
- **提出 / 開跑 / 結論**: 2026-10-20 / 2026-10-20 / 2026-10-20
- **一句話問題**: 原 SM 的 explore 臂（UCB）在 b89–b92 只有 24% > −5；改用變化量模型的 UCB（均值＋std），探索名額能否同時保有探索性與可用率？
- **一句話結論 (TL;DR)**: **成立**——explore 改用變化量模型 UCB 後，b96–b98（v6a 尺）中位 −0.88、> −1 占 67%，是三個模型臂中最好的；原 SM explore（b90–b92）中位 −4.87、7%，random −4.86、0%。形式判準（> −5 ≥ 40%）在 v6a 尺下無鑑別力，以相對比較為主。
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

### 2026-10-20 15:40：目標改為 v6a（−15）——判準注記（任何 R84 結果回來前）

- Ricky 決定先做 −15 版達標。b95（v6 尺）撤下未跑；R84 改為 b96–b98 三批（皆以 v6a 尺選樣）。
- 判準修訂（15:40，R84 尚無任何結果）：dexplore「> −5」改以 **v6a WM** 計（選樣目標已換，用同一把尺才有意義），門檻不變（≥ 40% 成立、< 20% 撤回）；歷史對照同步以 v6a 重算原 SM explore（b89–b92）。

### 2026-10-20 20:23：判讀（b96–b98，v6a 尺）與第三次自問檢查

| 臂 | n | 最佳 | 中位 | > −1 比例 | > −5 比例 |
|---|---|---|---|---|---|
| dclimb | 30 | −0.66 | −1.13 | 40% | 93% |
| dexploit | 21 | −0.61 | −1.02 | 48% | 100% |
| **dexplore** | 12 | −0.63 | **−0.88** | **67%** | 92% |
| random | 9 | −1.76 | −4.86 | 0% | 56% |
| 對照：原 SM explore（b90–b92） | 14 | — | −4.87 | 7% | — |

- 形式判準 > −5 ≥ 40%：92% 通過，但目標改 v6a 後 WM 普遍落在 −1 附近，此門檻已無鑑別力（目標切換時應同步把門檻換成相對指標，教訓記下）。以相對比較判定**成立**。
- 自問檢查：①v6a 最佳 `b99_003` −0.46，前 5 名全卡通帶（30 GHz 點）②選樣當下 ρ（v6a）：TF +0.48／+0.45／+0.42、CNN +0.29／+0.41／+0.60、原 SM −0.17／−0.07／+0.43——TF 與 CNN 互有勝負、MAE 相當 ③random 最差，留作對照 ④無新過時設定；37 HFSS 不穩（b96s3 7 筆 3 錯）待處理 ⑤下一步：兩模型平均打分（R86）。

## 5. 結論 (Conclude)

- **決策**：dexplore 定案。三個模型臂（climb／exploit／explore）全由變化量模型排序；原 SM 只剩 random 外的配額自檢。下一步 R86：CNN＋上下文模型平均打分。


## 6. 後續決策 (Next)

## 7. 歸檔指向 (Archive)
