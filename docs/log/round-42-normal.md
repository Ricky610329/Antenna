# Round 42 — 常態輪：批線續航 × 鏈線雙目標（攻深/清潔收復）

- **狀態**: running（2026-07-26 晨開輪;自主續輪宣告制;R41 收輪接棒,Ricky「R42 收測試回常態」）
- **提出 / 開跑 / 結論**: 2026-07-26 / 2026-07-26 / —
- **一句話問題**: 常態管線（two 主通道+V 臂+誤差錨迴圈）持續運轉下,鏈線雙目標
  （c41grp2 攻深/c41grp3 清潔收復）能不能把「第四筆合格解公證」或「展示級合格解」收下來？
- **指向**: [round-41](round-41-group-test.md)（組算子定案/刀鋒解/fill4 臨門）· decisions
  「組級變異算子定案」（2026-07-26）

## 1. 假設 (Propose)
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - 批線（≤3 批,select-r42=r41 配置）:V 臂常駐續讀;紀錄照公證鐵則（usable_lo −3.46/0.5 格
    =≤−3.96;usable_oob 7.78）。
  - 鏈線（跨輪續飛）:c41grp2 攻深（錨 p06_11 +0.03）;c41grp3 清潔收復（錨 fill4 −0.04,
    **合格∧diagb≤12=展示級候選→公證＋對照現王**）。
  - c41grp2p06_11（第四筆合格解,單次,餘裕厚）:**鏈收檔時若仍為鏈 best→公證**（非紀錄級,
    但家族帳/展示級底子值得 3/3 定錨）。
  - 組算子批線推廣:A/B 帳攢 ≥6 包後判（本輪不動批線臂）。
  - 鏡射抽驗（tier2 閒時,3-5 對）:Δwm 分布;系統性 >0.3 → train-two 增強降權另案。
- **配額**：批 61×≤3（G12/I12/V8/M5/O3/K2/D10/W10;seed 130+N）。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| 批線常態 | two 主通道+V+誤差錨 | 五軸面板;V 續讀 |
| c41grp2 | 攻深（tri,group） | 合格勝錨→公證評估;dry2 收 |
| c41grp3 | 清潔收復（tri,group,錨 diagb12） | 合格∧diagb≤12=展示級→公證 |
| 鏡射抽驗 | tier2 3-5 對 | Δwm 系統性>0.3=增強降權案 |

## 3. 執行紀錄 (Run)
```
# v71 重錨:
python -m script.sm_reanchor train --add "dedust_r41b2a,dedust_r41b2b" --out sm_reanchor71.pth --ds-mode response
python -m script.sm_reanchor train-two --out sm_reanchor71.pth
# 批線（seed 130+N）:
python -m script.sm_invert gen --sm sm_reanchor71.pth --rad-head rad_head71.pth --out-dir tmp/invert_stage_r42bN --n-free 6 --n-surg 0 --n-champ 0 --n-oob 6 --seed <130+N>
python -m script.dedust select-r42 --batch N --sm sm_reanchor71.pth --gstage tmp/invert_stage_r42bN --rad-head rad_head71.pth --novelty
# check-dup ×2 → jobs-add ×2 prio 3 → watch
```
| 批/包 | 狀態 |
|---|---|
| — | （開輪;v71 重錨中） |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 獨立艙凍結續（觸發=Ricky 點頭）;ens 換代候選;全負帶外教材（−4.2/−1.87）追蹤。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r42b*`;公證 `r42n*`;鏈帳 docs/chains/c41grp{2,3}.jsonl。
