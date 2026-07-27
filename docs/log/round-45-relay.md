# Round 45 — 接力輪：文法深左個體 × 組級鏈爬升（深水右爬）

- **狀態**: running（2026-07-27 晚開輪;自主續輪宣告制;R44 收輪接棒）
- **提出 / 開跑 / 結論**: 2026-07-27 / 2026-07-27 / —
- **一句話問題**: 「深水右爬」路線可行嗎——文法/探索臂產出的深左個體（lo≤−4）,
  組級鏈能不能把 wm 從 −8 帶爬回作戰區（≥−1）甚至合格？
- **指向**: [round-44](round-44-grammar-2.md)（文法六批判定/苗子斷層結構性）·
  assets/round-43/pareto.png（左側群島存量）· decisions「組級變異算子定案」

## 1. 假設 (Propose)
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **雙鏈接力（tier 0,--mutator group,goal tri,dry 2,max 20 包）**：
    c45g1=錨 g43b1_004_free_rand（wm−8.04∧rad−0.94∧lo−6.18,均衡深水）;
    c45d1=錨 d44b2_001_denovo（wm−9.07∧oob−0.73 全負帶外族∧lo−4.67,極端實驗）。
    **階梯判準**:任一鏈 wm 爬升 ≥2（−8→−6 帶）=路線活;爬進作戰區（≥−1）=重大成果;
    兩鏈 dry2 於 <1 爬升=「深水右爬」首試負結果記帳。
  - 批線（≤3 批,select-r45）:D 臂槽收斂 old 6/GDd 4（R44 判定）;其餘照 r44;V 臂常駐;
    紀錄照公證鐵則。
- **配額**：批 60×≤3（seed 160+N）;鏈 25/包。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| c45g1 | 均衡深水錨組級爬 | wm +2=活;作戰區=重大;dry2<+1=負 |
| c45d1 | 全負帶外族右爬 | 同上;oob 保持 <2 觀察 |
| 批線 | old6/GDd4 收斂配置 | 五軸;左側教材產率 |

## 3. 執行紀錄 (Run)
```
# v80 重錨:
python -m script.sm_reanchor train --add "dedust_r44b3a,dedust_r44b3b" --out sm_reanchor80.pth --ds-mode response
python -m script.sm_reanchor train-two --out sm_reanchor80.pth
# 鏈:
python -m script.dedust chain --name c45g1 --anchor g43b1_004_free_rand --source-input dedust_r43b1a_input --goal tri --anchor-score -8.19 --mutator group --n 25 --prio 1
python -m script.dedust chain --name c45d1 --anchor d44b2_001_denovo --source-input dedust_r44b2b_input --goal tri --anchor-score -9.22 --mutator group --n 25 --prio 1
# 批線（seed 160+N）: select-r45 照常
```
| 批/包 | 狀態 |
|---|---|
| 鏈線 | **c45d1 三包 +2.07（−9.22→−7.15）=「路線活」判準達標**（p03 手動代判,daemon 誤關零損失）;c45d2 接棒 dry2 收鏈（高原 −7.15）;c45g1 三包 +0.47 且 **rad −0.94→0.00 修復完成**、lo −4.49 深水保持;c45g2 接棒續爬 |
| b1 | ✅ 判讀完（07-28 01:4x,60 筆）:two 2 尺勝（2.45/ρ+0.757）;lohead ρ+0.787;三標 5/合格 3;紀錄零推進;多樣性正常 |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 獨立艙凍結續;GNN bakeoff 觸發線;鏡射 rad 修復旋鈕候選。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r45b*`、`dedust_c45{g1,d1}_p*`;鏈帳 docs/chains/。
