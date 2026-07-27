# Round 44 — 文法二輪：GA2 組義槽進場 × GDd 對角深化

- **狀態**: running（2026-07-27 晨開輪;自主續輪宣告制;R43 收輪接棒）
- **提出 / 開跑 / 結論**: 2026-07-27 / 2026-07-27 / —
- **一句話問題**: 組義槽採樣（GA2,語義正確版）能不能把文法的苗子率從零帶起來？
  ＋GDd 對角機制在更多樣本下左傾持不持續？
- **指向**: [round-43](round-43-grammar-maiden.md)（對角=左側門票方向性定案/GC 汰）·
  decisions「組文法生成系統」· scratch「組義字典」

## 1. 假設 (Propose)
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **D 臂 10 席=文法槽二版**：old 2/GA 2/GB 2/**GA2 2**/GD 1/GDd 1（GC 汰,R43 §5）。
  - 四尺跨輪續攢（R43 帳累積;每槽 ≥6 總樣本後可下汰換判）;苗子率（wm≥−3）=GA2 的主看點
    （組義槽=語義正確,若仍零苗=斷層在更深處）。
  - 批線其餘照 r43 配置;紀錄照公證鐵則;V 臂常駐。
  - 鏈線:視 b1 判讀擇錨（左側群島 wm −0.5~0∧lo≤−4 帶,pareto 圖提示;--mutator group）。
- **配額**：批 60×≤3（G12/I12/V8/M5/O3/K2/D10/W10;seed 150+N）。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| GA2 首航 | 組義槽採樣 ×2 席 | 苗子率>0=語義有效;仍零=斷層更深 |
| GDd 續讀 | 對角塊鏈 ×1 | lo 左傾持續性（R43 帳續） |
| 批線常態 | two+V+誤差錨 | 五軸面板 |

## 3. 執行紀錄 (Run)
```
# v77 重錨:
python -m script.sm_reanchor train --add "dedust_r43b3a,dedust_r43b3b" --out sm_reanchor77.pth --ds-mode response
python -m script.sm_reanchor train-two --out sm_reanchor77.pth
# 批線（seed 150+N）:
python -m script.sm_invert gen --sm sm_reanchor77.pth --rad-head rad_head77.pth --out-dir tmp/invert_stage_r44bN --n-free 6 --n-surg 0 --n-champ 0 --n-oob 6 --seed <150+N>
python -m script.dedust select-r44 --batch N --sm sm_reanchor77.pth --gstage tmp/invert_stage_r44bN --rad-head rad_head77.pth --novelty
# check-dup ×2 → jobs-add ×2 prio 3 → watch
```
| 批/包 | 狀態 |
|---|---|
| b1 | ✅ 判讀完（07-27 09:4x,60 筆）:**GA2 首航 n=2:苗子 0、lo +1.49 右傾（同 GA 系）——組義槽第一讀未帶起苗子**;GDd 第四讀續左（−3.63,4/4）;GD 單樣本 −8.32（雜訊帶）;old 續左 −4.64;多樣性正常;紀錄零推進 |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 獨立艙凍結續;GNN bakeoff 觸發線（pot ~30k 或 two 凍結連 5 版零斜率）;鏡射=rad 修復旋鈕候選。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r44b*`;公證 `r44n*`。
