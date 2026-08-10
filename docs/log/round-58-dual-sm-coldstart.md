# Round 58 — dual SM 冷啟動輪:排序器上線+對稱子空間引導搜尋

- **狀態**: proposed(2026-08-11 02:44 開檔;SM 施工先行,發車=SM 品質閘過後)
- **提出 / 開跑 / 結論**: 2026-08-11 / — / —
- **一句話問題**: 拿 harvest 10,023+R57 302 筆訓出 dual SM(3 通道),它引導的選批能不能
  勝過 R57 的盲選(鄰域/隨機),並驗「對稱⇒阻帶優勢」押注。
- **指向**: [round-57](round-57-dual-maiden.md)·[analysis-12](analysis-12-harvest-dual-audit.md)·
  docs/records_dual.json(王 −6.04/m4 +1.93)·decisions「Dual 線開局三原則」

## 1. 假設 (Propose)

**假設**:①dual 響應可由 SM 學到可用排序訊號(harvest 一萬筆離線即可冷啟動,不必等線上);
②SM 引導選批 > 盲選;③對稱子空間+SM=推進 wm_dual 與 m4 軸的最短路。

- **判準(發車前寫死,2026-08-11 02:44)**:
  1. **SM 品質閘(不過不發車)**:held-out(10%,分層抽樣)上
     (a) wm_dual(pred) 對 wm_dual(true) spearman ρ ≥ **+0.4**;
     (b) **top-K 命中率**(pred top30 ∩ true top10%)顯著 > 隨機基準 3(超幾何 P<0.05)
     ——雙閘,(b) 為主((analysis-10 教訓:驗收指標=使用方式的指標;ρ 只當健檢)。
     閘不過 → 回報+換架構,不硬發。
  2. **b1=SM 引導批(含對照臂,100 筆)**:候選池=對稱隨機 ~5k+王/紀錄鄰域 ~5k(零 HFSS 生成),
     g 臂=SM 排序 top 50、c 臂=同池均勻抽 50(對照)。判定:g 臂 wm_dual 中位 > c 臂中位
     (mann-whitney P<0.05)=**SM 引導有效**;g 臂 best > **−6.04**(records_dual)=王推進(公證 ×2)。
  3. **押注結帳「對稱⇒阻帶優勢」**:先用 R57 既有資料回算(symr 30 vs r 臂 20 的 m4 分佈),
     b1 對稱樣本續驗;成立判準=對稱組 m4 中位優勢 ≥ 2 dB 且方向跨批一致。
  4. 主指標=records_dual 王(現 −6.04);**停止線=連 3 批王零推進 → 回報 Ricky**。
  5. 紀錄級一律公證 ×2(/notarize dual 口徑=同機 bit 級+報 records_dual);**≤3 批必收輪**。
  6. 線上學習路徑維持延後(Ricky 2026-08-10 裁定);SM 只做批次線排序器。
- 修訂紀律:結果回來前+日期註記。

## 2. 實驗設計 (Design)

| 項 | 設計 | 判準 |
|---|---|---|
| 施工 | `script/sm_dual.py`(train/eval/rank;3×17 響應頭;harvest+R57 鍋;held-out 分層) | pytest 綠+品質閘§1-1 |
| b1 | g 臂 50(SM top)+c 臂 50(同池對照) | §1-2 |
| b2/b3 | 視 b1:引導有效 → 加碼+王鄰域精修;無效 → 特徵/架構迭代一次再判 | §1-2/§1-4 |

- 候選池生成決定性(seed);對稱樣本=上半鏡下(R57 symr 同法);饋墊必保;check-dup 分域照舊。
- 命名:夾 `dedust_r58b<批><a/b/c>`、id `d58b<批>_<臂>_<序>`、公證 `r58n*`。

## 3. 執行紀錄 (Run)

```
# 施工(Opus agent):script/sm_dual.py — train(鍋=harvest_dual+dedust_r57* 全店,10% held-out)
#   → eval(印 §1-1 雙閘數字) → rank(候選池 npz → top-N id 表)
# 開發機節流:單例、訓練段低優先權、GPU 可用則用
# 發車(品質閘過後):
#   python -m script.sm_dual rank --pool <池> --top 50 → 生 dedust_r58b1_input(g+c 兩臂)
#   check-dup --input dedust_r58b1a_input(×3 單獨跑) → jobs-add ×3 --config configs/dual_r1_eval.yaml --prio 3
```

| 事件 | 狀態 |
|---|---|
| 開檔 | 2026-08-11 02:44(SM 施工發包) |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- analyze batch/gain-check dual 分支工具化 [M];sm_reanchor port 參數化評估(vs 獨立 sm_dual.py)[L]。
- m5/m6 入 min 重議觸發:對稱樣本 m5/m6 出現 ≥0(R57 §6 延續)。

## 7. 歸檔指向 (Archive)
（待;結果夾 `dedust_r58*`;紀錄=docs/records_dual.json)
