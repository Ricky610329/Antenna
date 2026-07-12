---
name: notarize
description: 紀錄公證全鏈——候選重測 ×2 收檔後的判定、換王記帳（records.json＋champions.md＋對比圖＋memory）或假象攔截記帳（用法：/notarize <候選id> <公證store名>）
---

# 公證判定 $ARGUMENTS

> 鐵則：**紀錄級宣稱一律公證後才算數**。歷史假象案例（都被本流程攔下）：w17 +0.48→−0.06、
> b20 +0.32→−0.19、h6_010 +0.43→+0.35。發車部分（select-repeat ×2 → prio 2）由
> `/batch-cycle` step② 完成；本 skill 從「公證 store 收檔」開始。

## 1. 讀結果
```
python -m script.dedust report --input dedust_r<R>n<X>_input --store dedust_r<R>n<X>
```
或直接讀 `results.json`（兩筆 r00_rep/r01_rep 的 wm/rad_margin/oob_bad）。

## 2. 判定表（機械規則）

| 條件 | 判定 | 動作 |
|---|---|---|
| 兩重測與原測**三次一致**（挑戰指標差 ≤0.03） | **公證過** | 走第 3 步換王記帳 |
| 任一重測掉出 0.03（如 +0.43→+0.35） | **假象攔截** | 走第 4 步假象記帳 |
| 兩重測彼此一致但與原測差 >0.03（雙穩態,如 vg0258） | 特殊：取**多數值**定案 | 記 round §4＋scratch，通常=不換王 |

判定用「挑戰的那個指標」（wm 候選看 wm、rad 候選看 rad_margin、帶外看 oob_bad）；
其餘指標順帶核對三標資格（wm≥0＝過線、rad≥0）。

## 3. 公證過 → 換王記帳（順序固定）
1. **records.json 先改**（`docs/records.json`：對應鍵的 id/value/certified；`updated` 換今天——先 `date` 查時間）。
2. 渲染對比圖＋目檢：
```
python -m script.figs.champ_compare --new <新王id> --old <前任id> --out docs/log/assets/round-<R>/newking_<軸>_<id>.png --title "..."
```
   產出後 **Read 目檢**（重疊/裁切/豆腐字）再嵌 round 檔。
3. **champions.md 散文跟上**（頭銜表該列＋榜插行＋前任降級註記）。
4. round 檔 §4 加公證判定表（原測/重測×2/判定一行）＋嵌圖。
5. memory：`project_w17_champion.md` 對應行刷新＋MEMORY.md 索引行。
6. commit（`docs(champions): ★ <軸>王易主 <id> <值>(公證3/3)`）＋push＋**PushNotification**（換王=紀錄級,要推）。

## 4. 假象攔截 → 記帳（不動名鑑）
1. round 檔 §4 一行：「✗ <id> 原測 X → 重測 Y，假象攔截第 N 例」。
2. records.json **不動**；champions.md **不動**。
3. 若同機制先前出過真例（如 h6/h7 對照），把「有真有假」寫進 §4——這是量測敏感性的科學資料。
4. commit。不推播。

## 注意
- 公證 store 的 kind=repeat＝check-dup 豁免（蓄意重複），不要對它跑 check-dup 報重複。
- 「可用帶外」紀錄（wm≥buffer∧rad≥0 的 min oob）同樣走本流程；它的鍵是 records.json 的 `usable_oob`。
- 帶內紀錄（`inband`）持有者是非三標參考點——換它**不算換王**，champions.md 只動 🎯 那一列。
