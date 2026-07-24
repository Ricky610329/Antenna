# Round 39 — 左側家族化輪：F 臂鄰域變異 × two 絕對值通道 × lo 判別器進鍵

- **狀態**: running（2026-07-24 午後開輪;自主續輪宣告制;R38 里程碑〔左側合格解首例公證〕接棒）
- **提出 / 開跑 / 結論**: 2026-07-24 / 2026-07-24 / —
- **一句話問題**: 左側合格解首例（c8trip03_01）是**孤點還是一族**？——鄰域變異（R11 穩健化
  方法論）能不能產出 ≥5 筆合格變異體證明「族」成立,並讓 usable_lo/usable_oob 繼續壓？
- **指向**: [round-38](round-38-shadow-two.md)（里程碑/轉正）· records.json（7.78/−2.63）·
  MILESTONES 第八章

## 1. 假設 (Propose)
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **F 臂家族錨組 16 席**（錨=c8trip03_01〔首例〕/c10trip02_07〔oob 7.12〕/c6tri5p06_21/
    s38s1_18〔oob 6.82〕;d 1-25 梯度;半 ref 半 rej 續帳;diagb 方向性續）:
    **三批合格變異體累計 ≥5 = 「左側合格族」成立**（公證抽驗 1 筆);<2 = 孤點警訊回報。
  - **two 絕對值通道換裝**：F 臂打分已用 two;主 select 通道（pred_wm/LCB 基準）R39 期評估後換
    （ens 仍 MLP 家族=混口徑誠實註記,R40 ens 換代)。
  - **lo 判別器進鍵第二讀**：批前瞻 ρ≥0.5（b1/b2 任一批 + R38b3 的 0.756=連兩批）→ 進鍵
    （select 罰分或 F 臂 gate）。
  - 紀錄:usable_lo 0.5/格公證;usable_oob 續壓（現 7.78）;紀錄級一律公證。
  - selfgen 換種首讀（R38 欠帳）:b1 判讀時算 auto store 新增樣本王朝表型佔比。
  - tri 鏈群（c8/c10/c6tri5）續=家族挖掘機;批 ≤3;五軸面板;修訂註記。
- **配額（批 58=2 夾）**：G 12/F 16（家族）/I 8/M 5/O 3/K 2/D 6/W 6。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| F 臂家族 | 四錨 d1-25 變異×半ref半rej | 三批合格 ≥5=族成立 |
| two 換裝 | F 臂即用;主通道評估 | 凍結尺+批前瞻不劣化 |
| lo 進鍵 | 第二讀 ρ≥0.5 | 連兩批→進鍵 |
| selfgen 首讀 | b1 判讀時算 | 王朝佔比 <40% 目標 |

## 3. 執行紀錄 (Run)
```
# v63 輕量重錨（response）:
python -m script.sm_reanchor train --add "dedust_r38b3a,dedust_r38b3b" --out sm_reanchor63.pth --no-ens --no-shadow --ds-mode response
# train-two v63（F 臂打分用）:
python -m script.sm_reanchor train-two --out sm_reanchor63.pth
# 批線（seed 100+N）:
python -m script.sm_invert gen --sm sm_reanchor63.pth --rad-head rad_head63.pth --out-dir tmp/invert_stage_r39bN --n-free 6 --n-surg 0 --n-champ 0 --n-oob 6 --seed <100+N>
python -m script.dedust select-r39 --batch N --sm sm_reanchor63.pth --gstage tmp/invert_stage_r39bN --rad-head rad_head63.pth --novelty
# check-dup ×2 → jobs-add ×2 prio 3 → watch
```
| 批 | 狀態 |
|---|---|
| — | （開輪;v63 重錨中） |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- t07 觸發檢查（radhead2 讀其鄰域 rad 梯度）;獨立艙凍結續;R40 ens 換代候選。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r39b*`;公證 `r39n*`;鏈帳 docs/chains/。
