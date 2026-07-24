# Round 40 — 換裝與空洞輪：two 主通道換裝 × V 臂 response 空洞反演首航

- **狀態**: proposed（2026-07-25 晨開輪;自主續輪宣告制;R39 收輪接棒）
- **提出 / 開跑 / 結論**: 2026-07-25 / — / —
- **一句話問題**: 排序實權全面交給 two（cnn2）後管線是否整體受益？＋朝 response 空間
  「沒人去過的區域」反演生成,能不能真的開出新資料聚落（多樣性警報的結構解）？
- **指向**: [round-39](round-39-left-family.md)（孤點警訊/two 五批連勝/F 臂撤）·
  decisions「資料擴展主軸=response 空間」（2026-07-25）· assets/round-39/data_map.png

## 1. 假設 (Propose)
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **two 換裝主通道**：select 打分/絕對值/LCB 全換 sm_two;MLP 降 ens 成員（ens 混口徑保留）。
    判準=①凍結尺不劣化（two 凍結 ≤0.9 帶）②批前瞻 ρ 維持 +0.7 帶③**G 臂 adv 率下降**
    （mlp 時代 53-100% → 目標 <30%）。任一批 ρ<0.4=回退 MLP 並記反例。
  - **V 臂空洞反演首航（8 席）**：同鍋 response PCA 稀疏區質心 K=4 當反演目標
    （每目標 2 席:1 近解 1 遠解）。判準=**實測 response 投影回 PCA 後,V 臂樣本對既有雲
    最近鄰距離中位 > 全批中位**＝真的開了新區（資訊增益軸,不指望三標）;連兩批開不出=收案。
  - **F 臂撤**（R39 孤點結論）;家族擴張主力=鏈線（c6tri8 續飛）。
  - **佇列原子化先行**：jobs-add 寫 jobs.json 加鎖檔（O_EXCL）+回歸測試,發車前落地
    （R38 兩起並發壞檔欠帳）。
  - 多樣性恆溫:D/W 10/10;紀錄照公證鐵則（usable_oob 7.78/usable_lo −2.63/0.5 格制）。
- **配額（批 62=2 夾）**：G 12/I 12/**V 8**/M 5/O 3/K 2/D 10/W 10。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| two 換裝 | 打分/絕對值/LCB 全換 two,MLP→ens | 凍結≤0.9∧ρ≥0.7∧adv<30%;ρ<0.4 回退 |
| V 臂 | response PCA 空洞質心 K=4 反演 ×8 席 | 實測投影 NN 距離>批中位=開新區;兩批空=收案 |
| 佇列原子化 | jobs-add 鎖檔+回歸測試 | 併發 jobs-add 壓測不壞檔 |
| 鏈線 | c6tri8 續（錨 −0.05,lo −3.64 鄰域） | tri 合格即公證;usable_lo ≤−3.13=推 1 格 |

## 3. 執行紀錄 (Run)
```
# v66 重錨（response 模式;含 two/lohead）:
python -m script.sm_reanchor train --add "dedust_r39b3a,dedust_r39b3b" --out sm_reanchor66.pth --ds-mode response
python -m script.sm_reanchor train-two --out sm_reanchor66.pth
# 批線（seed 110+N）:
python -m script.sm_invert gen --sm sm_reanchor66.pth --rad-head rad_head66.pth --out-dir tmp/invert_stage_r40bN --n-free 6 --n-surg 0 --n-champ 0 --n-oob 6 --seed <110+N>
python -m script.dedust select-r40 --batch N --sm sm_reanchor66.pth --gstage tmp/invert_stage_r40bN --rad-head rad_head66.pth --novelty
# check-dup ×2 → jobs-add ×2 prio 3 → watch
```
| 批 | 狀態 |
|---|---|
| — | （開輪;佇列原子化+select-r40/V 臂實作中） |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- t07 觸發檢查（radhead2 讀鄰域 rad 梯度）;獨立艙凍結續;ens 換代（cnn2 成員）候選。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r40b*`;公證 `r40n*`;鏈帳 docs/chains/c6tri8.jsonl。
