# Round 46 — 接力二輪：c45g2 衝作戰區 × 批線常態

- **狀態**: running（2026-07-28 午後開輪;自主續輪宣告制;R45 收輪接棒）
- **提出 / 開跑 / 結論**: 2026-07-28 / 2026-07-28 / —
- **一句話問題**: c45g2（wm −3.34∧lo −2.31,g 線 +4.70 續飛）能不能爬進作戰區（wm≥−1）
  甚至合格——完成「深水右爬」全程首例？
- **指向**: [round-45](round-45-relay.md)（接力大成立/d 線教訓=起點均衡度）· docs/chains/c45g2.jsonl

## 1. 假設 (Propose)
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **c45g2 跨輪續爬**（tri,group,max 20 包內）:進作戰區（wm≥−1）=重大成果;
    合格（wm≥0.15∧rad≥0∧lo≤−2）=**里程碑級,照公證鐵則**（第一個全程「隨機原礦→合格」
    的非王朝非嫁接血統）;dry2=記終點。rad −0.81=綁束軸,tri 鍵自動逼修。
  - 批線（≤3 批,select-r46=r45 配置）:V 臂常駐;紀錄照公證鐵則。
  - 新原礦擇錨:R46 批產出中 lo≤−4∧rad≥−1（均衡型,d 線教訓）候選記名單,R47 用。
- **配額**：批 60×≤3（seed 170+N）;鏈 25/包。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| c45g2 | 續爬（rad 綁束段） | 作戰區=重大;合格=里程碑公證;dry2 記終點 |
| 批線 | 常態（old6/GDd4） | 五軸;均衡型原礦名單 |

## 3. 執行紀錄 (Run)
```
# v83 重錨:
python -m script.sm_reanchor train --add "dedust_r45b3a,dedust_r45b3b" --out sm_reanchor83.pth --ds-mode response
python -m script.sm_reanchor train-two --out sm_reanchor83.pth
# 批線（seed 170+N）: select-r46 照常;check-dup ×2 → jobs-add ×2 prio 3 → watch
# ⚠ select-r46 之前必跑 G 臂 staging（b3 首發漏此步 FileNotFoundError 教訓,07-29）:
#   python -m script.sm_invert gen --sm <vNN> --rad-head <rad_headNN> --n-free 6 --n-surg 0 \
#     --n-champ 0 --n-oob 6 --seed 46<批號> --out-dir tmp/invert_stage_r46b<N>
#   → select-r46 --gstage tmp/invert_stage_r46b<N>（不能用預設 tmp/invert_stage）
```
| 批/包 | 狀態 |
|---|---|
| 鏈線 | **c45g2 p12 跨苗子線（−2.99,wm −2.84;g 線總帳 +5.20）**,p13/p14 dry2 收鏈（14 包）;c45g3 接棒續衝作戰區 |
| 鏈線2 | c45g3 4 包 dry2 收鏈（07-28 晚）:終錨 **c45g3p02_06 −2.87**（wm −2.72∧rad −0.28∧lo −2.18;p01 −2.93→p02 −2.87,p03/p04 未翻）;**g 系總帳 +5.32**（−8.04→−2.72）;c45g4 接棒發車（同錨續爬,判準沿 §1） |
| 鏈線3 | **c45g4 2 包 dry2 收鏈（07-29 00:5x）——g 系全線終點**:p01 −2.98/p02 −2.87 平錨不勝;終錨維持 c45g3p02_06（wm −2.72）;g 系四段接力（g1→g4,g2–g4 計 20 包）總帳 **+5.32**,止於苗子帶、**未進作戰區（wm≥−1）**;照 §1 判準記終點。⚠ 高原判定條件①成立（scratch 07-28 三條件）——收輪結論處理 |
| b1 | ✅ 判讀完（07-28 17:5x,60 筆）:two ρ+0.805（誤差尺輸 mlp 1.78/1.59 單批）;lohead ρ+0.801;三標 10/合格 5;o46b1_000 wm+0.41∧rad+0.29 厚餘裕;紀錄零推進;多樣性正常 |
| v85 | 重錨完成（07-29 05:4x 全家族）:held-out 中位 **1.227 新低**（遠區 1.72→1.53=深水資料紅利）;⚠ 凍結遠 2.50→3.01/rad ρ 0.334→0.251（<0.4 續不進鍵）;ens85×2 held-out 0.909;two85 凍結 0.811;lohead85 ρ+0.826;訓練集 151k（反權重跳檔=無傷準度,見 scratch 07-28 監測條）;**b3 末批發車**（select-r46 --batch 3 --sm v85→check-dup→prio 3→watch,判讀後照 3 批上限收輪） |
| b2 | ✅ 判讀完（07-28 23:0x,60 筆零 error）:two 誤差 1.37/ρ+0.843 **雙尺贏**（b1 誤差輸→轉正計數重起,b3 再驗）;lohead ρ+0.831（連兩批 ≥0.8）;三標 9/合格 8（I 臂 5/12 強勢,oob 前緣 9.95 未破 7.78）;紀錄零推進;近王 3%/無親新血 27%/對全史最近鄰 28;作戰區 diagb 中位 0;v85 重錨發動 |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- GNN bakeoff 觸發線（pot ~30k,現 22.4k）;獨立艙凍結續;鏡射 rad 旋鈕候選。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r46b*`;鏈帳 docs/chains/c45g2.jsonl。
