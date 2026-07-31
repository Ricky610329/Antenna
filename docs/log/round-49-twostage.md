# Round 49 — 兩段式制度化輪：攻堅可複製性 × lo 進鍵首航 × 批線常態

- **狀態**: running（2026-07-31 09:1x 開輪;自主續輪宣告制;R48 收輪接棒）
- **提出 / 開跑 / 結論**: 2026-07-31 / 2026-07-31 / —
- **一句話問題**: c48nq1 的 +0.94 爬升是**方法**還是**個案**——雙閘選錨+兩段式攻堅能不能在第二條鏈複製？
- **指向**: [round-48](round-48-graft.md)（兩段式首王/雙空間雙閘/lohead 三連過線）· decisions「型態體系軸」
  （R50 預告,本輪收輪=其觸發條件之一）· assets/round-48/lineage_pca_dual.png（量尺）

## 1. 假設 (Propose)
- **背景**：R48 產出三工具——兩段式攻堅（嫁接體當錨+爬山→margin 王 +0.79）、lo 判別器（三連 ρ>0.5）、
  雙閘名單（pattern d_dyn＋response 對簇相關）。本輪=制度化驗證,並收正片線開放假設。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  1. **雙閘重掃**（零 HFSS）:近標帶 wm∈[−1,0.15) 全史,lineage_pca 口徑
     （pattern 對王簇與 c48nq1 簇 d>20 ∧ response 對兩簇 corr<0.9）→ 錨名單 top3;
     第 1 名須過 chain 發車閘（wm>−2）。
  2. **c49nq1 攻堅鏈**（第二條兩段式）:goal=wm、px d1、25 席/包 prio 1、dry2 終點;
     **三包內 best 推進 ≥+0.3=兩段式可複製**;dry2 快死=「c48nq1 是個案」警訊記負結果;
     合格照公證鐵則;紀錄門檻引 `docs/records.json`（不抄死數字）。
     ★ 修訂（2026-07-31 10:2x,**c49nq1 零結果回收前**）:雙閘重掃出的首名錨=**c41grp2p07_19
     （wm+0.14∧rad+0.10∧lo−2.66,雙閘最遠 corr 0.71/0.73,c41grp2=usable_lo 王族）**——左側個體,
     goal 由 wm 改 **tri**（lo≤−2 內爬 min(wm−0.15,rad)=左側合格解會師鍵的原設計用途;wm 鍵會把 lo 換掉）。
     可複製判準等價改述:三包內 tri score 推進 ≥+0.15（錨 −0.01 起跳;過 0=左側第 5 筆合格解,照公證）。
     副產:此錨同時兌現「左側兩段式」（原不發車項的合規版——它過發車閘）。
  3. **批線 ≤3 批**（select-r49=r48 配置,v92 配套顯式當版,seed 20260806）:五軸照常;
     停止線=可用帶外連三批零推進→/stall-protocol。**≤3 批必收輪**。
  4. **lo 進鍵首航**（先決已成立:lohead ρ 0.710/0.797/0.779 三連 >0.5）:補池線
     `select-r21harvest --lo 6`;存活判準=**兩批內 L 臂 ≥1 筆 lo≤−1（中繼帶）**;全空=退鍵+負結果。
  5. 紀錄級一律公證（/notarize）;判準修訂只能在結果回來前+日期註記。
  6. ★ 追加實驗（2026-07-31 14:0x,c49nq1 收鏈**後**新立項,非修訂——c49nq1 判定照原判準記負結果）:
     **c49nq2 裁決鏈**=同錨 c41grp2p07_19、`--mutator group`（組級 70/30）、tri、dry2——
     分離兩個競合解釋:①兩段式不可複製（c48nq1=個案）②**算子不對症**（c41grp2 族紀錄全出自
     grp 算子,左側碎片語言對 px d1 斜率低=R36 老結論;c49nq1 用 px d1=形制照抄 c48nq1 但錨型不同）。
     判準:兩包內 tri 勝錨（>−0.01）→ 解釋②成立（左側族鏈=組級算子對症）;再快死 → 解釋①強化。
- **不發車項（誠實記錄）**：左側兩段式（g48graft1 A 式 lo −2.32 個體當錨）——**被 chain 發車閘擋下
  （wm ≈−9 < −2 組級包閘）**,不硬闖護欄;待左側中繼帶錨（wm>−2∧lo≤−2 苗子）出現再開。
- **c48nq1 p06+ 續爬記本輪**（錨 +0.79=現任王;dry2 端點公證）。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| 雙閘重掃 | 全史近標帶 lineage_pca 雙閘 | 錨名單 top3;首名過發車閘 |
| c49nq1 | 25 席/包×dry2,prio 1 | 三包 ≥+0.3=可複製;快死=個案警訊 |
| 批線 | select-r49（=r48 配置） | 五軸;I 臂強勢續觀察 |
| lo 進鍵 | 補池 --lo 6 | 兩批 ≥1 筆 lo≤−1;全空退鍵 |

## 3. 執行紀錄 (Run)
```
# v92 重錨（b3+c48nq1 鏈 125 筆入鍋）:
python -m script.sm_reanchor train --add "dedust_r48b3a,dedust_r48b3b,dedust_c48nq1_p01,dedust_c48nq1_p02,dedust_c48nq1_p03,dedust_c48nq1_p04,dedust_c48nq1_p05" --epochs 30 --out sm_reanchor92.pth
python -m script.sm_reanchor train-two --epochs 30 --out sm_reanchor92.pth
# 批線（staging 前置;seed 49<批號>）:
python -m script.sm_invert gen --sm sm_reanchor92.pth --rad-head rad_head92.pth --n-free 6 --n-surg 0 --n-champ 0 --n-oob 6 --oversample 6 --seed 491 --out-dir tmp/invert_stage_r49b1
python -m script.dedust select-r49 --batch 1 --sm sm_reanchor92.pth --rad-head rad_head92.pth --gstage tmp/invert_stage_r49b1
python -m script.dedust check-dup --input dedust_r49b1a_input   # ×2 夾;exit 1 不發車
python -m script.dedust jobs-add --input dedust_r49b1a_input --store dedust_r49b1a --prio 3   # ×2 夾
python -m script.dedust watch --stores dedust_r49b1a,dedust_r49b1b
# 補池（lo 進鍵首航;池 <48 觸發）: python -m script.dedust select-r21harvest --tag r49g1 --lo 6 --o 0 --wild 0 --shards 3
# 攻堅鏈（雙閘重掃出錨後）: chain 命令照 c48nq1 形制,id 前綴 c49nq1
```
| 批/包 | 狀態 |
|---|---|
| c49nq2 | **拒發車（07-31 14:0x,護欄攔截）**:「近王帶(錨 wm+0.14>−2)組級變異包實證低效」= analysis-07 發車閘照設計工作——裁決鏈取消,不硬闖;兩解釋(個案 vs 算子不對症)的實驗分離**懸置**,c49nq1 負結果維持原判;R41 grp_grow 近帶成功例 vs analysis-07 統計的張力記 scratch |
| c49nq1★ | **收鏈（07-31 13:5x,dry 2/2 快死）**:2 包 50 筆,終錨未被超越（tri −0.01;p01 best −0.16/p02 −0.16）→ **判準②負結果分支成立（此錨此算子未複製）**;副產=深左苗子 4 筆（lo −3.89/−3.52/−3.47,皆 wm/rad 微負=「lo 換 wm/rad」蹺蹺板實錘）入鍋+記錨銀行;帳 `docs/chains/c49nq1.jsonl` |
| c48nq1★ | **收鏈（07-31 12:5x,dry 2/2）**:7 包 175 筆,終錨=margin 王 c48nq1p05_16 +0.79（r48n2 已公證）;p06/p07 best +0.75/+0.73=王鄰域高原無人再勝,單軸天花板乾淨;全鏈帳 −0.15→+0.79(+0.94),`docs/chains/c48nq1.jsonl` |
| c49nq1 | **發鏈（07-31 10:2x）**:錨 c41grp2p07_19、goal tri、25 席 prio 1、dry2;`chain --name c49nq1 --anchor c41grp2p07_19 --source-input dedust_c41grp2_p07_input --goal tri --anchor-score -0.01` |
| 雙閘重掃 | ✅（07-31 10:1x）:近標帶 6,190 筆過 pattern 閘,top40 進 response 閘,8 筆全過;首名=c41grp2p07_19（左側王族,雙閘最遠）;次名 a005_x00r10c0s2（x00 製造系,rad+0.21）記候補;c12trip02_07（oob 7.46 破紀錄帶但 rad −0.15）記 scratch |
| v92→b1 | 鏈跑動中（v92 主訓練完,two/staging/select 段;b3+c48nq1 125 筆入鍋） |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- R50 型態體系軸（負片）:觸發=本輪收輪+Ricky 點頭;判準=decisions 條目;生成端已備。
- 訓練二刀 WeightedRandomSampler（另案保 golden）。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r49b*`、`dedust_c49nq1_p*`、`dedust_c48nq1_p06+`（跨輪鏈段）。
