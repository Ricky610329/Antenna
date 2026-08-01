# Round 50 — 型態體系軸元年:負片 OOD × SM 域冷啟動曲線 × 正片保底

- **狀態**: running（2026-07-31 21:5x 開輪;自主續輪宣告制;R49 收輪+Ricky「計畫維持不變」背書=觸發到位）
- **提出 / 開跑 / 結論**: 2026-07-31 / 2026-07-31 / —
- **一句話問題**: 分布外的負片體系,SM 要多少資料才「準」——同時,批次系統能不能從零開出第一個文法外山頭？
- **指向**: decisions「型態體系軸評估法」+「R50 探索半算力配比草案」+「多山頭能力誠實帳」（三條=本輪憲法）·
  `script/neg_gen.py`（生成端,tests 綠）· assets/round-48/neg_gen_samples(_v2).png（Ricky 已核樣張）·
  assets/round-49/senior_pool_diversity.png(雙外軸的實證前提)

## 1. 假設 (Propose)
- **判準（發車前寫死;引 decisions 條目,不複製全文）**：
  1. **評估口徑=decisions「型態體系軸評估法」**:≥10 輪長 baseline、負片臂對三標/合格率**免疫**
     （停滯協議不觸發）;KPI=**SM 域冷啟動曲線**（OOD 凍結尺誤差 vs 鍋內負片 n,自 n=0 起）。
     「準」兩檔:前瞻 ρ≥0.3=排序可用（SM 升顧問）/誤差 ≤ 正片 held-out 2×=編入漏斗。
  2. **席位（=decisions 配比草案）**:每批 60=正片 30（select-r50,r49 配置縮編半,五軸照常、
     停止線/公證鐵則不變）＋探索 30。**b1=負片 30**（裝載批;select-neg,SM-blind farthest-point）;
     **b2 起=負片 20＋學長未殖民族 10**（73 領袖池值降冪,select 命令 b2 前實作）。
  3. **OOD 凍結尺協議**:負片 b1 收檔後,id 序偶數位切 **15 筆=凍結尺,永不入鍋**
     （機制=衍生店 `dedust_r50b1b_pot` 只收其餘 15,clean_stores 只加衍生店）;每次重錨量一次。
  4. **影子 pred 協議**:負片選席 SM-blind;收檔後以 **v95（本輪凍結版本）離線補算** pred——
     數學上等價於預先記錄（SM 凍結),n=0 誤差錨自此起帳。
  5. **臂分布預註冊規則**:select-neg 為覆蓋驅動,若單一生成臂佔比 >50%（煙霧測試見 bool_keep 偏抽）
     → b2 改分層選席（每臂配額);規則現在寫死,避免事後調整嫌疑。
  6. ≤3 批必收輪;lo 進鍵首航掛本輪補池（--lo 6,存活判準沿 R49 §1④）;紀錄級一律公證。
  ★ 修正三則（2026-08-01 03:1x,稽核 fanout F2/F5+涓流口徑;OOD 尺結果尚未量測=發生在結果前）:
  ①§1③ 實作=原店留 15+凍結 15 搬 `_frozen` 夾（等價於原文 _pot 設計,審計軌跡見 §3);
  ②新尺正名 **OOD 尺**（避免與 kpi.csv 的正片凍結尺 frozen_med 撞名);
  ③冷啟動曲線 x 軸=**鍋內負片總 n 含涓流池**（b0 90 筆同分布同生成器;下版 n=15→105 非 45）。
- **c 系鏈線本輪不開**（鏈位留白;負片首批不設鏈——先有資料再談爬山）。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| 負片 30 | select-neg 覆蓋選席 | KPI=冷啟動曲線;合格率免疫;凍結尺 15 切樣 |
| 正片 30 | select-r50 縮編 | 五軸照常;I 臂爆發續觀察 |
| 影子 n=0 | v95 離線補 pred | 誤差錨起帳;ρ 累積跨批 |

## 3. 執行紀錄 (Run)
```
# v95 重錨(b3 入鍋):
python -m script.sm_reanchor train --add "dedust_r49b3a,dedust_r49b3b" --epochs 30 --out sm_reanchor95.pth
python -m script.sm_reanchor train-two --epochs 30 --out sm_reanchor95.pth
# b1(正片 30+負片 30;seed 正 20260807/負 20260808):
python -m script.sm_invert gen --sm sm_reanchor95.pth --rad-head rad_head95.pth --n-free 3 --n-surg 0 --n-champ 0 --n-oob 3 --oversample 6 --seed 501 --out-dir tmp/invert_stage_r50b1
python -m script.dedust select-r50 --batch 1 --sm sm_reanchor95.pth --rad-head rad_head95.pth --gstage tmp/invert_stage_r50b1
python -m script.dedust select-neg --batch 1
python -m script.dedust check-dup --input dedust_r50b1a_input   # ×2 夾(a=正片/b=負片)
python -m script.dedust jobs-add --input dedust_r50b1a_input --store dedust_r50b1a --prio 3   # ×2 夾
python -m script.dedust watch --stores dedust_r50b1a,dedust_r50b1b
```
| 批/包 | 狀態 |
|---|---|
| 涓流池 | **prio4 常駐負片池點火（07-31 23:5x,Ricky 提議 216 轉負片→採已定案常駐池機制,優於固定 1:2=閒時 100% 負片）**:90 筆(z50b0_*,獨立店 dedust_r50b0b;臂分布 grf_lab 37/grf_inv 27/bool_keep 12/grf_neg 8/bool_cut 5/eng 1;查重 0);批單 prio 1-3 永遠優先;**凍結尺仍只從 b1 指定席切,涓流全入鍋** |
| b3 | **發車鏈(08-01 23:4x,末批)**:v98(b2 正片+學長店入主鍋——**分類裁定:senior=同形態異血統,主錨安全可食;負片 b2b→neg_stores**)→seed 503→三夾 select→查重→prio 3→watch;收檔後照 3 批上限收輪 |
| b2★ | ✅ 三軸判讀(08-01 23:3x,60 筆零 error):**正片縮編版合格 8/30=26.7% 新高**(I 4/6=67% 四連強/M 2/K 1/O 1);紀錄零推進/帕累托 +0;two 誤差尺 0.75 又贏/lohead 0.660 七連;**學長首包 10/10 全收:4 個近標帶個體**(F18644 −0.49/F21881/F24038/F6161 ~−0.55,漂移家族依賴再確認 Δ−0.03~−3.40)+**rad 天賦 F6161 +1.20**(超 rad 王帳面)/F9609 +0.7——錨銀行入貨 4+2;負片批 20 照曲線記帳。⚠ KPI② 分池漏網(近王/新血仍算全 60,d_dyn 中位 234 被 OOD 拉高)=待修小債 |
| v97★ | **雙頭制首版全過(08-01 12:2x)**:主錨凍結尺 **0.572→0.50 復原**(消融預測命中)+held-out 0.610;**OOD 尺自動量測首航**:mlp 4.30(外推基線)/**two 2.42**(n=107)——**對比 two96 的 3.83=−37%,冷啟動曲線首現正斜率**(對永凍樣本=真泛化);two 正片凍結 0.537→0.529 零代價。曲線帳:n=0→3.48/n=15(密度鍋 MLP)→5.47 死背/n=107(平鍋 two)→**2.42 學習中** |
| 生產線★ | **常備 OOD 生產線點火(08-01 12:0x,Ricky 拍板「直接產 5000+橋接 2500,算力對半」)**:負片 5k=池 b10~b26(17×300 分層,prio 4)+既有 420;**橋接 2.5k**=池 b30~b38(9×278,三式 bri_dil/ero/mix,母本=r48/r49 正片輸入夾 120 個真實個體;kind=bridge,two 課程教材);批單 prio 1-3 恆優先=算力對半自然形成;全店入 neg_stores(雙頭制);預估 7.5k×6min≈3-4 週閒時吃完 |
| b2 | **發車鏈(08-01 11:0x)**:v97=雙頭制首版(主錨純正片/two 吃負片 105)→staging seed 502→select-r50(正 30)+select-neg **--stratify**(負 20,每臂 2-3)+**select-senior**(學長領袖 10,池值降冪首包)→check-dup ×3→prio 3→watch。★ 判準修訂(b2 結果前,合規):負片選席 FPS→分層(稽核 M1:FPS 餓死工程臂 eng1/sierp0 於 120 席,原 >50% 規則量錯尾巴);學長臂自帶凍結尺=收檔後切 5(偶數位) |
| 消融★ | **判決:負片有罪(08-01 04:1x,v96_noneg 消融)**——同鍋只拔 15 筆負片,凍結尺 0.572→**0.52**(判定線 0.50±0.02 內);密度反權重孤點放大=確認機制。**待 Ricky 裁決 v97 起的負片入鍋策略**:①OOD 權重帽(z>3σ reps=1)②**分頭=負片只餵 two**(證據最強:two96 平鍋 OOD 尺 3.83/ρ+0.24 全場最佳,天然快取此職)③暫停入主鍋。裁決前不發 v97;kpi.csv 多一列 96_noneg(消融產物,非產線版) |
| OOD 尺★ | **冷啟動曲線前兩點(08-01 04:0x,稽核 agent3 量測;負斜率!)**:v95(n=0)\|err\| **3.48**/ρ −0.21 → v96(n=15)**5.47**/ρ **−0.42**——v96 把 12 筆鍋內負片背熟(MSE 18.1→0.44)但對凍結尺更瞎=**孤點過擬合簽名**(密度反權重給極端負片 ×24 帽=720 次曝光);**平鍋對照 two96 學會同樣本不付代價**(3.83/ρ+0.24)。正片凍結尺 v96 0.572(+3.1σ)歸因未定:v96 增量 ~170 筆中 74% 是 selfgen(premise 修正),有號誤差往樂觀移=不支持「被負片下拉」;**定案=消融重跑**(v96_noneg,~1hr,凍結尺回 0.50±0.02→負片有罪) |
| b1 | ✅ 判讀完(08-01 01:1x,60 筆零 error):正片合格 2/30(I best +0.35/O +0.36);**負片 0 合格=預期(三標免疫)但 best z50b1_017_grfi wm −5.88**——首批即逼近 wm>−5 生命跡象線(vs 首筆 −19.65);two 尺2 續贏;涓流池同步流動 |
| 凍結尺★ | **切樣完成(08-01 00:4x,協議§1③)**:負片 b1 30/30 零 error 收滿→偶數位 15 筆搬 `dedust_r50b1b_frozen`(名單=夾內 FROZEN_LIST.txt;grfn3/grfi3/grfl4/bkee4/eng1),**永不入鍋**;鍋內留 15;之後每次重錨量凍結尺=冷啟動曲線 y 軸 |
| b1 | 上機中(07-31 23:2x;a=正片@37/b=負片@218;216 自產→轉涓流) |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 學長族 select 命令(b2 前);tier2 prio4 常駐負片池;R50 後承重塊放寬掃描(pad 5→3→1)。

## 7. 歸檔指向 (Archive)
- 結果夾 `dedust_r50b*`(a=正片/b=負片);OOD 凍結尺名單記本檔 §3;`docs/kpi.csv` 加負片欄。
