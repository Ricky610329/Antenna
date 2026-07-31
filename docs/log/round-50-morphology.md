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
| （待） | |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 學長族 select 命令(b2 前);tier2 prio4 常駐負片池;R50 後承重塊放寬掃描(pad 5→3→1)。

## 7. 歸檔指向 (Archive)
- 結果夾 `dedust_r50b*`(a=正片/b=負片);OOD 凍結尺名單記本檔 §3;`docs/kpi.csv` 加負片欄。
