# Round 41 — 組測試輪：常態線續航 × 金屬組變異單元 A/B（並行測試組）

- **狀態**: concluded（2026-07-26 03:1x 收輪,2 批＋2 鏈＋1 探針;假設已回答）
- **提出 / 開跑 / 結論**: 2026-07-25 / 2026-07-25 / 2026-07-26
- **一句話問題**: 「組（8-連通金屬島）是不是比 pixel 更對的變異單元？」——Ricky 2026-07-25
  提案;常態批線照跑,組測試組並行,R42 收測試（常駐 or 關閉）。
- **指向**: [round-40](round-40-two-switch-void.md)（two 換裝定案/V 臂存續）· scratch 07-25
  （組分析三條目:骨架+星座實錘/A 臂口徑敏感/B 臂 SM 近可加）· commit 5aadbc5（--mutator group）

## 1. 假設 (Propose)
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **常態線**（批 ≤3,select-r41=r40 配置:two 主通道+V 臂 8+D/W 10）:V 臂四五讀續
    （同判準:實測投影 NN 中位>批中位;累計連兩批 ❌ 才收案）;紀錄照公證鐵則。
  - **C 臂組級 A/B 鏈**（tier 0,c41grp,錨=c6tri8p01_18,goal tri,--mutator group 70/30）:
    同包配對——**判準=①每包 grp 半 vs px 半的 best score 勝負帳;②連 3 包 grp 半劣於 px 半
    =關閉記負結果;③grp 半產出合格解（wm≥0.15∧rad≥0∧lo≤−2）=直接常駐**;
    骨架凍結版（>25px 組僅 diag 算子可動）;dry/收鏈規則照舊（dry 2）。
  - **A 臂（已預跑,零 HFSS）**:四分組口徑掃描——結論收錄 §4,分組口徑定案進 decisions（R42 前）。
  - **B 臂（已預跑,零 HFSS）**:SM 交互矩陣——對角開關假說 SM 不可見,**HFSS 裁決權在 C 臂**
    （grp_diag 算子的實測效應 vs 其他算子）。
- **配額**：批線 62×≤3（同 r40）;C 臂鏈 25/包 走 tier 0 不佔批額。

- **★修訂（2026-07-26 00:1x,b2/p03 結果未回前;Ricky 兩指示）**：附掛 **r41s1 探針包 16 筆**
  （錨=新王 c41grp2p02_02,prio 2）——A 段 9=星座件定向平移（skew 方向盤;Ricky「φ90 很不平衡,
  再探索一下」;家族性偏斜實錘=三筆合格解全部 φ90 峰位 −34°/右半窗弱 ~3dB）;
  B 段 6=**對角清潔階梯**（fill/brk × k2/4/8,diagb 15→9~13;Ricky「想辦法減少對角線」;
  判讀=可製造化 vs 合格保持〔wm≥0.15∧rad≥0∧lo≤−3 帶〕）;C 段 1=鏡射 sanity。
  判準=A:|Δskew|≥0.5∧合格保持=有效旋鈕;B:diagb↓∧合格保持=「清潔 pass」候選常駐;不進紀錄。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| C 臂 A/B | 包內 70% 組級（六算子,骨架凍）+30% px 對照,sel_by=grp/px | 連3包 grp 劣=關;grp 產合格解=常駐 |
| grp_diag | 對角開關算子（不受凍結限制） | 實測效應分佈 vs 其他算子（B 臂裁決） |
| V 臂續讀 | 同 R40 判準 | 累計連兩批 ❌=收案 |
| 常態線 | two 主通道+誤差錨迴圈 | 五軸面板照常 |

## 3. 執行紀錄 (Run)
```
# v69 重錨:
python -m script.sm_reanchor train --add "dedust_r40b3a,dedust_r40b3b" --out sm_reanchor69.pth --ds-mode response
python -m script.sm_reanchor train-two --out sm_reanchor69.pth
# C 臂鏈（tier 0）:
python -m script.dedust chain --name c41grp --anchor c6tri8p01_18 --source-input dedust_c6tri8_p01_input --goal tri --anchor-score -0.05 --mutator group --n 25 --prio 1
# 批線（seed 120+N）:
python -m script.sm_invert gen --sm sm_reanchor69.pth --rad-head rad_head69.pth --out-dir tmp/invert_stage_r41bN --n-free 6 --n-surg 0 --n-champ 0 --n-oob 6 --seed <120+N>
python -m script.dedust select-r41 --batch N --sm sm_reanchor69.pth --gstage tmp/invert_stage_r41bN --rad-head rad_head69.pth --novelty
# check-dup ×2 → jobs-add ×2 prio 3 → watch
```
| 批/包 | 狀態 |
|---|---|
| c41grp | ❌ 夭折（07-25 19:5x）:**20 包全數 check-dup 撞歷史零發車**——bug=grp_diag/grp_shrink 常產 diff=1 單 px 變異（等價 d1）,used 集只擋 px 半,撞 c6tri8 p02/p03 已測 50 px。修復=chain 啟動載全史 hash 本地防撞+單 px 組變異記 used+rng 種子改 md5（commit b891250）;重發=c41grp2 |
| c41grp2 | 🔵 發鏈（07-25 20:0x,含防撞修復;錨/判準同 §1）;**p01 A/B 首讀（21:4x）:best 平手 −0.11**,組半縱深優（top2-4 −0.12/−0.13/−0.17 vs px −0.17/−0.20/−0.21）,tri 有效率 grp 11/18 vs px 6/7;**組半登頂算子=grp_diag（d1 對角開關）——假說首個 HFSS 正面訊號**;dry 1/2,p02 自續 |
| b1 | ✅ 判讀完（07-25 22:3x,61 筆）:two ρ+0.783/adv 0 續勝（**誤差尺 2.64 輸 mlp 2.17=七批首次**,觀察）;lohead ρ+0.797;**V 四讀 ✅**（0.16>0.15,top 0.72;累計 3/4）;多樣性正常（53/35）;**d41b1_000 oob −4.2 全負帶外新深**（wm −10.78,教材）;o41b1_000 合格 hi−4.73 右側深水;帕累托+1（g41b1_004 G-free 首筆）;紀錄零推進 |
| c41grp2 p02 | ★★★（07-25 23:4x 公證過）**c41grp2p02_02＝第三筆左側合格解＋usable_lo 紀錄易主 −2.63→−3.46**（公證 r41n1 **3/3 bit 級一致**;wm+0.150∧rad+0.06∧lo−3.46∧hi+2.21）;**出自 grp 半 grp_grow d3（中件修邊+3px）——組級算子第二包破紀錄,「grp 半產合格解=直接常駐」判準達成**;p02 分半 grp +0.000 勝 px −0.050;鏈換錨 c41grp2p02_02 續爬（p03）;對比圖 assets/round-41/newking_usable_lo.png;records.json 已換帳 |

## 4. 分析 (Analyze)
- **A 臂預跑（07-25,scratchpad r41_arm_a_grouping.txt）**：口徑敏感——8conn 左側家族 9 組
  （骨架 193/80/30 凍結+小件星座）;**4conn 碎 24-25 組且 c8 vs c6 拓撲不同**（8conn 全同）
  →「1px 對角拓撲開關」假說;close1/dbscan2 全鍋合格解一律 1 組（整張=單一耦合體）。
- **B 臂預跑（07-25,sm_two68,scratchpad r41_arm_b_result.txt）**：對角 px 單翻效應 0.115 vs
  非對角 0.133（SM 看不見開關效應——平滑化嫌疑,裁決權交 C 臂）;高交互對 9/10 組內
  （幾何分組弱支持）;|I| 中位 0.008 vs 單翻 0.138（SM 眼中近可加=組級粒度安全）。
- **r41s1 探針（07-26 00:4x 收,16 筆）**：①**刀鋒解實錘——16 微擾合格保持 0**（wm+0.150 貼線,
  2px 擾動即掉線;critical coupling 讀法坐實）;②skew 舵=9-10px 中件（mv9dn Δskew+1.33/
  mv9rt+0.66/mv10lt+0.55）,3px 小件非舵（±0.07）但動了照樣崩 wm;A 判準（|Δskew|≥0.5∧合格）
  0 筆達標;③對角清潔:fill 溫和/brk 災難（brk8 wm−7.8）;**fill4=wm+0.11∧rad+0.07∧lo−2.93,
  diagb 15→12,差 0.04 wm 臨門**→c41grp3 清潔收復鏈（錨 fill4,−0.04,group,判準=合格∧diagb≤12
  =展示級候選）;B 判準 0 筆全達標;④**鏡射警訊:mir wm−0.14 vs 錨+0.15（Δ0.29,skew 正確翻號）**
  ——響應不變性受疑（最可能網格數值不對稱）,train-two 增強標籤噪聲 ~0.3 記帳待驗（tier2 抽驗計畫）。

- **b2（07-26 03:0x,61 筆）**：two 回勝三尺（1.50/**ρ+0.864 新高**/adv 0——b1 誤差尺失利=單批雜訊
  確認）;lohead ρ+0.829;**V 五讀 ✅**（0.22 vs 0.17 最明確一讀,累計 4/5=V 臂站穩）;三標 10/合格 7;
  多樣性正常（53/31）;帕累托+1;紀錄零推進。
- **c41grp2 夜間戰報**：p06 勝錨 **c41grp2p06_11＝第四筆左側合格解（單次,wm+0.18∧rad+0.28∧
  lo−2.74）——餘裕厚非刀鋒,展示級底子候選**;p07 續爬。c41grp3 p01 dry（best wm 0.00,差 0.15）,p02 續。

## 5. 結論 (Conclude)
1. **「組是變異單元」定案**：判準「grp 半產合格解=直接常駐」達成（p02 grp_grow→第三筆合格解
   +usable_lo 紀錄 −3.46）;A/B 帳 p01 平手縱深優/p02 grp 勝;**鏈線常駐,批線推廣待 A/B ≥6 包（R42+）**。
2. **左側家族結構學三發現**（r41s1）：刀鋒解（16 微擾合格保持 0）;skew 舵=9-10px 中件（付 wm 代價）;
   對角清潔 fill 式可行（fill4 臨門 0.04→c41grp3 收復中）——「展示級」不是免費的,是一條要爬的鏈。
3. two（ρ 0.783/0.864）/lohead（0.797/0.829）/V（4/5）全數站穩;佇列/機器零事故。
4. **鏡射不變性警訊**（Δwm 0.29,單筆）：train-two 增強標籤噪聲嫌疑,tier2 抽驗 3-5 對待做。
5. 合格解全史 4 筆（3 筆已公證級+p06_11 單次）;usable_lo −3.46（公證）。

## 6. 後續決策 (Next)
- R42 常態輪:批線 select-r42;鏈線續（grp2 攻深/grp3 清潔收復——跨輪慣例）;組算子鏈線常駐定案
  進 decisions;鏡射抽驗（tier2 閒時）;ens 換代（cnn2 成員）候選;獨立艙凍結續（觸發=Ricky 點頭）。
- d40b3_001（oob −1.87）/d41b1_000（oob −4.2）全負帶外教材追蹤。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r41b*`、`dedust_c41grp_p*`;鏈帳 docs/chains/c41grp.jsonl。
