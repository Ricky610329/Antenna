# Round 37 — 左側大陸殖民輪：tri 會師鏈 × ref/rej balance × 資料換系統

- **狀態**: concluded（2026-07-23 23:0x 收輪;同日午後開輪;Ricky 全程共同設計〔左右拆帳/換系統/balance/粉塵忽略/
  diagb/0.5 格〕,細節「跟你配置的一樣」拍板）
- **提出 / 開跑 / 結論**: 2026-07-23 / 2026-07-23 / 2026-07-23
- **一句話結論 (TL;DR)**: **殖民開工成立（證據三選三）**——tri 前緣 −0.49→**−0.31**（wm−0.08∧rad−0.31∧lo−3.02∧oob8.08=無人區座標）,左側斜率遠勝王系（d=1 零）;**Ricky 三提案一日全兌現**（response 反權重轉正〔左側域−26%〕/balance 常駐〔實抓假陰性〕/架構翻案〔凍結−56%,analysis-06〕）;i37b3_006 wm+0.600 全史最高值卡 rad 非三標;tiers 三連 ≥2×→鏈 3 位/批 50 觀察;selfgen 換種欠帳。
- **一句話問題**: trade-off=王系經驗律（非物理定律）＋王系內左側 d=1 斜率實測零（c4lo 定讞）——
  把 refine 機器空投到左側好的系統（tri 前緣=c2rad 系）,能不能產出**全史第一筆左側合格解**
  （wm≥0.15∧rad≥0∧lo≤−2）？同步:資料分布搬家（response 反權重 A/B+selfgen 換種）。
- **指向**: [round-36](round-36-oob-wall.md)（左右拆帳制誕生/c4lo 定讞）· decisions
  「左右側拆帳紀錄制」「★修正 trade-off」「結構準則更新」· scratch（balance/獨立艙凍結）

## 1. 假設 (Propose)
- **證據**：①tri 前緣掃描（lo≤−2 池 2,180 筆）top10 全為 c2rad 鏈系——rad goal 爬升時 lo 門檻
  鎖住、wm 順帶回升,最佳 c2radp10_21 tri −0.49（wm−0.17∧rad−0.49∧lo−3.48）;②t07/池頂族=
  wm 帶內正∧lo −1.7~−4.5,只卡 rad（trade-off 剖面族依賴的實體）;③response PCA:合格族=單一
  聚落泡在左側紅區——資料配比按 response 分布做（Ricky）。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **旗艦里程碑**：左側合格解 wm≥0.15∧rad≥0∧lo≤−2（全史 0 筆）——出現=公證（/notarize
    鐵則）+推播。次要:usable_lo 以 **0.5/格** 公證推進（雜訊防線）;usable_oob 總帳型推進降級記帳。
  - **tri 雙鏈**（goal=tri=lo≤−2 內 min(wm−0.15,rad);純隨機 d=1——expert 排序鍵=wm 與 tri 不對齊,
    照 c2rad 先例不帶）:c5tri 錨 c2radp10_21（−0.49）/c6tri 錨 c2radp09_16（−0.53）;單錨 dry 2
    收鏈換備選（備池=tri 前緣 top10）;兩鏈六錨全 dry=「c2rad 系 tri 高原」定讞,帶帳回報。
  - **L 臂新大陸錨組 12 席+SM 過濾 balance**（Ricky 拍板「一半參照一半放覺得不好的」）:
    錨=c2radp10_21/c2radp09_16/t07_top/l31b2_005;**半 ref（SM top）半 rej（SM 判死下半區均勻抽）**,
    sel_by 記帳——判準:rej 半連三批追平 ref 半（合格/三標率）=SM 過濾在新大陸無增值實錘→L 退全隨機。
  - **diagb 方向性規則**：變體不得比錨增對角橋（左側家族天生 diagb 14-16,絕對否決殺大陸——
    保守解=世代往下壓）;全域 select 罰分 --diagb-pen 2.0/橋（上限 5）。⚠與 Ricky 原句「不要對角線」
    的絕對讀法不同——**留 Ricky 確認**,要絕對制就改。
  - **response 反權重 A/B**（偶數版 v58 執行）:同鍋雙訓（--ds-mode pattern vs response）,
    比凍結尺+左側域（lo≤−2 held-out 子集）分層誤差——response 版兩尺不輸∧左側域贏=轉正預設。
  - **selfgen 種子換系統**：王朝表型種子 md5 決定性留 20%,左側家族全保——⚠生效需三台 git pull
    +worker 重啟（下次機器維護時）。
  - 殖民期 KPI:當批合格數不焦慮——看 ①鏈爬升斜率 ②SM 左側域誤差 ③L 臂 realized lo 分布左移
    （三選二=殖民開工證據）;asym 進鍵判定（R35 欠帳）b1 判讀結案;批 ≤3;紀錄公證鐵則;修訂註記。
- **配額（批 50=2 夾）**：G 12（free6/oobp6,SM 探測定位）/L 12（新大陸 ref6+rej6）/I 8/M 5/O 3/
  K 2/D 4/W 4。cnn-solo 關（R36 回退）;rad-key 關。

## 2. 實驗設計 (Design)
| 項 | 設計 | 判準 |
|---|---|---|
| tri 雙鏈 | c2rad 系前緣雙錨純隨機爬 | 左側合格解=里程碑;六錨 dry=定讞 |
| L balance | 半 ref 半 rej,sel_by 帳 | rej 連三批追平=過濾無增值 |
| response A/B | v58 雙訓雙尺 | 左側域贏+兩尺不輸=轉正 |
| selfgen 換種 | 王朝種子留 20% | 生效後 selfgen 結構佔比追蹤 |

## 3. 執行紀錄 (Run)
```
# tri 雙鏈（開發機;純隨機;錨已驗）:
python -m script.dedust chain --name c5tri --anchor c2radp10_21 --source-input dedust_c2rad_p10_input --anchor-score -0.49 --goal tri
python -m script.dedust chain --name c6tri --anchor c2radp09_16 --source-input dedust_c2rad_p09_input --anchor-score -0.53 --goal tri
# 批線（v57 輕量出爐後;seed 80+N）:
python -m script.sm_invert gen --sm sm_reanchor57.pth --rad-head rad_head57.pth --out-dir tmp/invert_stage_r37bN --n-free 6 --n-surg 0 --n-champ 0 --n-oob 6 --seed <80+N>
python -m script.dedust select-r37 --batch N --sm sm_reanchor57.pth --gstage tmp/invert_stage_r37bN --rad-head rad_head57.pth --novelty
python -m script.dedust check-dup --input dedust_r37bNa_input   # a/b;exit 1 停
python -m script.dedust jobs-add --input dedust_r37bNa_input --store dedust_r37bNa --prio 3  # ×2
python -m script.dedust watch --stores dedust_r37bNa,dedust_r37bNb
# v58（b1 收檔後,偶數全訓+A/B）: train --add "..." --out sm_reanchor58.pth
#                                train --add "" --out sm_reanchor58r.pth --ds-mode response --no-rad --no-ens --no-shadow
```
| 批 | 狀態 |
|---|---|
| 1（r37b1{a,b}） | 🔵 14:25 發車（v57 輕量凍結 **1.248**/near **0.997 首破 1.0**;L 臂新大陸錨組+ref/rej balance 首批;diagb 罰首批;查重 0×2。tier 0:c5tri/c6tri 首包在飛） |
| 2（r37b2{a,b}） | ✅ 20:3x 收（19:13 發車;v58r 導航首批;合格 4;ref/rej best 反轉;NN 8 警報） |
| 3（r37b3{a,b}）末批 | 🔵 21:23 發車（**v59=response 首版**〔遠域 2.345/P90 4.87 佳績,凍結 1.296〕;D/W 各+2〔恆溫,54 筆〕;查重 0×2） |

## 4. 分析 (Analyze)

### §4a — b1 判讀（16:2x 收,週期 ~115 分〔tri 雙鏈插隊〕,零 error）
- **L 臂新大陸首批**:0 合格但 **best wm −0.03**（nl_c2r09 變體,比錨 −0.10 好）;
  **realized lo 中位 −2.39/−1.92=殖民開工證據③首批成立**（lo 分布左移,vs 王系 +3.7）。
- **ref/rej 首讀（balance 儀表）**:ref 半勝——wm 中位 −1.45 vs −3.90/best −0.03 vs −0.35
  ——SM 排序在新大陸首批**有增值**（判準連三批;rej best 出自 t07 系）。
- **asym 進鍵判定（R35 欠帳結案）**:全臂 ρ −0.779（wm）/−0.649（lo）超線;**G 臂內 ρ −0.008=
  臂內無區分力**（G free asym 全高無變異）——原用途（G free 預過濾）**否決**,保留記錄鍵。
- M 60%/O 67% 老區照常（合格 3;最佳 10.08）;**denovo 首進帕累托**（d37b1_002 oob 1.21 極端點）;
  影子對決 MLP 勝（雙 rank 維持正確）;G free 100% ×7;多樣性警報（歷史 NN 11——tri/L 挖已測
  鄰域的殖民期預期效應,記帳觀察,b2 若 <10 動配額）。
- tier 0 同期:c5tri p01 dry1（best −0.57 但 **wm −0.17→−0.04 單軸大動**=左側 wm 快軸驗證,
  輸在 rad 同滑被 min 擋——會師目標的正確行為）;c6tri p01 dry1（−0.57）;雙鏈 p02 在飛。

### §4b — response 反權重 A/B 裁決（19:2x;判準三條全過→**轉正**）
| 尺 | v58 pattern | v58r response | 判定 |
|---|---|---|---|
| 凍結尺 | **1.214（平史上低點）** | 1.227 | 持平 ✓ |
| 全域 held-out 中位 | 1.305 | 1.314 | 持平 ✓ |
| **左側域（lo≤−2,n=516）** | 3.132 | **2.304** | **贏 26%** ✓✓ |
- **Ricky 的 response 多樣性提案 24hr 從想法到轉正**;左側導航儀升級。落地:b2 主模 v58r
  （配件借 v58 同鍋系——v58r 輕量無配件,混版誠實記帳）;**R38 起重錨預設 --ds-mode response**;
  registry 更新。附帶:v58r 遠域 2.413 反超 v58 2.532（補稀有區的預期效應）。

### §4c — b2 判讀（20:3x 收,零 error）
- 合格 4（**K 臂 2/2 全中**〔k37b2_000 wm+0.28∧oob9.73〕+I 2〔i37b2_002 **wm+0.50**〕）;紀錄零推進。
- **L 臂二讀**:best 衝到 **wm+0.29**（l31b2 錨系;b1 −0.03→b2 +0.29=新大陸 wm 上衝快）,仍 0 合格
  （rad/lo 未同步）。**ref/rej 二讀=中位 ref 續勝（−0.31 vs −5.00）但 best 反轉——rej 半 l37b2_006
  (+0.29) 是全批 L 最佳**:SM 判死區交出最好左側候選=balance 儀表首次抓到假陰性實體。
- **多樣性警報升級**:歷史 NN 8（<10 微調批線,b1 11→b2 8 連降——殖民期鄰域挖掘+v58r 導引集中）
  →b3 按恆溫規則 **D/W 各 +2**（G 不動,L 主戰臂不砍）。
- tier 0 同期:c5tri 收鏈（dry2,終錨 −0.31=tri 新前緣）→c5tri2 sideways（−0.41）;c6tri2 p02 dry1
  →c6tri3 同錨接棒。analysis-06 架構突破同晚（獨立事件,見該檔）。

### §4d — b3 判讀＋收輪彙總（2026-07-23 22:5x）
**b3**（23:0x 收,54 筆零 error）:合格 4（I 2+M 2）;**i37b3_006 wm+0.600=全史 wm 最高值但 rad −0.09
差線非三標**（I 系三度摸王座卡 rad;帕累托 +2）;M 臂 a218 系（selfgen 血統）兩筆合格;L 臂 0 合格
（best −0.28）;多樣性 NN 12（D/W+2 半有效,警報帶續）。

**判準逐條判定**：
1. **旗艦里程碑（左側合格解）=未達**,但缺口大幅收斂:tri 前緣 −0.49→**−0.31**（c5trip02_22,
   wm−0.08∧rad−0.31∧lo−3.02∧oob8.08=全史無人站過的座標）;鏈六段（c5tri/2+c6tri/2/3/4）。
2. **殖民開工證據=三選三全成立**（判準只要求三選二）:①鏈正斜率（+0.18 單步）②SM 左側域誤差
   改善（A/B −26%）③L 臂 realized lo 左移（三批中位 −2~−3.8）。**殖民成立,R38 續攻**。
3. **ref/rej balance=SM 過濾有增值成立**（中位 ref 3/3）;但 b2 best 反轉=假陰性實體——
   **balance 常駐定案**（半半制=儀表+保險）。
4. **response 反權重=轉正**（§4b;v59 首產版遠域 2.345/P90 4.87 佳績）。
5. selfgen 換種=**未生效（欠帳）**:機器未 pull——R38 期做。diagb 方向性規則=維持（Ricky 未回
   絕對制之問,標開放）。
6. 硬紀律:批 3 ✓/零 error×6 夾/公證 0 件（無候選）/紀錄零推進（合格三批 11/154=7.1%,殖民期
   KPI 口徑=不焦慮）。
7. **tiers 第三讀 2.07×**（三連 ≥2×）——階梯下一格「批 25+鏈 3」:引護欄保守裁決=**鏈 2→3 執行、
   批維持 50 延後一輪**（多樣性警報活躍,塌縮歸因不乾淨——判準內護欄,誠實記帳）。
8. 同日獨立事件:analysis-06 架構翻案（臂A 三尺全勝/lo 判別器可用）→ R38 影子二號。

## 5. 結論 (Conclude)
1. **殖民戰役開工成立**（證據三選三）——左側大陸的 tri 斜率遠勝王系高原（−0.49→−0.31 一日,
   vs 王系左側 d=1 斜率零）;缺口剩 wm 0.23/rad 0.31。
2. **Ricky 三提案一日全兌現**:response 反權重（→轉正,左側域 −26%）/SM 過濾 balance（→常駐,
   實抓假陰性）/架構復審（→翻案,凍結 −56%）——「人定軸+機器跑」的共同優化敘事最強實例日。
3. I 系困局清晰化:wm 上探 0.60 但 rad 屢差臨門——rad 軸=I 系天花板;lo 判別器+影子二號=下輪工具。
4. tier0:tier1 三連 ≥2×——鏈制擴編（3 位）;批 50 保留觀察一輪。

## 6. 後續決策 (Next)
- **R38 主軸**:影子二號（臂A 制度內對決,b1 間實作/b2 起盲測）+lo 判別器進鍵（select 記錄鍵
  pred_lo）+tri 鏈三位續攻+selfgen 換種生效（機器 pull）。
- 獨立艙凍結續;d=2 跳步備援;GNN=第三視角候選。

## 6. 後續決策 (Next)
- 獨立艙（設計凍結,scratch;觸發=Ricky 點頭/馬太惡化）;lo 軸判別器（A/B 後）;d=2 跳步備援。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r37b*`;公證 `r37n*`;鏈帳 docs/chains/c5tri/c6tri.jsonl。
