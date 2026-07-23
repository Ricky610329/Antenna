# Round 37 — 左側大陸殖民輪：tri 會師鏈 × ref/rej balance × 資料換系統

- **狀態**: running（2026-07-23 午後開輪;Ricky 全程共同設計〔左右拆帳/換系統/balance/粉塵忽略/
  diagb/0.5 格〕,細節「跟你配置的一樣」拍板）
- **提出 / 開跑 / 結論**: 2026-07-23 / 2026-07-23 / —
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
| — | （開輪;v57 訓練中） |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- 獨立艙（設計凍結,scratch;觸發=Ricky 點頭/馬太惡化）;lo 軸判別器（A/B 後）;d=2 跳步備援。

## 7. 歸檔指向 (Archive)
- 結果夾 `dataset/dedust_r37b*`;公證 `r37n*`;鏈帳 docs/chains/c5tri/c6tri.jsonl。
