# Round 57 — dual 元年:濾波器線首航(批次線+SM 搜尋,復用 single 設施)

- **狀態**: running(2026-08-10 smoke 過+b1 發車)
- **提出 / 開跑 / 結論**: 2026-08-10 / 2026-08-10 / —
- **一句話問題**: dual(輻射型二埠濾波器)用 single 的批次線+SM 搜尋設施能不能跑起來,
  首批 100 筆把「起跑線/鏡像假說/m5m6 可達性」三件事量清楚。
- **指向**: [proposal-dual-kickoff](proposal-dual-kickoff.md)(D0-D2 收斂)·
  [senior-thesis-dual-port](../reference/notes/senior-thesis-dual-port.md)·configs/dual_r1_eval.yaml·docs/records_dual.json

## 1. 假設 (Propose)

**假設**:設施復用成立(D2:~15 行 port 分派+一把新尺);學長 harvest 萬筆離規格遠(best −8.2),
SM 排序+錨點鄰域能把首批推得比 harvest 錨更好;上下鏡像 ⇒ S11≡S22 可實測成立。

- **判準(發車前寫死;Ricky 2026-08-10 核准)**:
  1. **尺=wm_dual=min(m1,m2,m3,m4)**(mask 口徑,worst_margin_dual);**m5/m6 記帳不進 min**
     (可達性未證;首批順帶驗:若 m5/m6 母體 max 仍 <−2 → 維持記帳;若有樣本逼近 0 → 重議)。
  2. **管線活**=首批 ≥95% 成功+energy_max≤1 全過+響應 17 點全過(壞檔=響亮報錯非靜默)。
  3. **鏡像假說**=鏡像臂 20 筆的 |S11−S22| 中位 <1dB(對照非鏡像臂中位 ~9dB)→ 成立則鏡像 generator
     升格主力(判準 6→4 項、自由度砍半)。
  4. **起跑線**=首批 wm_dual 最佳值;>harvest best=SM/鄰域有效的初證。
     ★ 修訂(2026-08-10 施工期,發車前:harvest 全掃真值 best=**−7.20**〔400 抽樣估 −8.23 修正〕,
     最佳 20=−7.20~−8.67;起跑線判準以 −7.20 為準);
     首個顯著最佳解照公證鐵則 ×2 → records_dual.json 開帳。
  5. 首批=select-dual 100 筆(harvest 錨 20/鄰域 40/隨機 20/鏡像 20;雙饋墊必保;seed 決定性;
     port=dual 查重分域);S0 尺=dual 預設求解(Fast sweep,與 harvest 同分佈——幾何亦不動,
     Ricky 裁:等找到好 pattern 後用消融確認)。
  6. 線上學習路徑=延後/忽略(Ricky 2026-08-10);loss 決策隨之延後。
- 修訂紀律:結果回來前+日期註記。

## 2. 實驗設計 (Design)

| 項 | 設計 | 判準 |
|---|---|---|
| 施工 A/B | 模擬器護欄+尺 / dedust 配管+select-dual | 全套 pytest 綠+golden 零漂移 |
| smoke | 1 筆 only_create_project 埠位渲染目檢+1 筆全跑(energy 自證) | §1② |
| 首批 b1 | 100 筆(四臂) | §1②③④ |

## 3. 執行紀錄 (Run)

```
# 施工:包A(dual_port 護欄+worst_margin_dual)+包B(dedust 配管+select-dual)→ 整合+全測試 → commit
# 部署:三台 git pull+重啟 worker(dual 化動 worker 路徑=部署事件,照 script/CLAUDE.md 第8條)
# smoke → select-dual → check-dup(dual 域)→ jobs-add --config configs/dual_r1_eval.yaml --prio 3
```

| 事件 | 狀態 |
|---|---|
| 開檔 | 2026-08-10(施工中) |
| 施工 A+B+審計必修 | 2026-08-10 綠(424 pytest;commit 2f7d797 前後系列) |
| 部署 | 2026-08-10 三台 pull+重啟(Ricky) |
| smoke(dedust_r57smoke,2 筆 harvest 交叉樣本) | 2026-08-10 20:41 收檔,0 error;**對帳=樣本 00 與學長存值 bit 級全同**(三通道 MAE 0.00/相關 1.000),樣本 01 求解噪音級(MAE 0.1-0.4dB);通道序錯位檢查過(對角相關 1.000 ≫ 交叉 0.68-0.81);energy_max 0.86-0.89≤1;m1..m4 與 D1 手算逐位吻合 → **判準② 管線活=通過** |
| b1 發車 | 2026-08-10 20:52 `jobs-add --input dedust_r57b1_input --store dedust_r57b1 --config configs/dual_r1_eval.yaml --prio 3`;**98 筆**(select-dual 100 − 2 筆與 smoke 重複移除:d57b1_a_04/a_10 即 smoke 兩樣本,判讀時併 smoke 結果補回 a 臂帳);check-dup 綠(dual 域);218 20:53 接單 |

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- SM 冷啟動訓練(patch_dual 暖啟+harvest_dual 鍋)=首批後另包;sm_reanchor port 參數化=後補 L。
- 幾何消融(0.01mm 對齊問題)=找到好 dual pattern 後照 R54 模式做。

## 7. 歸檔指向 (Archive)
- 結果夾 `dedust_r57b1`;紀錄=docs/records_dual.json。
