# Round 57 — dual 元年:濾波器線首航(批次線+SM 搜尋,復用 single 設施)

- **狀態**: proposed(2026-08-10 開檔;施工包 A/B 進行中,發車=施工綠+機台 pull 後)
- **提出 / 開跑 / 結論**: 2026-08-10 / — / —
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
  4. **起跑線**=首批 wm_dual 最佳值(預期 −8~−6);>harvest best(−8.23)=SM 排序有效的初證;
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

## 4. 分析 (Analyze)
（待）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
- SM 冷啟動訓練(patch_dual 暖啟+harvest_dual 鍋)=首批後另包;sm_reanchor port 參數化=後補 L。
- 幾何消融(0.01mm 對齊問題)=找到好 dual pattern 後照 R54 模式做。

## 7. 歸檔指向 (Archive)
- 結果夾 `dedust_r57b1`;紀錄=docs/records_dual.json。
