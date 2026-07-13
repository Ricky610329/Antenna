# Round 26 — 帶外前瞻復活驗證：oob ρ 中位 ≥0.3？× 退 rad 鍵 × I 加碼

- **狀態**: running（2026-07-14 00:2x 開輪;R25 3 批硬上限收輪後接棒,宣告制）
- **提出 / 開跑 / 結論**: 2026-07-14 / 2026-07-14 / —
- **一句話問題**: SM 的帶外排序長年半死（R19-R24 前瞻 ρ <0.1）,R25 吃進多樣性資料後出現爬升軌跡
  （+0.218→+0.310→**+0.577 p=0.008**）——這是復活還是曇花？復活的話,帶外導引（價值軸主戰武器）
  就能從「盲抽+事後篩」升級成「可信預篩」。
- **一句話結論 (TL;DR)**: 待跑
- **指向**: [round-25](round-25-diversity-dial.md)（§5 前瞻兩面判決）· analysis-04（範圍限制機理）·
  decisions「探索型介入的評估法」「漸進式成長條款」

## 1. 假設 (Propose)

- **證據**：①R25 M 臂 oob 前瞻逐批爬升 +0.218/+0.310/+0.577（b3 顯著 p=0.008）,長期基線 <0.1
  （R21 五批定案「oob 排序基本死」）;②時間上與 v23/v24 吸收多樣性資料（探索類 40%、錨池 399→428）
  同步;③analysis-04 機理（範圍限制）預測:批內跨度↑→排序 ρ↑——wm 軸已否證,oob 軸可能因
  「帶外變異天然更大」而成立。
- **判準（發車前寫死;Ricky 可隨時否決）**：
  - **主判準**：R26 三批 M 臂 oob 前瞻 ρ **中位 ≥0.3**（R25 中位 0.31 為起點基線）＝**復活成立**→
    帶外導引升級進 R27 議程（帶外過濾重啟/O 配額加碼——本輪中途不動機器）;
    中位 <0.1 ＝曇花,回「oob 排序死」定案;0.1-0.3 ＝續觀察一輪。
  - **rad 頭退鍵**（R25 觸發連兩批 <0.3）：b1 起 select 不帶 `--rad-key`（`--rad-head` 保留續記
    pred_rad 前瞻）;**復鍵條件=續記前瞻連兩批 ≥0.3**（達成→下一批帶回 --rad-key,留日期註記）。
  - **D 臂第二期判決批**（b1 湊滿 3 批:89.8→93.0→b1）：min sel 對第一期末值 75.2 有下降趨勢=續航;
    平/升=帶帳本回報 Ricky 裁決（不自動處決,漸進條款）。
  - **F 臂第二期（預設方案,Ricky 可否決）**：縮編 24→**12 席**續 1 期——依據=第一期三批趨勢單調向好
    （best wm −5.15→−4.11→−3.95、中位 sel 105.5→94.5→90.7、b1 rad 過閘個案）;KPI 照舊
    （前緣點 wm>−1∧oob<8.61=礦脈開張）;本期結束仍無 wm>−3∧oob<9 → 帶「修復法天花板」判定回報。
  - 主指標（可用帶外,紀錄見 `docs/records.json`）：連 3 批零推進 → 重跑 /stall-protocol 驗屍
    （R25 驗屍判「梯子施工期」;再觸發時任一道轉紅=升第二層換分布）。
  - 一般臂存活:三標率連兩批 <6% 收臂;紀錄級一律公證（/notarize 鐵則）;**≤3 批必收輪**;
    判準發車後修訂=只能在結果回來前＋留日期註記。
  - ⚠ I 臂 KPI「模型更新量」量測工具連兩輪未建——本輪期間閒時補;在那之前 I 臂以三標率+血統貢獻記帳（誠實）。

## 2. 實驗設計 (Design)

- 每批 150：**O 32／M 20／C 20**（王朝直系 48% 沿 R25）／**S 20**／**I 26**（61% 爆發加碼,18→26）／
  **F 12**（第二期縮編）／**D 8**（判決批）／**W 12**（回補彩票）。
- 機器沿 select-r25（r22mix 本體）:root-cap 0.6、novelty 0.02、吸收擴充（R23/R24/R25/自產）、
  FRAG 錨同 R25、`--d-sm sm_denovo3.pth`（每批 train-denovo 重訓）。
- **不帶 --rad-key**（退鍵;--rad-head 續記前瞻）。

## 3. 執行紀錄 (Run)
```
# 發車（開發機,conda ant;每批照 /batch-cycle）:
python -m script.dedust select-r26 --batch N --sm sm_reanchor<最新版> --rad-head rad_head2.pth --novelty --d-sm sm_denovo<最新>
python -m script.dedust check-dup --input dedust_r26bNa_input   # a..f,exit 1=停
python -m script.dedust jobs-add --input dedust_r26bNa_input --store dedust_r26bNa --prio 3   # ×6
# 收檔: dedust watch --stores dedust_r26bNa,...,f;判讀: analyze batch --round 26 --batch N
# 每批收檔後: sm_reanchor train --add "...六夾..." --out vN+1 ＋ train-denovo --out sm_denovoN+1
```
| 批 | 狀態 |
|---|---|
| 1（r26b1{a-f}） | 🔵 00:3x 發車（v25;**--d-sm sm_denovo2**——denovo3 對決輸 harvest（wm ρ −0.307/oob −0.266）,照「輸=不換導引」規則落選,b1 加料後再戰;錨點 476〔滾動吸收 R25〕;查重 0×6;退 rad 鍵） |

## 4. 分析 (Analyze)
（待收檔）

## 5. 結論 (Conclude)
（待）

## 6. 後續決策 (Next)
（待）

## 7. 歸檔指向 (Archive)
- 結果夾: `dataset/dedust_r26b*`;公證 `r26n*`;填空池 `r26g*`。
