# Round 8 — 乾淨子空間測繪（clean-subspace mapping，四臂批次驗證）

- **狀態**: proposed（輸入 97 筆已備妥 NAS，待使用者確認後發車）
- **提出 / 開跑 / 結論**: 2026-07-03 / — / —
- **一句話問題**: 「可製造世界」的前緣真值在哪、整塊型可拔是不是通則、補洞是否因果性幫 Gain、SM 能不能在乾淨區被餵亮？
- **一句話結論 (TL;DR)**: 待跑
- **指向**: 工具 `script/dedust.py`（select-r8/run/report,--input dedust_r8_input --store dedust_r8）· 輸入 `DATASET_PATH/dedust_r8_input/`（97 筆＋manifest 含 SM 預篩）· 前作 [round-07](round-07-dedust.md) · 框架討論 `docs/discuss/scratch.md`「三指標」塊

> 本檔只放**連結指向**。R7 收斂出的框架：每筆 HFSS 必須推進三個指標之一——**產品**（可製造最佳 wm,現 -2.68）/**能力**（SM 乾淨區誤差,現盲 5-15dB）/**知識**（驗證過的設計規則數）。R8 是一次把三格都推的測繪批次。

## 1. 假設 (Propose)

- **承接**: R7 實錘「乾淨解要用搜的」→ 搜之前要先測繪：起點多好（A）、規則哪些真（B）、導航儀能不能用（C）、以及對照組的地板在哪（D）。
- **假設**: ① 整塊型（main_frac≥0.9）除塵近零代價（p03 通則化）；② 補洞幫 Gain/rad（analysis-01 因果版）；③ SM 乾淨區盲是資料缺口、不是架構缺口（餵資料可治）；④ 池抽樣 ≫ 真 uniform random（R6 誠實缺口的量化）。
- **判準**（寫死在發車前）:
  - A：整塊型除塵 |Δwm| 中位 **<0.5dB** → 通則成立；前緣 HFSS 真值 best 落點（池值 -0.81 有 1-2dB 漂移帶）。
  - B：補洞後 Gain 與 rad **同號變好**（4 個編輯對）→ 規則升級為精修算子；b00_ref vs R7 p03_d3 = **HFSS 重跑噪聲地板**（判讀一切 Δ 的尺）。
  - C：收檔後離線重錨 SM（r7+r8 真值併訓），量乾淨區預測誤差**重錨前後**（前=本次 sm-screen 已存 manifest）→ 誤差進 ~2dB 帶=精修 round 解鎖。
  - D：random 10 筆的 margin 分布 vs 池抽樣線（R6 圖補一條真基線）。
- **依據**: [round-07](round-07-dedust.md) §5 · [analysis-01](analysis-01-pattern-anatomy.md) · [round-06](round-06-offline-expected-best.md) 侷限①

## 2. 實驗設計 (Design)

| 臂 | 內容 | 筆數 | 買什麼 |
|---|---|---|---|
| A | 乾淨前緣 main_frac≥0.9 top-15（互 Hamming>60）× orig+d3 | 30 | 通則檢驗＋可製造 warm-start 起點真值 |
| B | p03_d3 重跑（噪聲地板）＋補洞×4（p03_d3、A 臂前 3 名 d3；Δ 走 base_id） | 5 | 「Gain←少洞」因果檢驗 |
| C | 錨點（p03_d3＋A 前 7 名 d3）×翻{8,32}px 無粉塵修復×2 seed＝32＋平滑 blob 20 | 52 | SM 乾淨區校準資料（＋52 條 rad） |
| D | 真 uniform random（iid p=0.5、feed 強制、不修復） | 10 | 池≠隨機的量化基線 |

- 生成全決定性（seed 進 id）；「乾淨」暫定＝**全碎片≥4px＋feed pad ≥4px**（待使用者定像素 mm/最小可裁尺寸,改了 `select-r8` 重生成即可,零 HFSS）。
- **HFSS 預算**: 97 筆 ≈ **4.8 hr**（R7 實測 3 分/筆）；可中斷續跑。
- SM 預篩（重錨前基線）已跑入 manifest：A 臂除塵 Δ 混合 ±1.5（vs R7 碎片雲一致大跌——通則弱佐證）；b00_holes Gain 預測 +1.79（與 analysis-01 同向）。

## 3. 執行紀錄 (Run)

```
# 正式機 .37（先 git pull）
python -m script.dedust run --input dedust_r8_input --store dedust_r8
# 任一機看進度/收檔
python -m script.dedust report --input dedust_r8_input --store dedust_r8
```
- 事件: —

## 4. 分析 (Analyze)

**完整附圖報告 → [round-08-report.md](round-08-report.md)**（四臂逐筆數字＋4 張圖，圖在 `assets/round-08/`）。verdict 一行版：

| 臂 | 判定 | 關鍵數字 |
|---|---|---|
| A | ❌ 崩 | \|Δ\| 中位 1.17（判準<0.5）、變好 3/15；前緣真值 best −1.80；**池→現行 HFSS 漂移 14/15 向下（中位 −0.52）** |
| B | ❌ 敗 | rad 四筆全負、Gain 兩正兩負；噪聲地板 **0.00**（b00_ref≡R7 p03_d3） |
| C | 🟡 半亮 | 重錨前：池內 \|err\| 1.5–2.4（bias 一致樂觀）、池外 4.4–5.5（一致悲觀）；重錨待跑 |
| D | ✅ 實錘 | uniform best-of-10 = −8.38 vs 池抽樣 −3.47（差 ~5dB） |

（SM 重錨前後誤差對比＝C 臂判準正式答案，離線待跑）

## 5. 結論 (Conclude)

- 待。分岔預告：A 通則成立且前緣真值 ≥ -1.5 → 精修 round 直接發；A 崩 → 乾淨前緣是海市蜃樓 → 轉 DIP 生成式搜尋；C 餵不亮（重錨後仍 >3dB）→ SM 架構問題,精修改 trust-region 保守版。

## 6. 後續決策 (Next)

- 解鎖鏈：A+C → 「乾淨 warm-start 精修 round」（起點+導航儀齊備）；B → 精修的編輯算子清單；D → 指導者簡報的 random 基線。
- rad 累計 ~110 條（R7 15＋R8 97）→ rad head pretrain（Stage-3）資料充足。

## 7. 歸檔指向 (Archive)

- configs/README 列: 無（零新 config；`script/dedust.py` 擴充 select-r8/--input/--store，測試 tests/test_dedust.py 11 條）
- 結果夾: `DATASET_PATH/dedust_r8/`（NAS）
- memory: [[project_benchmark_vs_random]] · [[project_radiation_pattern]]
- ONGOING 動作: 發車時 🔵、收檔移 ✅
