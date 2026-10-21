# 進行中實驗追蹤（ONGOING）

> **live 操作板**：只放「現在在跑／待跑」＋重啟指令。為什麼與結論看 [研究日誌](../docs/log/README.md)，config 全集看 [README.md](README.md)。
> 2026-10-12 前的完整流水帳（R80 收尾、Codex 的 R81 release 機制、diffsim、資料工廠、🔜 候選、✅ 指標）原樣封存在
> [docs/log/archive/ongoing-2026-10-12.md](../docs/log/archive/ongoing-2026-10-12.md)。

最後更新：2026-10-21 17:53

## 🔵 R81 寬頻濾波器 spec v2（running）→ [round-81](../docs/log/round-81-wide-filter.md)

- **主目標（學長 10/15 23:00 定優先）＝v6**：量測 11–40 GHz，S21 通帶 26–30 ≥ −3、阻帶 11–19 與 36–40 ≤ −20（`configs/dual_r81_v6.yaml`；−15 版 `dual_r81_v6a.yaml` 參考），S11/S22 26.5–29.5 ≤ −10 照 Ricky 要求保留。11 GHz 直流直通，實際需兩埠斷開：`r81_curve_sm build --batch N --n 24 --climb-n 10 --climb-top 10 --max-flips 12 --exploit-lcb 0.5 --parent-score violation --p-cross 0.3 --disconnect --push 0.5 --reseam 0.2 --reseam-amp 2 --reseam-reach 1 --config configs/dual_r81_v6.yaml`，enqueue `--prio -2 --shards 3`。
- **現況（10-21 09:01）**：目標暫改 **v6a（−15）先達標**（Ricky 10/20），v6a 最佳 `b111_004` −0.21（單次，只卡通帶 30 GHz）；v6 最佳 `b92_016` −3.11。現行建批見 R88／R89 §3（`--climb-main ctx --model-arms ens --explore-model cnn …`）。通帶天花板：低阻一有抑制，通帶最佳封頂 −0.19（1217 筆中 1193 筆同血統 a0，見 round-81 末節）。
- **已達標**：v4a（b25_008 +0.21，公證）；v4x（18–40，`b61_008` +0.03，216 公證一致）——v4x 線停止。v5a 斷開線（Ricky 好奇，非規格）最佳 −2.69，降 prio 1 不再加批。
- **R80 對稱天線續探（學長要求，慢慢探索）**：最佳 `r80sc_b05_014` WM +0.599（公證）；`python -m script.r80_sym_climb build/enqueue/score`，量測同 R80，釘選 218、prio 1（排在 v6 之後）。
- **資料根**：`\\140.123.106.219\temp\碩二_鄒穎麒's\antenna\experiments\r81_wide_filter_20261007\v2_v001\dataset`
- **正式機重啟**（三台同一行，repo 根、原環境）：

  ```powershell
  git pull --ff-only origin GAN
  .\script\r81v2.cmd
  ```

- **HFSS 不空轉規則**：主批 prio 5；永遠墊一池 prio 9 填空池（未認領 < 3 個 job 就補）；主批全部被認領時就選下一批。
- **選下一批／補池**（開發機）：

  ```bash
  python -m script.r81_curve_sm score --stores <store dir> ...                 # 看實測 WM
  python -m script.r81_curve_sm build --batch N --out <dir>                     # 重訓 5 顆 SM、選 60 筆
  python -m script.r81_curve_sm enqueue --input <dir> --name r81v2b0N           # 切 6 片、check-dup、入佇列（填空池加 --prio 9、名稱 r81v2gNN）
  ```

## 🔜 候選：非 a0 家族探索輪（學費預算制）

- what：撥 1/3 HFSS（每批 8 筆或每 3 批 1 批）給非 a0 起點——①學長 24–32 歷史中通帶最好、結構與 a0 差 > 150 格者重量 11–40 篩低頻天然抑制；②把 a0 的低頻抑制區塊（R85 掃描：左下 4–7 欄）移植到通帶好的遠親。預算 6 批，KPI＝非 a0 最佳 v6a WM 的趨勢，距主線 1 dB 內畢業、預算盡回報。
- why：低阻 ≥ 0 的 806 筆全是 a0、與最佳差 ≤ 82 格；通帶天花板 −0.19 疑為家族性（同結構無阻帶要求時通帶可達 −1.08 dB）。
- **觸發條件**：R89 整合批＋R88 三批（b112–b114）收齊後 v6a 最佳仍 < −0.15（未破天花板 0.05 以上）→ 開輪。Ricky 10/21 提議「卡很久就多開探索輪」。

## 🔜 候選：空腔單格加銅／第二種裂口（Ricky 10/21 看圖提問）

- 空腔：最佳設計左側兩塊空白（上圈 4–11 列 0–7 欄、下圈 12–23 列 0–12 欄）共 109 空格，前段只單格加銅試過 18 格（中位 −0.04；(21,6) +0.12、(6,5) +0.09；(18,6)、(20,9) −3.5～−3.9）。R85 整塊加銅：下圈 4–7 欄毀低阻 −5.7～−11.5，左緣 0–3 欄對 WM 無害但耗高阻 −4～−8。→ 實測單格掃描優先掃上圈與下圈 0–3 欄、避開下圈 4–7 欄。
- 裂口：1248 筆共 421 種裂口（push 229、reseam 20…），改裂口子代中位 ΔWM −1.41、改善率 22%；前 50 名全同一種裂口（換裂口其他結構要重配）。第二種裂口群 `b104_011` 等（12–19 欄缺口位置不同）通帶 −0.47 但低阻 +1.3～1.5、高阻 +2～4，餘裕多 → 可當第二個底爬通帶，或與主裂口群做 R89 式整合。
- **觸發條件**：R89 整合批收齊後，依結果決定與探索輪合併開或單獨開。

## 🔵 R89 整合前段設計（running）→ [round-89](../docs/log/round-89-merge.md)

- **10-21 17:53 公證最佳 `mg07_003` +0.063**（n03：216＝218）；n02 `mg06_002` +0.019 亦通過。
- **10-21 17:22 公證通過**：`mg05_009` +0.006（216＝218 完全一致）。新高 `mg07_003` +0.063（單次，公證 n03）；切回 v6 需 ≥ +0.1。
- **10-21 16:30 Ricky：v6a 要做到 ≥ +0.1（公證）才切回 v6**。
- **10-21 16:27 v6a 首筆 ≥ 0**：`mg05_009` +0.006（218，單次）→ 公證 `r81v6n01` 釘 216。
- mg01（24 筆）收齊：最佳 `mg01_011` **−0.110**（單次），達「整合有效」門檻。mg02（疊加 12＋第二代移植 12，prio −3）已發車。
- mg02 指令：`python -m script.r89_merge --n 24 --base-mode top --rank small --stack 12 --stack-base r81v2_b111_010_0dcdaf1d --name r81v6mg02 --out <dir>`

## 🔵 R88 利用／探索臂只准小步（running）→ [round-88](../docs/log/round-88-small-step.md)

- b112–b114 `--climb-main ctx --model-arms ens --explore-model cnn --small-exploit 4 --small-explore 2`；目標 v6a（−15）。R87 已結（✅ 成立，差距小）；b110 為 R87 延續批、b111 為新設定首批。

## 🔵 R85 拉扯區外擾動＋修復（running）→ [round-85](../docs/log/round-85-perturb-repair.md)

- A 區塊掃描 `r81v6sc01`（24 筆，prio −3）已發車；B 擾動＋修復待 R84 收輪（b98 起）。

## R80 對稱探索：✅ 10/11 收尾凍結，不再派工 → [round-80](../docs/log/round-80-symmetry-exploration.md)
