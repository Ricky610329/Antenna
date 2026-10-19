# 進行中實驗追蹤（ONGOING）

> **live 操作板**：只放「現在在跑／待跑」＋重啟指令。為什麼與結論看 [研究日誌](../docs/log/README.md)，config 全集看 [README.md](README.md)。
> 2026-10-12 前的完整流水帳（R80 收尾、Codex 的 R81 release 機制、diffsim、資料工廠、🔜 候選、✅ 指標）原樣封存在
> [docs/log/archive/ongoing-2026-10-12.md](../docs/log/archive/ongoing-2026-10-12.md)。

最後更新：2026-10-20 06:40

## 🔵 R81 寬頻濾波器 spec v2（running）→ [round-81](../docs/log/round-81-wide-filter.md)

- **主目標（學長 10/15 23:00 定優先）＝v6**：量測 11–40 GHz，S21 通帶 26–30 ≥ −3、阻帶 11–19 與 36–40 ≤ −20（`configs/dual_r81_v6.yaml`；−15 版 `dual_r81_v6a.yaml` 參考），S11/S22 26.5–29.5 ≤ −10 照 Ricky 要求保留。11 GHz 直流直通，實際需兩埠斷開：`r81_curve_sm build --batch N --n 24 --climb-n 10 --climb-top 10 --max-flips 12 --exploit-lcb 0.5 --parent-score violation --p-cross 0.3 --disconnect --push 0.5 --reseam 0.2 --reseam-amp 2 --reseam-reach 1 --config configs/dual_r81_v6.yaml`，enqueue `--prio -2 --shards 3`。
- **現況（10/19 18:45）**：v6 最佳 `b78_006` −4.28（單次）；b72 起停滯，交界推移（`--push 0.5`）連兩批出最佳。選樣已取消 `--relax`。批次 b79 起照常跑；218 改從 `C:\antenna\Antenna` 啟動（Defender 擋 Documents）。
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

## 🔵 R83 生成池 exploit 改由變化量模型排序（running）→ [round-83](../docs/log/round-83-delta-exploit.md)

- b89–b91 加 `--delta-exploit`（dclimb 5／climb 5、rot 4 維持）；判準＝三批 dexploit「> −5」≥ 40% 且中位 ≥ −6.5。R82 已結（✅，[round-82](../docs/log/round-82-delta-surrogate.md)）。

## R80 對稱探索：✅ 10/11 收尾凍結，不再派工 → [round-80](../docs/log/round-80-symmetry-exploration.md)
