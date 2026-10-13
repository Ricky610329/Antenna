# 進行中實驗追蹤（ONGOING）

> **live 操作板**：只放「現在在跑／待跑」＋重啟指令。為什麼與結論看 [研究日誌](../docs/log/README.md)，config 全集看 [README.md](README.md)。
> 2026-10-12 前的完整流水帳（R80 收尾、Codex 的 R81 release 機制、diffsim、資料工廠、🔜 候選、✅ 指標）原樣封存在
> [docs/log/archive/ongoing-2026-10-12.md](../docs/log/archive/ongoing-2026-10-12.md)。

最後更新：2026-10-13 11:00

## 🔵 R81 寬頻濾波器 spec v2（running）→ [round-81](../docs/log/round-81-wide-filter.md)

- **spec v3**（學長 10/13 修訂，`configs/dual_r81_s21_v3.yaml`，現行）：S21 在 26–30 GHz ≥ −3 dB；22–24.5 GHz 與 36–40 GHz ≤ −20 dB；掃 22–40 GHz、Discrete。量測與 v2 相同，v2 的真值直接沿用。
- **現況（10/13 11:00）**：v2 下最好 WM −0.24（`r81v2_b08_006`，單次）；改 v3 重算 303 筆後最好 **−1.06**（`r81v2_b08_008`）：通帶 −0.74、兩阻帶各差約 1 dB。
  佇列：b10（v3 第一批，prio 3 插隊）＞ b09（v2 爬山，prio 4）＞ b03 剩餘（prio 5）＞ 填空池 g01（prio 9）。出現 WM>0 先重測公證。
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

## R80 對稱探索：✅ 10/11 收尾凍結，不再派工 → [round-80](../docs/log/round-80-symmetry-exploration.md)
