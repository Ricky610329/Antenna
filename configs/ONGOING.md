# 進行中實驗追蹤（ONGOING）

> **live 操作板**：只放「現在在跑／待跑」＋重啟指令。為什麼與結論看 [研究日誌](../docs/log/README.md)，config 全集看 [README.md](README.md)。
> 2026-10-12 前的完整流水帳（R80 收尾、Codex 的 R81 release 機制、diffsim、資料工廠、🔜 候選、✅ 指標）原樣封存在
> [docs/log/archive/ongoing-2026-10-12.md](../docs/log/archive/ongoing-2026-10-12.md)。

最後更新：2026-10-12 晚

## 🔵 R81 寬頻濾波器 spec v2（running）→ [round-81](../docs/log/round-81-wide-filter.md)

- **spec v2**（學長 10/12 修訂，`configs/dual_r81_s21_v2.yaml`）：S21 在 26–30 GHz ≥ −3 dB；22–24.5 GHz 與 36–40 GHz ≤ −15 dB；掃 22–40 GHz、Discrete。
- **現況**：v2 第一批 60 筆 `dedust_r81v2b01s1` … `s6` 三台跑中。殺手是 36–40 GHz 高阻帶。
- **資料根**：`\\140.123.106.219\temp\碩二_鄒穎麒's\antenna\experiments\r81_wide_filter_20261007\v2_v001\dataset`
- **正式機重啟**（三台同一行，repo 根、原環境）：

  ```powershell
  git pull --ff-only origin GAN
  .\script\r81v2.cmd
  ```

- **收檔後選下一批**（開發機，只讀 NAS）：

  ```bash
  python -m script.r81_curve_sm score --stores <store dir> ...       # 看本批實測 WM
  python -m script.r81_curve_sm build --batch N --out <dir>           # 重訓 5 顆 SM、選 60 筆（30/18/12）
  python -m script.dedust --dataset-root <v2 資料根> check-dup --input <dir>   # 發車前必跑，exit 1 不發車
  ```

  派工照 `script/CLAUDE.md` 第 5 條的 `jobs-add`（`--dataset-root` 放子命令前），必帶 `--scope symmetry_filter_20261007` 與 `--config`。

## R80 對稱探索：✅ 10/11 收尾凍結，不再派工 → [round-80](../docs/log/round-80-symmetry-exploration.md)
