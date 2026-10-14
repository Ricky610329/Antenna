# 進行中實驗追蹤（ONGOING）

> **live 操作板**：只放「現在在跑／待跑」＋重啟指令。為什麼與結論看 [研究日誌](../docs/log/README.md)，config 全集看 [README.md](README.md)。
> 2026-10-12 前的完整流水帳（R80 收尾、Codex 的 R81 release 機制、diffsim、資料工廠、🔜 候選、✅ 指標）原樣封存在
> [docs/log/archive/ongoing-2026-10-12.md](../docs/log/archive/ongoing-2026-10-12.md)。

最後更新：2026-10-14 22:30

## 🔵 R81 寬頻濾波器 spec v2（running）→ [round-81](../docs/log/round-81-wide-filter.md)

- **spec（10/13 再改，兩階段）**：S11/S22 也要過（26.5–29.5 GHz ≤ −10 dB）。先用 **v4a**（S21 阻帶 −15 dB，`configs/dual_r81_v4a.yaml`，現行搜尋目標）；過了再收緊到 **v4**（阻帶 −20 dB，`configs/dual_r81_v4.yaml`）。v2/v3 只剩歷史。量測一直沒變，全部真值沿用。
- **現況（10/14 22:30）**：v4a 已公證（b25_008 +0.21／+0.20 等四筆）；v4 最佳 −0.87（`b45_007`，單次）。
  學長 10/14 晚許願：量測 **18–40 GHz**、18–24.5 與 36–40 ≤ −20（`dual_r81_v5.yaml`）；試探版 −15（`dual_r81_v5a.yaml`）。老師尚未拍板。
  **Ricky 定**：先把 18 GHz 納入，持續跑。新批一律 18–40 量測，選樣兩線交替——A 線 `--config configs/dual_r81_v4x.yaml`（v4 計分，先收尾）、B 線 `--config configs/dual_r81_v5.yaml`（往 18 GHz 推）。
  `r81v5m01`（28 筆好設計 18–40 補量，prio 0）跑中；b47/b48 為舊 22–40 批，跑完即止。
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
