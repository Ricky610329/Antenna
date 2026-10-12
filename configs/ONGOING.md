# 進行中實驗追蹤（ONGOING）

> **live 操作板**：只放「現在在跑／待跑」＋重啟指令。為什麼與結論看 [研究日誌](../docs/log/README.md)，config 全集看 [README.md](README.md)。
> 2026-10-12 前的完整流水帳（R80 收尾、Codex 的 R81 release 機制、diffsim、資料工廠、🔜 候選、✅ 指標）原樣封存在
> [docs/log/archive/ongoing-2026-10-12.md](../docs/log/archive/ongoing-2026-10-12.md)。

最後更新：2026-10-13 02:30

## 🔵 R81 寬頻濾波器 spec v2（running）→ [round-81](../docs/log/round-81-wide-filter.md)

- **spec v2**（學長 10/12 修訂，`configs/dual_r81_s21_v2.yaml`）：S21 在 26–30 GHz ≥ −3 dB；22–24.5 GHz 與 36–40 GHz ≤ −15 dB；掃 22–40 GHz、Discrete。
- **現況**：b01 收齊（59/60），最好 WM −5.60（單次），未破首批 −4.28；SM 前瞻 ρ −0.21、exploit 不勝 random → b02 改小步變異 1–8 格。
  b02（prio 5）三台跑中；填空池 g01（prio 9）墊底。殺手仍是 36–40 GHz 高阻帶。
- **資料根**：`\140.123.106.219\temp\碩二_鄒穎麒's\antenna\experiments\r81_wide_filter_20261007\v2_v001\dataset`
81_wide_filter_202610072_v001\dataset`
- **正式機重啟**（三台同一行，repo 根、原環境）：

  ```powershell
  git pull --ff-only origin GAN
  .\script
81v2.cmd
  ```

- **HFSS 不空轉規則**：主批 prio 5；永遠墊一池 prio 9 填空池（未認領 < 3 個 job 就補）；主批全部被認領時就選下一批。
- **選下一批／補池**（開發機）：

  ```bash
  python -m script.r81_curve_sm score --stores <store dir> ...                 # 看實測 WM
  python -m script.r81_curve_sm build --batch N --out <dir>                     # 重訓 5 顆 SM、選 60 筆
  python -m script.r81_curve_sm enqueue --input <dir> --name r81v2b0N           # 切 6 片、check-dup、入佇列（填空池加 --prio 9、名稱 r81v2gNN）
  ```

## R80 對稱探索：✅ 10/11 收尾凍結，不再派工 → [round-80](../docs/log/round-80-symmetry-exploration.md)
