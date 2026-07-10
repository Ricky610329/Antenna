腳本 (Script)
==========

正常情況下是不能直接執行的，所以可以用以下方式解決。

1. 在Python檔的開頭加入下列程式碼
    ```python
    from sys import path
    from os.path import dirname, join
    path.append(join(dirname(__file__),'..'))
    ```
2. 在終端機 (根目錄) 執行以下指令
    ```bash
    python -m script.<腳本名>

    # Example
    python -m script.kill
    ```
## 主要腳本一覽（2026-07-06）

| 腳本 | 用途 |
|---|---|
| `dedust.py` | **批次 HFSS 驗證線**（R7 起的研究主力）：select-* 生輸入 → run 燒 HFSS → report 看結果（子命令地圖見檔頭 docstring） |
| `dedust.py worker` / `jobs-add` | **資料工廠**（2026-07-10）:NAS 佇列派工＋單筆 watchdog＋連敗保險絲（詳 `script/CLAUDE.md`） |
| `sm_reanchor.py` | SM 乾淨區重錨（train/eval;產 NAS `sm_reanchor.pth`） |
| `status.py` | NAS run 狀態掃描（`--md` 貼 ONGOING;`--alert --notify-topic` 當 watchdog） |
| `expected_best.py` | R6 期望基準尺（每 round 收檔可重跑疊圖） |
| `pattern_anatomy.py` | 池結構特徵快取（`collect-pool` → `tmp/pattern_anatomy/pool.npz`,多個 select 依賴） |
| `round_report.py` / `analyze.py` / `benchmark_vs_random.py` | 線上 run 的歸檔圖表 / 重現診斷 / worst-margin 對標 |

接手導覽：先讀 `docs/log/README.md`（時間軸）→ `configs/ONGOING.md`（live）→ 該 round 檔。
