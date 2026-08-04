---
name: nas-backup
description: NAS → 本機備份量測資料（HFSS 模擬很慢、重跑不回來,這是全專案最珍貴的資產）。增量、只讀 NAS、零刪除,跑完自動對帳。用法：/nas-backup（完整備份＋對帳）或 /nas-backup verify（只對帳,不複製）
---

# NAS → 本機備份

> **為什麼**：`pattern → HFSS → (S11, Gain, rad)` 一筆要跑 ~100 秒。六萬多筆量測 ≈ **兩千小時機時**。
> 程式碼有 git、權重能重訓、圖能重畫——**只有量測資料重跑不回來**。NAS 單點故障 = 整個研究歸零。

## 一、備份什麼（全部,不要聰明過濾）

| NAS 來源 | 內容 | 為什麼不可再生 |
|---|---|---|
| `dataset/` | 全部 store + SM 權重 | 量測資料本體 |
| `result/` | 38 個線上訓練 run | `online/` 是該 run 收集的量測、`patterns/` 是最佳 pattern |
| `rad圖/` | 方向圖圖檔 | 小、順手 |

**每個 store 的完整結構**（漏抄任何一項都是資料損失）：

```
<store>/
├─ <hash>.pt      = (pattern 25×25, y(2,17))   y = [S11 dB, Gain dB] × 17 頻點
├─ rad/<id>.pt    = {theta(181), phi0(181), phi90(181)}   ← 原始方向圖曲線
└─ results.json   = 每候選衍生指標 (wm/oob/contrast/rolloff/sel…)
```

⚠ **不要用 `/XF` `/XD` 做任何排除**。看起來像垃圾的東西可能不是：`rad/` 是子夾、
`results.json` 不是 `.pt`、`result/*/online/` 藏著量測。省下的容量遠不值那個風險。
（唯一例外見 §5 快速模式,且只在明確趕時間時用。）

## 二、執行（PowerShell,背景跑）

```powershell
$dst = "C:\Users\Ricky\antenna_nas_backup"
$src = "T:\碩二_鄒穎麒's\antenna"
$log = "$dst\_backup_log.txt"
robocopy "$src\dataset" "$dst\dataset" /E /XO /MT:16 /R:2 /W:5 /NFL /NDL /NP /LOG+:$log
robocopy "$src\result"  "$dst\result"  /E /XO /MT:16 /R:2 /W:5 /NFL /NDL /NP /LOG+:$log
robocopy "$src\rad圖"   "$dst\rad圖"   /E /XO       /R:2 /W:5 /NFL /NDL /NP /LOG+:$log
```

旗標的意思（**別亂改**）：

- `/E` 含空夾遞迴 ／ `/XO` 只抄比目的地新的 → **可中斷續跑、重跑不重抄**
- **沒有 `/MIR`** —— `/MIR` 會刪掉本機多出來的檔,一次手滑就毀掉舊備份。永遠不要加
- 只讀 NAS、只寫本機 → NAS 零改動,符合「NAS 唯讀」規矩
- exit code **< 8 才算成功**（0=無新檔、1=有複製、3=有複製+有跳過…都正常;≥8 才是真失敗）

跑的時候批次線 daemon 可能正在寫 store —— **不用停**。SampleStore 一筆一檔,
半寫的檔頂多下次增量補上,不會污染既有資料。

## 三、對帳（每次備份後必跑）

```
/c/Users/Ricky/miniforge3/envs/ant/python.exe .claude/skills/nas-backup/verify.py
```

逐類比對 NAS 與本機的筆數,任何一類短少就是 **exit 1**。短少時：重跑 §2 指令
（`/XO` 會自動只補缺的),再對帳一次。**別在沒對帳的情況下宣稱備份完成。**

## 四、記錄

備份根目錄的 `BACKUP_INFO.md` 記「上次備份時間 + 各類筆數 + schema 說明」。
對帳過了就更新它——這是未來的你（或還原時的你）唯一的說明書。

## 五、快速模式（只在趕時間時用,且要說出口）

`result/` 有 ~83 GB 是 `checkpoint/generator_<epoch>.pth` 逐 epoch 快照（單 run 449 個、
每個 19.8 MB;`sm.pth` 是覆蓋式只有 1 個）。這些**已驗證對做圖無用**：`script/figs/*` 與
`round_report.py` 零 `.pth` 載入,而每版 SM 的準度早已記在 `docs/kpi*.csv`（git 版控）。

趕時間時可對 `result/` 加 `/XF generator_*.pth`,把 105 GB 壓到 ~22 GB。
**但 `dataset/` 永遠全抄**,而且要在回報裡明講跳過了什麼——不准默默省略。

## 六、護欄

- 全程只讀 NAS、零刪除。學長 `碩二` 樹更是碰都不碰
- 不用 `/MIR`、不刪本機任何既有檔
- 對帳不過 → 不算完成,照實說
- 備份與「NAS 清理」是兩件事。要清理先確認本機這份對帳過了,而且單獨徵得同意
