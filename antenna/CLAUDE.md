# antenna/ 核心程式規範

> 這裡是核心（`G(spec) → pattern → SM/SIM → loss → 反傳`）。動手前讀 root CLAUDE.md
> 「北極星」與「架構不變式」兩節——本檔只重申最容易踩的三條，不複製全文。

## 三條硬的

1. **層級單向**：`antenna/` 核心零 legacy 依賴——只有 `train.py`、`script/`、`application/`
   可以 import `antenna/legacy/`。核心（pattern/response/training/models/optim/losses/zoo）
   互相引用 OK，**不得**反向依賴周邊（監控/容錯/視覺化＝外接模組）。
2. **`utils/utils.py` 的 `Path` 不可搬離**（有 `__reduce__`，被烘進舊 checkpoint）；
   `Data`/`DataManager`/`Record`/checkpoint 只 pickle 純 payload，搬類別 OK。
3. **tau 歸 ACP**：二值化是訓練管線固定一步，模型不碰 tau。

## 收尾標準

- 任何結構性改動：`python -m pytest tests/ -q`（repo 根）全綠＋golden 零漂移，才算完。
- 型別註解＝輕量文件；`TypeVar`/`Generic`/`ParamSpec`/`@overload` 一律不用。
- 看到 over-design 就簡化、能刪不要包——但**先 audit 既有 code 再動，不平行重建**。
