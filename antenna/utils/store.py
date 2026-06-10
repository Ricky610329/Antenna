"""
antenna/utils/store.py — 輕量樣本庫：一筆一檔，hash 即檔名。

新格式 (未來程式碼的標準)，取代 DataManager 單一 pickle 的「儲存」職責：

    <store 資料夾>/
      a3f9c2e1b4ddca01.pt    # 一筆 = 一檔，torch.save((x, y))；檔名 = 內容 SHA-1 前 16 碼

設計重點 (對比舊 DataManager)：
- append = 寫一個 ~3KB 小檔 (O(1))，不再每存一筆就全量重寫整個 pickle (NAS 上尤其痛)。
- 檔名 = 內容 hash → 「檔案存在」即重複，去重免維護指紋集。
- 半截壞檔只損一筆，不會毀掉整個資料集；寫入採 tmp → os.replace 原子搬移。
- 純 tensor tuple 落地，載入走 weights_only=True (無 pickle 任意物件的版本/安全問題)。
- 實作 torch Dataset 介面 → 可直接餵 DataLoader (train_by_datas 等不用改)。

舊 DataManager (.data 單一 pickle) 原樣保留：學長既有 code 與 NAS 上舊資料集
(patch_single_mirror / patch_dual) 仍走舊路；正式轉換用 script/convert_dataset.py
(待確認後一次執行)。
"""
import os
from hashlib import sha1

import torch
from torch import Tensor
from torch.utils.data import Dataset
from loguru import logger

from .utils import Path


def fingerprint(x: Tensor, y: Tensor) -> str:
    """(x, y) 內容指紋：shape + 原始 bytes 的 SHA-1 前 16 碼 (同內容必同名)。"""
    h = sha1()
    for t in (x, y):
        t = t.detach().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


class SampleStore(Dataset):
    """一筆一檔的 (pattern, response) 樣本庫。"""

    def __init__(self, rootdir, *, verbose: bool = True):
        self.rootdir = Path(rootdir)
        self.rootdir.mkdir(parents=True, exist_ok=True)
        for leftover in self.rootdir.glob("*.tmp"):   # 清掉先前崩潰殘留的半截暫存檔
            leftover.unlink()
        self._files = sorted(self.rootdir.glob("*.pt"))
        self._cache: dict = {}                        # 樣本小，讀過即留在 RAM (重複 epoch 不再打磁碟)
        if verbose:
            logger.info(f"SampleStore '{self.rootdir.name}' 就緒，共 {len(self._files)} 筆。")

    def add(self, x: Tensor, y: Tensor) -> bool:
        """寫入一筆；內容重複 (同 hash) 不落地並回傳 False。"""
        x, y = x.detach().cpu(), y.detach().cpu()
        path = self.rootdir.joinpath(f"{fingerprint(x, y)}.pt")
        if path.exists():
            return False
        tmp = path.with_suffix(".tmp")
        torch.save((x, y), tmp)
        os.replace(tmp, path)                         # 原子搬移：不留半截正式檔
        self._cache[len(self._files)] = (x, y)
        self._files.append(path)
        return True

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx) -> tuple:
        if idx not in self._cache:
            self._cache[idx] = torch.load(self._files[idx], weights_only=True)
        return self._cache[idx]

    def __repr__(self):
        return f"SampleStore(dir={self.rootdir}, n={len(self)})"
