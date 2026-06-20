"""
antenna/utils/replay.py — 輕量經驗回放緩衝 (Experience Replay Buffer)。

線上訓練 SM 時，取代「把單一最新樣本擬到收斂」的反模式 (catastrophic forgetting)：
改成把最近的 (pattern, response) 收進一個「定容 FIFO」緩衝，每 epoch 從緩衝
做少數步 minibatch/單樣本訓練 → SM 對「整個探索過的空間」都準、不會只記得最新一筆。
(對應 streaming-regression 的 experience replay；學長的 online/DLF 是這條路的雛形。)

設計：
- in-memory 定容 deque (滿了丟最舊) —— 不落 NAS、無 I/O，每 epoch 重用。
- 實作 torch Dataset 介面 (__len__/__getitem__) → SurrogateModel.train_by_datas / DataLoader 直接吃。
- 存 (pattern, response) 與 online SampleStore 同格式 → train_by_datas 走同一條前處理。
"""
from collections import deque

from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    """定容 FIFO 經驗回放緩衝 (in-memory)，存 (pattern, response) 張量對。"""

    def __init__(self, maxlen: int = 256):
        self._buf = deque(maxlen=maxlen)   #? 滿了自動丟最舊 (FIFO)，無需手動淘汰

    def add(self, pattern, response):
        """收一筆；detach+cpu 落地避免抓著計算圖 / 佔 GPU。"""
        self._buf.append((pattern.detach().cpu(), response.detach().cpu()))

    def __len__(self) -> int:
        return len(self._buf)

    def __getitem__(self, idx) -> tuple:
        return self._buf[idx]
