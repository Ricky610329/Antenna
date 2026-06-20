"""
antenna/utils/replay.py — 經驗回放緩衝 (Experience Replay Buffer) + 動態損失過濾 (DLF)。

線上訓練 SM 時，取代「把單一最新樣本擬到收斂」的反模式 (catastrophic forgetting)：
把最近的 (pattern, response, loss) 全收進定容 FIFO 緩衝，每 epoch 從緩衝做少數步訓練。

對應學長論文 §3.5「經驗回放與動態損失過濾」：
- **全收**：樣本無條件累積進緩衝 (不在寫入端篩；寫入端篩=論文點名的 baseline，會偽收斂)。
- **DLF**：每次 SM 重訓時，用「累計門檻 λ_t」對整個緩衝重新過濾，只取 loss ≤ λ_t 的菁英子集訓。
  門檻隨訓練自動收緊 → 前期重多樣性、後期重精準 (論文消融顯示比 baseline 改善 >50%)。

設計：in-memory 定容 deque (不落 NAS)、實作 torch Dataset → train_by_datas/DataLoader 直接吃。
"""
from collections import deque

import torch
from torch.utils.data import Dataset, Subset


class ReplayBuffer(Dataset):
    """定容 FIFO 經驗回放緩衝 (in-memory)，存 (pattern, response, loss)。loss 供 DLF 菁英過濾用。"""

    def __init__(self, maxlen: int = 256):
        self._buf = deque(maxlen=maxlen)   #? 滿了自動丟最舊 (FIFO)

    def add(self, pattern, response, loss):
        """收一筆 (全收、不在寫入端篩)；detach+cpu 落地、loss 記下供 DLF 用。"""
        self._buf.append((pattern.detach().cpu(), response.detach().cpu(), float(loss)))

    def __len__(self) -> int:
        return len(self._buf)

    def __getitem__(self, idx) -> tuple:
        pattern, response, _ = self._buf[idx]
        return (pattern, response)         #? train_by_datas 只要 (pattern, response)

    def elite(self, threshold: float) -> Subset:
        """DLF：回傳 loss ≤ threshold 的菁英子集 (torch Subset)。每輪用當前 λ_t 重新過濾整個緩衝。"""
        idx = [i for i, (_, _, loss) in enumerate(self._buf) if loss <= threshold]
        return Subset(self, idx)

    def patterns(self):
        """回傳所有已見 pattern 攤平堆疊 (M, N)，供 boundary loss 算 trust-region 距離；空則回 None。"""
        if not self._buf:
            return None
        return torch.stack([p.reshape(-1) for p, _, _ in self._buf])
