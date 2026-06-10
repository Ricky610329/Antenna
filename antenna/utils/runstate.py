"""
antenna/utils/runstate.py — 訓練狀態 (結果夾即資料庫)。

取代 Record(=TEMP) 在「訓練路徑」的角色，把單一 pickle 拆成可讀的檔案：

    <結果夾>/
      metrics.csv          # 純量時序：一個 epoch append 一行 (O(1)；pandas/Excel 直接讀)
      patterns/{hash}.pt   # 每筆「模擬過的」(pattern, response, loss)；
                           #   hash 即檔名 → 檔案存在 = 模擬過 = 去重快取

對比舊 temp.record：Record.save() 每 epoch 把「歷來全部 pattern/響應」重新 pickle
一次到 NAS (O(n²) I/O)；本類 append 一行 csv + 必要時寫一個小 .pt，皆 O(1)。

斷點續跑：建構時讀回 metrics.csv → last_epoch / loss 歷史 / 去重快取全部就位。
Record 與舊 temp.record 不動，留給既有結果與 app.py (歷史檔案館)。
"""
import csv
from collections import defaultdict

import torch
from loguru import logger

from .utils import Path
from .store import fingerprint

#? csv 欄位順序 (固定)：epoch 在前、pattern_hash 殿後 (指向 patterns/ 的檔)
SCALAR_KEYS = ("epoch", "real_loss", "fake_loss", "min_loss",
               "real_loss_average", "r_feed", "time", "pattern_hash")


class RunState:
    """一次訓練 run 的狀態：純量時序 + 模擬過的 pattern 快取。"""

    def __init__(self, rootdir, *, verbose: bool = True):
        self.rootdir = Path(rootdir)
        self.rootdir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.rootdir.joinpath("metrics.csv")
        self.patterns_dir = self.rootdir.joinpath("patterns")
        self.patterns_dir.mkdir(exist_ok=True)
        self._series: dict = defaultdict(list)
        if self.metrics_path.exists():
            self._load_metrics()
            if verbose:
                logger.info(f"RunState 載回 {self.last_epoch} 個 epoch 的歷史 (metrics.csv)")

    # ── 純量時序 (語義對齊 Record，golden 保真) ────────────────────────────────
    def append(self, key: str, value):
        self._series[key].append(value)

    def series(self, key: str) -> list:
        return self._series[key]

    def last(self, key: str, default=None):
        s = self._series[key]
        return s[-1] if s else default

    def average(self, key: str):
        s = self._series[key]
        return sum(s) / len(s) if s else None

    def early_stop(self, key: str, patience: int = 10) -> bool:
        """最近 patience 筆都沒優於先前最佳 → True (觸發 rollback)。語義照 Record.early_stop。"""
        values = self._series[key]
        if len(values) < patience + 1:
            return False
        best_so_far = min(values[: len(values) - patience])
        return all(v >= best_so_far for v in values[len(values) - patience:])

    def best_epoch(self, key: str = "real_loss") -> int:
        """key 最小值「首次」出現那筆的 epoch (rollback 載回用)。"""
        s = self._series[key]
        return self._series["epoch"][s.index(min(s))]

    @property
    def last_epoch(self) -> int:
        return int(self.last("epoch", 0))

    def save_row(self):
        """把本 epoch 各欄的最後一筆寫成 csv 一行 (append；首次自動寫表頭)。"""
        new_file = not self.metrics_path.exists()
        with open(self.metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(SCALAR_KEYS)
            w.writerow([self.last(k, "") for k in SCALAR_KEYS])

    def _load_metrics(self):
        with open(self.metrics_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for k in SCALAR_KEYS:
                    v = row.get(k, "")
                    if k == "pattern_hash":
                        self._series[k].append(v)
                    elif v != "":
                        self._series[k].append(int(v) if k == "epoch" else float(v))

    # ── patterns：模擬快取 (hash 即檔名 = 去重) ───────────────────────────────
    def lookup(self, pattern) -> tuple:
        """模擬過 → (response, loss, hash)；沒見過 → None。"""
        h = fingerprint(pattern)
        path = self.patterns_dir.joinpath(f"{h}.pt")
        if not path.exists():
            return None
        _, response, loss = torch.load(path, weights_only=True)
        return response, loss, h

    def add_pattern(self, pattern, response, loss: float) -> str:
        """記下一筆模擬結果，回傳 hash (寫進當 epoch 的 pattern_hash 欄)。"""
        h = fingerprint(pattern)
        path = self.patterns_dir.joinpath(f"{h}.pt")
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            torch.save((pattern.detach().cpu(), response.detach().cpu(), float(loss)), tmp)
            import os
            os.replace(tmp, path)
        return h

    def pattern_at(self, epoch: int):
        """某 epoch 的 (pattern, response)，供 summary/事後分析。"""
        idx = self._series["epoch"].index(epoch)
        h = self._series["pattern_hash"][idx]
        pattern, response, _ = torch.load(self.patterns_dir.joinpath(f"{h}.pt"), weights_only=True)
        return pattern, response

    def __repr__(self):
        return f"RunState(dir={self.rootdir}, epochs={self.last_epoch})"
