import math
import warnings

from torch.optim.lr_scheduler import _LRScheduler

from antenna.types import *
from antenna.utils import Record, config

# 模組常數：集中管理字串列舉，避免分散在 __init__ 與型別標註兩處
_ON_PLATEAU_MODES: tuple[str, ...] = ("peak", "reset", "linear")
_METRIC_MODES: tuple[str, ...] = ("min", "max")


def _default_tau_callback(tau: float) -> None:
    """預設 tau callback：向後相容，將溫度寫入 AntennaPattern.tau。

    延遲 import 以避免模組載入時的循環依賴。
    """
    from antenna import AntennaPattern

    AntennaPattern.tau = tau


class AdaptiveCyclicalScheduler(_LRScheduler, Generic[CustomOptimizer]):
    """
    一個融合了 OneCycle、CosineAnnealingWarmRestarts 和 ReduceLROnPlateau 思想的排程器。

    它在週期性的餘弦退火基礎上，為每個週期增加了暖身階段，並能根據監控指標
    在模型停滯時提前觸發重啟。同時，它也同步調整一個外部的溫度參數。
    """

    def __init__(
        self,
        optimizer: CustomOptimizer,
        T_0: int = 50,
        T_mult: int = 1,
        lr_max: float = 0.01,
        lr_min: float = 1e-6,
        temp_max: float = 10.0,
        temp_min: float = 0.1,
        warmup_ratio: float = 0.1,
        mode: str = "min",
        factor: float = 0.5,
        patience: int = 5,
        on_plateau: Literal["peak", "reset", "linear"] = "peak",
        threshold: float = 0.0,
        last_epoch: int = -1,
        tau_callback: Optional[Callable[[float], None]] = None,
    ):
        """

        :param optimizer: 要排程的優化器 (e.g., torch.optim.Adam)。
        :param T_0: 第一個週期的長度 (以 step/batch/epoch 計數，取決於您如何使用 step)。
        :param T_mult: 週期長度乘數。每當週期重啟時，新的週期長度將是當前週期長度乘以 T_mult。
                       若 T_mult=1，則所有週期長度相同。
        :param lr_max: 週期內達到的最高學習率。
        :param lr_min: 週期內達到的最低學習率。
        :param temp_max: 週期內達到的最高溫度值。
        :param temp_min: 週期內達到的最低溫度值。
        :param warmup_ratio: 暖身階段佔整個週期長度的比例 (0.0 到 1.0 之間)。
        :param mode: 監控的指標的優化方向。'min' 表示指標越小越好 (例如 loss)，'max' 則相反 (例如 accuracy)。
        :param factor: 強制重啟後，當前週期長度的縮減因子 (0.0 到 1.0 之間)。用於加速後續週期。
        :param patience: 耐心值。在觸發強制重啟前，容忍指標沒有改善的步數 (step/batch/epoch)。
        :param on_plateau: patience 觸發時的動作
        :param threshold: 判斷指標是否改善的閾值。當前指標與最佳指標的差距必須大於此值才算作改善。
        :param last_epoch: 最後一個已排程的步數/週期數。用於從中斷處恢復訓練。
        :param tau_callback: 每次 step 後接收當前溫度的 callback；未傳入時使用預設行為
                             (寫入 AntennaPattern.tau) 以維持向後相容。
        :raises ValueError: 如果 T_0, T_mult, on_plateau 或 mode 參數無效。
        """
        # --- Tau callback（預設向後相容：自動設定 AntennaPattern.tau）---
        self._tau_callback = tau_callback if tau_callback is not None else _default_tau_callback

        self.record = Record(self.__class__.__name__, config.get("RESULT_PATH"))
        # --- 參數驗證 ---
        if not isinstance(T_0, int) or T_0 <= 0:
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if not isinstance(T_mult, int) or T_mult < 1:
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if on_plateau not in _ON_PLATEAU_MODES:
            raise ValueError(f"on_plateau must be one of {_ON_PLATEAU_MODES}, got {on_plateau!r}")
        if mode not in _METRIC_MODES:
            raise ValueError(f"mode must be one of {_METRIC_MODES}, got {mode!r}")

        # --- 週期性參數 (來自 CosineAnnealing) ---
        self.T_0 = T_0
        self.T_mult = T_mult
        self.T_i = T_0  # 當前週期的長度
        self.T_cur = last_epoch
        self.on_plateau = on_plateau

        # --- 學習率與溫度範圍 ---
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.temp_max = temp_max
        self.temp_min = temp_min
        self.current_temp = temp_max

        # --- 暖身參數 (來自 OneCycleLR) ---
        self.warmup_ratio = warmup_ratio

        # --- 自適應參數 (來自 ReduceLROnPlateau) ---
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.patience_counter = 0
        self.best_metric = float("inf") if mode == "min" else float("-inf")

        super().__init__(optimizer, last_epoch)
        self.base_lrs = [lr_max] * len(self.optimizer.param_groups)

    def get_temp(self) -> float:
        """獲取當前計算出的溫度"""
        return self.current_temp

    def get_lr(self):
        """計算並返回當前的學習率，同步更新內部溫度狀態。

        副作用：更新 self.current_temp — 讓 lr 與 tau 共用同一條 cosine 曲線，
        外部請透過 get_temp() 讀取當前溫度。progress 代表「距離峰值的比例」
        (0 = 峰值 lr_max/temp_max，1 = 谷底 lr_min/temp_min)。
        """
        warmup_steps = int(self.T_i * self.warmup_ratio)
        cosine_span = self.T_i - warmup_steps

        if self.T_cur < warmup_steps:
            # 1. 暖身階段：從 1 (谷底) 線性下降到 0 (峰值)
            progress = 1 - self.T_cur / warmup_steps
        elif cosine_span > 0:
            # 2. 餘弦退火階段：從 0 (峰值) 平滑上升到 1 (谷底)
            cosine_progress = (self.T_cur - warmup_steps) / cosine_span
            progress = 1 - (1 + math.cos(math.pi * cosine_progress)) / 2
        else:
            # warmup_ratio == 1.0 的退化情況：整個週期都是暖身，結尾鎖定在峰值
            progress = 0.0

        lr = self.lr_max - (self.lr_max - self.lr_min) * progress
        self.current_temp = self.temp_max - (self.temp_max - self.temp_min) * progress

        return [lr for _ in self.optimizer.param_groups]

    def _is_metric_better(self, metric):
        if self.mode == "min":
            return metric < self.best_metric - self.threshold
        else:
            return metric > self.best_metric + self.threshold

    def step(self, metric: Optional[float] = None):
        if metric is None:
            warnings.warn(
                "AdaptiveCyclicalScheduler requires a metric to be passed to step() for adaptation.", UserWarning
            )
        else:
            # --- 自適應邏輯 ---
            if self._is_metric_better(metric):
                self.best_metric = metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                self.patience_counter = 0
                self.T_i = max(int(self.T_i * self.factor), self.T_0 // 2)  # 保持最小週期長度限制
                # 以下所有分支的設定值都會被 step() 尾端的 self.T_cur += 1 遞增一格
                warmup_steps = int(self.T_i * self.warmup_ratio)
                if self.on_plateau == "reset":
                    # 回到週期起點，重新開始暖身
                    self.T_cur = -1
                elif self.on_plateau == "peak":
                    # 直接跳到峰值：遞增後 T_cur == warmup_steps，對應 progress=0 (即 lr_max)
                    self.T_cur = warmup_steps - 1
                elif self.on_plateau == "linear":
                    # 從當前 lr 反推在暖身線上的位置，讓 lr 連續地繼續往上爬
                    if warmup_steps > 0:
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        lr_span = self.lr_max - self.lr_min
                        ratio = (current_lr - self.lr_min) / lr_span if lr_span > 0 else 0.0
                        ratio = max(0.0, min(1.0, ratio))
                        self.T_cur = int(round(ratio * warmup_steps))
                    else:
                        # 沒有暖身區間時退化為 reset 行為
                        self.T_cur = -1

        # --- 週期性邏輯 ---
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
        else:
            self.T_cur += 1

        # 更新學習率（get_lr() 同時會更新 self.current_temp）
        lrs = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr

        self._last_lr = lrs

        # 更新溫度(tau) — 透過 callback 而非直接修改全域狀態
        self._tau_callback(self.current_temp)

        self.record["lr"] = lrs[0]
        self.record["tau"] = self.current_temp

    def state_dict(self):
        """返回排程器的狀態字典。"""
        state = super().state_dict()
        state.update(
            {
                "T_i": self.T_i,
                "T_cur": self.T_cur,
                "current_temp": self.current_temp,
                "patience_counter": self.patience_counter,
                "best_metric": self.best_metric,
            }
        )
        return state

    def load_state_dict(self, state_dict):
        """載入排程器的狀態字典。"""
        super().load_state_dict(state_dict)
        self.T_i = state_dict["T_i"]
        self.T_cur = state_dict["T_cur"]
        self.current_temp = state_dict["current_temp"]
        self.patience_counter = state_dict["patience_counter"]
        self.best_metric = state_dict["best_metric"]

    def plot(self, axes: Optional[Axes] = None, show: bool = False, title: str = "LR & Tau"):
        from antenna.utils.utils import plt

        ax: Axes = plt.axes(axes)  # type: ignore
        ax_lr = ax
        ax_tau = ax_lr.twinx()
        (p1,) = ax_lr.plot(self.record["lr"], color="tab:blue", label="LR")
        (p2,) = ax_tau.plot(self.record["tau"], color="tab:orange", label="Tau")
        ax_lr.set_ylabel("Learning Rate", color="tab:blue")
        ax_tau.set_ylabel("Tau", color="tab:orange")
        ax_lr.tick_params(axis="y", labelcolor="tab:blue")
        ax_tau.tick_params(axis="y", labelcolor="tab:orange")
        ax_lr.legend(handles=[p1, p2])
        ax.set_title(title, fontsize=20)

        if show:
            plt.show()
        return ax
