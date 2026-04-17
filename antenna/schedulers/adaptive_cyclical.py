import math
import warnings

from torch.optim.lr_scheduler import _LRScheduler

from antenna.types import *
from antenna.utils import Record, config


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
        :raises ValueError: 如果 T_0, T_mult, 或 mode 參數無效。
        """
        # --- Tau callback（預設向後相容：自動設定 AntennaPattern.tau）---
        if tau_callback is not None:
            self._tau_callback = tau_callback
        else:

            def _default_tau_callback(tau: float):
                from antenna import AntennaPattern

                AntennaPattern.tau = tau

            self._tau_callback = _default_tau_callback

        self.record = Record(self.__class__.__name__, config.get("RESULT_PATH"))
        # --- 週期性參數 (來自 CosineAnnealing) ---
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if on_plateau not in ["peak", "reset", "linear"]:
            raise ValueError("on_plateau must be 'peak' or 'reset' or 'linear'")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.T_i = T_0  # 當前週期的長度
        self.T_cur = last_epoch if last_epoch != -1 else -1

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

        # 檢查模式
        if mode not in ["min", "max"]:
            raise ValueError("mode " + mode + " is unknown!")

        super().__init__(optimizer, last_epoch)
        self.base_lrs = [lr_max] * len(self.optimizer.param_groups)

    def get_temp(self) -> float:
        """獲取當前計算出的溫度"""
        return self.current_temp

    def get_lr(self):
        """計算並返回當前的學習率和溫度"""
        warmup_steps = int(self.T_i * self.warmup_ratio)

        if self.T_cur < warmup_steps:
            # 1. 暖身階段
            lr = self.lr_min + (self.lr_max - self.lr_min) * (self.T_cur / warmup_steps)
            self.current_temp = self.temp_min + (self.temp_max - self.temp_min) * (self.T_cur / warmup_steps)
        else:
            # 2. 餘弦退火階段
            cosine_progress = (self.T_cur - warmup_steps) / (self.T_i - warmup_steps)
            lr = self.lr_min + (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * cosine_progress)) / 2
            self.current_temp = (
                self.temp_min + (self.temp_max - self.temp_min) * (1 + math.cos(math.pi * cosine_progress)) / 2
            )

        return [lr for _ in self.optimizer.param_groups]

    def _is_metric_better(self, metric):
        if self.mode == "min":
            return metric < self.best_metric - self.threshold
        else:
            return metric > self.best_metric + self.threshold

    def step(self, metric: float = None):
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
                # print(f"\nMetric has not improved for {self.patience} steps. Forcing a warm restart!")
                self.patience_counter = 0
                self.T_i = max(int(self.T_i * self.factor), self.T_0 // 2)  # 保持最小週期長度限制

                # 2. 決定重啟位置 (Apply on_plateau strategy)
                match self.on_plateau:
                    case "reset":  # 回到起點 (最小值)，重新開始暖身
                        # 設定為 -1，因為 step() 尾端的 self.T_cur += 1 會將其變為 0
                        self.T_cur = -1

                    case "peak":  # 直接跳到峰值 (最大值)，跳過暖身
                        # 計算新週期中，暖身結束的那一點 (即 LR/Tau 最大的點)
                        warmup_steps = int(self.T_i * self.warmup_ratio)

                        # 設定為 warmup_steps - 1，因為 step() 尾端的 self.T_cur += 1 會將其變為 warmup_steps
                        # 根據 get_lr() 的邏輯，當 T_cur == warmup_steps 時，剛好是最大值
                        self.T_cur = warmup_steps - 1

                    case "linear":  # 從當前數值，線性爬升回最大值
                        # A. 取得當前 LR (假設所有 group LR 一致，取第一個)
                        current_lr = self.optimizer.param_groups[0]["lr"]

                        # B. 計算新週期中，暖身階段的總長度
                        warmup_steps = int(self.T_i * self.warmup_ratio)

                        if warmup_steps > 0:
                            # C. 反推：當前的 LR 在暖身線上對應的比例 (0.0 ~ 1.0)
                            # 公式: ratio = (目前 - 最小) / (最大 - 最小)
                            ratio = (current_lr - self.lr_min) / (self.lr_max - self.lr_min + 1e-10)
                            ratio = max(0.0, min(1.0, ratio))  # 限制範圍以防萬一

                            # D. 設定時間點：反推對應的步數
                            # 這樣下一次 get_lr() 就會從這個高度繼續往上走
                            self.T_cur = int(round(ratio * warmup_steps))
                        else:
                            # 如果沒有暖身區間，就直接設為 -1 (避免除以零)
                            self.T_cur = -1
                    case _:
                        pass

        # --- 週期性邏輯 ---
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
        else:
            self.T_cur += 1

        # 更新學習率
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        # 更新溫度(tau) — 透過 callback 而非直接修改全域狀態
        if self._tau_callback is not None:
            self._tau_callback(self.get_temp())

        self.record["lr"] = self.get_lr()[0]
        self.record["tau"] = self.get_temp()

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
                # 可以選擇性儲存初始參數，但通常在 __init__ 中處理
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
