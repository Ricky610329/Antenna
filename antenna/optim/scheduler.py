"""
antenna/optim/scheduler.py — ACP (Adaptive Cyclical Policy，論文核心機制)。

AdaptiveCyclicalScheduler：lr 與二值化溫度 tau 的雙耦合排程
(OneCycle 暖身 + 餘弦退火週期重啟 + 高原偵測強制重啟)。
超參數由 YAML 的 scheduler 區段調整 (antenna.training.build_scheduler)。
"""
import math
import warnings
from typing import Literal, Optional

import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler

from antenna.utils.types import Axes


class AdaptiveCyclicalScheduler(_LRScheduler):
    """
    一個融合了 OneCycle、CosineAnnealingWarmRestarts 和 ReduceLROnPlateau 思想的排程器。
    
    它在週期性的餘弦退火基礎上，為每個週期增加了暖身階段，並能根據監控指標
    在模型停滯時提前觸發重啟。同時，它也同步調整一個外部的溫度參數。
    """
    #* 在閉迴路中的角色：本排程器同時驅動兩條曲線——
    #*   (1) GEN 優化器的學習率 lr：暖身→退火，決定每步更新 pattern 的步幅。
    #*   (2) 二值化溫度 tau：高溫探索→低溫定形(見檔頭 STE 說明)。由 get_temp() 取出、迴圈顯式傳給 GEN。
    #* 三種思想融合：
    #*   OneCycle：每個週期開頭先「暖身」緩升，避免一開始大步長把 GEN 帶歪。
    #*   CosineAnnealingWarmRestarts：暖身後餘弦退火，週期性回到高點(warm restart)
    #*       讓 GEN 有機會跳出破碎的局部解、重新廣域探索。
    #*   ReduceLROnPlateau：監控真實 loss，停滯(patience 耗盡)時「強制重啟」並縮短週期。
    #? lr 與 tau 由同一條進度(同一 T_cur)計算 → 永遠同步退火，不可期望單獨調整其一。
    #? tau 不寫入全域：排程器只「產生」tau(get_temp())，由訓練迴圈讀取後顯式傳給 binarization。
    def __init__(
        self,
        optimizer,
        T_0: int = 50,
        T_mult: int = 1,
        lr_max: float = 0.01,
        lr_min: float = 1e-6,
        temp_max: float = 10.0,
        temp_min: float = 0.1,
        warmup_ratio: float = 0.1,
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 5,
        on_plateau:Literal['peak', 'reset','linear'] = 'peak',
        threshold: float = 0.0,
        boundary_suppress_cap: int = 3,
        last_epoch: int = -1
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
        from antenna.utils import config, Record
        self.record = Record(self.__class__.__name__, config.get('RESULT_PATH'))  #* 記錄每步 lr/tau，供事後 plot() 視覺化
        # --- 週期性參數 (來自 CosineAnnealing) ---
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if on_plateau not in ['peak', 'reset','linear']:
             raise ValueError("on_plateau must be 'peak' or 'reset' or 'linear'")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.T_i = T_0  # 當前週期的長度
        self.T_cur = last_epoch if last_epoch != -1 else -1   #* 當前週期內的進度計數(step 末會 +1)

        self.on_plateau = on_plateau

        # --- 學習率與溫度範圍 ---
        #? lr 與 temp 共用同一條「暖身+餘弦」進度(T_cur/T_i)，故兩者永遠同相位升降。
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.temp_max = temp_max
        self.temp_min = temp_min
        self.current_temp = temp_max

        # --- 暖身參數 (來自 OneCycleLR) ---
        self.warmup_ratio = warmup_ratio   #* 每週期前段佔比，這段內 lr/tau 由 min 線性升到 max

        # --- 自適應參數 (來自 ReduceLROnPlateau) ---
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.patience_counter = 0   #* 累計「未改善」步數，達 patience 即觸發強制重啟
        self.best_metric = float('inf') if mode == 'min' else float('-inf')   #* 依方向初始化歷史最佳

        # 檢查模式
        if mode not in ['min', 'max']:
            raise ValueError('mode ' + mode + ' is unknown!')

        #! 父類別 _LRScheduler.__init__ 會呼叫一次 _initial_step()→step() 來定初始 lr，
        #  那一步本就「沒有 metric」(尚未訓練) → 不該警告。用旗標把建構期的初始 step 與
        #  訓練期的 step 區分開：只有訓練期 metric 缺失才警告 (純門控，不動排程數值)。
        # boundary-gated warm restart (opt-in，見 step())：boundary=None 時完全不影響行為 (golden 安全)。
        self.boundary_suppress_cap = boundary_suppress_cap   #* 連續抑制上限 → 強制放行一次 (防餓死)；<=0 不設限
        self._suppress_streak = 0                            #* 目前連續抑制次數
        self._last_restart_suppressed = False                #* 上一步是否抑制了 warm restart (供 TB 診斷)
        self._adapt_ready = False
        super(AdaptiveCyclicalScheduler, self).__init__(optimizer, last_epoch)
        #! base_lrs 覆寫為 lr_max：本排程器自行算 lr(get_lr)，不依賴父類用 base_lr 縮放
        self.base_lrs = [lr_max] * len(self.optimizer.param_groups)
        self._adapt_ready = True   #* 建構完成 → 之後的 step(metric=None) 才視為呼叫端漏傳

    def get_temp(self) -> float:
        """獲取當前計算出的溫度"""
        return self.current_temp

    def get_lr(self):
        """計算並返回當前的學習率和溫度"""
        #? 注意：此方法有副作用——同時更新 self.current_temp(tau)。lr 為回傳值，
        #? tau 透過 get_temp() 取出。兩者依同一 T_cur 計算，保證同步。
        warmup_steps = int(self.T_i * self.warmup_ratio)   #* 本週期暖身佔的步數

        if self.T_cur < warmup_steps:
            # 1. 暖身階段
            #* 由 min 線性升到 max：lr 緩升避免初期大步長破壞 pattern；tau 升高→輸出
            #* 變模糊灰階，先做廣域探索而非過早把像素逼成 0/1。
            lr = self.lr_min + (self.lr_max - self.lr_min) * (self.T_cur / warmup_steps)
            self.current_temp = self.temp_min + (self.temp_max - self.temp_min) * (self.T_cur / warmup_steps)
        else:
            # 2. 餘弦退火階段
            #* 由 max 平滑退回 min：cos 從 0→π 時 (1+cos)/2 由 1→0。lr 降低→精修；
            #* tau 降低→sigmoid 變陡，pattern 逐步定形成乾淨可製造的 0/1(STE 收斂)。
            cosine_progress = (self.T_cur - warmup_steps) / (self.T_i - warmup_steps)   #* 退火進度 0→1
            lr = self.lr_min + (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * cosine_progress)) / 2
            self.current_temp = self.temp_min + (self.temp_max - self.temp_min) * (1 + math.cos(math.pi * cosine_progress)) / 2

        return [lr for _ in self.optimizer.param_groups]   #* 所有 param_group 套用同一 lr

    def _is_metric_better(self, metric):
        #* 是否較歷史最佳「顯著」改善：需超過 threshold 才算數，避免雜訊抖動誤判為進步。
        if self.mode == 'min':
            return metric < self.best_metric - self.threshold   #* loss 類：越小越好
        else:
            return metric > self.best_metric + self.threshold   #* accuracy 類：越大越好

    def step(self, metric: float = None, boundary: float = None, boundary_threshold: float = None):
        #* 每個訓練步呼叫一次；務必傳入監控指標(通常是 HFSS 真實 loss)，否則自適應失效。
        #* boundary/boundary_threshold (opt-in)：boundary≥τ_b 代表 G 衝出 SM 可信區 → 抑制 plateau
        #* warm restart (不加熱、繼續冷卻就地固化)；None → 與現行逐位元相同 (golden 安全)。
        self._last_restart_suppressed = False
        if metric is None:
            #* 建構期的初始 step (尚未訓練) 本就無 metric → 不警告；只有訓練期漏傳才提醒。
            if getattr(self, "_adapt_ready", True):
                warnings.warn("AdaptiveCyclicalScheduler requires a metric to be passed to step() for adaptation.", UserWarning)
        else:
            # --- 自適應邏輯 ---
            if self._is_metric_better(metric):
                self.best_metric = metric        #* 有進步 → 更新最佳並歸零耐心計數
                self.patience_counter = 0
            else:
                self.patience_counter += 1       #* 停滯 → 耐心計數累加

            if self.patience_counter >= self.patience:
                # print(f"\nMetric has not improved for {self.patience} steps. Forcing a warm restart!")
                self.patience_counter = 0
                #? boundary 閘門 (opt-in)：boundary≥τ_b → G 已衝出 SM 可信區 → 抑制 warm restart (不加熱)，
                #  讓餘弦退火冷卻、就地固化 (下一筆 HFSS 在邊界、SM 在此被訓)；boundary<τ_b (區內卡住) 才
                #  放行往外探。防餓死：連續抑制達 boundary_suppress_cap → 強制放行一次。boundary=None → 必放行。
                out_of_region = (boundary is not None and boundary_threshold is not None
                                 and boundary >= boundary_threshold)
                if out_of_region and (self.boundary_suppress_cap <= 0
                                      or self._suppress_streak < self.boundary_suppress_cap):
                    self._suppress_streak += 1
                    self._last_restart_suppressed = True   # 抑制：不縮週期、不重定位 → 續退火冷卻
                else:
                    self._suppress_streak = 0
                    self._force_warm_restart()

        # --- 週期性邏輯 (每步都跑) ---
        #* 自然走完一個週期(T_cur 追上 T_i)：歸零並依 T_mult 拉長下個週期(warm restart)；
        #* 否則單純前進一步。注意強制重啟已透過改寫 T_cur 改變這裡的落點。
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult   #* T_mult>1 → 週期越來越長(後期更穩定)
        else:
            self.T_cur += 1

        # 更新學習率
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr   #* 把新算出的 lr 實際寫回優化器

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        # 更新溫度(tau)
        #? tau 不再由此寫入全域：排程器只負責「產生」tau(get_temp())，由訓練迴圈讀取後
        #? 顯式傳給 GEN.forward / binarization。此處僅記錄 lr/tau 供繪圖。
        self.record['lr'] = self.get_lr()[0]   #* 記錄供 plot() 畫雙軸曲線
        self.record['tau'] = self.get_temp()

    def _force_warm_restart(self):
        #* 停滯太久：強制重啟 (boundary 閘門放行時呼叫)。縮短週期(乘 factor)讓後續探索更密集，
        #* 但不低於 T_0//2 以免週期過短、暖身/退火失去意義。
        self.T_i = max(int(self.T_i * self.factor), self.T_0 // 2)
        # 決定重啟位置 (on_plateau 策略)：激進程度 peak > linear > reset。
        #   peak  —— 最激進：直接拉到最高溫/最高 lr 全力跳出局部解(可能震盪)。
        #   reset —— 最溫和：完全重來、重新暖身，較穩但較慢回到探索強度。
        #   linear—— 折衷：從目前高度沿暖身線往上爬，保留部分已收斂的進度。
        match self.on_plateau:
            case 'reset':   # 回到起點 (最小值)，重新開始暖身 (step() 尾端 +1 → 變 0)
                self.T_cur = -1
            case 'peak':    # 直接跳到峰值 (最大值)，跳過暖身 (step() 尾端 +1 → 變 warmup_steps)
                warmup_steps = int(self.T_i * self.warmup_ratio)
                self.T_cur = warmup_steps - 1
            case 'linear':  # 從當前數值，線性爬升回最大值
                current_lr = self.optimizer.param_groups[0]['lr']
                warmup_steps = int(self.T_i * self.warmup_ratio)
                if warmup_steps > 0:
                    #! +1e-10 防 lr_max==lr_min 時除零；ratio 即「目前 lr 在暖身線上的相對高度」。
                    ratio = (current_lr - self.lr_min) / (self.lr_max - self.lr_min + 1e-10)
                    ratio = max(0.0, min(1.0, ratio))
                    self.T_cur = int(round(ratio * warmup_steps))
                else:
                    self.T_cur = -1
            case _:
                pass   #* 理論上 __init__ 已驗證，不會走到這裡

    def state_dict(self):
        """返回排程器的狀態字典。"""
        state = super().state_dict()
        state.update({
            'T_i': self.T_i,
            'T_cur': self.T_cur,
            'current_temp': self.current_temp,
            'patience_counter': self.patience_counter,
            'best_metric': self.best_metric,
            '_suppress_streak': self._suppress_streak,
            # 可以選擇性儲存初始參數，但通常在 __init__ 中處理
        })
        return state

    def load_state_dict(self, state_dict):
        """載入排程器的狀態字典。"""
        super().load_state_dict(state_dict)
        self.T_i = state_dict['T_i']
        self.T_cur = state_dict['T_cur']
        self.current_temp = state_dict['current_temp']
        self.patience_counter = state_dict['patience_counter']
        self.best_metric = state_dict['best_metric']
        self._suppress_streak = state_dict.get('_suppress_streak', 0)   # 舊 checkpoint 無此鍵 → 預設 0
    
    def plot(self, axes:Optional[Axes] = None, show:bool = False, title:str = "LR & Tau"):
        ax:Axes = plt.axes(axes) # type: ignore
        ax_lr = ax
        ax_tau = ax_lr.twinx()
        p1, = ax_lr.plot(self.record['lr'], color='tab:blue', label='LR')
        p2, = ax_tau.plot(self.record['tau'], color='tab:orange', label='Tau')
        ax_lr.set_ylabel('Learning Rate', color='tab:blue')
        ax_tau.set_ylabel('Tau', color='tab:orange')
        ax_lr.tick_params(axis='y', labelcolor='tab:blue')
        ax_tau.tick_params(axis='y', labelcolor='tab:orange')
        ax_lr.legend(handles=[p1, p2])
        ax.set_title(title, fontsize=20)

        if show: plt.show()
        return ax


