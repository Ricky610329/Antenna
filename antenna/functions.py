import torch
from torch import Tensor, nn, nn
import torch.nn.functional as F
from enum import Enum
from antenna.types import *
from antenna.utils.utils import Figure, plt
from collections import defaultdict
import numpy as np
from loguru import logger

def custom_loss_interval(prediction:Tensor, target_low:Tensor, target_high:Tensor, loss_type='SmoothL1Loss'):
    """
    計算基於目標區間的自定義 loss。
    如果 prediction 在 [target_low, target_high] 區間內，則 loss 為 0。
    否則，計算 prediction 與最近的區間邊界之間的 loss。
    """
    criterion = nn.SmoothL1Loss(reduction='none') if loss_type == 'SmoothL1Loss' else nn.MSELoss(reduction='none')

    # 初始化 loss tensor
    losses = torch.zeros_like(prediction)

    # 1. 處理 prediction > target_high 的情況
    mask_above = prediction > target_high
    if mask_above.sum() > 0:
        losses[mask_above] = criterion(prediction[mask_above], target_high.expand_as(prediction)[mask_above])

    # 2. 處理 prediction < target_low 的情況
    mask_below = prediction < target_low
    if mask_below.sum() > 0:
        losses[mask_below] = criterion(prediction[mask_below], target_low.expand_as(prediction)[mask_below])

    # 3. prediction 在區間內的情況 (target_low <= prediction <= target_high)
    #    此時 losses[mask_in_interval] 仍然是 0，不需要額外處理

    return losses.mean() # 返回平均 loss

class FlipMode(Enum):
    """鏡像模式"""
    horizontal = '|'    # 水平翻轉所以是切垂直的
    vertical = '-'      # 垂直翻轉所以是切水平的
    both = '*'

def mirror(input: Tensor, mode: Union[FlipMode, Literal['-','|','*']]  = '*') -> Tuple[Tensor, ...]:
    """
    對給定的輸入進行鏡像處理。可依據 mode 參數控制。

    - 'horizontal': 水平翻轉，回傳 2 個 Tensor。
    - 'vertical': 垂直翻轉，回傳 2 個 Tensor。
    - 'both': 以四個象限為基礎，產生 4 個同時滿足水平和垂直鏡像的 Tensor。

    Args:
        input (Tensor): 一個 2D tensor，形狀為 (H, W)。
        mode (str): 鏡像模式，可選 'horizontal', 'vertical', 'both'。
                    預設為 'horizontal'。

    Returns:
        Tuple[Tensor, ...]: 根據模式回傳 2 或 4 個鏡像處理後的 Tensor。

    Raises:
        ValueError: 如果提供了無效的 mode。
    
    Example::

        x = torch.tensor([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ])
        mirroreds = mirror(x, mode='-|*)
        for n in mirroreds:
            print(n)
    """

    def _get_horizontal_mirrors(tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """輔助函數：水平翻轉"""
        H, W = tensor.shape
        mid_w = W // 2
        if W % 2 == 0:
            left_half, right_half = tensor[:, :mid_w], tensor[:, mid_w:]
            ltr = torch.cat([left_half, torch.flip(left_half, dims=[1])], dim=1)
            rtl = torch.cat([torch.flip(right_half, dims=[1]), right_half], dim=1)
        else:
            left_half, center_col, right_half = tensor[:, :mid_w], tensor[:, mid_w:mid_w+1], tensor[:, mid_w+1:]
            ltr = torch.cat([left_half, center_col, torch.flip(left_half, dims=[1])], dim=1)
            rtl = torch.cat([torch.flip(right_half, dims=[1]), center_col, right_half], dim=1)
        return ltr, rtl

    def _get_vertical_mirrors(tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """輔助函數：垂直翻轉"""
        H, W = tensor.shape
        mid_h = H // 2
        if H % 2 == 0:
            top_half, bottom_half = tensor[:mid_h, :], tensor[mid_h:, :]
            ttb = torch.cat([top_half, torch.flip(top_half, dims=[0])], dim=0)
            btt = torch.cat([torch.flip(bottom_half, dims=[0]), bottom_half], dim=0)
        else:
            top_half, center_row, bottom_half = tensor[:mid_h, :], tensor[mid_h:mid_h+1, :], tensor[mid_h+1:, :]
            ttb = torch.cat([top_half, center_row, torch.flip(top_half, dims=[0])], dim=0)
            btt = torch.cat([torch.flip(bottom_half, dims=[0]), center_row, bottom_half], dim=0)
        return ttb, btt

    def _get_quadrant_mirrors(tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """輔助函數：'both' 模式的象限翻轉"""
        H, W = tensor.shape
        mid_h = H // 2
        mid_w = W // 2

        # 根據維度奇偶決定切片終點
        # 如果 H 是奇數, mid_h_ceil 會是中間那一行之後的索引
        # 如果 H 是偶數, mid_h_ceil 會是中間那一行之後的索引 (等於 mid_h)
        mid_h_ceil = (H + 1) // 2
        mid_w_ceil = (W + 1) // 2

        # 1. 精確取得四個象限 (對於奇數維度，中心行列會被包含在多個象限中，這沒關係)
        top_left_q = tensor[:mid_h_ceil, :mid_w_ceil]     # 包含中心點/線 (如果 H/W 為奇數)
        top_right_q = tensor[:mid_h_ceil, mid_w:]        # 從中間寬度開始 (不包含中心線，如果 W 為奇數)
        bottom_left_q = tensor[mid_h:, :mid_w_ceil]    # 從中間高度開始 (不包含中心線，如果 H 為奇數)
        bottom_right_q = tensor[mid_h:, mid_w:]       # 不包含中心行列

        # 2. 從每個象限建構一個全對稱的 Tensor

        # --- 從左上角 (top_left_q) 建構 ---
        # 水平翻轉左上角 (不含中心列，如果 W 為奇數)
        flipped_tl_h = torch.flip(top_left_q[:, :mid_w], dims=[1])
        # 組合上半部分 (若 W 為奇數，中心列來自 top_left_q)
        top_half_from_tl = torch.cat([top_left_q, flipped_tl_h], dim=1)
        # 垂直翻轉上半部分 (不含中心行，如果 H 為奇數)
        flipped_tl_v = torch.flip(top_half_from_tl[:mid_h, :], dims=[0])
        # 組合完整圖像 (若 H 為奇數，中心行來自 top_half_from_tl)
        result_from_tl = torch.cat([top_half_from_tl, flipped_tl_v], dim=0)

        # --- 從右上角 (top_right_q) 建構 ---
        # 水平翻轉右上角 (包含中心列，如果 W 為奇數)
        flipped_tr_h = torch.flip(top_right_q, dims=[1])
        # 組合上半部分 (若 W 為奇數，中心列來自 flipped_tr_h)
        top_half_from_tr = torch.cat([flipped_tr_h, top_right_q[:, (W % 2):]], dim=1) # 如果 W 是奇數，跳過第一列 (中心列)
        # 垂直翻轉上半部分 (不含中心行，如果 H 為奇數)
        flipped_tr_v = torch.flip(top_half_from_tr[:mid_h, :], dims=[0])
        # 組合完整圖像 (若 H 為奇數，中心行來自 top_half_from_tr)
        result_from_tr = torch.cat([top_half_from_tr, flipped_tr_v], dim=0)

        # --- 從左下角 (bottom_left_q) 建構 ---
        # 水平翻轉左下角 (不含中心列，如果 W 為奇數)
        flipped_bl_h = torch.flip(bottom_left_q[:, :mid_w], dims=[1])
        # 組合下半部分 (若 W 為奇數，中心列來自 bottom_left_q)
        bottom_half_from_bl = torch.cat([bottom_left_q, flipped_bl_h], dim=1)
        # 垂直翻轉下半部分 (包含中心行，如果 H 為奇數)
        flipped_bl_v = torch.flip(bottom_half_from_bl, dims=[0])
        # 組合完整圖像 (若 H 為奇數，中心行來自 flipped_bl_v)
        result_from_bl = torch.cat([flipped_bl_v, bottom_half_from_bl[(H % 2):, :]], dim=0) # 如果 H 是奇數，跳過第一行 (中心行)

        # --- 從右下角 (bottom_right_q) 建構 ---
        # 水平翻轉右下角 (包含中心列，如果 W 為奇數)
        flipped_br_h = torch.flip(bottom_right_q, dims=[1])
        # 組合下半部分 (若 W 為奇數，中心列來自 flipped_br_h)
        bottom_half_from_br = torch.cat([flipped_br_h, bottom_right_q[:, (W % 2):]], dim=1) # 如果 W 是奇數，跳過第一列 (中心列)
        # 垂直翻轉下半部分 (包含中心行，如果 H 為奇數)
        flipped_br_v = torch.flip(bottom_half_from_br, dims=[0])
        # 組合完整圖像 (若 H 為奇數，中心行來自 flipped_br_v)
        result_from_br = torch.cat([flipped_br_v, bottom_half_from_br[(H % 2):, :]], dim=0) # 如果 H 是奇數，跳過第一行 (中心行)


        # --- 驗證形狀 (可選，用於除錯) ---
        expected_shape = (H, W)
        assert result_from_tl.shape == expected_shape, f"Shape mismatch TL: {result_from_tl.shape} != {expected_shape}"
        assert result_from_tr.shape == expected_shape, f"Shape mismatch TR: {result_from_tr.shape} != {expected_shape}"
        assert result_from_bl.shape == expected_shape, f"Shape mismatch BL: {result_from_bl.shape} != {expected_shape}"
        assert result_from_br.shape == expected_shape, f"Shape mismatch BR: {result_from_br.shape} != {expected_shape}"

        return result_from_tl, result_from_tr, result_from_bl, result_from_br

    
    if isinstance(mode, FlipMode):
        mode = [mode.value] 
    else:
        # 驗證 mode 字串中的所有字元是否合法
        valid_modes = {'|', '-', '*'}
        if not set(mode).issubset(valid_modes):
            invalid_chars = set(mode) - valid_modes
            raise ValueError(f"無效的 mode 字元: {invalid_chars}。請只使用 '|', '-', '*' 的組合。")


    results = []
    # 迭代處理 mode 中的每個字元，並收集結果, 使用 sorted(set(mode)) 可以確保執行順序固定，且避免重複執行
    for char_mode in sorted(list(set(mode))):
        if char_mode == '-':
            results.extend(_get_horizontal_mirrors(input))
        elif char_mode == '|':
            results.extend(_get_vertical_mirrors(input))
        elif char_mode == '*':
            results.extend(_get_quadrant_mirrors(input))
            
    return tuple(results)

def gumbel_sinkhorn_rectangular(logits: torch.Tensor, tau: float = 1.0, n_iters: int = 20, hard: bool = False):
    """
    適用於長方形矩陣的 Gumbel-Sinkhorn 演算法。
    
    Args:
        logits (torch.Tensor): 輸入的分數矩陣，形狀為 (..., K, M)，
                               其中 K 是位置數，M 是物件數。
        tau (float): 溫度參數。
        n_iters (int): Sinkhorn 迭代次數。
        hard (bool): 是否回傳離散的指派結果。
    
    Returns:
        torch.Tensor: 形狀為 (..., K, M) 的 (軟性/硬性) 分配矩陣。
    """
    # Gumbel 雜訊擾動 (這裡我們手動實現，因為 F.gumbel_softmax 假設維度是 logits.shape[-1])
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    perturbed_logits = (logits + gumbels) / tau
    
    # 為了數值穩定性，在 log-space 進行迭代
    log_alpha = perturbed_logits
    
    for _ in range(n_iters):
        # 沿著 M 維度 (物件) 進行正規化
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        # 沿著 K 維度 (位置) 進行正規化
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)

    soft_assignment = torch.exp(log_alpha)

    if hard:
        # 取得離散的指派結果 (不可微分)
        _, indices = torch.max(soft_assignment, dim=-1)
        hard_assignment = F.one_hot(indices, num_classes=logits.shape[-1]).to(logits.dtype)
        return hard_assignment
        
    return soft_assignment

def total_variation_loss(img, weight=0.01):
    """計算 Total Variation Loss 以抑制過度破碎的圖樣"""
    from .utils.data import size_converter
    from . import AntennaPattern
    img = size_converter(AntennaPattern, img, output_shape="B, 1, H, W")
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:] - img[:,:,:-1,:], 2).sum()
    tv_w = torch.pow(img[:,:,:,1:] - img[:,:,:,:-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

import torch
import math
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import warnings

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
        mode: str = 'min',
        factor: float = 0.5,
        patience: int = 5,
        on_plateau:Literal['peak', 'reset','linear'] = 'peak',
        threshold: float = 0.0,
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
        from .utils import config, Record
        self.record = Record(self.__class__.__name__, config.get('RESULT_PATH'))
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
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # 檢查模式
        if mode not in ['min', 'max']:
            raise ValueError('mode ' + mode + ' is unknown!')

        super(AdaptiveCyclicalScheduler, self).__init__(optimizer, last_epoch)
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
            self.current_temp = self.temp_min + (self.temp_max - self.temp_min) * (1 + math.cos(math.pi * cosine_progress)) / 2

        return [lr for _ in self.optimizer.param_groups]

    def _is_metric_better(self, metric):
        if self.mode == 'min':
            return metric < self.best_metric - self.threshold
        else:
            return metric > self.best_metric + self.threshold

    def step(self, metric: float = None):
        if metric is None:
            warnings.warn("AdaptiveCyclicalScheduler requires a metric to be passed to step() for adaptation.", UserWarning)
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
                self.T_i = max(int(self.T_i * self.factor), self.T_0 // 2) # 保持最小週期長度限制

                # 2. 決定重啟位置 (Apply on_plateau strategy)
                match self.on_plateau :
                    case 'reset':   # 回到起點 (最小值)，重新開始暖身
                        # 設定為 -1，因為 step() 尾端的 self.T_cur += 1 會將其變為 0
                        self.T_cur = -1
                    
                    case 'peak':    # 直接跳到峰值 (最大值)，跳過暖身
                        # 計算新週期中，暖身結束的那一點 (即 LR/Tau 最大的點)
                        warmup_steps = int(self.T_i * self.warmup_ratio)
                        
                        # 設定為 warmup_steps - 1，因為 step() 尾端的 self.T_cur += 1 會將其變為 warmup_steps
                        # 根據 get_lr() 的邏輯，當 T_cur == warmup_steps 時，剛好是最大值
                        self.T_cur = warmup_steps - 1

                    case 'linear':  # 從當前數值，線性爬升回最大值
                        
                        # A. 取得當前 LR (假設所有 group LR 一致，取第一個)
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        # B. 計算新週期中，暖身階段的總長度
                        warmup_steps = int(self.T_i * self.warmup_ratio)
                        
                        if warmup_steps > 0:
                            # C. 反推：當前的 LR 在暖身線上對應的比例 (0.0 ~ 1.0)
                            # 公式: ratio = (目前 - 最小) / (最大 - 最小)
                            ratio = (current_lr - self.lr_min) / (self.lr_max - self.lr_min + 1e-10)
                            ratio = max(0.0, min(1.0, ratio)) # 限制範圍以防萬一

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
            param_group['lr'] = lr
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

        # 更新溫度(tau)
        from . import AntennaPattern
        AntennaPattern.tau = self.get_temp()

        self.record['lr'] = self.get_lr()[0]
        self.record['tau'] = self.get_temp()

    def state_dict(self):
        """返回排程器的狀態字典。"""
        state = super().state_dict()
        state.update({
            'T_i': self.T_i,
            'T_cur': self.T_cur,
            'current_temp': self.current_temp,
            'patience_counter': self.patience_counter,
            'best_metric': self.best_metric,
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
    
    def plot(self, axes:Optional[Axes] = None, show:bool = False, title:str = "LR & Tau"):
        from .utils.utils import plt
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


class SpectralConnectivityLoss(nn.Module):
    def __init__(self, height=25, width=25, epsilon=1e-2):
        super().__init__()
        self.H, self.W = height, width
        self.num_nodes = height * width
        self.epsilon = epsilon  # 基礎連通性，防止 lambda_2 鎖死在 0
        
        # 預建鄰接索引
        src, dst = [], []
        for r in range(height):
            for c in range(width):
                idx = r * width + c
                if r + 1 < height: src.append(idx); dst.append((r + 1) * width + c)
                if c + 1 < width: src.append(idx); dst.append(r * width + (c + 1))
        self.register_buffer('src', torch.tensor(src).long())
        self.register_buffer('dst', torch.tensor(dst).long())

    def forward(self, antenna_map):
        # 確保輸入是 (B, 1, H, W)
        if antenna_map.dim() == 2: antenna_map = antenna_map.unsqueeze(0).unsqueeze(0)
        batch_size = antenna_map.shape[0]
        flat_map = antenna_map.view(batch_size, -1)
        losses = []

        for b in range(batch_size):
            # 改進權重邏輯：金屬與金屬=1.0, 其他部分至少有 epsilon 的連通性
            node_vals = flat_map[b]
            # 邊權重 = 節點相乘的加權 + 基礎連通性
            w = (node_vals[self.src] * node_vals[self.dst]) + self.epsilon
            
            # 建立對稱拉普拉斯
            A = torch.zeros(self.num_nodes, self.num_nodes, device=antenna_map.device)
            A[self.src, self.dst] = w
            A[self.dst, self.src] = w
            D = torch.diag(torch.sum(A, dim=1))
            L = D - A
            
            # 計算特徵值
            eigvals = torch.linalg.eigvalsh(L)
            lambda_2 = eigvals[1]  # Fiedler Value
            
            # 目標：讓 lambda_2 越大越好（金屬連通後 lambda_2 會顯著增加）
            # 使用負對數或反比，讓差異更明顯
            losses.append(1/(lambda_2))
            # losses.append(-torch.log(lambda_2+1e6))
            # losses.append(torch.exp(-lambda_2)) 

        return torch.mean(torch.stack(losses))

class GapClosingLoss(nn.Module):
    def __init__(self):
        """
        Closing = Dilation(膨脹) + Erosion(侵蝕)
        """
        super().__init__()

    def forward(self, antenna_map):
        # R = (kernel_size-1)/2

        # 1. Soft Dilation (膨脹) - 填補裂縫
        # 使用 MaxPool 模擬 
        dilated = F.max_pool2d(antenna_map, kernel_size=3, stride=1, padding=1)
        
        # 2. Soft Erosion (腐蝕) - 恢復外形
        # Erosion(x) = -Max(-x)
        closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)
        
        # 3. 計算 Loss
        # 如果 antenna_map 有裂縫，closed 會把裂縫填滿 (數值變大)
        # 我們希望 antenna_map 本身就沒有裂縫，即 antenna_map 應該接近 closed
        # Loss = || Closed - Original ||
        
        loss = torch.mean((closed - antenna_map) ** 2)
        return loss

class FeedReachability: # R_feed
    def __init__(self, feed_positions:list[tuple[int, int]]):
        """
        計算共同連通指標 (Mutual Feed Connectivity Index)
        只有當所有饋電點都連通在同一個金屬塊上時，才計算該塊的佔比。

        :param feed_positions: 座標列表 [(r1, c1), (r2, c2), ...]
        """
        from scipy.ndimage import label
        assert len(feed_positions)>0, ""

        self.feed_positions = feed_positions
        """潰入點"""
        self.rate = None
        """電流導通率"""
        self.mask = 0
        """電流導通的遮罩"""
        # self.structure = np.ones((3, 3))
        self.structure = np.array([
            [0.0,1.0,0.0],
            [1.0,1.0,1.0],
            [0.0,1.0,0.0]])
        """連通性, 預設採用 8-連通 (8-connectivity), 若要4連通可以是十字架"""
        self.record:list[FeedReachabilityDictType] = []
        self.r_feed_str = "$R_{{feed}}$"

        self._label = label
    
    @classmethod
    def single_feed(cls):
        shape = (25, 25)
        return cls([(shape[0]-1, int((shape[1])/2))])

    @classmethod
    def dual_feed(cls):
        shape = (25, 25)
        return cls([(shape[0]-1, int((shape[1])/2)),(0, int((shape[1])/2))])
    
    def __call__(self, pattern: Union[Tensor, np.ndarray], *, record:bool=False, title:str = "Pattern ($R_{{feed}}$={rate:.2%})"):
        """
        :param pattern: 2D array (1=金屬, 0=介質)
        :return: Feed Reachability Rate
        """
        
        if isinstance(pattern, Tensor):
            pattern = pattern.numpy()
        
        labeled_array, _ = self._label(pattern, structure=self.structure)
        
        # 1. 取得所有饋電點所在的 Label IDs
        feed_labels = []
        
        for pos in self.feed_positions:
            
            # 檢查座標是否越界或該處無金屬
            if 0 <= pos[0] < pattern.shape[0] and 0 <= pos[1] < pattern.shape[1]:
                lbl = labeled_array[pos]
                if lbl > 0:
                    feed_labels.append(lbl)
                else:
                    logger.warning("其中一個潰入點沒金屬，直接失敗")
                    return 0.0, np.zeros_like(pattern) # 其中一個點沒金屬，直接失敗
            else:
                logger.error("潰入點座標是否越界")
                return 0.0, np.zeros_like(pattern)

        # 2. 「AND」邏輯檢查：判斷所有饋電點的 Label 是否完全相同
        unique_labels = set(feed_labels)
        
        if len(unique_labels) == 1:
            # 所有饋電點都在同一個連通塊上
            shared_label = list(unique_labels)[0]
            shared_mask = (labeled_array == shared_label)
            
            total_metal_pixels = np.sum(pattern)
            connected_pixels = np.sum(shared_mask)
            mutual_index = connected_pixels / total_metal_pixels

        else:
            # 饋電點分布在不同的連通塊上，或彼此斷開
            mutual_index = 0.0
            shared_mask = np.zeros_like(pattern)
        
        self.rate = mutual_index
        self.mask = shared_mask
        self.pattern = pattern
        self.title = title.format(rate=mutual_index)
        
        if record:
            self.record.append(
                {
                    'pattern': pattern,
                    'feed_positions': self.feed_positions,
                    'rate': mutual_index,
                    'mask': shared_mask,
                    "title": self.title
                }
            ) 
        return mutual_index
    
    @property
    def r_feed_dict(self):
        result = defaultdict(list)
        for entry in self.record:
            result[entry['title']].append(entry['rate'])
        
        return result
    
    @property
    def r_feed_list(self):
        return [_['rate'] for _ in self.record]
    
    @property
    def rate_list(self):
        return [_['rate']*100 for _ in self.record]
    
    @property
    def r_feed_avg(self):
        return np.mean(self.r_feed_list)
    
    def plot(self, axes = None, show=False, data:FeedReachabilityDictType=None):
        pattern = self.pattern

        pattern = data['pattern'] if data else self.pattern
        mask = data['mask'] if data else self.mask
        rate = data['rate'] if data else self.rate
        title = data['title'] if data else self.title
        feed_positions = data['feed_positions'] if data else self.feed_positions

        ax:Axes = axes if axes else plt.axes(axes) # type: ignore
        ax.set_title(title)

        #* 初始化底圖
        display_img = np.full((pattern.shape[0], pattern.shape[1], 3), [0.96, 0.96, 0.97]) # 淺冷灰色
        
        #* 標示所有原始金屬區域
        display_img[pattern == 1] = [0.74, 0.76, 0.78] # 中灰色
        
        #* 疊加有效連通區域
        display_img[mask == 1] = [0.1, 0.7, 0.1]
        
        #* 繪製影像
        ax.imshow(display_img, interpolation='nearest')
        
        #* 標註饋電點
        for feed_pos in feed_positions:
            ax.plot(feed_pos[1], feed_pos[0], 'ro', markersize=8, markeredgecolor='yellow')

        ax.axis('off') # on/off
        # plt.grid(False)
        if show: plt.show()
        return ax

    def plot_records(self, cols: int = 4, show: bool = True):
        record_n = len(self.record)
        with Figure("FeedReachability", ncols=(record_n,cols), show=show) as fig:
            for n, r_feed in enumerate(self.record):
                ax = fig.index(-1)
                self.plot(ax, show=False, data=r_feed)

    def plot_records_rate(self, axes = None, show=False):

        plt.rcParams.update({
            'font.size': 16,
        })

        ax:plt.Axes = axes if axes else plt.axes(axes) # type: ignore
        # ax.set_title(f'Feed Reachability (Avg. = {self.r_feed_avg:.2%})')

        for key, rate in self.r_feed_dict.items():
            ax.plot(rate, label=f"{key} (Avg. = {np.mean(rate):.2%})")

        ax.set_xlabel('Epoch')  # x 軸名稱
        ax.set_ylabel('$R_{{feed}}$')  # y 軸名稱
        ax.set_ylim(0, 1)

        # plt.grid(False)
        plt.legend()
        if show: plt.show()
        return ax

    def plot_one_record_rate(self, axes = None, show=False):

        plt.rcParams.update({
            'font.size': 16,
        })

        ax:plt.Axes = axes if axes else plt.axes(axes) # type: ignore
        ax.set_title(f'Feed Reachability (Avg. = {self.r_feed_avg:.2%})')

        ax.plot(self.rate_list)

        ax.set_xlabel('Epoch')  # x 軸名稱
        ax.set_ylabel('$R_{{feed}}$ (%)')  # y 軸名稱
        ax.set_ylim(0, 100)

        # plt.grid(False)
        if show: plt.show()
        return ax