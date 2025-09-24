import torch
from torch import Tensor, nn
from enum import Enum
from typing import *

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
        # 取整數，自動處理奇偶數的中心點
        # math.ceil(W / 2)
        mid_w_ceil = (W + 1) // 2
        # math.ceil(H / 2)
        mid_h_ceil = (H + 1) // 2

        # 1. 取得四個象限（包含中心線）
        top_left_q = tensor[:mid_h_ceil, :mid_w_ceil]
        top_right_q = tensor[:mid_h_ceil, W // 2:]
        bottom_left_q = tensor[H // 2:, :mid_w_ceil]
        bottom_right_q = tensor[H // 2:, W // 2:]
        
        # 2. 從每個象限建構一個全對稱的 Tensor
        
        # 從左上角建構
        top_half = torch.cat([top_left_q, torch.flip(top_left_q, dims=[1])[:, W % 2:]], dim=1)
        result_from_tl = torch.cat([top_half, torch.flip(top_half, dims=[0])[H % 2:, :]], dim=0)

        # 從右上角建構
        top_half = torch.cat([torch.flip(top_right_q, dims=[1])[:, :- (W % 2 if W % 2 else W)], top_right_q], dim=1)
        result_from_tr = torch.cat([top_half, torch.flip(top_half, dims=[0])[H % 2:, :]], dim=0)
        
        # 從左下角建構
        bottom_half = torch.cat([bottom_left_q, torch.flip(bottom_left_q, dims=[1])[:, W % 2:]], dim=1)
        result_from_bl = torch.cat([torch.flip(bottom_half, dims=[0])[:- (H % 2 if H % 2 else H)], bottom_half], dim=0)

        # 從右下角建構
        bottom_half = torch.cat([torch.flip(bottom_right_q, dims=[1])[:, :- (W % 2 if W % 2 else W)], bottom_right_q], dim=1)
        result_from_br = torch.cat([torch.flip(bottom_half, dims=[0])[:- (H % 2 if H % 2 else H)], bottom_half], dim=0)

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

