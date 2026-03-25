from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor

from antenna.types import *


class FlipMode(Enum):
    """鏡像模式"""

    horizontal = "|"  # 水平翻轉所以是切垂直的
    vertical = "-"  # 垂直翻轉所以是切水平的
    both = "*"


def mirror(input: Tensor, mode: Union[FlipMode, Literal["-", "|", "*"]] = "*") -> Tuple[Tensor, ...]:
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
            left_half, center_col, right_half = tensor[:, :mid_w], tensor[:, mid_w : mid_w + 1], tensor[:, mid_w + 1 :]
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
            top_half, center_row, bottom_half = tensor[:mid_h, :], tensor[mid_h : mid_h + 1, :], tensor[mid_h + 1 :, :]
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
        top_left_q = tensor[:mid_h_ceil, :mid_w_ceil]  # 包含中心點/線 (如果 H/W 為奇數)
        top_right_q = tensor[:mid_h_ceil, mid_w:]  # 從中間寬度開始 (不包含中心線，如果 W 為奇數)
        bottom_left_q = tensor[mid_h:, :mid_w_ceil]  # 從中間高度開始 (不包含中心線，如果 H 為奇數)
        bottom_right_q = tensor[mid_h:, mid_w:]  # 不包含中心行列

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
        top_half_from_tr = torch.cat(
            [flipped_tr_h, top_right_q[:, (W % 2) :]], dim=1
        )  # 如果 W 是奇數，跳過第一列 (中心列)
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
        result_from_bl = torch.cat(
            [flipped_bl_v, bottom_half_from_bl[(H % 2) :, :]], dim=0
        )  # 如果 H 是奇數，跳過第一行 (中心行)

        # --- 從右下角 (bottom_right_q) 建構 ---
        # 水平翻轉右下角 (包含中心列，如果 W 為奇數)
        flipped_br_h = torch.flip(bottom_right_q, dims=[1])
        # 組合下半部分 (若 W 為奇數，中心列來自 flipped_br_h)
        bottom_half_from_br = torch.cat(
            [flipped_br_h, bottom_right_q[:, (W % 2) :]], dim=1
        )  # 如果 W 是奇數，跳過第一列 (中心列)
        # 垂直翻轉下半部分 (包含中心行，如果 H 為奇數)
        flipped_br_v = torch.flip(bottom_half_from_br, dims=[0])
        # 組合完整圖像 (若 H 為奇數，中心行來自 flipped_br_v)
        result_from_br = torch.cat(
            [flipped_br_v, bottom_half_from_br[(H % 2) :, :]], dim=0
        )  # 如果 H 是奇數，跳過第一行 (中心行)

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
        valid_modes = {"|", "-", "*"}
        if not set(mode).issubset(valid_modes):
            invalid_chars = set(mode) - valid_modes
            raise ValueError(f"無效的 mode 字元: {invalid_chars}。請只使用 '|', '-', '*' 的組合。")

    results = []
    # 迭代處理 mode 中的每個字元，並收集結果, 使用 sorted(set(mode)) 可以確保執行順序固定，且避免重複執行
    for char_mode in sorted(list(set(mode))):
        if char_mode == "-":
            results.extend(_get_horizontal_mirrors(input))
        elif char_mode == "|":
            results.extend(_get_vertical_mirrors(input))
        elif char_mode == "*":
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
