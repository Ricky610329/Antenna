"""鏡像與 Gumbel-Sinkhorn 相關損失 / 幫手函數。"""

from enum import Enum
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


class FlipMode(Enum):
    """鏡像模式。

    - ``horizontal`` (``"|"``) 水平翻轉（沿垂直中線對折）
    - ``vertical`` (``"-"``) 垂直翻轉（沿水平中線對折）
    - ``both`` (``"*"``) 同時對兩軸對稱（由四個象限各自建構）
    """

    horizontal = "|"
    vertical = "-"
    both = "*"


ModeType = FlipMode | Literal["-", "|", "*"]
_VALID_MODE_CHARS = {"|", "-", "*"}


def _mirror_axis(tensor: Tensor, dim: int) -> tuple[Tensor, Tensor]:
    """沿指定軸向將 tensor 對稱化，回傳兩個 (從前半、從後半建構) 對稱版本。"""
    size = tensor.shape[dim]
    half = size // 2
    has_center = size % 2 == 1

    # 將 tensor 切分成 (前半, [中心], 後半)
    front = tensor.narrow(dim, 0, half)
    back = tensor.narrow(dim, half + (1 if has_center else 0), half)

    flipped_front = torch.flip(front, dims=[dim])
    flipped_back = torch.flip(back, dims=[dim])

    if has_center:
        center = tensor.narrow(dim, half, 1)
        from_front = torch.cat([front, center, flipped_front], dim=dim)
        from_back = torch.cat([flipped_back, center, back], dim=dim)
    else:
        from_front = torch.cat([front, flipped_front], dim=dim)
        from_back = torch.cat([flipped_back, back], dim=dim)

    return from_front, from_back


def _quadrant_mirrors(tensor: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """由四個象限各自建構一個同時水平、垂直對稱的 tensor。

    回傳順序：``(result_from_top_left, result_from_top_right,
    result_from_bottom_left, result_from_bottom_right)``。
    """
    # 先沿水平軸做對稱：from_left 使用左半建構、from_right 使用右半建構
    from_left, from_right = _mirror_axis(tensor, dim=1)
    # 再沿垂直軸做對稱：from_top 使用上半建構、from_bottom 使用下半建構
    tl, bl = _mirror_axis(from_left, dim=0)
    tr, br = _mirror_axis(from_right, dim=0)
    return tl, tr, bl, br


def mirror(input: Tensor, mode: ModeType = "*") -> tuple[Tensor, ...]:
    """對給定的 2D tensor 進行鏡像處理。

    mode 可以是 :class:`FlipMode`，或由 ``'-'``, ``'|'``, ``'*'`` 組合而成的字串：

    - ``'|'``：水平翻轉（沿垂直中線），回傳 2 個 Tensor。
    - ``'-'``：垂直翻轉（沿水平中線），回傳 2 個 Tensor。
    - ``'*'``：同時水平與垂直對稱，回傳 4 個 Tensor。

    多字元組合時會依字元排序後的固定順序串接結果，重複字元只會處理一次。

    Args:
        input: 形狀為 ``(H, W)`` 的 2D tensor。
        mode: 鏡像模式。

    Returns:
        依序排列的鏡像 Tensor tuple。

    Raises:
        ValueError: 如果 mode 字串含有非法字元。

    Example::

        x = torch.tensor([[1, 2, 3, 4, 5],
                          [6, 7, 8, 9, 10],
                          [11, 12, 13, 14, 15]])
        for m in mirror(x, mode="-|*"):
            print(m)
    """
    if isinstance(mode, FlipMode):
        mode_chars = [mode.value]
    else:
        invalid = set(mode) - _VALID_MODE_CHARS
        if invalid:
            raise ValueError(f"無效的 mode 字元: {invalid}。請只使用 '|', '-', '*' 的組合。")
        # sorted(set(...)) 以確保執行順序固定，並避免重複執行
        mode_chars = sorted(set(mode))

    results: list[Tensor] = []
    for char in mode_chars:
        if char == "-":
            results.extend(_mirror_axis(input, dim=0))
        elif char == "|":
            results.extend(_mirror_axis(input, dim=1))
        elif char == "*":
            results.extend(_quadrant_mirrors(input))

    return tuple(results)


def gumbel_sinkhorn_rectangular(
    logits: Tensor,
    tau: float = 1.0,
    n_iters: int = 20,
    hard: bool = False,
) -> Tensor:
    """適用於長方形矩陣的 Gumbel-Sinkhorn 演算法。

    Args:
        logits: 輸入的分數矩陣，形狀為 ``(..., K, M)``，其中 K 是位置數、M 是物件數。
        tau: 溫度參數。
        n_iters: Sinkhorn 迭代次數。
        hard: 是否回傳離散 (不可微分) 的指派結果。

    Returns:
        形狀為 ``(..., K, M)`` 的 (軟性或硬性) 分配矩陣。
    """
    # Gumbel 雜訊擾動（手動實作，因為 F.gumbel_softmax 僅對 logits.shape[-1] 做正規化）
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    log_alpha = (logits + gumbels) / tau

    # 為了數值穩定性，在 log-space 進行 Sinkhorn 迭代
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)  # 沿 M 軸正規化
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)  # 沿 K 軸正規化

    soft_assignment = torch.exp(log_alpha)

    if hard:
        _, indices = torch.max(soft_assignment, dim=-1)
        return F.one_hot(indices, num_classes=logits.shape[-1]).to(logits.dtype)

    return soft_assignment
