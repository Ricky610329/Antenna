import torch
from torch import Tensor

def mirror(input: Tensor):
    """
    對給定的輸入進行左右鏡像處理，回傳兩種對稱結果。

    Args:
        input (Tensor): 一個 2D tensor，形狀為 (H, W)

    Returns:
        Tuple[Tensor, Tensor]: 
            - 第一個 tensor: 左半翻轉貼到右半. shape = (H, W)
            - 第二個 tensor: 右半翻轉貼到左半, shape = (H, W)

    Example:
        >>> x = torch.tensor([[1, 2, 3],
                              [4, 5, 6]])
        >>> ltr, rtl = mirror(x)
        >>> print(ltr)
        tensor([[1, 2, 1],
                [4, 5, 4]])
        >>> print(rtl)
        tensor([[3, 2, 3],
                [6, 5, 6]])
    """
    mid = input.shape[1] // 2

    if input.shape[1] % 2 == 0:
        # Even width
        left_half = input[:, :mid]
        right_half = input[:, mid:]

        # 左翻轉到右
        left_to_right = torch.cat([left_half, torch.flip(left_half, dims=[1])], dim=1)

        # 右翻轉到左
        right_to_left = torch.cat([torch.flip(right_half, dims=[1]), right_half], dim=1)

    else:
        # Odd width
        left_half = input[:, :mid]
        center = input[:, mid:mid+1]
        right_half = input[:, mid+1:]

        # 左翻轉到右
        left_to_right = torch.cat([left_half, center, torch.flip(left_half, dims=[1])], dim=1)

        # 右翻轉到左
        right_to_left = torch.cat([torch.flip(right_half, dims=[1]), center, right_half], dim=1)

    return left_to_right, right_to_left

def mutate(matrix:Tensor, rate):
    total = matrix.numel()
    n = total * rate
    indices = torch.randperm(total).tolist()
    selected_indices = indices[:n]
    
    for idx in selected_indices:
        i, j = divmod(idx, matrix.size(1))
        matrix[i, j] = 1 - matrix[i, j]
    return matrix