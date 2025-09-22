import torch
from torch import Tensor, nn

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