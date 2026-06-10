"""
================================================================================
antenna.patch — 微帶貼片天線「反向設計」的損失函數 (Loss Functions)
================================================================================

【這個檔案在做什麼？】
    本模組集中定義「衡量預測響應 vs 目標響應」的損失函數，是整條反向設計閉迴路
    (GEN 目標響應→pattern；SM pattern→預測響應；SIM 真實 HFSS 模擬) 的「評分尺」。
    GEN 生成 pattern 後，SM 算出預測響應，再用這裡的 loss 與「想要的目標」比較，
    梯度沿 loss → SM → pattern → GEN 反向傳播以更新 GEN。
    這些函數由 antenna.training.setup_responses 經 `spec.register_loss_fn(...)`
    綁進響應規格 (TargetResponse)，成為各條 S 參數曲線的評分函數。

【共同的核心設計哲學：為什麼不直接用 MSE？】
    天線規格本質上是「不等式」而非「等式」：
      - 反射 (S11/S22)：只要「夠低」(匹配夠好) 即可，更低不會是壞事；
      - 增益 (Gain/S21)：只要「夠高」即可，更高同樣不是壞事。
    若直接對整條目標曲線做 MSE，會把「比目標更好」的預測也當成誤差去懲罰，
    反而把已經達標的解硬拉回目標線，與物理需求相悖。因此本檔的損失皆採
    「單邊懲罰 (one-sided)」或「區間容差 (interval)」設計 ──「夠好就不罰、過頭也不罰」，
    只在「沒達到規格」的方向上產生梯度，引導 GEN 往滿足規格的可行域移動。

【本檔兩個損失函數】
    - custom_loss_minmax 單邊極值損失：method='low' 只罰「目標最低點處預測偏高」、
                         method='high' 只罰「目標最高點處預測偏低」(單埠 train_single 使用)。
    - interval_loss      區間損失：要求預測落在 [target+lower, target+upper] (相對) 或
                         [lower, upper] (絕對) 區間內，區間內 loss=0 (雙埠 train_dual 使用)。
"""
from typing import Literal, Union, overload
from torch import Tensor, nn
# patch_simulator：HFSS COM 介接相關 (本檔損失函數不直接用到，屬套件子模組匯出)。
from .patch_simulator import  com_error
from .patch_simulator.dual_port import DualPortSimulator      # 雙埠 HFSS 模擬器 (供 train_dual 匯入)
from .patch_simulator.single_port import SinglePortSimulator  # 單埠 HFSS 模擬器 (供 train_single 匯入)

import torch

def custom_loss_minmax(prediciton:Tensor, target:Tensor, method:Literal['low', 'high'], loss_type='SmoothL1Loss'):
    """
    單邊極值損失 (Min/Max One-sided Loss)：單埠 train_single 的主損失函數。

    設計意圖：
        相對於同時管目標「最高點與最低點」兩端的寫法，本函數「只挑一個極值點」
        並只做單一方向的懲罰，把「達標即可」的不等式規格表達得最乾淨：
          - method='high'：只看「目標最高點」(如 Gain 中央 +4dB)，只罰「預測偏低」
                           (預測 < 目標)；預測更高視為更好，不罰。
          - method='low' ：只看「目標最低點」(如 S11 中央凹陷 -10dB)，只罰「預測偏高」
                           (預測 > 目標)；預測更低 (匹配更好) 不罰。
        故 train_single 中：S11 用 method='low' (反射夠低即可)、Gain 用 method='high'
        (增益夠高即可)。核心精神同為「夠好就不罰、過頭也不罰」,
        只在「未滿足規格」的方向產生梯度。

    :param prediciton: SM 對該條響應的預測值。
    :param target: 目標響應曲線。
    :param method: 'low' (罰目標最低點處預測偏高) 或 'high' (罰目標最高點處預測偏低)。
    :param loss_type: 'SmoothL1Loss' 或 'MSELoss'。
    """
    criterion = nn.SmoothL1Loss() if loss_type=='SmoothL1Loss' else nn.MSELoss()
    loss_zero = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # 達標時回傳的零損失 (仍保留梯度需求)

    match method:
        case 'high':
            #* 高點規格：要求「預測 >= 目標最高值」(如增益要夠高)
            target_high = target.max()
            mask_high = target == target_high           # 目標最高點所在的頻點
            mask_b_high = prediciton[mask_high] < target_high  # 其中「預測偏低 (未達標)」者才罰
            # 全部達標 → 0；否則只對未達標頻點計誤差，不懲罰「更高」的預測。
            return loss_zero if mask_b_high.sum() == 0 else criterion(
                prediciton[mask_high][mask_b_high], target[mask_high][mask_b_high]
            )

        case 'low':
            #* 低點規格：要求「預測 <= 目標最低值」(如反射要夠低)
            target_low = target.min()
            mask_low = target == target_low             # 目標最低點所在的頻點
            mask_b_low = prediciton[mask_low] > target_low     # 其中「預測偏高 (未達標)」者才罰
            # 全部達標 → 0；否則只對未達標頻點計誤差，不懲罰「更低」的預測。
            return loss_zero if mask_b_low.sum() == 0 else criterion(
                prediciton[mask_low][mask_b_low], target[mask_low][mask_b_low]
            )

        case _:
            # 防呆：method 僅允許 'low' / 'high'。
            raise ValueError('The method must be `low` or `high`.')

# interval_loss 提供兩種呼叫介面 (以下兩個 @overload 僅供型別檢查/IDE 提示，無執行體)：
#   (1) 相對模式：lower/upper 為 float 偏移，邊界 = target + 偏移 (需傳 target)。
#   (2) 絕對模式：lower/upper 為 Tensor，直接當成上下界 (不需 target)。
# 設計意圖：天線規格常以「目標 ± 容差」表達 (如 [target-1, target+1])，比 minmax 更柔性 ──
#   允許預測在容差帶內自由浮動而不受罰，只懲罰「超出帶外」的部分。
@overload
def interval_loss(
    prediction: Tensor, lower_response: float,   upper_response: float,
    target: Tensor = None, *,  loss_type: str = 'SmoothL1Loss',   reduction: str = 'mean',
) -> torch.Tensor:
    """
    Interval Loss: 視為相對於 Target 的誤差容許值[target + lower, target + upper], 限制 prediction 必須在此動態邊界內。

    :param prediction: 預測值。
    :param lower_response: 相對於 Target 的下限偏移 (如 -0.5)
    :param upper_response: 相對於 Target 的上限偏移 (如 0.5)
    :param target: 真實標籤
    :param loss_type: 'SmoothL1Loss' 或 'MSELoss'。
    :param reduction: 'mean' 或 'sum'。
    """
    ...
@overload
def interval_loss(
    prediction: Tensor, lower_response: Tensor,   upper_response: Tensor, *,
    loss_type: str = 'SmoothL1Loss',   reduction: str = 'mean',
) -> torch.Tensor:
    """
    Interval Loss: 限制 prediction 必須在 [lower, upper] 之間。
    
    :param prediction: 預測值
    :param lower_response: 絕對下限值
    :param upper_response: 絕對上限值
    :param loss_type: 'SmoothL1Loss' 或 'MSELoss'。
    :param reduction: 'mean' 或 'sum'。
    """    
    ...


def interval_loss(
    prediction: Tensor,  lower_response: Union[float, Tensor],  upper_response: Union[float, Tensor], 
    target: Tensor = None,* , loss_type: str = 'SmoothL1Loss', reduction: str = 'mean'
) -> Tensor:
    """
    區間損失 (Interval Loss) 的核心運算函數。
    
    :param prediction: 預測值。
    :param lower_response: 
        - Float: 相對於 Target 的下限偏移 (如 -0.5)。
        - Tensor: 絕對下限值。
    :param upper_response: 
        - Float: 相對於 Target 的上限偏移 (如 0.5)。
        - Tensor: 絕對上限值。
    :param target (Tensor, optional): 真實標籤。若使用 float 模式 (相對偏移) 則為必填。
    :param loss_type: 'SmoothL1Loss' 或 'MSELoss'。
    :param reduction: 'mean' 或 'sum'。
    """
    # 底層距離度量：超出容差帶後，依超出量大小計罰 (SmoothL1 對大偏差較穩健)。
    if loss_type == 'SmoothL1Loss':
        loss_fn = nn.SmoothL1Loss(reduction=reduction)
    elif loss_type == 'MSELoss':
        loss_fn = nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    if isinstance(lower_response, Tensor) and isinstance(upper_response, Tensor):
        #* 絕對模式：直接以傳入的 Tensor 當上下界，不依賴 target。
        min_bound = lower_response
        max_bound = upper_response

    else:   #* Target + Offset
        #* 相對模式：邊界隨目標曲線逐點平移，形成「目標 ± 容差」的動態容差帶。
        if target is None:
            raise ValueError("使用 Float (相對偏移模式) 時，必須傳入 target。")

        min_bound = target + lower_response   # 逐頻點下界 = 目標 + 下限偏移 (如 target + (-1))
        max_bound = target + upper_response   # 逐頻點上界 = 目標 + 上限偏移 (如 target + (+1))

    #* Universal Clamp Logic
    # 我們將 Prediction 限制在 [min_bound, max_bound] 範圍內，得到一個「參考目標 (Reference Target)」。
    # - 若 Prediction 在範圍內：Ref = Prediction。 Loss = 0。
    # - 若 Prediction 超出範圍：Ref = 邊界值。 Loss = |Pred - 邊界值|。
    # clamp 把 prediction 夾進 [min_bound, max_bound]：帶內者夾後等於自己 (loss=0)，
    # 帶外者夾到最近邊界；.detach() 讓此「參考目標」不帶梯度、被當成常數。
    target_clamped = torch.clamp(prediction, min=min_bound, max=max_bound).detach() # 確保參考目標被視為常數，讓梯度正確指向 Prediction
    # 以「prediction vs 夾後參考目標」算損失：等價於只懲罰超出容差帶的部分，
    # 梯度方向把帶外預測往最近邊界拉、帶內預測則完全自由 (零梯度)。
    loss = loss_fn(prediction, target_clamped)

    return loss   # 帶內 → 0；帶外 → 與最近邊界的距離