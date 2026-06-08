"""
================================================================================
antenna.patch — 微帶貼片天線「反向設計」的損失函數 (Loss Functions)
================================================================================

【這個檔案在做什麼？】
    本模組集中定義「衡量預測響應 vs 目標響應」的損失函數，是整條反向設計閉迴路
    (GEN 目標響應→pattern；SM pattern→預測響應；SIM 真實 HFSS 模擬) 的「評分尺」。
    GEN 生成 pattern 後，SM 算出預測響應，再用這裡的 loss 與「想要的目標」比較，
    梯度沿 loss → SM → pattern → GEN 反向傳播以更新 GEN。
    這些函數同時也用來在 train_single.py / train_dual.py 中經
    `AntennaResponse.registerLossHook(...)` 註冊成各條 S 參數曲線的評分函數。

【共同的核心設計哲學：為什麼不直接用 MSE？】
    天線規格本質上是「不等式」而非「等式」：
      - 反射 (S11/S22)：只要「夠低」(匹配夠好) 即可，更低不會是壞事；
      - 增益 (Gain/S21)：只要「夠高」即可，更高同樣不是壞事。
    若直接對整條目標曲線做 MSE，會把「比目標更好」的預測也當成誤差去懲罰，
    反而把已經達標的解硬拉回目標線，與物理需求相悖。因此本檔的損失皆採
    「單邊懲罰 (one-sided)」或「區間容差 (interval)」設計 ──「夠好就不罰、過頭也不罰」，
    只在「沒達到規格」的方向上產生梯度，引導 GEN 往滿足規格的可行域移動。

【本檔四個損失函數】
    - custom_loss_r      反射損失：只在目標「最高點」與「最低點」處，且預測「未達標」時才罰。
    - custom_loss_g      增益損失：同上邏輯，方向對應「增益要高」的需求。
    - custom_loss_minmax 單邊極值損失：method='low' 只罰「目標最低點處預測偏高」、
                         method='high' 只罰「目標最高點處預測偏低」(單埠 train_single 使用)。
    - interval_loss      區間損失：要求預測落在 [target+lower, target+upper] (相對) 或
                         [lower, upper] (絕對) 區間內，區間內 loss=0 (雙埠 train_dual 使用)。
"""
# ..utils：帶入 nn、Tensor、config、Literal/Union/overload 等型別與工具 (見 antenna/utils 與 antenna/types)。
from ..utils import *
# patch_simulator：HFSS COM 介接相關 (本檔損失函數不直接用到，屬套件子模組匯出)。
from .patch_simulator import  com_error
from .patch_simulator.dual_port import DualPortSimulator      # 雙埠 HFSS 模擬器 (供 train_dual 匯入)
from .patch_simulator.single_port import SinglePortSimulator  # 單埠 HFSS 模擬器 (供 train_single 匯入)

import torch

def custom_loss_r(prediciton, target, loss_type='SmoothL1Loss'):
    """
    反射損失 (Reflection Loss)：針對 S11/S22 等反射型 S 參數的單邊懲罰損失。

    設計意圖：
        反射響應的目標曲線通常呈「兩端高、中央低」(如 side=-1.25dB、center=-12dB)，
        中央凹陷代表諧振頻段匹配良好 (能量幾乎全進天線、反射很小)。
        我們只在「目標的兩個極值點」(最高點 high_response、最低點 low_response) 上
        檢查預測是否達標，且只在「未達標的方向」才計入損失：
          - 在最高點 (兩端、要求反射高/未諧振)：只罰「預測 < 目標」(預測太低)；
          - 在最低點 (中央、要求反射夠低/匹配好)：只罰「預測 > 目標」(凹陷不夠深)。
        如此即可避免用 MSE 把「比目標更好」的預測也拉回目標線，符合「夠好就不罰」的物理需求。

    :param prediciton: SM 對該條 S 參數的預測響應。
    :param target: 目標響應曲線。
    :param loss_type: 'SmoothL1Loss' (對離群值較穩健) 或 'MSELoss'。
    """
    # 距離度量：SmoothL1 在誤差小時近似 L2、大時近似 L1，對偶發大偏差較不敏感。
    criterion_r = nn.SmoothL1Loss() if loss_type=='SmoothL1Loss' else nn.MSELoss()

    high_response = target.max()   # 目標最高點 (反射型：對應兩端、要求未諧振處反射高)
    low_response = target.min()    # 目標最低點 (反射型：對應中央諧振凹陷、要求匹配最好處)

    mask_25 = target == high_response  # mask == -2.5 index  #* 選出所有「等於目標最高值」的頻點
    mask_b_25 = prediciton[mask_25] < high_response          #* 其中「預測低於目標」者才算未達標 (要罰)

    if mask_b_25.sum()==0:
        # 全部達標 (預測都 >= 目標最高值)：此項不貢獻梯度，回傳 0。
        loss_25 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        # 只對「未達標」的頻點計算誤差，避免懲罰「比目標更好」的預測。
        loss_25 = criterion_r(prediciton[mask_25][mask_b_25], target[mask_25][mask_b_25])

    mask_10 = target == low_response  # mask == -2.5 index  #* 選出所有「等於目標最低值」的頻點
    mask_b_10 = prediciton[mask_10] > low_response          #* 其中「預測高於目標」者才算未達標 (凹陷不夠深)

    if mask_b_10.sum()==0:
        # 全部達標 (預測都 <= 目標最低值，匹配夠好甚至更好)：不罰，回傳 0。
        loss_10 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        loss_10 = criterion_r(prediciton[mask_10][mask_b_10], target[mask_10][mask_b_10])

    loss = loss_25 + loss_10   # 兩個極值點的單邊懲罰相加
    return loss

def custom_loss_g(prediciton, target, loss_type='SmoothL1Loss'):
    """
    增益損失 (Gain Loss)：針對 Gain/S21 等「越大越好」響應的單邊懲罰損失。

    設計意圖：
        增益型目標曲線通常呈「兩端低、中央高」(如 side=-19dB、center=+4dB)，
        代表只在工作頻段 (中央) 要有高增益、頻段外則希望低 (抑制旁波/雜訊)。
        與 custom_loss_r 同為單邊懲罰，但方向對調以符合增益的物理需求：
          - 在最低點 (兩端、要求頻段外增益低)：只罰「預測 > 目標」(沒壓夠低)；
          - 在最高點 (中央、要求工作頻段增益高)：只罰「預測 < 目標」(增益不足)。
        「增益比目標更高、或頻段外比目標更低」皆視為達標，不予懲罰 (避免 MSE 把好解拉回)。

    :param prediciton: SM 對 Gain/S21 的預測響應。
    :param target: 目標增益曲線。
    :param loss_type: 'SmoothL1Loss' 或 'MSELoss'。
    """
    criterion_g = nn.SmoothL1Loss() if loss_type=='SmoothL1Loss' else nn.MSELoss()

    high_gain = target.max()   # 目標最高增益 (對應工作頻段中央，要求增益高)
    low_gain = target.min()    # 目標最低增益 (對應頻段外兩端，要求增益被壓低)

    mask_10 = target == low_gain  # mask == -10 index  #* 選出「等於目標最低增益」的頻點 (兩端)
    mask_b_10 = prediciton[mask_10] > low_gain         #* 其中「預測高於目標」者未達標 (沒壓夠低)

    if mask_b_10.sum()==0:
        # 兩端增益都已壓到目標以下：達標，不罰。
        loss_10 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        loss_10 = criterion_g(prediciton[mask_10][mask_b_10], target[mask_10][mask_b_10])

    mask_4 = target == high_gain  # mask == 4 index  #* 選出「等於目標最高增益」的頻點 (中央)
    mask_b_4 = prediciton[mask_4] < high_gain         #* 其中「預測低於目標」者未達標 (增益不足)

    if mask_b_4.sum()==0:
        # 中央增益都已達到或超過目標：達標 (更高更好)，不罰。
        loss_4 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    else:
        loss_4 = criterion_g(prediciton[mask_4][mask_b_4], target[mask_4][mask_b_4])

    loss = loss_10 + loss_4   # 兩端「壓低」與中央「拉高」兩項單邊懲罰相加
    return loss

def custom_loss_minmax(prediciton:Tensor, target:Tensor, method:Literal['low', 'high'], loss_type='SmoothL1Loss'):
    """
    單邊極值損失 (Min/Max One-sided Loss)：單埠 train_single 的主損失函數。

    設計意圖：
        custom_loss_r/g 同時管目標的「最高點與最低點」兩端；本函數則「只挑一個極值點」
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