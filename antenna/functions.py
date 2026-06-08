###* ============================================================================
###* antenna/functions.py
###* ----------------------------------------------------------------------------
###* 本檔集中「反向設計閉迴路」所需的可微分損失(loss)、對稱性工具與排程器。
###* 角色定位 (對應專案 pipeline)：
###*   GEN(SigmoidGEN)：目標響應 → 25x25 二元 pattern（STE 可微分二值化）
###*   SM (OldSM/HFSSNet)：pattern → 預測響應，是 HFSS 的可微分替身
###*   SIM(Single/DualPortSimulator)：以 COM 驅動 Ansys HFSS 取得真實響應
###* 閉迴路：GEN 生 pattern → HFSS 得真實 loss → 線上訓練 SM → loss 經 SM 反傳
###*         更新 GEN → 套用 pattern 正則化 → early-stop rollback。
###*
###* 為何需要本檔的正則化 / 排程器？
###*   單純讓 SM-loss 變小，往往會產生「破碎、不連通、無法製造」的金屬圖樣
###*   （像隨機點陣的孤島）。本檔的 loss 把「可製造性 / 連通性」這類物理與工程
###*   約束注入梯度，逼 GEN 收斂到實際做得出來、且電流能流通的天線結構：
###*     - total_variation_loss      ：抑制過度破碎（鼓勵大塊連續區域）
###*     - SpectralConnectivityLoss  ：用圖拉普拉斯 Fiedler 值衡量整體金屬連通性
###*     - GapClosingLoss            ：形態學閉運算填補細小裂縫
###*     - FeedReachability(R_feed)  ：檢核饋電點是否落在同一連通金屬塊上
###*   AdaptiveCyclicalScheduler 則同步調整 GEN 的學習率 lr 與二值化溫度 tau，
###*   在停滯時強制重啟，避免卡在破碎的局部極小值。
###* ----------------------------------------------------------------------------
###* tau 與 STE 的關係（全檔關鍵概念，後續多處引用）：
###*   GEN 的二值化是「sigmoid(logits / tau) → 前向取硬門檻 0/1，反向用軟性梯度」
###*   的 Straight-Through Estimator(STE)。
###*     - tau 大 → sigmoid 平緩，輸出接近 0.5 的灰階，梯度順暢但 pattern 模糊
###*               （適合暖身期廣域探索）。
###*     - tau 小 → sigmoid 陡峭，輸出趨近乾淨的 0/1，pattern 銳利可製造，但梯度
###*               稀疏易卡住（適合退火後期收斂定形）。
###*   因此排程器「lr 與 tau 同步退火」：高溫探索 → 低溫定形，與閉迴路的
###*   early-stop rollback 一起把 GEN 推向可製造解。
###* ============================================================================
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
    #* 為何用「區間」而非單一目標值？天線規格(如 |S11| 需 < -10dB)通常是一段
    #* 可接受範圍，而非精確點。容差帶(dead-zone)讓任何達標的響應都 0 懲罰，
    #* 避免 GEN 為了硬湊單一目標值而過度扭曲 pattern，給足合法解的探索空間。
    #* reduction='none'：逐元素回傳，後面再用 mask 對「超出邊界」的元素挑出計算。
    criterion = nn.SmoothL1Loss(reduction='none') if loss_type == 'SmoothL1Loss' else nn.MSELoss(reduction='none')
    #? SmoothL1(Huber) 對離群點較魯棒：誤差大時近似 L1，可避免響應遠離規格時梯度爆掉。

    # 初始化 loss tensor
    losses = torch.zeros_like(prediction)   #* 預設 0；只有越界的元素才會被填入懲罰值

    # 1. 處理 prediction > target_high 的情況
    mask_above = prediction > target_high   #* 超過上界 → 對齊到上界計算距離
    if mask_above.sum() > 0:
        #* expand_as 把(可能是純量/廣播形狀的)邊界展開成與 prediction 同形，再用同一遮罩取出對應元素
        losses[mask_above] = criterion(prediction[mask_above], target_high.expand_as(prediction)[mask_above])

    # 2. 處理 prediction < target_low 的情況
    mask_below = prediction < target_low    #* 低於下界 → 對齊到下界計算距離
    if mask_below.sum() > 0:
        losses[mask_below] = criterion(prediction[mask_below], target_low.expand_as(prediction)[mask_below])

    # 3. prediction 在區間內的情況 (target_low <= prediction <= target_high)
    #    此時 losses[mask_in_interval] 仍然是 0，不需要額外處理
    #?   區間內梯度為 0：達標即停止施力，是此 loss 的核心「容差」行為。

    return losses.mean() # 返回平均 loss

class FlipMode(Enum):
    """鏡像模式"""
    #* 對稱性對天線設計的意義：許多天線(如對稱偶極、貼片)在結構對稱時，輻射場型
    #* 與阻抗才會對稱、可控；強制對稱也大幅縮小 GEN 的搜尋空間(只需決定 1/2 或
    #* 1/4 的像素)，使收斂更穩、結果更易製造。
    #! 命名陷阱：成員名(horizontal/vertical)與其「切割軸」相反——水平翻轉是把
    #! 左右對調，對稱軸是「垂直線」，故水平翻轉「切的是垂直線」。
    horizontal = '|'    # 水平翻轉所以是切垂直的
    vertical = '-'      # 垂直翻轉所以是切水平的
    both = '*'          #* 同時水平+垂直對稱(四象限對稱)

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
        #* 沿垂直中軸把左右弄成鏡像。回傳兩個版本：以左半為準(ltr) / 以右半為準(rtl)，
        #* 讓上層可挑選或平均，避免只用單側資訊導致偏向。
        H, W = tensor.shape
        mid_w = W // 2
        if W % 2 == 0:  #* 偶數寬：恰好對半切，無中心列
            left_half, right_half = tensor[:, :mid_w], tensor[:, mid_w:]
            ltr = torch.cat([left_half, torch.flip(left_half, dims=[1])], dim=1)  #* 左半 + 左半的鏡像
            rtl = torch.cat([torch.flip(right_half, dims=[1]), right_half], dim=1)  #* 右半的鏡像 + 右半
        else:           #* 奇數寬：保留正中央那一列(center_col)避免遺漏像素
            left_half, center_col, right_half = tensor[:, :mid_w], tensor[:, mid_w:mid_w+1], tensor[:, mid_w+1:]
            ltr = torch.cat([left_half, center_col, torch.flip(left_half, dims=[1])], dim=1)
            rtl = torch.cat([torch.flip(right_half, dims=[1]), center_col, right_half], dim=1)
        return ltr, rtl

    def _get_vertical_mirrors(tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """輔助函數：垂直翻轉"""
        #* 與水平版同理，但沿水平中軸把上下弄成鏡像；回傳以上半為準(ttb)/以下半為準(btt)。
        H, W = tensor.shape
        mid_h = H // 2
        if H % 2 == 0:  #* 偶數高：對半切，無中心行
            top_half, bottom_half = tensor[:mid_h, :], tensor[mid_h:, :]
            ttb = torch.cat([top_half, torch.flip(top_half, dims=[0])], dim=0)  #* 上半 + 上半的鏡像
            btt = torch.cat([torch.flip(bottom_half, dims=[0]), bottom_half], dim=0)  #* 下半的鏡像 + 下半
        else:           #* 奇數高：保留正中央那一行(center_row)
            top_half, center_row, bottom_half = tensor[:mid_h, :], tensor[mid_h:mid_h+1, :], tensor[mid_h+1:, :]
            ttb = torch.cat([top_half, center_row, torch.flip(top_half, dims=[0])], dim=0)
            btt = torch.cat([torch.flip(bottom_half, dims=[0]), center_row, bottom_half], dim=0)
        return ttb, btt

    def _get_quadrant_mirrors(tensor: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """輔助函數：'both' 模式的象限翻轉"""
        #* 四象限對稱：把某一象限同時水平+垂直鏡像，鋪成上下左右皆對稱的完整圖。
        #* 四個象限各自當「種子」可得 4 種候選；對 25x25(奇數)而言，中心行列必須
        #* 被「恰好一次」納入，否則會重複/遺漏一條線，這正是下方奇偶判斷的重點。
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
        #?   建構手法固定為「先左右鏡像補滿一半 → 再上下鏡像補滿整圖」。
        #?   差別只在各象限「是否含中心行列」，故下方對奇數維度用 (W%2)/(H%2)
        #?   切片來「跳過會重複的那一條中心線」。

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

    
    #* mode 可傳 FlipMode 列舉或字串(可組合，如 '-|*' 一次要多種對稱)
    if isinstance(mode, FlipMode):
        mode = [mode.value]   #* 列舉 → 取其 value 字元，統一成可迭代字元序列
    else:
        # 驗證 mode 字串中的所有字元是否合法
        valid_modes = {'|', '-', '*'}
        if not set(mode).issubset(valid_modes):
            invalid_chars = set(mode) - valid_modes
            raise ValueError(f"無效的 mode 字元: {invalid_chars}。請只使用 '|', '-', '*' 的組合。")


    results = []
    # 迭代處理 mode 中的每個字元，並收集結果, 使用 sorted(set(mode)) 可以確保執行順序固定，且避免重複執行
    #? set 去重避免同一模式重複展開；sorted 固定順序讓輸出 tuple 的排列可重現(利於測試/比對)。
    for char_mode in sorted(list(set(mode))):
        if char_mode == '-':
            results.extend(_get_horizontal_mirrors(input))  #* 每種水平對稱貢獻 2 個 tensor
        elif char_mode == '|':
            results.extend(_get_vertical_mirrors(input))    #* 每種垂直對稱貢獻 2 個 tensor
        elif char_mode == '*':
            results.extend(_get_quadrant_mirrors(input))    #* 四象限對稱貢獻 4 個 tensor

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
    #* 用途：把離散的「指派/排列」問題鬆弛成可微分的(近)雙隨機矩陣，讓梯度能流過
    #* 原本不可微的 argmax/匈牙利指派。tau 在此同樣是「鬆弛溫度」：tau 越小越接近
    #* 硬指派(0/1)但梯度越尖；tau 越大越平滑。與 GEN 二值化的 tau 概念一致。

    # Gumbel 雜訊擾動 (這裡我們手動實現，因為 F.gumbel_softmax 假設維度是 logits.shape[-1])
    #? 兩層 -log 即反 CDF 抽 Gumbel(0,1)；+1e-20 防 log(0)。加雜訊是為了「重參數化」
    #? 採樣——每次得到不同的可微分樣本，鼓勵探索而非永遠取同一指派。
    gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    perturbed_logits = (logits + gumbels) / tau   #* 除以 tau 控制鬆弛尖銳度

    # 為了數值穩定性，在 log-space 進行迭代
    #? Sinkhorn 正規化：在 log 空間交替對列/行做 log-softmax，等價於反覆對指數空間
    #? 做 row/column normalize，收斂到(近)雙隨機矩陣(每列每行和趨近 1)。
    log_alpha = perturbed_logits

    for _ in range(n_iters):
        # 沿著 M 維度 (物件) 進行正規化
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)  #* 每個位置對所有物件的分配和 → 1
        # 沿著 K 維度 (位置) 進行正規化
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)  #* 每個物件對所有位置的分配和 → 1

    soft_assignment = torch.exp(log_alpha)   #* 回到機率空間：軟性指派矩陣(可微分)

    if hard:
        # 取得離散的指派結果 (不可微分)
        #! hard=True 直接回傳 one-hot，會切斷梯度；若需可微分硬指派應在外層自行做
        #! STE(forward 用此硬值、backward 用 soft_assignment)，本函式不代為處理。
        _, indices = torch.max(soft_assignment, dim=-1)
        hard_assignment = F.one_hot(indices, num_classes=logits.shape[-1]).to(logits.dtype)
        return hard_assignment

    return soft_assignment

def total_variation_loss(img, weight=0.01):
    """計算 Total Variation Loss 以抑制過度破碎的圖樣"""
    #* 為何重要：TV loss 懲罰相鄰像素的差異，鼓勵大塊「連續同值」區域、抑制
    #* 棋盤格般的高頻雜訊。對天線而言，破碎像素＝無法蝕刻製造、且電氣行為不可控；
    #* TV 把 GEN 推向平滑連續的金屬塊，是「可製造性」最基本的正則化。
    #! 與 STE 的互動：在較大 tau(輸出偏灰階)時 TV 梯度最有意義；tau 太小、pattern
    #! 已硬二值化後，相鄰差只剩 0/1，TV 主要懲罰邊界周長(鼓勵更圓潤、更少邊界)。
    from .utils.data import size_converter
    from . import AntennaPattern
    img = size_converter(AntennaPattern, img, output_shape="B, 1, H, W")  #* 統一成 (B,1,H,W) 才能做維度差分
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:,:,1:,:] - img[:,:,:-1,:], 2).sum()   #* 垂直方向相鄰列差平方和(高頻能量)
    tv_w = torch.pow(img[:,:,:,1:] - img[:,:,:,:-1], 2).sum()   #* 水平方向相鄰行差平方和
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)   #* 除以元素數做正規化，使量級不隨解析度漂移

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
    #* 在閉迴路中的角色：本排程器同時驅動兩條曲線——
    #*   (1) GEN 優化器的學習率 lr：暖身→退火，決定每步更新 pattern 的步幅。
    #*   (2) 二值化溫度 tau(AntennaPattern.tau)：高溫探索→低溫定形(見檔頭 STE 說明)。
    #* 三種思想融合：
    #*   OneCycle：每個週期開頭先「暖身」緩升，避免一開始大步長把 GEN 帶歪。
    #*   CosineAnnealingWarmRestarts：暖身後餘弦退火，週期性回到高點(warm restart)
    #*       讓 GEN 有機會跳出破碎的局部解、重新廣域探索。
    #*   ReduceLROnPlateau：監控真實 loss，停滯(patience 耗盡)時「強制重啟」並縮短週期。
    #! 關鍵副作用：step() 內會寫 AntennaPattern.tau —— lr 與 tau 永遠同步，不可
    #! 期望單獨調整其一。
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

        super(AdaptiveCyclicalScheduler, self).__init__(optimizer, last_epoch)
        #! base_lrs 覆寫為 lr_max：本排程器自行算 lr(get_lr)，不依賴父類用 base_lr 縮放
        self.base_lrs = [lr_max] * len(self.optimizer.param_groups)

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

    def step(self, metric: float = None):
        #* 每個訓練步呼叫一次；務必傳入監控指標(通常是 HFSS 真實 loss)，否則自適應失效。
        if metric is None:
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
                #* 停滯太久：強制重啟。縮短週期(乘 factor)讓後續探索更密集，但不低於
                #* T_0//2 以免週期過短、暖身/退火失去意義。
                self.patience_counter = 0
                self.T_i = max(int(self.T_i * self.factor), self.T_0 // 2) # 保持最小週期長度限制

                # 2. 決定重啟位置 (Apply on_plateau strategy)
                #? 三種策略差在「停滯後 lr/tau 跳到哪個高度」，激進程度 peak > linear > reset：
                #?   peak  —— 最激進：直接拉到最高溫/最高 lr 全力跳出局部解(可能震盪)。
                #?   reset —— 最溫和：完全重來、重新暖身，較穩但較慢回到探索強度。
                #?   linear—— 折衷：從目前高度沿暖身線往上爬，保留部分已收斂的進度。
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
                            #! +1e-10 防 lr_max==lr_min 時除零；ratio 即「目前 lr 在暖身線上的相對高度」。
                            ratio = (current_lr - self.lr_min) / (self.lr_max - self.lr_min + 1e-10)
                            ratio = max(0.0, min(1.0, ratio)) # 限制範圍以防萬一

                            # D. 設定時間點：反推對應的步數
                            # 這樣下一次 get_lr() 就會從這個高度繼續往上走
                            self.T_cur = int(round(ratio * warmup_steps))
                        else:
                            # 如果沒有暖身區間，就直接設為 -1 (避免除以零)
                            self.T_cur = -1
                    case _:
                        pass   #* 理論上 __init__ 已驗證，不會走到這裡

        # --- 週期性邏輯 ---
        #* 自然走完一個週期(T_cur 追上 T_i)：歸零並依 T_mult 拉長下個週期(warm restart)；
        #* 否則單純前進一步。注意上面的強制重啟已透過改寫 T_cur 改變這裡的落點。
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
        #! 關鍵跨模組副作用：直接設定 AntennaPattern.tau(類別屬性)，GEN 下次二值化即套用
        #! 新溫度。這是 lr 與 tau「同步退火」的實際接點。
        from . import AntennaPattern
        AntennaPattern.tau = self.get_temp()

        self.record['lr'] = self.get_lr()[0]   #* 記錄供 plot() 畫雙軸曲線
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
    """以圖拉普拉斯(graph Laplacian)的代數連通度衡量金屬連通性的可微分 loss。

    原理(譜圖理論)：把 25x25 pattern 視為網格圖，金屬像素間的邊權重高、介質間低。
    拉普拉斯矩陣 L=D-A 的第二小特徵值 λ2(Fiedler value)是「代數連通度」：
        λ2 ≈ 0 → 圖近乎斷裂(存在孤島)；λ2 越大 → 整體越連通、越難被切開。
    天線需電流能在整塊金屬上流動，故希望 λ2 大；本 loss 取 1/λ2，最小化它即等於
    鼓勵 λ2 增大、消除孤島。相較只看局部鄰接的 TV/GapClosing，這是「全域連通」約束。
    """
    #! 計算成本高：對 num_nodes×num_nodes(625×625)做 eigvalsh，且 forward 內逐 batch
    #! for 迴圈跑特徵分解，屬重型正則化，通常以較小權重、較低頻率使用。
    def __init__(self, height=25, width=25, epsilon=1e-2):
        super().__init__()
        self.H, self.W = height, width
        self.num_nodes = height * width
        self.epsilon = epsilon  # 基礎連通性，防止 lambda_2 鎖死在 0
        #? epsilon 的妙用：給每條邊一個微小底權重，使圖永遠是「連通」的(λ2>0)，
        #? 避免 λ2≡0 造成 1/λ2 爆炸或梯度消失；金屬-金屬邊則在此底值上再加 1.0。

        # 預建鄰接索引
        #* 只連「右」與「下」鄰居即可涵蓋所有 4-鄰接無向邊(避免重複建邊)，下方 forward
        #* 再對稱填入 A[src,dst] 與 A[dst,src]。索引預建並註冊為 buffer，隨模型搬到對的 device。
        src, dst = [], []
        for r in range(height):
            for c in range(width):
                idx = r * width + c
                if r + 1 < height: src.append(idx); dst.append((r + 1) * width + c)   #* 與下方鄰居連邊
                if c + 1 < width: src.append(idx); dst.append(r * width + (c + 1))    #* 與右方鄰居連邊
        self.register_buffer('src', torch.tensor(src).long())
        self.register_buffer('dst', torch.tensor(dst).long())

    def forward(self, antenna_map):
        # 確保輸入是 (B, 1, H, W)
        if antenna_map.dim() == 2: antenna_map = antenna_map.unsqueeze(0).unsqueeze(0)
        batch_size = antenna_map.shape[0]
        flat_map = antenna_map.view(batch_size, -1)   #* 攤平成節點向量，索引對應 r*W+c
        losses = []

        for b in range(batch_size):
            # 改進權重邏輯：金屬與金屬=1.0, 其他部分至少有 epsilon 的連通性
            node_vals = flat_map[b]
            # 邊權重 = 節點相乘的加權 + 基礎連通性
            #? 乘積讓「兩端皆金屬(≈1*1)」的邊權重高、「一端為介質(≈0)」的邊權重趨近 epsilon。
            #? 因 pattern 此時多為連續(可微分)值，梯度可經此乘積回傳到 GEN，是可微分的關鍵。
            w = (node_vals[self.src] * node_vals[self.dst]) + self.epsilon

            # 建立對稱拉普拉斯
            A = torch.zeros(self.num_nodes, self.num_nodes, device=antenna_map.device)
            A[self.src, self.dst] = w   #* 無向圖 → 對稱填入兩個方向
            A[self.dst, self.src] = w
            D = torch.diag(torch.sum(A, dim=1))   #* 度矩陣(對角=各節點邊權重和)
            L = D - A                             #* 拉普拉斯 L=D-A，半正定，特徵值升序

            # 計算特徵值
            eigvals = torch.linalg.eigvalsh(L)   #* 對稱矩陣用 eigvalsh：較快且數值穩定
            lambda_2 = eigvals[1]  # Fiedler Value
            #? λ1 必為 0(常數向量)，λ2 才反映連通強度，故取索引 1。

            # 目標：讓 lambda_2 越大越好（金屬連通後 lambda_2 會顯著增加）
            # 使用負對數或反比，讓差異更明顯
            losses.append(1/(lambda_2))   #* 反比：λ2 小(快斷裂)→loss 大，強力推連通
            # losses.append(-torch.log(lambda_2+1e6))
            # losses.append(torch.exp(-lambda_2))   #? 上兩行為替代懲罰式(已停用)，曲線形狀不同

        return torch.mean(torch.stack(losses))   #* 對 batch 取平均

class GapClosingLoss(nn.Module):
    """形態學「閉運算(Closing)」的可微分版，用來懲罰金屬中的細小裂縫/孔洞。

    閉運算 = 先膨脹(Dilation)再侵蝕(Erosion)：膨脹會把細縫填滿、侵蝕再縮回原本外形，
    淨效果是「補洞但幾乎不改變大塊外輪廓」。若 pattern 本來就沒裂縫，closed≈原圖、
    loss≈0；有裂縫時 closed 與原圖差異大、loss 上升，逼 GEN 自己把縫補起來。
    與 SpectralConnectivityLoss 互補：此 loss 處理「局部、小尺度」斷點，連通 loss 顧
    「全域」連通；兩者都服務於「電流可流通且可製造」的最終目標。
    """
    def __init__(self):
        """
        Closing = Dilation(膨脹) + Erosion(侵蝕)
        """
        super().__init__()

    def forward(self, antenna_map):
        # R = (kernel_size-1)/2   #* kernel=3 → 影響半徑 R=1，只補 1 像素級的細縫

        # 1. Soft Dilation (膨脹) - 填補裂縫
        # 使用 MaxPool 模擬
        #? MaxPool 取鄰域最大值＝形態學膨脹；對連續值是「可微分」的軟膨脹(梯度走最大值處)。
        dilated = F.max_pool2d(antenna_map, kernel_size=3, stride=1, padding=1)

        # 2. Soft Erosion (腐蝕) - 恢復外形
        # Erosion(x) = -Max(-x)
        #? 對偶關係：取負→MaxPool(=膨脹)→再取負，即等價於鄰域最小值＝侵蝕。
        closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)

        # 3. 計算 Loss
        # 如果 antenna_map 有裂縫，closed 會把裂縫填滿 (數值變大)
        # 我們希望 antenna_map 本身就沒有裂縫，即 antenna_map 應該接近 closed
        # Loss = || Closed - Original ||
        #! 注意：此 loss 只在「裂縫處」非零，故不會強迫填滿大空洞，僅整治細縫，
        #! 不致過度肥大化金屬而破壞天線設計意圖。
        loss = torch.mean((closed - antenna_map) ** 2)
        return loss

class FeedReachability: # R_feed
    """饋電連通度指標 R_feed：評估「所有饋電點是否落在同一塊連通金屬上」。

    物理意義：天線必須由饋電點(feed)把訊號送進金屬輻射體；若某饋電點下沒有金屬、
    或多個饋電點各自連到不同的金屬孤島，天線就無法正常工作。R_feed 因此是「pattern
    在工程上是否可用」的硬性檢核。
    與本檔其他 loss 的差別：R_feed 走 scipy 連通元件標記(label)，是「不可微分的評估
    指標(metric)」，用於監看/early-stop/視覺化，而非直接回傳梯度的訓練 loss。
    回傳值定義為「共同連通塊像素 / 全部金屬像素」的佔比：全部饋電點同塊時介於 0~1，
    否則為 0。single_feed/dual_feed 為 25x25 預設饋電佈局的便捷建構。
    """
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
        # self.structure = np.ones((3, 3))   #* 註解掉的此版才是真 8-連通(含對角)
        #! docstring 與實作不符的陷阱：下方十字結構其實是 4-連通(只連上下左右)，
        #! 與下方字串敘述的「8-連通」相反；要改 8-連通需改回上一行的 np.ones((3,3))。
        self.structure = np.array([
            [0.0,1.0,0.0],
            [1.0,1.0,1.0],
            [0.0,1.0,0.0]])
        """連通性, 預設採用 8-連通 (8-connectivity), 若要4連通可以是十字架"""
        self.record:list[FeedReachabilityDictType] = []   #* 累積各次評估結果，供逐 epoch 繪圖
        self.r_feed_str = "$R_{{feed}}$"   #* matplotlib LaTeX 標籤用字串

        self._label = label   #* 綁定 scipy.ndimage.label(連通元件標記)以利重複呼叫
    
    @classmethod
    def single_feed(cls):
        """單埠佈局：饋電點在底邊中央 (對應 SinglePortSimulator)。"""
        shape = (25, 25)
        return cls([(shape[0]-1, int((shape[1])/2))])   #* (最底列, 中間行)

    @classmethod
    def dual_feed(cls):
        """雙埠佈局：饋電點在底邊中央與頂邊中央 (對應 DualPortSimulator)。"""
        shape = (25, 25)
        return cls([(shape[0]-1, int((shape[1])/2)),(0, int((shape[1])/2))])   #* 底邊中央 + 頂邊中央
    
    def __call__(self, pattern: Union[Tensor, np.ndarray], *, record:bool=False, title:str = "Pattern ($R_{{feed}}$={rate:.2%})"):
        """
        :param pattern: 2D array (1=金屬, 0=介質)
        :return: Feed Reachability Rate
        """
        
        if isinstance(pattern, Tensor):
            pattern = pattern.numpy()   #* scipy 只吃 ndarray；此處等同切斷梯度(本類為評估指標)

        #* label：把 pattern 中相連的金屬像素標成同一整數 ID，0 代表背景(介質)
        labeled_array, _ = self._label(pattern, structure=self.structure)

        # 1. 取得所有饋電點所在的 Label IDs
        feed_labels = []

        for pos in self.feed_positions:

            # 檢查座標是否越界或該處無金屬
            if 0 <= pos[0] < pattern.shape[0] and 0 <= pos[1] < pattern.shape[1]:
                lbl = labeled_array[pos]   #* 取該饋電點所屬的連通塊 ID
                if lbl > 0:
                    feed_labels.append(lbl)
                else:
                    #! 任一饋電點下方無金屬(lbl==0)即整體失敗：天線根本無法被饋電
                    logger.warning("其中一個潰入點沒金屬，直接失敗")
                    return 0.0, np.zeros_like(pattern) # 其中一個點沒金屬，直接失敗
            else:
                logger.error("潰入點座標是否越界")
                return 0.0, np.zeros_like(pattern)

        # 2. 「AND」邏輯檢查：判斷所有饋電點的 Label 是否完全相同
        #?   只有所有饋電點的塊 ID 相同(set 長度為 1)才算「共同連通」；只要有人在
        #?   不同塊上就算失敗，因為各埠的電流路徑沒有真正接在一起。
        unique_labels = set(feed_labels)

        if len(unique_labels) == 1:
            # 所有饋電點都在同一個連通塊上
            shared_label = list(unique_labels)[0]
            shared_mask = (labeled_array == shared_label)   #* 此共同塊的布林遮罩

            total_metal_pixels = np.sum(pattern)            #* 全部金屬像素
            connected_pixels = np.sum(shared_mask)          #* 落在共同塊上的金屬像素
            #* 佔比越高代表越多金屬實際參與饋電路徑、孤島越少(理想接近 1.0)
            mutual_index = connected_pixels / total_metal_pixels

        else:
            # 饋電點分布在不同的連通塊上，或彼此斷開
            mutual_index = 0.0
            shared_mask = np.zeros_like(pattern)
        
        self.rate = mutual_index
        self.mask = shared_mask
        self.pattern = pattern
        self.title = title.format(rate=mutual_index)   #* 把 rate 填入標題模板，供繪圖顯示

        if record:   #* 需逐 epoch 追蹤趨勢時才存檔，避免無謂記憶體累積
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
        #* 依 title 分組的 rate 序列：同一標題(同一條曲線)的歷次 rate 彙整成 list，
        #* 方便在同一張圖上畫多條(例如不同饋電佈局)。
        result = defaultdict(list)
        for entry in self.record:
            result[entry['title']].append(entry['rate'])

        return result

    @property
    def r_feed_list(self):
        return [_['rate'] for _ in self.record]   #* 所有歷史 rate(0~1)

    @property
    def rate_list(self):
        return [_['rate']*100 for _ in self.record]   #* 同上但換算成百分比(0~100)

    @property
    def r_feed_avg(self):
        return np.mean(self.r_feed_list)   #* 整段訓練的平均 R_feed(單一總結指標)
    
    def plot(self, axes = None, show=False, data:FeedReachabilityDictType=None):
        """視覺化單筆結果：底圖灰、金屬中灰、共同連通塊綠、饋電點紅(黃框)。"""
        #* 傳入 data(來自 record 的某筆)即可重畫歷史結果；不傳則畫最近一次 __call__ 的狀態。
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