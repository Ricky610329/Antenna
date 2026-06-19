"""
antenna/utils/torch_utils.py — PyTorch 張量輔助工具

封裝目的
--------
反向設計閉迴路系統（GEN / SM / SIM）中，各模組（尤其是 SM 的
train_one_data）需要頻繁建立張量，且幾乎都要求：
  * 自動搬到 config.device（GPU 或 CPU，由全域設定決定）
  * 以 float64 作為梯度計算的預設 dtype
  * 視情況開啟 requires_grad

直接呼叫原生 torch.tensor() 需要每次手動傳入 device / dtype / requires_grad，
容易遺漏且散落各處難以維護。本檔將這些慣例集中管理，讓呼叫端只需在意
「要不要算梯度」，其餘細節交由包裝函式一致處理。

重新匯出的原生符號
------------------
stack / concat 直接從 torch 重新匯出，讓其他模組只需 import torch_utils
即可取得常用張量操作，不必再分別 import torch。
"""
from torch import (
    __version__,
    nn,
    tensor as _tensor,
    Tensor,
    cuda,
    manual_seed as _manual_seed,
    load as _torch_load,
    save as _torch_save,
    device as _torch_device,
    # get_default_device,
    set_default_device,
    stack,      # 重新匯出：沿 new axis 堆疊多個張量，等同 torch.stack
    concat,     # 重新匯出：沿既有 axis 串接多個張量，等同 torch.cat / torch.concat
    float64,    # 重新匯出：float64 dtype 常數，供預設 dtype 判斷使用
    set_grad_enabled, is_grad_enabled # with no_grad():...
)
import numpy as np
from numpy import (
    ndarray,
    random
)
from .utils import config  #? 全域設定物件，包含 config.device（目標運算裝置）
from .types import Sizable, Tensor_B_N, Tensor_B_W_H, Tensor_N, Tensor_W_H
from typing import Any, Literal, Optional, overload
import torch  #? size_converter 的型別註解需要 torch.Tensor


try:
    from torch.utils.tensorboard import SummaryWriter # type:ignore pip install tensorboard
    def getTensorBoardWriter(log_dir:str = './runs') -> SummaryWriter:
        """
        建立 TensorBoard SummaryWriter，用於訓練過程的純量、直方圖等視覺化記錄。

        啟動 TensorBoard 的方式：

        ## Usage
        ```bash
        tensorboard --logdir=runs
        ```

        ## Example
        ```
        tbwriter = getTensorBoardWriter()
        for n_iter in range(100):
            tbwriter.add_scalar('Loss/train', np.random.random(), n_iter)
            tbwriter.add_scalar('Loss/test', np.random.random(), n_iter)
            tbwriter.add_scalar('Accuracy/train', np.random.random(), n_iter)
            tbwriter.add_scalar('Accuracy/test', np.random.random(), n_iter)
        ```
        """
        return SummaryWriter(log_dir)
except ModuleNotFoundError:
    pass  #! tensorboard 未安裝時靜默略過，不影響主流程

# ─────────────────────────────────────────────
# tensor()  —  輕量版張量建構包裝
# ─────────────────────────────────────────────
def tensor(data: Any,dtype= None, device=None, requires_grad: bool = False):
    ###* 與原生 torch.tensor() 的差異：
    ###* device 預設值改為 config.device（全域裝置），省去每次手動指定
    ###* 呼叫端仍可覆寫 device；requires_grad 預設 False 與原生相同
    ###* 使用場景：不需要特殊 dtype 判斷、需要快速建立固定精度張量時
    if isinstance(data, Tensor):
        #? data 已是張量：torch.tensor(t) 會複製並印 copy-construct 警告 →
        #  改用 detach().clone() 取得等價的乾淨葉節點 (值/dtype/device 不變，只是不再警告)。
        out = data.detach().clone().to(device=device or config.device)
        if dtype is not None:
            out = out.to(dtype=dtype)
        return out.requires_grad_(requires_grad)
    return _tensor(data, dtype=dtype, device=device or config.device, requires_grad=requires_grad)

# ─────────────────────────────────────────────
# cTensor()  —  「有無梯度計算」條件式張量建構
# ─────────────────────────────────────────────
def cTensor(data:Any, requires_calculate:bool, *, device=None, dtype = None):
        """
        Creating Tensors.

        「c」代表 conditional（條件式）：根據 requires_calculate 旗標自動決定
        裝置與梯度設定，是 SM train_one_data 的核心輔助工具。

        與原生 torch.tensor() 的差異
        ----------------------------
        * requires_calculate=True  → 確保 floating-point dtype（不足則轉 float64）
                                     → 搬至 config.device
                                     → requires_grad=True（可參與反向傳播）
        * requires_calculate=False → 強制留在 CPU、不開梯度（僅儲存或回報用）

        Args:
            requires_calculate (bool):
                - If True: device=device or config.device, requires_grad=True
                - If Fasle: device='cpu', requires_grad=False
            device:
                Not applicable when requires_calculate is Fasle.

        Example:
            ```
            cTensor([1, 2, 3], requires_calculate=True)     # tensor([1., 2., 3.], dtype=torch.float64, requires_grad=True)
            cTensor([1, 2, 3], requires_calculate=False)    # tensor([1, 2, 3])
            ```

        """
        t = _tensor(data)
        if requires_calculate:
            t = t if t.dtype.is_floating_point else t.type(dtype or float64)  #* 整數型別自動升為 float64，確保梯度可計算
            t = t.to(device or config.device)  #* 搬至目標運算裝置（GPU 優先）
            t = t.requires_grad_(True)         #* 開啟梯度追蹤，供 backward() 使用
        else:
            t = t.cpu()  #* 僅作記錄/輸出用，強制回 CPU 節省 GPU 記憶體
        return t


###* ============================================================================
###* size_converter：★通用張量形狀轉換器★
###*   GEN / SM / 正則化在 pipeline 中反覆需要把「一團展平或半成形的張量」整回正確形狀。
###*   它不靠呼叫端硬編形狀，而是向 `sizer`(具 .size() 的物件，描述單一樣本的 N 與 (H, W))
###*   詢問每筆樣本的尺寸，再據此自動推出 batch 大小 B 並 reshape。
###*
###*   兩種模式(由 output_shape 是否為 None 決定)：
###*     ◆ 模式 1（output_shape=None，用 flatten / batch 兩個旗標）：
###*         flatten=True  → 每筆攤平成 (N,)，整體 (B, N) 或去批次後 (N,)
###*         flatten=False → 每筆還原成 (H, W)，整體 (B, H, W) 或 (H, W)
###*         batch=True    → 強制保留批次維 (B, ...)，即使 B=1
###*         batch=False   → B=1 時擠掉批次維；B>1 卻要非批次輸出則直接報錯(無法壓非單例維)。
###*     ◆ 模式 2（output_shape 是字串，如 "B,1,H,W" 或 "B,N,1"）：
###*         忽略 flatten/batch，逐段把 'B'/'N'/'H'/'W'/數字 翻成實際維度，精準塑形，
###*         並驗證總元素量一致、未含 'B' 時不可有 B>1。常用於要插入 channel 維度的情境(如 1)。
###*   #! 共同前提：input 的總元素數必須是「單筆 N」的整數倍，否則無法推出乾淨的 B。
###* ============================================================================
@overload
def size_converter(
    sizer: Sizable,tensor: torch.Tensor, flatten:Literal[True], batch: Literal[True]
) -> Tensor_B_N: ...
@overload
def size_converter(
    sizer: Sizable,tensor: torch.Tensor, flatten:Literal[True], batch: Literal[False]
) -> Tensor_N: ...
@overload
def size_converter(
    sizer: Sizable,tensor: torch.Tensor, flatten:Literal[False], batch: Literal[True]
) -> Tensor_B_W_H: ...
@overload
def size_converter(
    sizer: Sizable,tensor: torch.Tensor, flatten:Literal[False], batch: Literal[False]
) -> Tensor_W_H: ...
@overload
def size_converter(
    sizer: Sizable,tensor: torch.Tensor, output_shape: str
) -> torch.Tensor: ...

def size_converter(
    sizer: Sizable,
    tensor: torch.Tensor, 
    flatten: bool = False, 
    batch: bool = False,
    output_shape: Optional[str] = None
) -> torch.Tensor:
    """
    General-purpose tensor size converter.

    Mode 1 (output_shape is None):
        Transformation using the `flatten` and `batch` parameters.
        
    Mode 2 (output_shape is a string):
        Ignore the `flatten` and `batch` parameters,
        and perform the transformation exactly as indicated by the `output_shape` string.

    Args:
        tensor (torch.Tensor): The input tensor to be transformed.
        sizer (Sizable): An object or class that has a `.size(flatten: bool)` method

        flatten (bool): Only output_shape is None.
            True - 輸出形狀為 (B, N) 或 (N,)。
            False - 輸出形狀為 (B, H, W) 或 (H, W)。
        batch (bool): Only output_shape is None.
            True - 強制輸出為批次形式 (B, ...)，即使 B=1。
            False - 如果計算出的 B=1，則移除批次維度 (...,)。
        output_shape (Optional[str]): [B, H, W, N] Priority use. EX: "B, 1, H, W" or "B, N, 1"

    Returns:
        torch.Tensor: The reshaped tensor.
    """
    #* 先向 sizer 問清「單筆樣本」的尺寸：
    #*   N_per_sample = 攤平後元素數(N)；components = (H, W) 二維形狀。
    #!   sizer 介面是這個轉換器的關鍵抽象 —— 只要物件提供 .size(flatten)，就能被它整形，
    #!   因此 GEN/SM 的 pattern 與 response 都能共用同一支函式而不必各寫 reshape。
    try:
        N_per_sample = sizer.size(flatten=True)
        components = sizer.size(flatten=False)
        H_comp = components[0]
        W_comp = components[1]
    except Exception as e:
        raise ValueError(f"Unable to obtain size information from sizer({sizer})\n{e}")

    #* 用「總元素數 ÷ 單筆 N」反推批次大小；不能整除代表輸入根本不是這種樣本的整數倍 → 報錯。
    total_input_numel = tensor.numel()
    if total_input_numel % N_per_sample != 0:
        raise ValueError(
            f"The total number of elements in the input tensor ({total_input_numel}) "
            f"must be an integer multiple of {N_per_sample}."
        )

    #* Batch size
    B_calc = total_input_numel // N_per_sample    #* 推算出的批次大小，後續所有塑形都以它為準。

    #* Use the string output_shape
    #* ===== 模式 2：字串塑形 =====
    #*   逐段解析 output_shape，把符號翻成實際維度數值，組成 final_shape_list。
    if output_shape is not None:
        try:
            shape_parts = [part.strip() for part in output_shape.split(',')]
            final_shape_list = []
            has_batch_dim = False              #* 記錄字串裡是否出現 'B'，供後續批次合法性檢查。

            for part in shape_parts:
                if part == 'B':                #* 'B' → 推算出的批次大小
                    final_shape_list.append(B_calc)
                    has_batch_dim = True
                elif part == 'N':              #* 'N' → 單筆攤平長度
                    final_shape_list.append(N_per_sample)
                elif part == 'H':              #* 'H' → 單筆高度
                    final_shape_list.append(H_comp)
                elif part == 'W':              #* 'W' → 單筆寬度
                    final_shape_list.append(W_comp)
                elif part.isdigit():           #* 純數字 → 固定維度(常見如插 channel 維 "1")
                    final_shape_list.append(int(part))
                else:
                    raise ValueError(f"'{part}'")    #* 不認得的符號 → 觸發下方統一錯誤訊息。

        except ValueError as e:
            raise ValueError(
                f"The string `output_shape` contains an unrecognized component: {e}."
                "Please only use 'B', 'N', 'H', 'W', or numbers."
            )
        
        #* Validate: batch dimension
        #!   護欄一：B>1 卻沒在字串裡放 'B'，等於要把多個樣本硬塞進不含批次維的形狀 →
        #!           會默默把 batch 揉進其他維度造成資料錯亂，故直接擋下。
        if not has_batch_dim and B_calc > 1:
            raise ValueError(
                f"輸入的計算批次大小為 {B_calc}, 但 output_shape "
                f"'{output_shape}' 中未包含 'B'。 "
                "無法壓縮非單例的批次維度。"
            )

        #* Validate: the final total number of elements
        #!   護欄二：reshape 前先確認目標形狀的總元素量與輸入完全一致，
        #!           提早給出清楚錯誤，而不是讓 torch.reshape 丟出較難讀的訊息。
        target_numel = 1
        for dim in final_shape_list:
            target_numel *= dim

        if target_numel != total_input_numel:
            raise ValueError(
                f"Output shape '{output_shape}' (解析為 {final_shape_list}) "
                f"的總元素量 ({target_numel}) 與 "
                f"輸入張量的總元素量 ({total_input_numel}) 不匹配。"
            )

        return tensor.reshape(final_shape_list)    #* 通過兩道護欄後才真正塑形。

    else: #* Use the flatten and batch parameters
        #* ===== 模式 1：flatten / batch 旗標塑形 =====
        #*   先決定「每筆樣本」的目標形狀，再前綴 B 組成完整形狀。
        if flatten:
            target_shape_per_sample = (N_per_sample,)      #* 攤平 → (N,)
        else:
            target_shape_per_sample = components # (H_comp, W_comp)    #* 還原二維 → (H, W)

        final_shape = (B_calc, *target_shape_per_sample)   #* 一律先 reshape 成含批次維的形狀。
        output_tensor = tensor.reshape(final_shape)

        if not batch:
            #* 要求非批次輸出：唯有 B=1 才能安全擠掉批次維。
            if B_calc == 1:
                #? (1, H, W) -> (H, W) or (1, N) -> (N,)
                return output_tensor.squeeze(dim=0)
            else:
                # B > 1, 但要求 non-batch output
                #!   B>1 又要 batch=False 是矛盾請求(會遺失樣本維度) → 報錯而非靜默壓縮。
                raise ValueError(
                    f"輸入的計算批次大小為 {B_calc}, 但請求了 'batch=False' "
                    "(非批次輸出)。無法壓縮非單例的批次維度。"
                )
        else:
            #? (B, H, W) -> (B, H, W)
            #* batch=True：保留批次維直接回傳(即使 B=1 也維持 (1, ...) 形狀)。
            return output_tensor
