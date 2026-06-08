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
from typing import Any


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
