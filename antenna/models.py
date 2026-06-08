###* 模型家族 (Model Family) ###
#? 本檔集中定義整個反向設計閉迴路會用到的「模型」抽象：
#?   1. Models      — 模型管理外殼 (把 model/optimizer/scheduler/criterion/Record 綁成一包，統一存讀/換檔/凍結)。
#?   2. GEN 生成器  — SigmoidGEN / OldGEN / GumbelSigmoidGEN / SPGEN / CVAE / MirrorCVAE：
#?                    目標響應 (spec) → 25x25 二元 pattern。生成器需要「可微分的二值化」才能讓
#?                    SM 反傳回來的梯度更新 MLP，故大量使用 STE 技巧 (見下方各 autograd Function)。
#?   3. 工具元件    — BiScaleNorm / GradientEstimator / 各種 STE Function。
#?
#? 與 SM / SIM 的關係：GEN 不直接接觸不可微分的 HFSS(SIM)，而是透過可微分的代理模型 SM
#? 取得梯度；GEN 生 pattern → SM 預測響應 → loss 經 SM 反傳到 GEN。STE 的角色就是讓
#? 「forward 是離散硬值 (0/1)、backward 卻有可用梯度」，繞過二值化本身不可導的斷點。

from antenna.utils import *
from antenna.types import *
from antenna import *

import torch
from torch.autograd.function import (
    Function,          #? 自訂 autograd 運算的基底：可分別定義 forward(離散) 與 backward(給梯度)
    FunctionCtx ,      #? forward 中的 context，save_for_backward 暫存張量供 backward 使用
    BackwardCFunction,
)
from torch.autograd import Variable


import numpy as np
from math import sqrt


from functools import partial  #? 用來把 gumbel 函式的固定 kwargs 預先綁定 (見 SPGEN)


###* Models — 模型管理外殼 (model/optimizer/scheduler/criterion + Record 打包) ###
#? 泛型參數讓型別檢查能追蹤被包住的 module/optimizer 等具體型別。
#? 設計動機：訓練腳本 (train_single.py / train_dual.py) 想用單一物件就完成
#? 「呼叫模型、存讀 checkpoint、依 label 換存檔、凍結梯度、early-stop rollback」等操作，
#? 不必各自手動管理 state_dict。GEN 與 SM 都會被包進各自的 Models 實例。
class Models(Generic[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams]):
    def __init__(
            self, 
            name:str = "models_{label}", 
            rootdir:Optional[Union[str, Path]] = None, 
            model:Optional[ CustomModule | CallableModule[ModelParams, ReturnType]] = None, 
            optimizer:Optional[CustomOptimizer] = None, 
            scheduler:Optional[CustomScheduler] = None, 
            criterion:Optional[Callable[LossParams, Tensor]] = None,
            *,
            load:bool = False,
            device = config.device
        ):
        #? name 內若含 "{label}" 佔位符 → 走「可換存檔」模式：先把樣板存進 self._name，
        #  真正的檔名 self.name 留空 (None)，等 change(label) 帶入 label 後才確定。
        #  這讓同一個 Models 物件能依任務/批次切換不同 .pth 檔 (例如不同目標頻段)。
        if "{label}" in name:
            self._name = name
            self.name = None
        else:
            #? 沒有佔位符 → 檔名固定，self._name 設 None 代表「不可換檔」。
            self._name = None
            self.name = name

        self._rootdir = rootdir or config.checkpoint_save_path  #? 預設落在全域 config 的 checkpoint 目錄

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        #? Record：訓練紀錄器，與模型一起存進同一個 checkpoint。
        #! 只有當 load=True 且檔案已存在時才從檔案載入紀錄，避免首次訓練去讀不存在的檔。
        self.record = Record(self.__class__.__name__, rootdir=self._rootdir, load=load and self.model_file.exists())

        self.device = device  #? 透過 device.setter 直接把 model.to(device)
        if load: self.load()
    
    #? 讓 Models 實例可直接像函式般呼叫，等同呼叫底層 model 的 forward。
    #  在 pipeline 中 gen(target) / sm(pattern) 都是走這條。
    def __call__(self, *args:ModelParams.args, **kwargs:ModelParams.kwargs) -> ReturnType:
        return  self.model(*args, **kwargs)

    #? __str__ 會被當成 checkpoint 的 "title"。load() 時用它比對檔案是否與當前
    #  (模型+優化器+排程器+損失) 組合一致，避免把錯誤的權重載進不相容的架構。
    def __str__(self):
        _str = "{class_name}(Model={model}, Optimizer={optimizer}, Scheduler={scheduler}, Criterion={criterion})"
        return _str.format(
            class_name = self.__class__.__name__,
            model = self.model.__class__.__name__,
            optimizer = self.optimizer.__class__.__name__,
            scheduler = self.scheduler.__class__.__name__,
            criterion = self.criterion.__name__ if isinstance(self.criterion, FunctionType) else self.criterion.__class__.__name__
        )
    
    @property
    def model_file(self) -> Path:
        """The full path to the model archive."""
        #! 若處於 "{label}" 換檔模式但尚未 change()，self.name 仍是 None → 直接擋下，
        #  提醒先呼叫 change() 決定 label，否則組不出檔名。
        assert self.name, "Please use `Models.change()` first."
        return Path(self._rootdir).joinpath(f"{self.name}.pth")

    @property
    def device(self):
        """Model parameters of the device."""
        #? 以「模型第一個參數所在的裝置」作為當前裝置的真實來源 (single source of truth)。
        return next(self.model.parameters()).device

    @device.setter
    def device(self, device):
        #? 設定裝置等同把整個 model 搬過去；故 self.device = x 是合法且有副作用的寫法。
        self.model.to(device = device)

    @property
    def FloatTensor(self):
        #? 依當前裝置回傳對應的 FloatTensor 類別 (CPU/CUDA)，方便建立與模型同裝置的張量。
        return torch.FloatTensor if str(self.device) == 'cpu' else torch.cuda.FloatTensor # type: ignore
    
    def change(self, label:str, *, load:bool=False, save:bool=False):
        """
        Change model label.

        :param label: models label. You can enter `{label}` in name.
        :param load: Load the changed models.
        :param save: Save the models before the change.
        """
        #? 換 label 流程：可選擇先存舊檔 → 把 label 套進樣板算出新檔名 → 可選擇載入新檔。
        if save:
            self.save()                              #! 先存「舊」label 的檔，避免換名後丟失進度
        if self._name:
            self.name = self._name.format(label=label)  #? 只有換檔模式 (self._name 非 None) 才更新檔名
        if load:
            self.load()                              #? 載入「新」label 對應的檔
        return self.name

    #? 從 model_file 還原整包狀態 (模型/優化器/排程器/Record)。
    def load(self, force:bool = False):
        checkpoint_loaded = self.checkpoint(load=True)
        #! 預設只允許載入 title 完全相符的檔，確保架構一致；force=True 才略過此檢查 (debug/遷移用)。
        if checkpoint_loaded['title'] == self.__str__() or force:
            self.device = checkpoint_loaded['device']

            self.model.load_state_dict(checkpoint_loaded['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint_loaded['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint_loaded['scheduler_state_dict'])
            self.record.load_state_dict(checkpoint_loaded['record_state_dict'])

        else:
            raise RuntimeError(f"Please use the correct model file.\nFile: {checkpoint_loaded['title']}\nCurrent: {self.__str__()}")

    def save(self) -> Path:
        return self.save_as(self.model_file)

    def save_as(self, filename:Union[str, Path]) -> Path:
        """
        :param filename: 檔案完整路徑，含副檔名(suffix)
        """

        #? 原子寫入 (atomic save)：先寫 .tmp，成功後再 replace 成正式檔，
        #  避免存檔過程中斷 (例如訓練被 Ctrl-C) 導致 .pth 半寫壞、之後讀檔崩潰。
        filename = Path(filename)
        temp_file = filename.with_suffix(filename.suffix + '.tmp')
        try:
            torch.save(self.checkpoint(load=False), temp_file)
            temp_file.replace(filename)              #? replace 在多數平台是原子操作
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()                   #! 失敗時清掉殘留的 .tmp，保持目錄乾淨
            raise
        return filename

    #? pre_load_model：只載「權重 + 優化器」而不比對 title，用於把預訓練 (pretrain) 的
    #  SM/GEN 權重灌進當前模型作為閉迴路的起點，跳過 load() 的嚴格架構檢查。
    def pre_load_model(self, path:Union[str, Path]):
        path = Path(path)
        checkpoint_loaded:Checkpoint = path.load_torch()
        self.model.load_state_dict(checkpoint_loaded['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_loaded['optimizer_state_dict'])
        #! 載入後立即檢查所有參數有限性：預訓練檔若含 NaN/inf 會在閉迴路一開始就汙染梯度，
        #  在此早期擋下比讓訓練跑到一半才爆掉更易除錯。
        for name, param in self.model.state_dict().items():
            if not torch.all(torch.isfinite(param)):
                raise RuntimeError(f"!!! 在參數 '{name}' 中發現無效值 (NaN 或 inf) !!!")
        logger.success(f'Successfully loaded the pre-trained model. ({path})')

    #? 一步推進：先走優化器再走排程器 (若有)。把兩者包成一個呼叫方便訓練腳本使用。
    def step(self, optimizer_param=None, scheduler_param=None):
        self.optimizer.step(optimizer_param)
        if self.scheduler: self.scheduler.step(scheduler_param)

    #? 組裝/載入 checkpoint 字典。load=True 直接從檔讀；load=False 則即時蒐集當前狀態用於存檔。
    def checkpoint(self, load:bool = False) -> Checkpoint:
        if load:
            checkpoint:dict = self.model_file.load_torch()
        else:
            #? title 用 __str__ 字串標記架構身分；各 state_dict 在對應元件存在時才取，
            #  不存在則填 None (允許只有 model 而無 scheduler 等的精簡配置)。
            checkpoint = {
                "title": self.__str__(),
                'model_state_dict': None if not self.model else self.model.state_dict(),
                'optimizer_state_dict': None if not self.optimizer else self.optimizer.state_dict(),
                'scheduler_state_dict': None if not self.scheduler else self.scheduler.state_dict(),
                'device': self.device,
                'record_state_dict': self.record.state_dict()
            }
        return checkpoint

    #? requires_grad：一鍵切換整個模型的「可訓練 / 凍結」狀態。
    #  關鍵用途：閉迴路中常需凍結 SM 只更新 GEN (或反之)，以及讓 MirrorCVAE 把 SM 當固定評估器。
    def requires_grad(self, mode:bool = True, train:Optional[bool] = None):
        for param in self.model.parameters():
            param.requires_grad = mode

        #? train 另外控制 train()/eval() 模式 (影響 Dropout/BatchNorm 行為)，與梯度開關正交：
        #  可以「凍結權重但仍處 eval」或「開梯度且 train」自由組合；None 則不動模式。
        match train:
            case True:
                self.model.train()
            case False:
                self.model.eval()
            case _:
                pass

        return next(self.model.parameters()).requires_grad

###* BiScaleNorm — 正負分開的雙尺度正規化 ###
#? GEN 的 MLP 最後一層常接這個，把輸出壓到 [-1, 1] 區間但「保留 0 這個中性點」。
#? 為什麼不用一般 LayerNorm/標準化：二值化以 0 (或平均值) 為分界，因此希望正半邊與負半邊
#? 各自獨立縮放——正值除以全域最大值、負值除以全域最小值的絕對值，使最大正值映到 +1、
#? 最絕對負值映到 -1，而 0 仍保持 0。這讓後續 sigmoid/STE 的分界更穩定、梯度尺度更一致。
class BiScaleNorm(nn.Module):
    def __init__(self):
        super(BiScaleNorm, self).__init__()

    def forward(self, input_vector):
        # 大於 0 的值的正規化
        #? 正半邊：每個正值除以全域最大值 → 落在 (0, 1]；非正值位置填 0。
        max_val = torch.max(input_vector)
        positive_normalized = torch.where(input_vector > 0, input_vector / max_val, torch.tensor(0.0, device=input_vector.device))

        # 小於 0 的值的正規化
        #? 負半邊：每個負值除以最小值的絕對值 → 落在 [-1, 0)；非負值位置填 0。
        min_val = torch.min(input_vector)
        negative_normalized = torch.where(input_vector < 0, input_vector / torch.abs(min_val), torch.tensor(0.0, device=input_vector.device))

        # 合併正規化結果
        #? 兩半邊在彼此互斥的位置上各填 0，相加即還原成完整向量 (值域 [-1, 1])。
        normalized_vector = positive_normalized + negative_normalized
        return normalized_vector

###* sign_f — 符號函數的 STE 版本 (hard-tanh 直通) ###
#? forward 輸出硬性的 ±1 (不可導的階梯)；backward 則用 hard-tanh 的梯度近似：
#? 只在 |input| <= 1 的區間讓梯度通過，超出此區間 (已飽和) 的梯度歸零。
#? OldGEN 用它做二值化：sign(x)/2 + 0.5 把 ±1 映到 {0, 1}，同時保有可訓練梯度。
class sign_f(Function):
    """
    sign function
    """
    @staticmethod
    def forward(ctx:BackwardCFunction, inputs:Tensor):
        #? forward：硬性符號化。>=0 給 +1、<0 給 -1，產生離散輸出 (不可導)。
        output = inputs.new(inputs.size())
        output[inputs >= 0.] = 1
        output[inputs < 0.] = -1
        ctx.save_for_backward(inputs)  #? 暫存原始 inputs，backward 要靠它判斷飽和區
        return output

    @staticmethod
    def backward(ctx:BackwardCFunction, grad_output:Tensor):
        #? backward：STE 的核心。把 sign 視為近似恆等函數，直接讓上游梯度通過，
        #  但對已飽和 (|input|>1) 的位置截斷梯度 (hard-tanh 行為)，避免在平坦區白費更新。
        input_, = ctx.saved_tensors
        grad_output[input_>1.] = 0
        grad_output[input_<-1.] = 0
        return grad_output

###* _GumbelSigmoid — Gumbel-Sigmoid 採樣 (自訂 autograd，含 tau 梯度) ###
#? 二值 logits 的「可微分隨機採樣」：forward 加 Gumbel 雜訊後過 sigmoid 得到接近 0/1 的軟值，
#? backward 自行推導 logits 與溫度 tau 的解析梯度，使 tau 也能被學習 (退火由模型自動調)。
#! 注意：此為 GumbelSigmoid 的早期版本，目前主路徑用的是下方的 GumbelSigmoid (帶 scale 縮放)。
class _GumbelSigmoid(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, logits, tau_tensor, eps=1e-10):
        """
        Gumbel-Sigmoid采樣方法
        logits: 輸入的logits（可以是實數）
        tau: 溫度，控制離散度
        eps: 防止除以0的小常數
        """
        #? Gumbel 雜訊：對 U~Uniform 取雙重 -log 得到 Gumbel(0,1)；加進 logits 後再過 sigmoid，
        #  等價於對二元類別做 Gumbel-Softmax 的二類特例，提供採樣隨機性又保留可微分性。
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
        y = torch.sigmoid((logits + gumbel_noise) / tau_tensor)  #? tau 越小越接近硬 0/1

        # 保存為 backward 方法提供所需的變數
        ctx.save_for_backward(logits, y, gumbel_noise)
        ctx.tau = tau_tensor  # 保存 tau 以便在 backward 中使用
        return y

    @staticmethod
    def backward(ctx:BackwardCFunction, grad_output):
        # 讀取 forward 傳遞的變數
        logits, y, gumbel_noise = ctx.saved_tensors
        tau = ctx.tau  # 從 ctx 中讀取 tau

        # 計算 gradient
        sigmoid_grad = y * (1 - y)  # Sigmoid 梯度
        grad_input = grad_output * sigmoid_grad / tau  # 給 logits 的梯度

        # 計算 tau 的梯度
        #? 對 (logits+noise)/tau 中的 tau 微分得 -(.)/tau^2；下式以加總成標量回傳給 tau 參數。
        # grad_tau = (grad_output * sigmoid_grad * (logits - y)).sum() / tau**2  # 給 tau 的梯度
        grad_tau = (grad_output * sigmoid_grad * (logits + gumbel_noise)).sum() / tau**2  # 給 tau 的梯度

        return grad_input, grad_tau, None  #? 第三個 None 對應 forward 的 eps (常數，不需梯度)

###* GumbelSigmoid — Gumbel-Sigmoid 採樣 (主用版本，雜訊縮小 scale) ###
#? 與 _GumbelSigmoid 同理，但把 Gumbel 雜訊乘上 scale=0.1 以「降低隨機擾動強度」，
#? 讓 GumbelSigmoidGEN 在訓練早期不至於因雜訊過大而難以收斂。tau 同樣可學習。
class GumbelSigmoid(Function):
    @staticmethod
    def forward(ctx:Function, logits, tau, eps=1e-20):
        # tau = max(0.1, ctx.tau - 0.001 * ctx.tau) if hasattr(ctx, 'tau') else tau
        U = torch.rand_like(logits)
        scale = 0.1  # 降低到 0.1
        #? 縮小後的 Gumbel 雜訊：scale 把擾動幅度壓低，避免採樣過度隨機。
        gumbel_noise = -torch.log(-torch.log(U + eps) + eps) * scale
        y = torch.sigmoid((logits + gumbel_noise) / tau)

        ctx.save_for_backward(logits, y, gumbel_noise, tau)  #? 此版把 tau 也存進 saved_tensors

        return y

    @staticmethod
    def backward(ctx:Function, grad_output):
        logits, y, gumbel_noise, tau = ctx.saved_tensors

        ###* Sigmoid 函數的梯度 ###
        sigmoid_grad = y * (1 - y)

        ###* logits 的梯度 ###
        grad_input = grad_output * sigmoid_grad / tau

        ###* tau 的梯度 ###
        #! 符號與 _GumbelSigmoid 版相反 (此處帶負號)，反映對 1/tau 縮放因子的方向；
        #  最終 sum 成標量回傳給可學習的 tau 參數。
        grad_tau = -grad_output * sigmoid_grad * (logits + gumbel_noise) / (tau ** 2)
        grad_tau = grad_tau.sum()  # 總和作為標量梯度
        return grad_input, grad_tau, None  #? None 對應 eps (常數)

###* BinarizeSTE — 0.5 閾值二值化的 STE ###
#? forward 以 0.5 為界硬性二值化 (得 0/1 遮罩)；backward 用一個簡化的 STE：
#? 只讓「被選中 (mask=1) 的位置」回傳梯度，未選中位置梯度歸零。
#! 這是有偏 (biased) 的近似——它不等同 sign_f 的對稱直通，而是只更新「目前為 1」的像素，
#  在本檔中目前被各 GEN 註解掉、未實際啟用，保留作為替代二值化策略。
class BinarizeSTE(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, input:Tensor):
        mask = (input >= 0.5).float()
        ctx.save_for_backward(mask)
        return mask

    @staticmethod
    def backward(ctx:BackwardCFunction, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask  # 只保留 mask 區域的梯度

# %%
from .functions import gumbel_sinkhorn_rectangular  #? Gumbel-Sinkhorn：把連續 logits 變成近似排列/指派矩陣

###* SPGEN — Small-Pattern GEN (Gumbel-Sinkhorn 小圖樣拼接) ###
#? 另一種生成器思路：不直接逐像素生成，而是維護一張「小圖樣表 (pattern_table)」，
#? 用可學習 logits 經 Gumbel-Sinkhorn 在每個網格位置「軟性指派」一個小圖樣，再拼成大 pattern。
#? 優點：把搜尋空間限制在合法的小圖樣組合上 (結構先驗)，比逐像素更易得到可製造的規則圖樣。
#? hard=False (訓練) 給軟指派以保梯度；hard=True (推論) 給離散指派。
class SPGEN(nn.Module, Generic[CallableParam]):
    def __init__(self ,pattern_table:Tuple, size=40, gumbel_fn:Callable[CallableParam, Tensor]=gumbel_sinkhorn_rectangular, **gumbel_fn_kwargs):
        super(SPGEN,self).__init__()

        self.pattern_table = pattern_table              #? 候選小圖樣的集合 (tuple)
        self.pattern_table_tensor = self._to_tensor()   #? 攤平成 [Channels, small_h*small_w] 供矩陣乘法
        self.num_patterns = len(pattern_table) # [Channels] How many small patterns.
        self.grid_size = size // self.patern_size # [big_h, big_w] Composition of small patterns (|===|---|---|---|)
        #? 唯一可學習參數：每個網格位置對各小圖樣的偏好分數 (logits)。整個生成器的「設計變數」就是它。
        self.logits = nn.Parameter(
            #? [batch, big_h, big_w, Channels]
            torch.randn(1, self.grid_size, self.grid_size, self.num_patterns),
            requires_grad=True
        )
        #? 用 partial 預先綁定 gumbel 函式的固定 kwargs，之後 forward 只需傳 tau/n_iters/hard。
        self.gumbel_fn:Callable[CallableParam, Tensor] =  partial(gumbel_fn,  **gumbel_fn_kwargs)

    def __str__(self):
        return f"SPGEN(total={self.patern_size*self.grid_size}(small[{self.patern_size}]xbig[{self.grid_size}))"
    
    def _to_tensor(self):
        """
        >>> torch.Size([Channels, small_h * small_w])
        """
        #? 把每個小圖樣攤平成一列、堆成 [num_patterns, small_h*small_w] 的查表矩陣，
        #  forward 才能用一次矩陣乘法 (assignment @ table) 完成「指派→取出圖樣」。
        #! 副作用：迴圈中順手把 self.patern_size 設為小圖樣邊長 (假設所有小圖樣同尺寸)。
        _reselt = []
        for pattern in self.pattern_table:
            self.patern_size: int = len(pattern)
            _reselt.append(np.array(pattern, dtype=np.int16).reshape(-1))

        return torch.tensor(
            np.stack(_reselt), 
            dtype = torch.float32
        )

    #? forward 不吃外部輸入 (target)，而是直接把內部 logits 解碼成 pattern——
    #  SPGEN 是「單一任務最佳化」式生成器：logits 本身就是被梯度更新的解。
    def forward(self, tau: float = 1.0, n_iters: int = 20, hard: bool = True):

        # 原始 logits 形狀: [1, grid_h, grid_w, num_patterns]
        batch_size, grid_h, grid_w, num_patterns = self.logits.shape
        
        # 1. 重塑 logits 以符合 gumbel_sinkhorn_rectangular 的輸入
        # 將 grid_h 和 grid_w 維度合併為一個「位置」維度
        # 新形狀: [1, grid_h * grid_w, num_patterns]
        num_positions = grid_h * grid_w
        reshaped_logits = self.logits.view(batch_size, num_positions, num_patterns)

        # 2. 使用新的 Gumbel-Sinkhorn 函式
        # 輸出 assignment_matrix 形狀: [1, grid_h * grid_w, num_patterns]
        # 注意：訓練時 hard 應為 False，推斷時可設為 True
        assignment_matrix = self.gumbel_fn(reshaped_logits, tau=tau, hard=hard)

        # pattern_table_tensor: [num_patterns, small_h * small_w]
        # 3. 執行矩陣乘法來選擇圖案
        # torch.matmul: [1, K, M] @ [M, S*S] -> [1, K, S*S]
        # K = num_positions, M = num_patterns, S = patern_size
        selected_patterns = torch.matmul(assignment_matrix, self.pattern_table_tensor) # [1, H*W, small_h*small_w]
        
        # 4. 將結果重塑回最終的圖像形狀
        # selected_patterns 現在的空間維度是攤平的，需要重新構建
        soft_output = selected_patterns.view(
            batch_size, self.grid_size, self.grid_size,  # batch, grid_h, grid_w
            self.patern_size, self.patern_size           # 小圖案大小 (small_h, small_w)
        ).permute(
            0, 1, 3, 2, 4  # (batch, grid_h, small_h, grid_w, small_w)
        ).reshape(
            batch_size,
            self.grid_size * self.patern_size,
            self.grid_size * self.patern_size
        )

        self.output_image = soft_output
        return self.output_image

    def save(self, nrowcol:tuple, result_path, pattern_dict:dict=None):
        with Figure("SPGEN Small Pattern", nrowcol, save=True, rootdir=result_path, size=(18, 12), default_axes_title_size=10, default_tick_size=6) as fig:
            if pattern_dict:
                for name, pattern in pattern_dict.items():
                    ax:Axes = fig.index(-1)
                    ax.axis('off') 
                    ax.set_title(f"{name}")
                    ax.imshow(pattern, cmap='viridis')
            else:
                for n in range(len(self)):
                    ax:Axes = fig.index(-1)
                    ax.axis('off') 
                    ax.set_title(f"Small Pattern {n+1}")
                    ax.imshow(self[n], cmap='viridis')

    #? 索引/長度代理到 pattern_table，方便 save() 視覺化每個小圖樣。
    def __getitem__(self, idx):
        return self.pattern_table[idx]

    def __len__(self):
        return len(self.pattern_table)


###* GumbelSigmoidGEN — MLP + Gumbel-Sigmoid 生成器 ###
#? 結構：response → MLP(逐層放大再縮回) → BiScaleNorm → 得 logits → GumbelSigmoid 採樣成軟二值。
#? 與 SM 的關係：輸出的軟 pattern 餵給 SM 取得預測響應與 loss，梯度經 GumbelSigmoid 的解析
#? backward 回流，同時更新 MLP 權重與可學習溫度 tau。
#! 屬實驗性生成器；主用的是 SigmoidGEN (見下)。此處 GumbelSigmoid 路徑帶來採樣隨機性。
class GumbelSigmoidGEN(nn.Module):
    """
    Generator Model
    """
    def __init__(self):
        super(GumbelSigmoidGEN,self).__init__()
        pattern_size = AntennaPattern.size(flatten=True)  #? 輸出維度 = 攤平後的 pattern 像素數 (25*25)
        #? MLP：response_size → pattern → pattern*2 → pattern → pattern，最後接 BiScaleNorm。
        #  中間放大到 2 倍再縮回，給網路較大的表示容量；末層用 BiScaleNorm 把 logits 壓到 [-1,1]。
        self.fc_patch = nn.Sequential(
            nn.Linear(AntennaResponse.size(flatten=True), pattern_size),
            nn.PReLU(),
            nn.Linear(pattern_size, pattern_size*2),
            nn.PReLU(),
            nn.Linear(pattern_size*2, pattern_size),
            nn.PReLU(),
            nn.Linear(pattern_size, pattern_size),
            BiScaleNorm(),
        )
        self.tau = nn.Parameter(torch.tensor(5.0, requires_grad=True))  #? 溫度設為可學習參數，初值偏大 (5.0) → 早期較平滑
        self.tau_history = []   #? 記錄 tau 隨訓練變化，便於觀察退火趨勢

        #? 權重初始化：Linear 用 Kaiming (配 ReLU 系列激活)；bias 給 1.0 讓初期輸出略偏正；
        #  PReLU 的負斜率初始化為 0.25 (PyTorch 預設)。良好初始化可避免 STE 一開始就飽和。
        for m in self.fc_patch:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1.0)
            if isinstance(m, nn.PReLU):
                m.weight.data.fill_(0.25)

        self.to(config.device)

    def forward(self, input, *, is_trainig:bool = True):
        """
        輸出 Gumbel-Sigmoid 處理過的結果
        """
        #! clamp 到 [-5,5]：防止 logits 過大導致 sigmoid 飽和、梯度爆炸或消失。
        self.logits  = torch.clamp( # 防止梯度爆炸
            self.fc_patch(input), min=-5.0, max=5.0
        )
        # # 在訓練階段使用 Gumbel-Sigmoid 來保持梯度
        # if is_trainig:
        #     x = GumbelSigmoid.apply(x, tau)  # 訓練階段使用 Gumbel-Sigmoid 進行輸出
        # else:
        #     x = (x >= 0.5).float()  # 推論階段，硬性 binarize

        #? 用自訂 GumbelSigmoid.apply 而非直接函式呼叫，才能掛上前面定義的解析 backward
        #  (同時對 logits 與 tau 給梯度)。
        x = GumbelSigmoid.apply(self.logits, self.tau)  # 訓練階段使用 Gumbel-Sigmoid 進行輸出
        # self.anneal_tau()
        self.tau_history.append(self.tau.detach().cpu().item())  #? detach 後存純量，避免把計算圖留在歷史清單


        # x = BinarizeSTE.apply(x)

        return x

    #? anneal_tau：溫度退火。早期大 tau (輸出軟、利於探索)，後期小 tau (輸出趨近硬 0/1、利於收斂)。
    #! 目前 forward 中已被註解停用，tau 改為純靠梯度自學；此方法保留作手動退火的備案。
    def anneal_tau(self, rate=0.995, min_tau=0.1):
        """
        Annealing (退火)
        
        在訓練初期，較大的 tau 值會使得輸出更為平滑，有利於模型探索不同的解空間。

        在訓練後期，較小的 tau 值會使輸出更接近離散的 0 和 1，從而幫助模型收斂到一個確定的離散解。
        """
        # self.tau = max(min_tau, self.tau * rate)
        self.tau = torch.clamp(self.tau, min=0.1)
        self.tau_recoed.append(self.tau.detach().cpu())

    #? 推論用硬二值化：對 logits 過 sigmoid 再以 threshold 切出 0/1，包成 AntennaPattern。
    #! 第二個 return 永遠不會執行 (前一行已 return)，是被保留的舊寫法殘留。
    def binarize(self, threshold = 0.5):
        binarized_output = (torch.sigmoid(self.logits) > threshold).float()
        return  AntennaPattern(binarized_output)
        return  AntennaPattern((self.x >= threshold).float())

###* OldGEN — 早期生成器 (MLP + BiScaleNorm + sign_f STE) ###
#? 結構：response → MLP(1024-1024) → BiScaleNorm → sign_f → 線性映回 {0,1}。
#? 二值化靠 sign_f：forward 出 ±1，再 /2 + 0.5 變 {0,1}；backward 透過 sign_f 的 hard-tanh STE 給梯度。
#! 「Old」代表已被 SigmoidGEN 取代；保留以對照不同二值化策略 (sign_f vs binarization)。
class OldGEN(nn.Module):
    """
    Generator Model
    """
    def __init__(self):
        super(OldGEN,self).__init__()
        self.fc_patch = nn.Sequential( # Can use BiScaleNorm or nn.PReLU, except the last layer.
            nn.Linear(AntennaResponse.size(flatten=True), 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, AntennaPattern.size(flatten=True)),
            BiScaleNorm(),
        )

        self.r = sign_f.apply  #? 綁定 sign_f 的 STE 二值化算子
        self.to(config.device)

    def forward(self, input) -> Tensor:
        x = self.fc_patch(input)
        x = self.r(x) / 2 + 0.5 # type: ignore  #? sign(±1) → /2+0.5 → {0,1}，且梯度經 STE 直通
        return x

###* SigmoidGEN — 主用生成器 (MLP + BiScaleNorm + AntennaPattern.binarization) ###
#? 本專案實際主用的生成器，入口 train_single.py / train_dual.py 即用它。
#? 結構與 OldGEN 幾乎相同 (response → MLP 1024-1024 → BiScaleNorm)，差別在二值化改用
#? AntennaPattern.binarization：它內部用「sigmoid 軟近似 + STE」做可微分二值化——
#? forward 取硬 0/1、backward 用陡峭 sigmoid (steepness=1/tau) 的平滑梯度繞過不可導斷點，
#? 閾值預設取輸入平均值。tau 由外部 (排程) 控制退火，故 forward 多收一個 tau 參數。
class SigmoidGEN(nn.Module):
    """
    Generator Model
    """
    def __init__(self):
        super(SigmoidGEN,self).__init__()
        self.fc_patch = nn.Sequential( # Can use BiScaleNorm or nn.PReLU, except the last layer.
            nn.Linear(AntennaResponse.size(flatten=True), 1024),
            nn.PReLU(),
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, AntennaPattern.size(flatten=True)),
            BiScaleNorm(),
        )
        self.to(config.device)

    def forward(self, input, tau:Optional[float] = None) -> Tensor:
        x = self.fc_patch(input)
        #? 把 MLP 輸出的 logits 交給 AntennaPattern.binarization 做 STE 二值化 (tau 控制軟硬程度)。
        return AntennaPattern.binarization(x, tau)

###* GradientEstimator — 梯度估計器 (pattern → response 的另類 SM) ###
#? 一個小型 CNN+MLP：把 25x25 pattern 經卷積抽特徵後攤平，再過 MLP 預測 response。
#? 角色類似輕量代理模型，可用於估計「pattern 對 loss 的梯度方向」，輔助繞過不可微分的 HFSS。
class GradientEstimator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.PReLU(),
            nn.Linear(2048, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, output_dim)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Flatten()
            
        )
        self.to(config.device)

    def forward(self, A:Tensor):
        A = A.unsqueeze(0) #? [batch, W, H]  補上 channel 維度給 Conv2d
        # print(A.shape)
        output = self.conv(A)               #? 卷積抽特徵並攤平
        output = self.net(output)           #? MLP 映成 response 維度
        return AntennaResponse(output)      #? 包成領域型別，沿用後續 criterion/視覺化

###* CVAE — 條件變分自編碼器 (生成器的機率式版本) ###
#? 反向設計的另一條路線：以「目標響應 (response)」為條件 c，從潛在空間 z 解碼出 pattern。
#? 訓練 (forward) 需要成對 (pattern, response) 走 Encoder→z→Decoder 重建 pattern；
#? 推論/生成 (generate/inference) 則只用 Decoder：給目標響應 + 隨機 z → 候選 pattern，
#? 一次可採樣多個候選 (對應 inverse 問題多解的本質)。
#? 與 SM 的關係：CVAE 本身不含 SM；候選 pattern 之後仍須交給 SM/SIM 評估挑選。
class CVAE(nn.Module):
    """
    條件變分自動編碼器 (CVAE)
    """
    def __init__(self, latent_dim: int, pattern_size:Optional[int] = None, response_size:Optional[int] = None, binary_fn:Callable[..., Tensor] = AntennaPattern.binarization):
        """
        初始化 CVAE 模型。

        Args:
            pattern_size (int): 天線圖案展平後的大小
            response_size (int): EM 響應展平後的大小
            latent_dim (int): 潛在空間 (z) 的維度。
        """
        super(CVAE, self).__init__()
        self.pattern_size = pattern_size or AntennaPattern.size(flatten=True)       #? x  (要重建/生成的目標)
        self.response_size = response_size or AntennaResponse.size(flatten=True)    #? c  (條件：目標響應)
        self.latent_dim = latent_dim
        hidden_dim = 256

        #* Encoder: [Pattern + Response] -> [Latent Params]
        #! Only Training  ← Encoder 僅訓練時用；推論/生成階段只走 Decoder。
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.pattern_size + self.response_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.fc_binary = binary_fn
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # 均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # 變異數對數

        #* Decoder: [Latent z + Response] -> [Pattern Logits]
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim + self.response_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.pattern_size),
        )
        
        self.to(config.device)
        # logger.info(f"CVAE Model Initialized: pattern_size={pattern_size}, response_size={response_size}, latent_dim={latent_dim}")

    def encode(self, pattern: Tensor_B_N, response: Tensor_B_N) -> Tuple[Tensor, Tensor]:
        """
        (pattern, response) -> (mu, logvar)

        Args:
            pattern (Tensor): 批次的二進制天線圖案 (B, pattern_size)。
            response (Tensor): 批次的對應 EM 響應 (B, response_size)。

        Returns:
            Tuple[Tensor, Tensor]: 潛在空間的 (mu, logvar)。
        """
        inputs = torch.cat([pattern, response], dim=-1)  #? 把 pattern 與條件 response 串接後送進 encoder
        h = self.encoder_fc(inputs)
        #! logvar clamp 到 [-10,10]：防止 exp(logvar) 溢位或塌縮造成 KLD / reparameterize 出 NaN。
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        z = mu + epsilon * std

        :param mu: 潛在空間的平均。
        :param logvar: 潛在空間的 log 變異數。
        :return Tensor: 採樣出的潛在向量 z。
        """
        #? 重參數化技巧 (reparameterization trick)：把隨機性移到與參數無關的 eps 上，
        #  使 z = mu + eps*std 對 mu/logvar 可微，梯度才能穿過「採樣」這一步反傳到 encoder。
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # 從標準常態分佈中採樣雜訊
        return mu + eps * std

    def decode(self, z: Tensor, response: Tensor_B_N) -> Tensor:
        """
        (z, Response) -> Logits 

        :param z: 批次的潛在向量 (B, latent_dim)。
        :param response: 批次的目標 EM 響應 (B, response_size)。
        :return Tensor: 重建圖案的 Logits (B, pattern_size)。
        """
        #? Decoder 以「潛在向量 z + 條件 response」串接為輸入，輸出 pattern 的 logits (未過 sigmoid)。
        inputs = torch.cat([z, response], dim=-1)
        return self.decoder_fc(inputs)

    def forward(self, pattern: Tensor, response: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        (Encoder + Decoder) -> (recon_logits, mu, logvar)

        Args:
            pattern (Tensor): 輸入的真實圖案。
            response (Tensor): 輸入的真實響應。

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (recon_logits, mu, logvar)
        """
        #? 完整 VAE 前向：統一形狀 → 編碼成 (mu,logvar) → 重參數採樣 z → 解碼回 pattern logits。
        #  回傳 mu/logvar 供損失函式計算 KLD。
        pattern = size_converter(AntennaPattern, pattern, flatten=True, batch=True)
        response = size_converter(AntennaResponse, response, flatten=True, batch=True)
        mu, logvar = self.encode(pattern, response)
        z = self.reparameterize(mu, logvar)
        recon_pattern = self.decode(z, response)
        return recon_pattern, mu, logvar

    def inference(self, target_response:Tensor, z:Optional[Tensor] = None, best:Optional[tuple[Tensor, Tensor]] = None, noise_scale = 0.0):
        """
        推論
        """
        self.eval()
        target_response = size_converter(AntennaResponse, target_response, flatten=True, batch=True)
        #? 三種採樣策略，決定 z 從何而來：
        if best is not None: # 基於歷史最佳解進行微調
            #? (1) 局部探索：把歷史最佳 (pattern,response) 編碼出 mu，在其鄰域加小雜訊採樣，
            #     圍繞已知好解做微調 (exploitation)。
            best_pattern, best_response = best
            with torch.no_grad():
                best_pattern = size_converter(AntennaPattern, best_pattern, flatten=True, batch=True)
                best_response = size_converter(AntennaResponse, best_response, flatten=True, batch=True)
                mu, _ = self.encode(best_pattern, best_response)
                z = mu + torch.randn_like(mu)*noise_scale

        elif z is None:  # 全域隨機探索
            #? (2) 全域探索：z 直接從 N(0,1) 採樣 (exploration)。(3) 若外部已給 z 則直接沿用。
            z = torch.randn(target_response.size(0), self.fc_mu.out_features).to(target_response.device)

        #? 解碼後做 STE 二值化輸出最終 0/1 pattern。
        return AntennaPattern.binarization(self.decode(z, target_response))

    def elbo_logits(self, recon_logits: Tensor, target: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1) -> Tensor:
        """
        基於原始 Logits 計算。
        
        Args:
            recon_logits: Decoder 的直接輸出 (未經 Sigmoid)
            target: 真實圖樣 (0 或 1)
        """
        #? ELBO = 重建項 (BCE) + beta * 正則項 (KLD)。beta 即 Beta-VAE 的權重，調控潛在空間的解耦/壓縮。
        # BCEWithLogitsLoss 內部整合了 Sigmoid，能防止 log(sigmoid(x)) 的溢位問題
        BCE = F.binary_cross_entropy_with_logits(recon_logits, target, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  #? 使 q(z|x) 靠近標準常態先驗

        return BCE + beta * KLD

    def elbo_binarized(self, recon_bin: Tensor, target: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1) -> Tensor:
        """
        基於二值化後的結果計算。
        """
        #? 此版直接吃「已二值化 (STE 後)」的 recon，用一般 BCE；故需先 clamp 遠離 0/1 邊界，
        #! 否則 log(0) 會產生 -inf 使 loss 變 NaN。
        eps = 1e-6
        recon_safe = torch.clamp(recon_bin, min=eps, max=1.0 - eps)

        # 2. 確保 target 也是浮點數 (雖然通常已經是)
        target = target.float()
        
        # 3. 計算 BCE
        # 建議改用 mean (平均)，避免因 pattern_size 很大導致 Loss 數值過大
        BCE = F.binary_cross_entropy(recon_safe, target, reduction='mean') 
        # 如果您堅持用 sum，請務必加上梯度裁剪 (Clip Grad)
        # BCE = F.binary_cross_entropy(recon_safe, target, reduction='sum')

        # 4. 計算 KLD
        # 這裡您之前已經加了 clamp logvar，應該是安全的
        # 若上方改用 mean，這裡建議也改用 mean 以維持比例
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = BCE + beta * KLD

        # 5. [最後防線] 檢查 NaN/Inf
        #! 若仍出現 NaN/Inf，回傳一個帶梯度的 0 讓這一步等同 no-op，避免整個訓練崩潰中斷。
        if torch.isinf(total_loss) or torch.isnan(total_loss):
            print(f"[Warning] Inf/NaN Loss Detected! BCE={BCE.item()}, KLD={KLD.item()}")
            # 回傳帶梯度的 0，避免程式崩潰，讓 Optimizer 跳過這一步
            return torch.tensor(0.0, requires_grad=True, device=total_loss.device)

        return total_loss

    def fit(self, 
            data_source, 
            optimizer: torch.optim.Optimizer, 
            epochs: int = 100, 
            batch_size: int = 32,
            beta: float = 1, 
            use_ste: bool = True,
            ste_params: dict = {'tau': 1.0, 'threshold': 0.0}) -> dict:
        """
        封裝好的訓練迴圈。
        
        Args:
            data_source: 可以是 Tuple(patterns, responses) 的 Tensors，或是 PyTorch DataLoader。
            optimizer: 優化器 (e.g. Adam)。
            epochs (int): 訓練輪數。
            batch_size (int): 若 data_source 為 Tensor 時的批次大小。
            beta (float): KL Divergence 的權重 (Beta-VAE)。
            use_ste (bool): 是否啟用 STE 二值化優化 (True 使用 elbo_binarized, False 使用 elbo_logits)。
            ste_params (dict): 傳給 binarization 的參數 (僅在 use_ste=True 時有效)。
            
        Returns:
            dict: 包含 'total_loss', 'bce', 'kld' 的歷史紀錄 list。
        """
        if hasattr(data_source, '__len__') and len(data_source) == 0:
            print(f"[Warning] Data source is empty (0 items). Skipping training loop.")
            return {'total': [], 'bce': [], 'kld': []}
        
        # 1. 確保模型處於訓練模式
        self.train()

        # 2. 準備數據迭代器
        #? 容忍三種 data_source：DataLoader 直接用；Dataset 包成 DataLoader；
        #  (patterns, responses) 的 Tensor tuple 則先包 TensorDataset 再包 DataLoader。
        if isinstance(data_source, torch.utils.data.DataLoader):
            dataloader = data_source
            
        # 其次檢查是否為 Dataset (需封裝 Batch)
        elif isinstance(data_source, torch.utils.data.Dataset):
            dataloader = torch.utils.data.DataLoader(
                data_source, 
                batch_size=batch_size, 
                shuffle=True
            )
            
        # 最後檢查是否為原始 Tensor Tuple/List (需封裝 Dataset + Batch)
        elif isinstance(data_source, (tuple, list)) and len(data_source) == 2:
            patterns, responses = data_source
            
            # 安全檢查：確保內容物確實是 Tensor
            if not (torch.is_tensor(patterns) and torch.is_tensor(responses)):
                raise TypeError("Data source tuple/list must contain PyTorch Tensors.")
                
            dataset = torch.utils.data.TensorDataset(patterns, responses)
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True
            )
            
        else:
            raise TypeError(
                f"Unsupported data_source type: {type(data_source)}. "
                "Expected DataLoader, Dataset, or (Pattern_Tensor, Response_Tensor)."
            )

        history = {'total_loss': [], 'bce': [], 'kld': []}

        # 3. 訓練迴圈
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_bce = 0.0
            epoch_kld = 0.0
            steps = 0
            
            for batch_x, batch_c in dataloader:
                batch_x = size_converter(AntennaPattern, batch_x.to(config.device), flatten=True, batch=True)
                batch_c = size_converter(AntennaResponse,  batch_c.to(config.device), flatten=True, batch=True)
                
                # --- Forward ---
                recon_logits, mu, logvar = self.forward(batch_x, batch_c)
                
                # --- Loss Calculation ---
                #? 兩種損失模式：use_ste 用「二值化後」的 recon 配 elbo_binarized (貼近最終離散目標)；
                #  否則直接用 logits 配 elbo_logits (數值更穩定的標準 VAE 路徑)。
                if use_ste: # Binary Optimization (STE)
                    recon_ste = AntennaPattern.binarization(
                        recon_logits,
                        # tau=ste_params.get('tau', 1.0),
                        # threshold=ste_params.get('threshold', 0.0)
                    )
                    loss = self.elbo_binarized(recon_ste, batch_x, mu, logvar, beta)
                else:   # Logits Optimization (Standard)
                    loss = self.elbo_logits(recon_logits, batch_x, mu, logvar, beta)

                # --- Backward ---
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)  #! 梯度裁剪：配 sum reduction 時尤其重要，防梯度爆炸
                optimizer.step()
                
                # --- Record ---
                # 為了記錄方便，我們重新算一下單項 Loss (不含 backward graph)
                with torch.no_grad():
                    epoch_loss += loss.item()
                    # 簡單估算拆解項 (僅供參考)
                    kld_val = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    epoch_kld += beta * kld_val.item()
                    epoch_bce += (loss.item() - (beta * kld_val.item()))
                    steps += 1

            # 平均 Loss
            if steps > 0:
                history['total_loss'].append(epoch_loss / steps)
                history['bce'].append(epoch_bce / steps)
                history['kld'].append(epoch_kld / steps)

        return history
    
    def generate(self, response: Tensor, n_samples: int = 1) -> Tensor:
        """
        用於反向設計的生成函數。
        給定一個「目標響應」(條件)，從潛在空間隨機採樣 z, 並使用「解碼器」生成 n_samples 個候選圖案。

        Args:
            response (Tensor): 目標 EM 響應 (1, response_size)。
            n_samples (int): 要生成的候選圖案數量。

        Returns:
            Tensor: 生成的候選圖案 Logits (n_samples, pattern_size)。
        """
        #? 反向設計生成：固定目標響應為條件，採樣 n_samples 個不同 z → 得多個候選 pattern。
        #  這正利用 VAE 的多模態性質涵蓋 inverse 問題的多解，後續再由 SM/SIM 篩出最佳。
        # 從標準常態分佈 N(0, 1) 中隨機採樣 z
        z = torch.randn(n_samples, self.latent_dim).to(config.device)

        # 將 "目標響應" (條件) 複製 n_samples 次，以匹配 z 的批次大小
        response_batch = response.repeat(n_samples, 1)

        # 只使用解碼器生成圖案 logits
        logits = self.decode(z, response_batch)
        return logits

import torch
import torch.nn as nn
from torch.functional import F
from typing import Optional, Tuple, List

# 匯入您專案所需的模組
from antenna import AntennaPattern, AntennaResponse, MultiResponses, config
from antenna.functions import mirror  #? 可微分鏡像：把單一 pattern 翻出多個對稱版本

###* MirrorCVAE — 生成 + 鏡像 + 用 SM 評估選最佳 ###
#? 整合「CVAE 風格生成器 + 對稱鏡像 + SM 評估器」的一站式生成模組，是與 SM 耦合最深的生成器。
#? 流程：解碼出基礎 pattern → mirror() 產生多個對稱變體 → 全部餵 SM 取 fake_loss →
#?       以 argmin 挑最佳 (但選擇本身不回傳梯度) → 回傳排序後的候選清單。
#? 設計動機：天線多為對稱結構，先驗地枚舉鏡像對稱版本能擴大有效候選並提升命中率；
#?           SM 在此被當「固定的可微分評估器」，故初始化時即凍結其梯度。
class MirrorCVAE(nn.Module, Generic[CustomSModel]):
    """
    結合了 CVAE 解碼器 (生成器) 與鏡像/評估/選擇機制的模組。

    forward() 方法會執行以下操作：
    1. 根據輸入的條件 (c) 和一個隨機採樣的潛在向量 (z) 生成一個 "基礎 pattern"。
    2. 使用 mirror() 函數 [cite: 331-332] 產生多個鏡像版本的 pattern 。
    3. 使用傳入的 surrogate model (smodel) 評估所有鏡像 pattern 。
    4. 找出 "最佳" (smodel 損失最低) 的 pattern 。
    5. 回傳這個最佳的 pattern 及其對應的 smodel 損失，兩者都帶有梯度，
       可直接用於反向傳播 。
    """
    def __init__(self,
                 latent_dim: int,
                 smodel: CustomSModel, # 您預先訓練好的 surrogate model
                 lower_pattern: AntennaPattern = None # 靜態的 'lower' pattern
                ):
        """
        初始化 MirrorCVAE 生成器。

        Args:
            latent_dim (int): CVAE 的潛在向量維度 (例如: 128)。
            smodel (nn.Module): 一個預先訓練好的代理模型 (Surrogate Model)。
                             這個模型將被用來 "評估" 鏡像 pattern 的好壞。
            lower_pattern (AntennaPattern): 要添加到每個 pattern 上的靜態 'lower' 部分。
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        condition_dim = AntennaResponse.size(flatten=True)
        pattern_dim = AntennaPattern.size(flatten=True)
        
        # --- 儲存外部模組 ---
        if not callable(smodel):
            raise TypeError("smodel 必須是一個可呼叫的 nn.Module")
        
        # 儲存 smodel，並凍結其參數 (如果 smodel 在別處訓練)
        #! 凍結 SM：MirrorCVAE 只更新解碼器，SM 僅作評估器；梯度仍會「穿過」SM 流回解碼器，
        #  但 SM 自身權重不被更新 (requires_grad(False) 來自 Models 外殼)。
        self.smodel = smodel
        self.smodel.requires_grad(False)

        self.lower = lower_pattern
        
        # --- CVAE 解碼器 (生成器) ---
        # !!
        # !! 請將這個 self.decoder 替換為您自己的 CVAE 解碼器架構
        # !!
        # 這裡使用一個基於 OldGEN [cite: 405-407] 的範例架構
        self.decoder = nn.Sequential(
            # 輸入維度 = 潛在向量 + 條件
            nn.Linear(latent_dim + condition_dim, 1024),
            nn.PReLU(), 
            nn.Linear(1024, 1024),
            nn.PReLU(), 
            nn.Linear(1024, pattern_dim),
            nn.Sigmoid() # 確保輸出在 0 到 1 之間
        )
            

    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """ 
        解碼器：將潛在向量 z 和條件 c 轉換為基礎 pattern 張量。
        """
        #! 此處沿 dim=0 串接 (非 dim=-1)，對應 forward 中以「無 batch 維」的 1D 向量處理；
        #  與 CVAE.decode 的批次串接寫法不同，使用時須留意 z/c 皆為單一樣本。
        inputs = torch.cat([z, c], dim=0)
        base_pattern_tensor = self.decoder(inputs)
        return base_pattern_tensor

    def forward(self, 
                c: torch.Tensor, 
                z: Optional[torch.Tensor] = None
               ) -> List[ResultType]:
        """
        執行 "產生-鏡像-評估-選擇" 的前向傳播。

        Args:
            c (torch.Tensor): 
條件向量 (例如 AntennaResponse.target.concat())。
            z (Optional[torch.Tensor], optional): 一個固定的潛在向量 (用於可重現的生成)。
                                                  如果為 None，將隨機採樣。

        Returns:
            Tuple[AntennaPattern, torch.Tensor]:
            - best_pattern (AntennaPattern): 
評估後最佳的鏡像 pattern 物件。
                                           梯度會連結到這個物件。
            - best_fake_loss (torch.Tensor): 
來自 smodel 對 best_pattern 的評估損失。
                                            這是您應該呼叫 .backward() 的損失張量。
        """
        if z is None:
            # 如果未提供 z，則隨機採樣一個
            z = torch.randn(self.latent_dim, device=c.device)

        # 1. 解碼 (生成) 基礎 pattern 張量
        # 這個張量帶有來自解碼器的梯度
        base_pattern_tensor = self.decode(z, c)
        base_pattern_obj = AntennaPattern(base_pattern_tensor)


        # 2. 產生鏡像 patterns
        #? mirror 全程用 cat/flip 等可微分運算，故梯度可從鏡像結果一路回流到基礎 pattern。
        #  mode='-|*' 同時產生水平、垂直、雙向對稱版本；若有 lower 則疊加靜態底層。
        # 這個鏡像操作 (cat, flip) 是可微分的 [cite: 331-343]
        mirrored_tensors = mirror(base_pattern_obj.merge(), mode='-|*')
        mirrored_patterns = [
            AntennaPattern(t) + self.lower
            if self.lower else AntennaPattern(t)
            for t in mirrored_tensors
        ]
        # 3. 評估所有鏡像 patterns
        all_losses: List[torch.Tensor] = []
        all_results: List[MultiResponses] = []
        self.smodel.model.eval() # 確保 smodel 處於評估模式

        #! 即使 SM 被凍結 (參數不更新)，仍需 enable_grad 讓梯度「穿過」SM 計算圖，
        #  否則回不到解碼器，反向設計就學不動。
        with torch.enable_grad(): # 確保在 smodel 內部計算時保留梯度
            for pattern in mirrored_patterns:
                # 讓梯度流經 smodel

                output_result = self.smodel(pattern.series)  #? SM 預測響應
                fake_loss = output_result.criterion()        #? 與目標比較得「代理損失」(非真實 HFSS loss)
                all_losses.append(fake_loss)
                all_results.append(output_result)

        # 4. 找出最佳損失的索引
        #! 關鍵技巧：argmin 在 detach 後的張量上做，使「選哪個鏡像」這個離散決策不接收梯度
        #  (argmin 不可導)；但被選中的 best_pattern 物件仍保有原始計算圖，可正常 backward。
        # 我們在 .detach() 後的張量上執行 argmin，
        # 這樣 "選擇" 操作本身 (argmin) 就不會接收梯度。
        losses_tensor_detached = torch.stack([l.detach() for l in all_losses])
        best_loss_index = torch.argmin(losses_tensor_detached)
        
        # (可選) 填充您在 `train_single_mirror.py`  中使用的 results 列表，用於繪圖
        #? 每個鏡像候選打包成 ResultType：real_* 欄位先留 None/0，待 HFSS(SIM) 真實模擬後才回填；
        #  fake_* 為 SM 的即時評估；is_best 標記 argmin 選中者。
        results: List[ResultType] = []
        for i, pattern in enumerate(mirrored_patterns):
            result_dict: ResultType = {
                "pattern": pattern,
                "real_result": None,
                "fake_result": all_results[i],
                "real_loss": None,
                "fake_loss": all_losses[i],
                "sm_loss": [], # sm_loss 在 HFSS 模擬後才更新
                "time": 0,     # time 在 HFSS 模擬後才更新
                "sort_key": all_losses[i].item(),
                "is_best": (i == best_loss_index.item())
            }
            results.append(result_dict)

            #? 依 SM 代理損失由小到大排序，回傳整串候選 (而非只回最佳)，讓上層自行決定送幾個進 HFSS。
            results_sorted: List[ResultType] = sorted(results, key=lambda x: x["sort_key"])

        return results_sorted