###* 模型家族 (Model Family) ###
#? 本檔集中定義整個反向設計閉迴路會用到的「模型」抽象：
#?   1. Models      — 模型管理外殼 (把 model/optimizer/scheduler/criterion/Record 綁成一包，統一存讀/換檔/凍結)。
#?   2. GEN 生成器  — SigmoidGEN：
#?                    目標響應 (spec) → 25x25 二元 pattern。生成器需要「可微分的二值化」才能讓
#?                    SM 反傳回來的梯度更新 MLP，故使用 STE 技巧 (見 BiScaleNorm 與 AntennaPattern.binarization)。
#?   3. 工具元件    — BiScaleNorm / 各種 STE Function。
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
    #  關鍵用途：閉迴路中常需凍結 SM 只更新 GEN (或反之)。
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

###* SigmoidGEN — 主用生成器 (MLP + BiScaleNorm + AntennaPattern.binarization) ###
#? 本專案實際主用的生成器，入口 train_single.py / train_dual.py 即用它。
#? 結構：response → MLP 1024-1024 → BiScaleNorm，最後接
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

