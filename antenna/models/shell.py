"""
antenna/models/shell.py — Models 管理外殼。

把 model/optimizer/scheduler/criterion 綁成一包：呼叫、存讀 checkpoint、
依 label 換存檔 (rollback 用)、凍結梯度。GEN 與 SM 都包在各自的 Models 裡
(SM 經 surrogates.SurrogateModel 繼承)。
"""
from types import FunctionType
from typing import Optional, Union

import torch
from torch import nn

from antenna.utils import Path, Record, config, logger


#? GEN 與 SM 都會被包進各自的 Models 實例 (SM 經 SurrogateModel 繼承)。
class Models:
    def __init__(
            self,
            name: str = "models_{label}",
            rootdir=None,
            model: Optional[nn.Module] = None,
            optimizer=None,
            scheduler=None,
            criterion=None,
            *,
            load: bool = False,
            device=config.device
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
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

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

    def change(self, label, *, load: bool = False, save: bool = False):
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
        except Exception:
            if temp_file.exists():
                temp_file.unlink()                   #! 失敗時清掉殘留的 .tmp，保持目錄乾淨
            raise
        return filename

    #? pre_load_model：只載「權重 + 優化器」而不比對 title，用於把預訓練 (pretrain) 的
    #  SM/GEN 權重灌進當前模型作為閉迴路的起點，跳過 load() 的嚴格架構檢查。
    #  strict=False：預訓練檔是本模型的「子集」(如方向圖版 SM 多了 head_rad，但 trunk/freq
    #    head 與舊 sm.pth 同名同形) → 只灌入共用層、缺的頭維持隨機初始；optimizer 狀態
    #    依「參數順序」對位部分載入 (新增的 head 在序末、自然保持新鮮)。
    #    ⚠ 一定要暖啟動 optimizer：實測「冷 optimizer + 暖權重」第一步就過衝、train_one_data
    #    3 步發散爆 NaN (warm: 2136→2079 緩降；cold: 2136→90→4.9e6→inf)。
    def pre_load_model(self, path: Union[str, Path], strict: bool = True):
        path = Path(path)
        checkpoint_loaded = path.load_torch()
        if strict:
            self.model.load_state_dict(checkpoint_loaded['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint_loaded['optimizer_state_dict'])
        else:
            incompatible = self.model.load_state_dict(checkpoint_loaded['model_state_dict'], strict=False)
            if incompatible.missing_keys:      # 本模型有、檔裡沒有 → 維持隨機初始 (如 head_rad.*)
                logger.info(f"部分載入：{len(incompatible.missing_keys)} 個鍵維持隨機初始 "
                            f"(如 {incompatible.missing_keys[:2]})")
            if incompatible.unexpected_keys:   # 檔裡有、本模型沒有 → 忽略
                logger.warning(f"部分載入：忽略檔中 {len(incompatible.unexpected_keys)} 個多餘鍵 "
                               f"(如 {incompatible.unexpected_keys[:2]})")
            self._warm_start_optimizer(checkpoint_loaded.get('optimizer_state_dict'))
        #! 載入後立即檢查所有參數有限性：預訓練檔若含 NaN/inf 會在閉迴路一開始就汙染梯度，
        #  在此早期擋下比讓訓練跑到一半才爆掉更易除錯。
        for name, param in self.model.state_dict().items():
            if not torch.all(torch.isfinite(param)):
                raise RuntimeError(f"!!! 在參數 '{name}' 中發現無效值 (NaN 或 inf) !!!")
        logger.success(f'Successfully loaded the pre-trained model. ({path}, strict={strict})')

    def _warm_start_optimizer(self, opt_sd):
        """部分載入 optimizer 狀態：把預訓練檔的 per-param 狀態依「參數順序」對位灌進現有
        optimizer。前提：本模型新增的參數 (如 head_rad) 接在原參數序之後 (golden-safe 建構保證)，
        故索引 0..N-1 正好對應預訓練檔的 trunk 參數；新增 head 在序末、沒有對應狀態 → 保持新鮮。
        只搬 per-param 狀態 (動量/方差/step)，不動 param_groups 的超參 (lr/betas 用本模型設定)。"""
        if not opt_sd or 'state' not in opt_sd:
            return
        params = [p for g in self.optimizer.param_groups for p in g['params']]
        for idx, st in opt_sd['state'].items():
            i = int(idx)
            if i < len(params):
                self.optimizer.state[params[i]] = {
                    k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in st.items()
                }

    #? 一步推進：先走優化器再走排程器 (若有)。把兩者包成一個呼叫方便訓練腳本使用。
    def step(self, optimizer_param=None, scheduler_param=None):
        self.optimizer.step(optimizer_param)
        if self.scheduler: self.scheduler.step(scheduler_param)

    #? 組裝/載入 checkpoint 字典。load=True 直接從檔讀；load=False 則即時蒐集當前狀態用於存檔。
    def checkpoint(self, load: bool = False) -> dict:
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
