###* 代理模型 (Surrogate Model, SM) ###
#? 本檔集中定義反向設計閉迴路裡的「SM」這一角色。閉迴路三方分工：
#?   GEN(生成器)：目標響應(spec) → 25x25 二元 pattern (見 models.py)。
#?   SM (本檔)  ：pattern → 預測響應 (S11 / Gain / S21 ...)，是不可微分 HFSS 的「可微分快速替身」。
#?   SIM(模擬器)：pattern → 真實響應 (COM 驅動 Ansys HFSS，準但慢且不可 backward)。
#?
#? 為什麼需要 SM？──── 兩個 HFSS 無法滿足的需求 ────
#?   1. 可微分：GEN 要被訓練，必須讓 loss 的梯度能反傳回 GEN 的權重。
#?      HFSS 不可微，故改讓梯度走 GEN → pattern → SM → loss 這條可微分路徑。
#?   2. 快速：HFSS 一次模擬要數分鐘，GEN 每個 epoch 都得評估 pattern，承擔不起。
#?      SM 是個小網路，前向/反向都在毫秒級。
#? 代價是 SM 只是「近似」HFSS，故每跑一次真實 HFSS，就用 (pattern, 真實響應) 線上訓練 SM，
#? 讓 SM 在「GEN 當前常產生的圖樣」附近越來越貼近真實電磁行為。
#?
#? 本檔內容：
#?   SurrogateModel — SM 基類 (繼承 Models 管理外殼)，封裝 train_one_data(單筆線上微調)、
#?                    train_by_datas(整批重訓)、__call__(回傳 MultiResponses)、early-stop 等。
#?   HFSSNet        — 純 MLP 骨幹：625 像素 → (3,17) 響應 (學長版 OldSM 實際使用)。
#?   OldSM — 工廠函式：把 model + criterion + optimizer + scheduler 組成一個 SurrogateModel。

from antenna.utils import *
from antenna.models import *          #? 取得 Models 管理外殼 (存讀檔/換 label/step/凍結梯度) 與型別變數
from antenna.ranger import Ranger     #? Ranger = RAdam + Lookahead，SM 訓練用的優化器
from antenna import *                 #? AntennaPattern / AntennaResponse / MultiResponses / config / size_converter 等核心物件

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from antenna.utils.data import DataManager  #? 可持久化/可去重/可當 PyTorch Dataset 的資料集容器 (供 train_by_datas 使用)

###* SurrogateModel — SM 基類 (繼承 Models 管理外殼) ###
#? 把 model(骨幹網路) + criterion(損失) + optimizer + scheduler 綁成一包，
#? 並提供兩種訓練入口與一個推論入口，是 GEN/SIM 之外的閉迴路第三方。
#?   __call__       — 推論：pattern → MultiResponses (預測響應)，供 GEN 反傳取梯度。
#?   train_one_data — 單筆線上微調：每跑一次新 HFSS 就用該筆 (pattern, 真實響應) 把 SM 訓到收斂。
#?   train_by_datas — 整批重訓：rollback 時用整個 online_dataset 重訓，糾正 SM 的整體偏差。
#? 泛型參數沿用 Models 的型別變數，讓型別檢查能追蹤被包住的具體 module/optimizer 等型別。
class SurrogateModel(
    Models[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams],
    Generic[CustomModule, ModelParams, ReturnType, CustomOptimizer, CustomScheduler, LossParams]
):
    def __init__(self, model:CustomModule, criterion:Callable[CallableParam, Tensor], optimizer:CustomOptimizer, scheduler:Optional[CustomScheduler]=None, *, rootdir=None):
        """
        Global Variable
        ---------------
        ```
        config['HFSS.min_loss'] = ...
        config['HFSS.max_epoch'] = ...
        ```

        Parameters
        ----------
        progress_callback: function 
            A callback function that will be called for every frame to notify
            the saving progress. It must have the signature ::

                def func(current_frame: int, total_frames: int) -> Any

            where *current_frame* is the current frame number and
            *total_frames* is the total number of frames to be saved.
            *total_frames* is set to None, if the total number of frames can
            not be determined. Return values may exist but are ignored.

            Example code to write the progress to stdout::

                progress_callback = lambda i, n: print(f'Saving frame {i}/{n}')
        """
        #? config['HFSS.min_loss'] / config['HFSS.max_epoch'] 是 train_one_data 單筆收斂的全域預設門檻
        #? (呼叫時若未明確傳入就回退到這兩個值)；config['HFSS.lr'] 則由 OldSM 工廠用來設定 Ranger 學習率。
        super().__init__(
            name='sm',          #? 固定名稱 → checkpoint 存成 sm.pth (SM 不像 GEN 需要逐 epoch rollback)
            rootdir=rootdir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            load=False # 避免呼叫父類別未覆寫的 load
        )

        self.device = config['device']
        self.pattern_size = AntennaPattern.size      #? 輸入維度資訊 (25x25=625 像素)
        self.response_size = AntennaResponse.size    #? 輸出維度資訊 (e.g. (3,17))
        self.size_converter = size_converter         #? 在 (B,N) 攤平 / (B,H,W) 影像 / 批次維度間轉換的工具

        self.epoch = 0  #? 累計被 GEN 呼叫推論的次數 (僅作計數，不影響訓練)

    #? 推論入口：GEN 生成 pattern → 餵入 SM 骨幹 → 包成 MultiResponses。
    #? 回傳物件帶梯度，GEN 對其 .criterion() 算 loss 後 backward，梯度即經 SM 反傳回 GEN。
    def __call__(self, pattern) -> MultiResponses:
        self.epoch += 1
        return MultiResponses(self.model(pattern))

    def train_by_datas(self, dataset:DataManager, epochs: int = 100, batch_size: Optional[int] = None, *, verbose:bool = True) -> List[float]:
        """
        Train the model using the provided dataset.

        Args:
            dataset (DataManager): Data set used for training.
            epochs (int): Total number of training cycles.
            batch_size (Optional[int]): Size of each batch.
            verbose (bool): Enable progress bar.

        Returns:
            List[float]: List of average losses per epoch.
        """
        #? 【何時呼叫】整批訓練 (batch training)。閉迴路中於兩種時機使用：
        #?   (1) 訓練前的暖身 (warm-up)：用既有 data_manager 先把 SM 訓到大致可用。
        #?   (2) rollback 時：GEN 被回滾到歷史最佳後，用整個 online_dataset 重訓 SM，
        #?       糾正單筆微調 (train_one_data) 累積下來的整體偏差，避免 SM 過度偏向最近少數圖樣。
        #? 與 train_one_data 的差異：這裡掃整個資料集多個 epoch，是「全域校正」而非「就地微調」。
        self.requires_grad(True, train=True)  #? 解凍參數並切到 train 模式 (BatchNorm/Dropout 生效)
        self.record.reset()

        if dataset is None or len(dataset) <= 0:
            return []  #? 空資料集 → 無事可做，直接回傳空清單 (常見於閉迴路初期還沒收集到資料)
        elif batch_size is None:
            pass
        else:
            batch_size = min(len(dataset), batch_size)  #? batch 不可超過資料總量

        dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True,
            generator=torch.Generator(device=config.device)  #? 指定 device 上的亂數產生器，避免 shuffle 的 device 不符
        )

        epoch_bar = tqdm(range(epochs), desc='Training...', disable=not verbose, **TQDM_CONFIG)
        for epoch in epoch_bar:
            for n, (patterns, real_responses) in enumerate(cast(tuple[Tensor, Tensor], dataloader)):

                #? 統一形狀：pattern 攤平成 (B, 625) 餵 MLP；響應保留 (B, C, L) 不攤平 (與骨幹輸出對齊)
                patterns = self.size_converter(AntennaPattern, patterns, flatten=True, batch=True)
                real_responses = self.size_converter(AntennaResponse, real_responses, flatten=False, batch=True)

                inputs:Tensor = patterns.flatten(start_dim=1).to(config.device)   #? SM 輸入：pattern
                labels:Tensor = real_responses.to(config.device)                  #? SM 目標：HFSS 真實響應

                self.optimizer.zero_grad()
                outputs: Tensor = self.model(inputs)            #? SM 預測響應
                loss: Tensor = self.criterion(outputs, labels)  #? 預測 vs. 真實 的回歸誤差 (OldSM 用 MSE)

                loss.backward()
                self.step(scheduler_param=loss)  #? optimizer.step() + scheduler.step(loss) (ReduceLROnPlateau 看 loss)

                self.record['loss'] = loss.item()  #? 記錄每個 batch 的 loss，供下方求 epoch 平均

            avg_epoch_loss = self.record.average('loss')  #? 本 epoch 所有 batch 的平均 loss
            self.record.reset('loss', delete=True)        #? 清掉 batch 級暫存，下個 epoch 重新累積
            self.record['epoch_loss'] = avg_epoch_loss    #? 推進到 epoch 級曲線 (即最終回傳值)

            epoch_bar.set_postfix({"Loss": f"{avg_epoch_loss:.4e}"})

            #? 早停：若最近 epochs/2 個 epoch 的 epoch_loss 都沒比之前更好，就提前結束 (省時，避免過擬合)
            if self.record.early_stop('epoch_loss', int(epochs / 2)):
                logger.success(f'Early Stopping triggered at epoch {epoch + 1}!')
                break

        self.model.eval()  #? 訓練結束切回 eval，後續 GEN 推論時 BatchNorm/Dropout 維持確定性
        return self.record['epoch_loss']  #? 回傳每 epoch 平均 loss 清單 (訓練腳本拿去畫收斂曲線)
    
    def train_one_data(self, pattern:Tensor, real_response:Tensor, min_loss=None, max_epoch=None, *, verbose:bool = True):
        """
        The model is trained using a single set of data.

        Args:
            pattern (Tensor): Real antenna pattern
            real_response (Tensor): The real response of the antenna pattern
            min_loss: Minimum loss limit
            max_epoch: Maximum epoch limit
            verbose (bool): Enable progress bar.

        Returns:
            List[float]: List of average losses per epoch.
        """
        #? 【何時呼叫】單筆線上訓練 (online fine-tuning)。閉迴路中：每當 GEN 產生一個「沒看過」的
        #? pattern 並真的跑了一次 HFSS，就立刻用這一筆 (pattern, 真實響應) 把 SM 訓到收斂。
        #? 目的：讓 SM 在「GEN 此刻正在探索的圖樣」附近貼緊真實 HFSS，使隨後 GEN 經 SM 反傳的梯度更可信。
        #? 與 train_by_datas 的差異：這裡只盯這一筆資料反覆迭代到 loss < min_loss，是「就地過擬合」式的微調。
        self.requires_grad(True, train=True)
        self.record.reset()

        self.record['loss'] = float('inf')  #? loss 初值設為無限大 → 確保 while 迴圈至少進入一次
        self.record['epoch'] = 0            #? 本筆資料的內層迭代計數 (與外層閉迴路 epoch 無關)

        input = tensor(pattern,  requires_grad=True)        #? 該 pattern (GEN 產生 / 模擬過的圖樣)
        label = tensor(real_response,  requires_grad=True)  #? 對應的 HFSS 真實響應 (訓練目標)

        min_loss = min_loss or config['HFSS.min_loss']    #? 收斂門檻：loss 降到此值以下即停 (未傳入則用全域預設)
        max_epoch = max_epoch or config['HFSS.max_epoch'] #? 迭代上限：避免某些難擬合的單筆資料無限迭代

        epoch_bar = tqdm(
            total=max_epoch, desc="Training one data",
            bar_format=TQDM_BAR_SIMPLE, disable=not verbose, **TQDM_CONFIG
        )
        #? 停止條件：loss 仍高於門檻「且」尚未到達迭代上限 → 任一不滿足就跳出 (達標或超時)
        while self.record('loss', 0) > min_loss and self.record('epoch', float('inf')) < max_epoch:
            self.optimizer.zero_grad()

            outputs_result:Tensor = self.model(input)  #? SM 對這筆 pattern 的當前預測

            #? reshape 成 (-1, *response_size) 讓預測與目標形狀對齊，再算回歸誤差
            loss:Tensor = self.criterion(
                outputs_result.reshape(-1, *AntennaResponse.size()),
                label.reshape(-1, *AntennaResponse.size())
            )

            loss.backward()
            self.step(scheduler_param=loss)  #? 更新 SM 權重 (+ scheduler)

            self.record['loss'] = loss.item()  #? 更新當前 loss → 驅動 while 的收斂判斷
            self.record.add('epoch', 1)        #? 內層迭代次數 +1 → 驅動 while 的上限判斷

            epoch_bar.update()
            epoch_bar.set_postfix(
                {'loss': f"{self.record('loss'):.2f}/{min_loss}"}
            )

        self.model.eval()
        return self.record['loss']  #? 回傳收斂後的最終 loss (訓練腳本記為該筆的 sm_loss)

###* HFSSNet — 純 MLP 骨幹 (學長版 OldSM 實際採用的網路) ###
#? 角色：SM 的「身體」(SurrogateModel 是外殼，HFSSNet 是被包住的 model)。
#? 結構最簡：把攤平的 625 個像素，經一連串全連接層 (逐步收斂的瓶頸 2048→1024→512→128→64)，
#? 直接映射到 (3,17) 響應，不利用像素的 2D 空間鄰接資訊 (那是下方 U-Net 版才做的)。
#? 用 PReLU 而非 ReLU：負區間保留可學斜率，避免 dead neuron、利於回歸這種需細緻擬合的任務。
class HFSSNet(nn.Module):

    #? num_pattern_pixel: 輸入像素數 (25x25=625)；num_response: 輸出響應形狀 (通道數 x 頻點數)
    def __init__(self, num_pattern_pixel = 625, num_response:tuple = (3, 17), hidden=(2048, 1024, 512, 128, 64)):
        super(HFSSNet, self).__init__()
        self.num_response = num_response
        self.num_pattern_pixel = num_pattern_pixel

        #? hidden 由 config 指定 (預設 (2048,1024,512,128,64) 與原架構完全相同)。每層 Linear→PReLU，
        #? 末層 Linear 無激活 (回歸輸出)，維度 = 響應總元素數 (如 3*17=51)。
        layers, prev = [], num_pattern_pixel
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.PReLU()]
            prev = h
        layers += [nn.Linear(prev, num_response[0] * num_response[1])]
        self.fc_patch = nn.Sequential(*layers)
        self.to(config.device)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_pattern_pixel={self.num_pattern_pixel}, num_response={self.num_response}"

    def forward(self, input):
        x = self.fc_patch(input)
        x = x.reshape(self.num_response)  #? 把攤平輸出還原成 (3,17) 響應形狀
        return x

###* OldSM — 工廠：學長版 SM (HFSSNet + Ranger + MSE + ReduceLROnPlateau) ###
#? 訓練腳本 (train_single.py / train_dual.py) 實際使用的就是這個工廠 (見各腳本 from ... import OldSM)。
#? 輕量穩定：純 MLP 骨幹 + MSE 回歸 + 「loss 卡住就降 lr」的 ReduceLROnPlateau，
#? 適合 SM 這種需頻繁線上微調、且每筆資料都要快速收斂的場景。
def OldSM(checkpoint, in_dim, response_shape, hidden=(2048, 1024, 512, 128, 64)):
    """
    學長的做法。維度由訓練端傳入 (不碰全域註冊狀態)；hidden 可由 config 指定。
    """
    model_ge = HFSSNet( # Pattern -> Response
        in_dim, response_shape, hidden=hidden
    )
    criterion_ge = nn.MSELoss()  #? 均方誤差：SM 是響應回歸任務，懲罰大偏差
    optimizer_ge = Ranger(
        params=model_ge.parameters(), lr=config['HFSS.lr']  #? lr 取自全域 config['HFSS.lr] (train 腳本設 0.001)
    )
    #? 高原排程：當 loss 連續 patience 個 step 不再下降，就把 lr 砍半 (factor=0.5)，下限 min_lr，幫助收斂後期細修
    scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    return SurrogateModel(
        model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint
    )

