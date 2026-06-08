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
#?   SelfAttention / DoubleConvWithDropout / EnhancedHFSSUNet — U-Net 版骨幹 (含自注意力)。
#?   UNetSM / OldSM — 工廠函式：把 model + criterion + optimizer + scheduler 組成一個 SurrogateModel。

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
    def __init__(self, num_pattern_pixel = 625, num_response:tuple = (3, 17)):
        super(HFSSNet, self).__init__()
        self.num_response = num_response
        self.num_pattern_pixel = num_pattern_pixel

        #? 6 層 MLP：每層後接 PReLU，逐步壓縮維度，末層輸出攤平的響應 (3*17=51) 個值
        self.fc_patch = nn.Sequential(
            nn.Linear(num_pattern_pixel, 2048),
            nn.PReLU(),
            nn.Linear(2048, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, num_response[0]*num_response[1])  #? 末層維度 = 響應總元素數，無激活 (回歸輸出)
        )
        self.to(config.device)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_pattern_pixel={self.num_pattern_pixel}, num_response={self.num_response}"

    def forward(self, input):
        x = self.fc_patch(input)
        x = x.reshape(self.num_response)  #? 把攤平輸出還原成 (3,17) 響應形狀
        return x
###* SelfAttention — 簡化版自注意力 (供 U-Net bottleneck 使用) ###
#? 角色：插在 U-Net 最深層 (bottleneck) 之後，讓特徵圖上「任意兩個位置」互相加權，
#? 補足卷積只看局部鄰域的限制 — 對天線而言，相隔很遠的金屬區塊也會共同影響響應，
#? 自注意力能直接建模這種長距離相依，理論上比純 MLP/純卷積更貼近真實電磁耦合。
class SelfAttention(nn.Module):
    """ 簡化的自注意力層 """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # 使用 // 8 可能會導致通道數過少，特別是如果 in_channels 本身不大
        # 改用固定的 attention_channels 或 min(in_channels // 8, 某個固定值)
        attention_channels = max(1, in_channels // 8) # 確保至少為 1
        self.query_conv = nn.Conv2d(in_channels, attention_channels, kernel_size=1) #
        self.key_conv = nn.Conv2d(in_channels, attention_channels, kernel_size=1) #
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1) #
        #? gamma 初始化為 0 → 訓練初期注意力分支貢獻為零，整層退化成 identity，
        #? 讓模型先學好卷積特徵、再逐步「開啟」注意力，是穩定訓練的常見技巧。
        self.gamma = nn.Parameter(torch.zeros(1)) #
        self.softmax = nn.Softmax(dim=-1) #

    def forward(self, x):
        batch_size, C, width, height = x.size()
        # [B, C', W*H] -> [B, W*H, C']
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1) #
        # [B, C', W*H]
        proj_key = self.key_conv(x).view(batch_size, -1, width * height) #
        # [B, W*H, C'] @ [B, C', W*H] -> [B, W*H, W*H]
        energy = torch.bmm(proj_query, proj_key) #
        attention = self.softmax(energy) # 在 W*H 維度上 softmax
        # [B, C, W*H]
        proj_value = self.value_conv(x).view(batch_size, -1, width * height) #

        # [B, C, W*H] @ [B, W*H, W*H] -> [B, C, W*H] (注意permute)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) #
        out = out.view(batch_size, C, width, height) #

        out = self.gamma * out + x # 殘差連接
        return out #

# --- 2. 包含 Dropout 的 DoubleConv ---
###* DoubleConvWithDropout — U-Net 的基本卷積積木 (含 BN + Dropout) ###
#? U-Net 編碼/解碼器每一層的標準單元：連兩次 (Conv→BN→ReLU→Dropout)。
#? 加 Dropout 是為了抑制 SM 過擬合 — SM 線上資料量小且分布偏向 GEN 當前圖樣，
#? 不加正則化容易記住雜訊、在 GEN 探索的新區域給出失真梯度。
class DoubleConvWithDropout(nn.Module):
    """(convolution => [BN] => ReLU => [Dropout]) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout_prob=0.15): # 增加 dropout 概率
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), #
            nn.BatchNorm2d(mid_channels), #
            nn.ReLU(inplace=True), # ***修改點：使用 ReLU***
            nn.Dropout(dropout_prob), # ***修改點：添加 Dropout***
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), #
            nn.BatchNorm2d(out_channels), #
            nn.ReLU(inplace=True), # ***修改點：使用 ReLU***
            nn.Dropout(dropout_prob)  # ***修改點：添加 Dropout***
        )

    def forward(self, x):
        return self.double_conv(x) #

# --- 3. 整合修改後的 EnhancedHFSSUNet ---
###* EnhancedHFSSUNet — U-Net 版 SM 骨幹 (含自注意力) ###
#? HFSSNet 的進階替代品 (由 UNetSM 工廠採用)。把 pattern 視為單通道影像，
#? 用編碼器(下採樣)→bottleneck→解碼器(上採樣) 的 U-Net 結構抽取多尺度空間特徵，
#? 並在 bottleneck 後接 SelfAttention 捕捉長距離耦合，最後用一個小 head 回歸成 (3,17) 響應。
#? 動機：天線像素的「空間鄰接 / 連通形狀」直接決定電磁行為，純 MLP(HFSSNet) 看不到這層結構，
#? U-Net 的卷積與 skip-connection 能保留這些空間資訊，理論上對 HFSS 的逼近更準。
class EnhancedHFSSUNet(nn.Module):
    def __init__(self, base_channels=64, dropout_prob=0.15):
        """
        增強版的 HFSSUNet，包含增加通道數、Dropout 和 Self-Attention。

        Args:
            num_pattern_pixel (int): 輸入 Pattern 的像素總數 (應為平方數)。
            num_response (tuple): 輸出響應的形狀 (e.g., (3, 17))。
            base_channels (int): U-Net 第一層的基礎通道數，控制模型容量。
            dropout_prob (float): 應用於 DoubleConv 層的 Dropout 概率。
        """
        super(EnhancedHFSSUNet, self).__init__()

        # --- 自動獲取大小 ---
        #? 直接從 AntennaPattern/AntennaResponse 的全域定義讀形狀，網路維度自動對齊，不必手動傳參
        _pattern_size = AntennaPattern.size(flatten=False) # (H, W)
        _response_size = AntennaResponse.size(flatten=False) # (C, L) or (H, W)

        # 確保 response_size 是二維的
        if len(_response_size) != 2:
             raise ValueError(f"AntennaResponse.size() 應返回二維形狀，但得到 {_response_size}")

        self.num_response = _response_size
        self.num_pattern_pixel = _pattern_size[0] * _pattern_size[1]
        self.input_dim_h, self.input_dim_w = _pattern_size # 分別獲取高和寬
        #--------------------

        self.base_channels = base_channels
        self.dropout_prob = dropout_prob

        # 不再需要檢查平方數，因為我們直接用 H 和 W
        # if self.input_dim * self.input_dim != self.num_pattern_pixel:
        #     raise ValueError("num_pattern_pixel 不是一個完美的平方數，無法轉換為 2D 圖像")

        n_channels_in = 1                   #? 輸入視為單通道灰階影像 (像素值即 0/1 金屬有無)
        n_channels_out = base_channels // 2 # Decoder 最後輸出的通道數, 64 // 2 = 32

        # --- Encoder (通道數增加, 使用 DoubleConvWithDropout) ---
        #? 編碼器：每經一次 down+pool，空間尺寸減半、通道數加倍，逐層抽取由細到粗的空間特徵。
        #? 各層 down 的輸出 (x1/x3/x5) 之後會以 skip-connection 接回解碼器對應層，保留高解析度細節。
        self.down1 = DoubleConvWithDropout(n_channels_in, base_channels, dropout_prob=dropout_prob)             # 1 -> 64
        self.pool1 = nn.MaxPool2d(2) #
        self.down2 = DoubleConvWithDropout(base_channels, base_channels * 2, dropout_prob=dropout_prob)         # 64 -> 128
        self.pool2 = nn.MaxPool2d(2) #
        self.down3 = DoubleConvWithDropout(base_channels * 2, base_channels * 4, dropout_prob=dropout_prob)     # 128 -> 256
        self.pool3 = nn.MaxPool2d(2) #

        # --- Bottleneck (通道數增加, 使用 DoubleConvWithDropout) ---
        #? 最深層：通道數最多、空間最小，承載最抽象的全域特徵
        self.bottleneck = DoubleConvWithDropout(base_channels * 4, base_channels * 8, dropout_prob=dropout_prob) # 256 -> 512

        # --- Self-Attention (在 Bottleneck 之後) ---
        #? 在最抽象的特徵上做自注意力，建模相隔遠的金屬區塊之間的電磁耦合 (卷積看不到的長距離關係)
        self.attention = SelfAttention(base_channels * 8) # 輸入通道數 = 512

        # --- Decoder (通道數增加, 使用 DoubleConvWithDropout) ---
        #? 解碼器：每層先用 ConvTranspose2d 上採樣放大空間，再 concat 編碼器對應的 skip 特徵 (故 up_conv 輸入通道翻倍)，
        #? 由粗到細逐步還原；skip-connection 讓被下採樣丟失的細節 (邊界/小結構) 得以回流。
        self.up1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2) # 512 -> 256
        self.up_conv1 = DoubleConvWithDropout(base_channels * 8, base_channels * 4, dropout_prob=dropout_prob)  # Skip:256 + Up:256 -> 256
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2) # 256 -> 128
        self.up_conv2 = DoubleConvWithDropout(base_channels * 4, base_channels * 2, dropout_prob=dropout_prob)  # Skip:128 + Up:128 -> 128
        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)     # 128 -> 64
        self.up_conv3 = DoubleConvWithDropout(base_channels * 2, n_channels_out, dropout_prob=dropout_prob)     # Skip:64 + Up:64 -> 32

        # --- Head (包含 Dropout, 使用 ReLU) ---
        #? 輸出頭：把解碼器特徵圖用 AdaptiveAvgPool 壓成 1x1 (彙整全域空間資訊)，再經小 MLP 映射成攤平響應。
        #? 注意 SM 的最終目標是回歸「響應曲線」而非還原影像，故 U-Net 末端不是分割圖而是接 FC 回歸。
        self.head_pool = nn.AdaptiveAvgPool2d((1, 1)) #
        self.head_fc = nn.Sequential(
            nn.Linear(n_channels_out, 128), # 輸入通道數 32, 增加中間層大小
            nn.ReLU(inplace=True),         # ***修改點：使用 ReLU***
            nn.Dropout(0.25),              # ***修改點：在 Head 中加入 Dropout, 稍微提高比例***
            nn.Linear(128, self.num_response[0] * self.num_response[1]) #
        )

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"num_pattern_pixel={self.num_pattern_pixel}, "
                f"num_response={self.num_response}, "
                f"base_channels={self.base_channels}, "
                f"dropout_prob={self.dropout_prob})") #

    def forward(self, x:Tensor):
        #? 整體資料流：攤平 pattern → 還原成 (B,1,H,W) 影像 → 編碼器 → bottleneck+注意力 → 解碼器(含 skip) → head 回歸 → (B,3,17) 響應
        # 0. Reshape Input: (B, num_pixels) -> (B, 1, H, W)
        x = x.unsqueeze(0)
        if x.dim() > 2:
             x = torch.flatten(x, 1)
        # if x.shape[1] != self.num_pattern_pixel:
        #      raise ValueError(f"Input has {x.shape[1]} features, but expected {self.num_pattern_pixel}")
        # 使用初始化時獲取的高和寬
        x_img = x.view(-1, 1, self.input_dim_h, self.input_dim_w) #

        # 1. Encoder
        x1 = self.down1(x_img); x2 = self.pool1(x1) #
        x3 = self.down2(x2); x4 = self.pool2(x3) #
        x5 = self.down3(x4); x6 = self.pool3(x5) #

        # 2. Bottleneck
        bottle = self.bottleneck(x6) #

        # 3. Attention
        attn_bottle = self.attention(bottle)

        # 4. Decoder
        #? F.interpolate 把上採樣結果對齊到對應 skip 特徵的空間尺寸，
        #? 因 25x25 經多次 pool 後的尺寸不一定能用 ConvTranspose 精準還原，故補一步插值再 concat。
        u1 = self.up1(attn_bottle) # <-- 使用 attn_bottle
        u1 = F.interpolate(u1, size=x5.shape[2:], mode='bilinear', align_corners=True) #
        cat1 = torch.cat([x5, u1], dim=1) #
        c1 = self.up_conv1(cat1) #

        u2 = self.up2(c1) #
        u2 = F.interpolate(u2, size=x3.shape[2:], mode='bilinear', align_corners=True) #
        cat2 = torch.cat([x3, u2], dim=1) #
        c2 = self.up_conv2(cat2) #

        u3 = self.up3(c2) #
        u3 = F.interpolate(u3, size=x1.shape[2:], mode='bilinear', align_corners=True) #
        cat3 = torch.cat([x1, u3], dim=1) #
        c3 = self.up_conv3(cat3) #

        # 5. Head
        out_pool = self.head_pool(c3) #
        out_flat = torch.flatten(out_pool, 1) #
        out_fc = self.head_fc(out_flat) #

        # 6. Final Reshape
        out = out_fc.view(-1, self.num_response[0], self.num_response[1]) #  #? 攤平輸出還原成 (B,3,17) 響應形狀，對齊 SM 介面

        return out

# --- 4. 更新 UNetSM 函數 ---
###* UNetSM — 工廠：組裝 U-Net 版 SM (EnhancedHFSSUNet + Ranger + L1/MSE + 週期排程器) ###
#? 工廠函式把骨幹/損失/優化器/排程器一次組好，回傳可直接丟進訓練腳本的 SurrogateModel。
#? 與 OldSM 的取捨：U-Net 容量更大、能利用空間結構，但較重、較易過擬合小量線上資料；
#? 預設用 L1Loss (對離群響應點較 MSE 穩健) 與 weight_decay 正則化來緩解。
def UNetSM(checkpoint, base_channels=64, dropout_prob=0.15, learning_rate=1e-4, scheduler_patience=15, weight_decay=1e-4, loss_type='L1'):
    """
    創建一個使用 EnhancedHFSSUNet 的 SurrogateModel 實例，並調整超參數。

    Args:
        checkpoint (str or Path): 儲存/載入模型權重的路徑。
        base_channels (int): U-Net 第一層的基礎通道數。
        dropout_prob (float): 應用於 DoubleConv 層的 Dropout 概率。
        learning_rate (float): 優化器的學習率。
        scheduler_patience (int): ReduceLROnPlateau 排程器的耐心值。
        weight_decay (float): 優化器的權重衰減值。
        loss_type (str): 使用的損失函數類型 ('L1' or 'MSE')。
    """
    # pattern_size 和 response_shape 會在 EnhancedHFSSUNet 內部自動獲取
    model_ge = EnhancedHFSSUNet(
        base_channels=base_channels,
        dropout_prob=dropout_prob
    )

    # ***修改點：選擇損失函數***
    if loss_type == 'L1':
        criterion_ge = nn.L1Loss() # 使用 L1 Loss
    elif loss_type == 'MSE':
        criterion_ge = nn.MSELoss() #
    else:
        raise ValueError("loss_type 必須是 'L1' 或 'MSE'")

    optimizer_ge = Ranger( # 保持 Ranger 優化器
        params=model_ge.parameters(),
        lr=learning_rate,           # ***修改點：使用較低的學習率***
        weight_decay=weight_decay   # ***修改點：明確設置權重衰減***
    )
    from antenna.functions import AdaptiveCyclicalScheduler
    #? 週期性學習率排程：在閉迴路長時間線上更新中週期性升降 lr，幫助 SM 跳出局部最小、適應資料分布漂移
    scheduler_ge = AdaptiveCyclicalScheduler( #
        optimizer_ge,
    )
    return SurrogateModel( #
        model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint
    )

###* OldSM — 工廠：學長版 SM (HFSSNet + Ranger + MSE + ReduceLROnPlateau) ###
#? 訓練腳本 (train_single.py / train_dual.py) 實際使用的就是這個工廠 (見各腳本 from ... import OldSM)。
#? 比 UNetSM 輕量穩定：純 MLP 骨幹 + MSE 回歸 + 「loss 卡住就降 lr」的 ReduceLROnPlateau，
#? 適合 SM 這種需頻繁線上微調、且每筆資料都要快速收斂的場景。
def OldSM(checkpoint):
    """
    學長的做法
    """
    model_ge = HFSSNet( # Pattern -> Response
        AntennaPattern.size(flatten=True), AntennaResponse.size()
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

# def UNetSM(checkpoint):
#     model_ge = HFSSUNet( # Pattern -> Response
#         AntennaPattern.size(flatten=True), AntennaResponse.size()
#     )
#     criterion_ge = nn.MSELoss()
#     optimizer_ge = Ranger(
#         params=model_ge.parameters(), lr=config['HFSS.lr']
#     )
#     scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
#     )
#     return SurrogateModel(
#         model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint
#     )
