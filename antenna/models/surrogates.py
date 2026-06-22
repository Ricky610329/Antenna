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
#?   HFSSNet        — 純 MLP 骨幹：625 像素 → (3,17) 響應 (學長版 MLPSurrogate 實際使用)。
#?   MLPSurrogate — 工廠函式：把 model + criterion + optimizer + scheduler 組成一個 SurrogateModel。

import math
from typing import List, Optional

import torch
from torch import Tensor, nn
from tqdm import tqdm

from antenna import AntennaPattern, AntennaResponse, MultiResponses
from .shell import Models             #? Models 管理外殼 (存讀檔/換 label/step/凍結梯度)
from antenna.optim import Ranger      #? Ranger = RAdam + Lookahead，SM 訓練用的優化器
from antenna.utils import TQDM_BAR_SIMPLE, TQDM_CONFIG, config, logger, tensor
from antenna.utils.torch_utils import size_converter

from torch.utils.data import DataLoader, Dataset  #? Dataset：SampleStore 與舊 DataManager 共同基類 (train_by_datas 收任一)

###* SurrogateModel — SM 基類 (繼承 Models 管理外殼) ###
#? 把 model(骨幹網路) + criterion(損失) + optimizer + scheduler 綁成一包，
#? 並提供兩種訓練入口與一個推論入口，是 GEN/SIM 之外的閉迴路第三方。
#?   __call__       — 推論：pattern → MultiResponses (預測響應)，供 GEN 反傳取梯度。
#?   train_one_data — 單筆線上微調：每跑一次新 HFSS 就用該筆 (pattern, 真實響應) 把 SM 訓到收斂。
#?   train_by_datas — 整批重訓：rollback 時用整個 online_dataset 重訓，糾正 SM 的整體偏差。
class SurrogateModel(Models):
    def __init__(self, model: nn.Module, criterion, optimizer, scheduler=None, *,
                 rootdir=None, min_loss: float = 0.1, max_epoch: int = 20000,
                 response_shape: Optional[tuple] = None,
                 rad_response: Optional[tuple] = None):
        """
        SM 外殼：包住骨幹網路 + 優化器 + 損失，提供線上/離線訓練與存讀檔。

        min_loss / max_epoch 是 train_one_data 單筆收斂的預設門檻 (呼叫時可逐次覆寫)；
        response_shape 是響應形狀 (label數, 點數)，供 train_one_data reshape 對齊。
        rad_response (選用) 是方向圖頭輸出形狀 (n_phi, n_theta)，供 train_one_data_rad reshape 對齊。
        全部由建構端顯式傳入，不讀全域 config / AntennaResponse 類別狀態。
        """
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
        self.response_shape = tuple(response_shape) if response_shape else None  #? 響應形狀 (label數, 點數)，建構端傳入
        self.rad_response = tuple(rad_response) if rad_response else None        #? 方向圖頭形狀 (n_phi, n_theta)，選用
        self.size_converter = size_converter         #? 在 (B,N) 攤平 / (B,H,W) 影像 / 批次維度間轉換的工具

        self.min_loss = min_loss     #? train_one_data 預設收斂門檻 (建構時顯式傳入)
        self.max_epoch = max_epoch   #? train_one_data 預設迭代上限
        self.epoch = 0  #? 累計被 GEN 呼叫推論的次數 (僅作計數，不影響訓練)

    #? 推論入口：GEN 生成 pattern → 餵入 SM 骨幹 → 包成 MultiResponses。
    #? 回傳物件帶梯度，GEN 對其 .criterion() 算 loss 後 backward，梯度即經 SM 反傳回 GEN。
    def __call__(self, pattern) -> MultiResponses:
        self.epoch += 1
        return MultiResponses(self.model(pattern))

    def train_by_datas(self, dataset:Dataset, epochs: int = 100, batch_size: Optional[int] = None, *, verbose:bool = True) -> List[float]:
        """
        Train the model using the provided dataset.

        Args:
            dataset (Dataset): 訓練資料集 (SampleStore / 舊 DataManager 皆可)。
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

        #? batch_size=None → DataLoader 一次吐一筆，一個 epoch = len(dataset) 步。
        #  外層只有 epoch 級 bar 時，大資料集 (如 harvest_single 2 萬餘筆) 會讓外層長時間卡在
        #  同一格、看起來像當掉 → 多掛一條「逐筆」內層 bar (leave=False，跑完即收) 顯示實際進度。
        n_steps = len(dataloader)
        epoch_bar = tqdm(range(epochs), desc='Training...', disable=not verbose, **TQDM_CONFIG)
        for epoch in epoch_bar:
            batch_bar = tqdm(dataloader, total=n_steps, desc=f"  epoch {epoch + 1}/{epochs}",
                             leave=False, disable=not verbose, **TQDM_CONFIG)
            for n, (patterns, real_responses) in enumerate(batch_bar):

                #? 統一形狀：pattern 攤平成 (B, 625) 餵 MLP；響應保留 (B, C, L) 不攤平 (與骨幹輸出對齊)
                patterns = self.size_converter(AntennaPattern, patterns, flatten=True, batch=True)
                real_responses = self.size_converter(AntennaResponse, real_responses, flatten=False, batch=True)

                inputs:Tensor = patterns.flatten(start_dim=1).to(config.device)   #? SM 輸入：pattern
                labels:Tensor = real_responses.to(config.device)                  #? SM 目標：HFSS 真實響應

                self.optimizer.zero_grad()
                outputs: Tensor = self.model(inputs)            #? SM 預測響應
                #? forward 把 batch 維 reshape 進 num_response → outputs 可能少一個 batch 維 (如 (2,17))，
                #  而 labels 經 size_converter(batch=True) 帶 batch 維 (如 (1,2,17))。對齊形狀再算 MSE，
                #  消除 (2,17) vs (1,2,17) 的廣播警告 (純形狀對齊、值不變)。
                loss: Tensor = self.criterion(outputs.reshape(labels.shape), labels)  #? 回歸誤差 (MSE)

                if not torch.isfinite(loss):     #! NaN/inf 防護網：跳過壞 batch，不讓一筆壞資料炸掉整批訓練
                    logger.warning(f"train_by_datas: loss 非有限 ({loss.item()})，跳過此 batch")
                    continue
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

        min_loss = min_loss or self.min_loss    #? 收斂門檻：loss 降到此值以下即停 (未傳入則用建構時的預設)
        max_epoch = max_epoch or self.max_epoch #? 迭代上限：避免某些難擬合的單筆資料無限迭代

        epoch_bar = tqdm(
            total=max_epoch, desc="Training one data",
            bar_format=TQDM_BAR_SIMPLE, disable=not verbose, **TQDM_CONFIG
        )
        #? 停止條件：loss 仍高於門檻「且」尚未到達迭代上限 → 任一不滿足就跳出 (達標或超時)
        while self.record('loss', 0) > min_loss and self.record('epoch', float('inf')) < max_epoch:
            self.optimizer.zero_grad()

            outputs_result:Tensor = self.model(input)  #? SM 對這筆 pattern 的當前預測

            #? reshape 成 (-1, *response_shape) 讓預測與目標形狀對齊，再算回歸誤差
            loss:Tensor = self.criterion(
                outputs_result.reshape(-1, *self.response_shape),
                label.reshape(-1, *self.response_shape)
            )

            if not torch.isfinite(loss):     #! NaN/inf 防護網：壞掉的一步不反傳，避免炸掉整個 HFSS run
                logger.warning(f"train_one_data: loss 非有限 ({loss.item()})，跳過此筆 SM 更新")
                break
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

    #? 方向圖推論入口 (選用)：pattern → 方向圖預測 (可微，供 beam_coverage_loss / GEN 反傳)。
    def rad_predict(self, pattern) -> Tensor:
        """pattern → 方向圖預測 (n_phi, n_theta)，可微。需 SM 建有方向圖頭。"""
        return self.model.forward_rad(pattern)

    def train_one_data_rad(self, pattern: Tensor, rad_response: Tensor, min_loss=None, max_epoch=None,
                           *, freeze_trunk: bool = True, verbose: bool = True):
        """
        單筆線上訓練「方向圖頭」(對應 train_one_data，但走 forward_rad / 方向圖 label)。

        與 train_one_data 對稱：每跑一次真實 HFSS 取得方向圖，就用該筆 (pattern, 真實方向圖)
        把方向圖頭訓到收斂。

        freeze_trunk (預設 True)：只更新 head_rad、凍住共用 backbone (fc_patch)。
          → 方向圖頭再不穩也「碰不到」S11/Gain 的 backbone，避免隨機 rad 頭把 trunk 帶歪、
            連累 S11/Gain 訓練爆 NaN (踩過的雷)。False 才放梯度回 backbone (兩者互相牽動)。
        """
        if self.rad_response is None:
            raise RuntimeError("此 SM 未建方向圖頭 (rad_response=None)，不能 train_one_data_rad。")
        self.requires_grad(True, train=True)
        #? 凍 trunk：把 head_rad 以外 (fc_patch) 的參數設 requires_grad=False，
        #  Ranger 對 grad=None 的參數會自動跳過 (見 optim/ranger.py) → 只更新 head_rad。
        trunk = [p for n, p in self.model.named_parameters() if not n.startswith("head_rad")]
        if freeze_trunk:
            for p in trunk:
                p.requires_grad_(False)
        self.record.reset()
        self.record['loss'] = float('inf')
        self.record['epoch'] = 0

        input = tensor(pattern, requires_grad=True)
        label = tensor(rad_response, requires_grad=True)

        min_loss = min_loss or self.min_loss
        max_epoch = max_epoch or self.max_epoch

        epoch_bar = tqdm(
            total=max_epoch, desc="Training one rad data",
            bar_format=TQDM_BAR_SIMPLE, disable=not verbose, **TQDM_CONFIG
        )
        try:
            while self.record('loss', 0) > min_loss and self.record('epoch', float('inf')) < max_epoch:
                self.optimizer.zero_grad()
                outputs_result: Tensor = self.model.forward_rad(input)   #? 走方向圖頭 (trunk 凍與否由 freeze_trunk)
                loss: Tensor = self.criterion(
                    outputs_result.reshape(-1, *self.rad_response),
                    label.reshape(-1, *self.rad_response)
                )
                if not torch.isfinite(loss):     #! NaN/inf 防護網：壞掉的一步不反傳
                    logger.warning(f"train_one_data_rad: loss 非有限 ({loss.item()})，跳過此筆方向圖更新")
                    break
                loss.backward()
                self.step(scheduler_param=loss)
                self.record['loss'] = loss.item()
                self.record.add('epoch', 1)
                epoch_bar.update()
                epoch_bar.set_postfix({'loss': f"{self.record('loss'):.2f}/{min_loss}"})
        finally:
            if freeze_trunk:                     #? 還原 trunk 的 requires_grad，避免影響後續 S11/Gain 訓練
                for p in trunk:
                    p.requires_grad_(True)

        self.model.eval()
        return self.record['loss']

###* HFSSNet — 純 MLP 骨幹 (學長版 MLPSurrogate 實際採用的網路) ###
#? 角色：SM 的「身體」(SurrogateModel 是外殼，HFSSNet 是被包住的 model)。
#? 結構最簡：把攤平的 625 個像素，經一連串全連接層 (逐步收斂的瓶頸 2048→1024→512→128→64)，
#? 直接映射到 (3,17) 響應，不利用像素的 2D 空間鄰接資訊 (那是下方 U-Net 版才做的)。
#? 用 PReLU 而非 ReLU：負區間保留可學斜率，避免 dead neuron、利於回歸這種需細緻擬合的任務。
class HFSSNet(nn.Module):

    #? num_pattern_pixel: 輸入像素數 (25x25=625)；num_response: 輸出響應形狀 (通道數 x 頻點數)
    #? rad_response: 選用的「方向圖頭」輸出形狀 (n_phi, n_theta)，None=不建方向圖頭 (與原架構相同)。
    #? rad_n_basis: 方向圖頭的平滑基底數 K (僅 rad_response 有給時生效)；K 個 cosine 係數展開成 n_theta 點。
    def __init__(self, num_pattern_pixel = 625, num_response:tuple = (3, 17), hidden=(2048, 1024, 512, 128, 64),
                 rad_response:Optional[tuple] = None, rad_n_basis:int = 16):
        super(HFSSNet, self).__init__()
        self.num_response = num_response
        self.num_pattern_pixel = num_pattern_pixel
        self.rad_response = tuple(rad_response) if rad_response else None

        #? hidden 由 config 指定 (預設 (2048,1024,512,128,64) 與原架構完全相同)。每層 Linear→PReLU，
        #? 末層 Linear 無激活 (回歸輸出)，維度 = 響應總元素數 (如 3*17=51)。
        layers, prev = [], num_pattern_pixel
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.PReLU()]
            prev = h
        layers += [nn.Linear(prev, num_response[0] * num_response[1])]
        self.fc_patch = nn.Sequential(*layers)
        #? 方向圖頭 (選用)：共用 backbone = fc_patch「末層 Linear 之前」的 64 維特徵，再接一層 Linear。
        #! 刻意在 fc_patch 之後才建 head_rad：從零建構時 freq 參數 (fc_patch) 的 RNG 抽取序列「完全不變」
        #! → 純頻率模型 byte-identical、golden 零漂移；fc_patch 名稱/結構也不動 → 舊 sm.pth 零 remap。
        #? 【平滑基底頭】head_rad 不直接吐 n_theta 個獨立值 (裸 Linear 無平滑先驗 + 凍 trunk 下擬不到收斂 → 鋸齒)，
        #? 改吐 K=rad_n_basis 個「cosine 基底係數」，再乘固定基底 B 展開成 n_theta 點：
        #?   pred = coeffs(n_phi,K) @ B(K,n_theta) → band-limited、結構上必平滑，且只擬 K 個數 → 收斂快。
        #? B 是不可訓的 buffer，且用 torch.cos/arange 建構不吃 RNG → fc_patch RNG 序列不變、golden 零漂移。
        if self.rad_response:
            n_phi, n_theta = self.rad_response
            self.rad_n_basis = max(1, min(rad_n_basis, n_theta))   # 基底數 ≤ 角度點數 (band-limit 才平滑)
            self.head_rad = nn.Linear(prev, n_phi * self.rad_n_basis)
            #? 預設基底建在 [-180,180] 等分網格 (整 run 第一次拿到真 θ 後由 set_rad_theta 重建對齊)。
            self.register_buffer(
                "rad_basis",
                self._build_cos_basis(torch.linspace(-180.0, 180.0, n_theta), self.rad_n_basis),
            )
        else:
            self.rad_n_basis = None
            self.head_rad = None
        self.to(config.device)

    @staticmethod
    def _build_cos_basis(theta: Tensor, n_basis: int) -> Tensor:
        """固定 cosine 基底 B (n_basis, n_theta)：B[k,i]=cos(k·φ_i)，φ_i 由 θ_i 線性映射到 [0,π]。
        任一係數組合 coeffs@B 在 θ 上都平滑 (band-limited)；k=0 為常數項 (承載整體增益準位)。
        逐欄由各自 θ_i 計算 → θ 即使是 HFSS 匯出序 (未排序) 也對位正確 (排序後必平滑)。"""
        theta = theta.reshape(-1).float()
        tmin, tmax = float(theta.min()), float(theta.max())
        span = (tmax - tmin) or 1.0                         # 防全等/單點 → 除零
        phi = math.pi * (theta - tmin) / span               # → [0, π]
        k = torch.arange(n_basis, dtype=theta.dtype).reshape(-1, 1)  # (K,1)
        return torch.cos(k * phi.reshape(1, -1))            # (n_basis, n_theta)

    def set_rad_theta(self, theta: Tensor):
        """用實際 HFSS θ 網格 (整個 run 固定) 重建平滑基底，使 forward_rad 的第 i 點對齊 θ_i。
        θ 可為 HFSS 匯出序 (未排序)：基底逐欄獨立算 → 對位正確、與順序無關。無方向圖頭則略過。"""
        if self.head_rad is None:
            return
        self.rad_basis = self._build_cos_basis(
            theta.detach().to("cpu"), self.rad_n_basis
        ).to(self.rad_basis.device, self.rad_basis.dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}(num_pattern_pixel={self.num_pattern_pixel}, num_response={self.num_response}"

    def forward(self, input):
        x = self.fc_patch(input)
        x = x.reshape(self.num_response)  #? 把攤平輸出還原成 (3,17) 響應形狀 (只回 freq，與原樣相同)
        return x

    def forward_rad(self, input):
        #? 方向圖預測：共用 backbone (fc_patch 除最後一層 Linear 外的所有層 → 64 維特徵) → head_rad
        #? 出 K 個 cosine 係數 → 乘固定基底 B 展開成 (n_phi, n_theta)。輸出 band-limited → 平滑 (不再鋸齒)。
        #? 梯度可經 B (固定、不可訓) 流回 head_rad / backbone；trunk 凍與否由 train_one_data_rad 控制。
        if self.head_rad is None:
            raise RuntimeError("此 HFSSNet 未建方向圖頭 (rad_response=None)，不能呼叫 forward_rad。")
        feat = input
        for layer in self.fc_patch[:-1]:        #? 末層 Linear 之前 = 共用 backbone 的 64 維特徵
            feat = layer(feat)
        coeffs = self.head_rad(feat).reshape(self.rad_response[0], self.rad_n_basis)  # (n_phi, K)
        return coeffs @ self.rad_basis      # (n_phi, n_theta)：matmul 即定形，n_theta 跟著實際基底走

###* MLPSurrogate — 工廠：學長版 SM (HFSSNet + Ranger + MSE + ReduceLROnPlateau) ###
#? 訓練腳本 (train_single.py / train_dual.py) 實際使用的就是這個工廠 (見各腳本 from ... import MLPSurrogate)。
#? 輕量穩定：純 MLP 骨幹 + MSE 回歸 + 「loss 卡住就降 lr」的 ReduceLROnPlateau，
#? 適合 SM 這種需頻繁線上微調、且每筆資料都要快速收斂的場景。
def MLPSurrogate(checkpoint, in_dim, response_shape, hidden=(2048, 1024, 512, 128, 64),
          lr=0.001, min_loss=0.1, max_epoch=20000, rad_response=None, rad_n_basis=16):
    """
    學長的做法。維度與超參數 (lr / 單筆訓練門檻) 全由訓練端顯式傳入，
    不讀全域 config (對應 YAML 的 hfss 區段)。

    rad_response (選用)：方向圖頭輸出形狀 (n_phi, n_theta)；給了就多建一個方向圖頭，
    且其參數一併進 optimizer (trunk 不凍 → 方向圖梯度會更新共用 backbone)。None=與原樣相同。
    rad_n_basis：方向圖頭的平滑 cosine 基底數 K (對應 YAML radiation.n_basis，預設 16)。
    """
    model_ge = HFSSNet( # Pattern -> Response (+ 選用方向圖頭)
        in_dim, response_shape, hidden=hidden, rad_response=rad_response, rad_n_basis=rad_n_basis
    )
    criterion_ge = nn.MSELoss()  #? 均方誤差：SM 是響應回歸任務，懲罰大偏差
    optimizer_ge = Ranger(
        params=model_ge.parameters(), lr=lr   #? 含 head_rad (若有)：故方向圖訓練會同時更新 backbone
    )
    #? 高原排程：當 loss 連續 patience 個 step 不再下降，就把 lr 砍半 (factor=0.5)，下限 min_lr，幫助收斂後期細修
    scheduler_ge = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ge, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    return SurrogateModel(
        model_ge, criterion_ge, optimizer_ge, scheduler_ge, rootdir=checkpoint,
        min_loss=min_loss, max_epoch=max_epoch, response_shape=response_shape,
        rad_response=rad_response,
    )

