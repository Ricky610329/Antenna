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

    def train_by_datas(self, dataset:Dataset, epochs: int = 100, batch_size: Optional[int] = None, *, min_loss: Optional[float] = None, verbose:bool = True, snapshot_epochs=None, early_stop: bool = True) -> List[float]:
        """
        Train the model using the provided dataset.

        Args:
            dataset (Dataset): 訓練資料集 (SampleStore / 舊 DataManager 皆可)。
            epochs (int): Total number of training cycles.
            min_loss (Optional[float]): 設了 → 某 epoch 平均 loss ≤ 此值即提前結束 (「訓到 fit」用,
                                        對應學長 DLF 的「訓到收斂」;None=不啟用、行為與原樣相同)。
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
        self._probe_snapshots = {}            #? 自適應訓練量：{epoch(1-indexed): 權重快照}；snapshot_epochs 給了才填

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

            if 'loss' not in self.record:                  #! 整個 epoch 的 batch 全非有限被 NaN guard 跳過 → 沒有任何
                continue                                   #  有效 loss → 跳過此 epoch (避免 average('loss') 撞 KeyError)
            avg_epoch_loss = self.record.average('loss')  #? 本 epoch 所有 batch 的平均 loss
            self.record.reset('loss', delete=True)        #? 清掉 batch 級暫存，下個 epoch 重新累積
            self.record['epoch_loss'] = avg_epoch_loss    #? 推進到 epoch 級曲線 (即最終回傳值)
            if snapshot_epochs and (epoch + 1) in snapshot_epochs:   #? 自適應：快照此刻權重 (供下一輪 held-out 探測訓練量)
                self._probe_snapshots[epoch + 1] = {k: v.detach().cpu().clone()
                                                    for k, v in self.model.state_dict().items()}

            epoch_bar.set_postfix({"Loss": f"{avg_epoch_loss:.4e}"})

            #? 「訓到 fit」提前結束：平均 loss 已達門檻 → 停 (DLF 訓到收斂用；min_loss=None 不啟用、原樣)。
            if min_loss is not None and avg_epoch_loss <= min_loss:
                if verbose: logger.success(f'Fit reached (loss {avg_epoch_loss:.4e} ≤ {min_loss}) at epoch {epoch + 1}.')
                break

            #? 早停：若最近 epochs/2 個 epoch 的 epoch_loss 都沒比之前更好，就提前結束 (省時，避免過擬合)。
            #  early_stop=False (自適應訓練量) → 關掉，確保訓滿 epochs、所有排定的快照都拿得到。
            if early_stop and self.record.early_stop('epoch_loss', int(epochs / 2)):
                logger.success(f'Early Stopping triggered at epoch {epoch + 1}!')
                break

        self.model.eval()  #? 訓練結束切回 eval，後續 GEN 推論時 BatchNorm/Dropout 維持確定性
        #? 回傳每 epoch 平均 loss 清單；若每個 epoch 都全 NaN 被跳過 (epoch_loss 從未寫入) → 回空清單,
        #  不讓 record['epoch_loss'] 撞 KeyError (呼叫端都容忍空回傳)。
        return self.record['epoch_loss'] if 'epoch_loss' in self.record else []

    def eval_snapshot(self, state_dict, pattern: Tensor, real_response: Tensor) -> float:
        """把一份權重快照載進「暫存複本」(不碰線上 self.model)、對這張 pattern 預測、回傳與真實響應的 MSE。
        供自適應訓練量控制器做 held-out 泛化評估。複本 lazy 深拷貝一次、之後每次載入重用 (省事、不重建架構)。"""
        if getattr(self, "_scratch", None) is None:
            import copy
            self._scratch = copy.deepcopy(self.model)
        self._scratch.load_state_dict(state_dict)
        self._scratch.eval()
        with torch.no_grad():
            pred = self._scratch(tensor(pattern))
            err = self.criterion(pred.reshape(-1, *self.response_shape),
                                 tensor(real_response).reshape(-1, *self.response_shape))
        return float(err)

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
        return self.record['loss'][1:]  #? 回傳逐步 loss 清單 (去掉 index0 的 inf 初值；末位=收斂後 loss)

    #? 方向圖推論入口 (選用)：pattern → 方向圖預測 (可微，供 beam_coverage_loss / GEN 反傳)。
    def rad_predict(self, pattern) -> Tensor:
        """pattern → 方向圖預測 (n_phi, n_theta)，可微。需 SM 建有方向圖頭。"""
        return self.model.forward_rad(pattern)

    #? 對齊方向圖頭的角度基底 (整個 run 第一次拿到真 θ 網格時呼叫)；委派給骨幹 (無方向圖頭則 no-op)。
    #  放在 SM 介面層 → 集成 (EnsembleSurrogate) 能以同一介面對所有成員 fan-out (見 set_rad_theta)。
    def set_rad_theta(self, theta):
        if hasattr(self.model, "set_rad_theta"):
            self.model.set_rad_theta(theta)

    #? 把線上更新的 lr 拉回「建構值」，並重置 plateau 排程器的內部狀態。
    #  用途：offline 預訓練 / 暖身 / strict 暖啟動會把 ReduceLROnPlateau 的 lr 砍到 min_lr，或從 checkpoint
    #  繼承塌掉的 lr → 線上 train_one_data 幾乎不更新 (每筆撞滿 max_epoch 還收斂不了)。只重置「步長 (lr)」
    #  與排程器狀態，不動動量/二階矩 (那是 warm optimizer 防過衝的本錢)。prepare_models 在載入策略尾端呼叫。
    def reset_online_lr(self):
        base_lr = self.optimizer.defaults.get("lr") if self.optimizer is not None else None
        if base_lr is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = base_lr
        if self.scheduler is not None and hasattr(self.scheduler, "_reset"):
            self.scheduler._reset()       # ReduceLROnPlateau：清掉 best / num_bad_epochs / cooldown

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
        return self.record['loss'][1:]  #? 同 train_one_data：去掉 index0 的 inf 初值 (末位=方向圖頭擬合 loss)

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
        #? 1-D 輸入 (單張 pattern)：還原成 num_response (3,17)，與原樣 byte-identical (golden 不動)。
        #? 2-D 輸入 (K, N) batch：保留批次維 → (K, *num_response)，供同批多候選平行推論。
        if input.dim() == 1:
            return x.reshape(self.num_response)
        return x.reshape(input.shape[0], *self.num_response)

    def forward_rad(self, input):
        #? 方向圖預測：共用 backbone (fc_patch 除最後一層 Linear 外的所有層 → 64 維特徵) → head_rad
        #? 出 K 個 cosine 係數 → 乘固定基底 B 展開成 (n_phi, n_theta)。輸出 band-limited → 平滑 (不再鋸齒)。
        #? 梯度可經 B (固定、不可訓) 流回 head_rad / backbone；trunk 凍與否由 train_one_data_rad 控制。
        if self.head_rad is None:
            raise RuntimeError("此 HFSSNet 未建方向圖頭 (rad_response=None)，不能呼叫 forward_rad。")
        feat = input
        for layer in self.fc_patch[:-1]:        #? 末層 Linear 之前 = 共用 backbone 的 64 維特徵
            feat = layer(feat)
        n_phi = self.rad_response[0]
        #? 1-D 輸入：(n_phi, K) → @B → (n_phi, n_theta)，與原樣 byte-identical (golden 不動)。
        #? 2-D 輸入 (B, N)：保留批次維 (B, n_phi, K) → @B → (B, n_phi, n_theta)，供同批多候選。
        if input.dim() == 1:
            coeffs = self.head_rad(feat).reshape(n_phi, self.rad_n_basis)            # (n_phi, K)
        else:
            coeffs = self.head_rad(feat).reshape(input.shape[0], n_phi, self.rad_n_basis)  # (B, n_phi, K)
        return coeffs @ self.rad_basis      # (..., n_theta)：matmul 即定形，n_theta 跟著實際基底走

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


###* EnsembleSurrogate — K 個獨立 SM 成員的集成 (不確定性 = 成員分歧) ###
#? 動機 (攻 SM 品質)：SM-guided 搜尋的命門是「SM 在它說好的地方準不準」。單一 SM 給不出
#? 「我對這張 pattern 有多沒把握」；deep ensembles 用 K 個獨立成員的「預測分歧」當便宜的
#? epistemic 不確定性 proxy —— 資料密處成員收斂趨同 (分歧小)、資料稀疏/外插處彼此發散 (分歧大)。
#? 這個分歧 (花 HFSS 前就能算) 用來：① guidance loss 的信任懲罰 λ_trust·u(x)(把候選推離自己沒
#? 把握的洞)；② acquisition (挑送 HFSS 的候選)。資料缺乏不是障礙：成員「共享全部資料」(不切分)，
#? 多樣性來自不同 init + 暖啟動擾動 + 不同洗牌序 (文獻：random init 已足夠，小資料別 bootstrap)。
#?
#? 設計 (低風險)：每個成員＝完整 MLPSurrogate (自有 optimizer/scheduler/checkpoint)，訓練/暖啟動/
#? 存讀「全部委派給既有、已測過」的單模型路徑 → 集成只做 fan-out (對每個成員) + 聚合 (mean/std)，
#? 不重寫訓練迴圈。推論回傳成員平均 (guidance 用、可微)；uncertainty 回傳成員間標準差 (可微)。
class EnsembleSurrogate:
    """K 個獨立 SurrogateModel 成員的集成。介面與 SurrogateModel 對齊
    (__call__ / train_one_data / train_by_datas / rad_* / save / load / pre_load_model /
    requires_grad / set_rad_theta) + 多一個 uncertainty()；所有訓練/暖啟動委派給各成員。"""

    def __init__(self, members: List[SurrogateModel], init_perturb: float = 0.02):
        if len(members) < 2:
            raise ValueError(f"EnsembleSurrogate 需 ≥2 個成員 (得到 {len(members)})")
        self.members = members
        self._init_perturb = float(init_perturb)
        #! init_perturb<=0 + 暖啟動 → 所有成員載同一檔、權重逐位元相同 → uncertainty 恆 0 (信任懲罰/
        #  acquisition 全失效，且不會被 has_unc 偵測到)。明確示警，避免「以為開了 ensemble 其實退化成單一 SM」。
        if self._init_perturb <= 0:
            logger.warning("EnsembleSurrogate init_perturb<=0 → 暖啟動後成員權重相同、uncertainty 恆 0 "
                           "(信任懲罰/acquisition 失效)。要 ensemble 不確定性請設 init_perturb>0。")
        #? 成員存到不同檔名 (sm0.pth .. smK-1.pth)，避免共用 'sm.pth' 互相覆蓋。
        for i, m in enumerate(self.members):
            m.name = f"sm{i}"
        #? 對齊單模型介面屬性 (呼叫端會讀)：響應形狀 / 方向圖頭形狀 / 單筆收斂門檻 (取成員 0)。
        self.response_shape = members[0].response_shape
        self.rad_response = members[0].rad_response
        self.min_loss = members[0].min_loss
        self.max_epoch = members[0].max_epoch

    #? 推論 (guidance/inference)：成員預測「平均」→ MultiResponses。可微 (梯度經各成員流回 pattern)。
    def __call__(self, pattern) -> MultiResponses:
        preds = torch.stack([m.model(pattern) for m in self.members])   # (K, *resp)
        return MultiResponses(preds.mean(0))

    #? 不確定性 (epistemic proxy)：成員預測的標準差，對響應元素取平均 → 純量。可微 (供信任懲罰反傳)。
    def uncertainty(self, pattern) -> Tensor:
        preds = torch.stack([m.model(pattern) for m in self.members])   # (K, *resp)
        return preds.std(dim=0).mean()

    def requires_grad(self, mode: bool = True, train: Optional[bool] = None):
        for m in self.members:
            m.requires_grad(mode, train)
        return mode

    def train_one_data(self, pattern, real_response, min_loss=None, max_epoch=None, *, verbose: bool = True):
        #? fan-out：每個成員各自把這一筆擬到收斂 (各自 loss / optimizer) → 共同學會這個已見點。
        out = []
        for m in self.members:
            out = m.train_one_data(pattern, real_response, min_loss=min_loss, max_epoch=max_epoch, verbose=verbose)
        return out   # 回傳最後一成員的逐步 loss (僅監控；呼叫端多半忽略)

    def train_by_datas(self, dataset, epochs: int = 100, batch_size=None, *, min_loss=None, verbose: bool = True,
                       snapshot_epochs=None, early_stop: bool = True):
        #? fan-out：每個成員各自重訓整批。多樣性主要來自「不同 init + 暖啟動擾動」(文獻結論)；
        #  洗牌序差異是次要 (各成員 new 一個 Generator、序可能相同)，不靠它撐分歧。
        #? snapshot_epochs 只傳 member0：探測訓練量用它一條軌跡代表全體 (各成員同架構同資料、只差 init 擾動)。
        out = []
        for i, m in enumerate(self.members):
            out = m.train_by_datas(dataset, epochs=epochs, batch_size=batch_size, min_loss=min_loss,
                                   verbose=verbose, snapshot_epochs=(snapshot_epochs if i == 0 else None),
                                   early_stop=early_stop)
        self._probe_snapshots = getattr(self.members[0], "_probe_snapshots", {})
        return out

    def eval_snapshot(self, state_dict, pattern, real_response) -> float:
        #? held-out 泛化評估委派給 member0 (探測用的快照就是它存的)。
        return self.members[0].eval_snapshot(state_dict, pattern, real_response)

    def rad_predict(self, pattern) -> Tensor:
        return torch.stack([m.rad_predict(pattern) for m in self.members]).mean(0)

    def train_one_data_rad(self, pattern, rad_response, min_loss=None, max_epoch=None,
                           *, freeze_trunk: bool = True, verbose: bool = True):
        out = []
        for m in self.members:
            out = m.train_one_data_rad(pattern, rad_response, min_loss=min_loss, max_epoch=max_epoch,
                                       freeze_trunk=freeze_trunk, verbose=verbose)
        return out

    def set_rad_theta(self, theta):
        for m in self.members:
            m.set_rad_theta(theta)

    def reset_online_lr(self):
        for m in self.members:
            m.reset_online_lr()

    def save(self):
        for m in self.members:
            m.save()

    def load(self, force: bool = False):
        for m in self.members:
            m.load(force=force)

    def pre_load_model(self, path, strict: bool = True):
        #? 暖啟動：所有成員從同一個預訓練檔載入 (含 optimizer 暖啟動，沿用單模型已測邏輯)，再對
        #  「成員 1..K-1」加微小權重擾動 → 成員從同一最優點附近「岔開」，使後續分歧反映 epistemic
        #  不確定性 (成員 0 保持精確預訓練作錨)。擾動 = 各參數張量 std × init_perturb 的高斯噪聲。
        for i, m in enumerate(self.members):
            m.pre_load_model(path, strict=strict)
            if i > 0 and self._init_perturb > 0:
                with torch.no_grad():
                    for p in m.model.parameters():
                        if p.numel() > 1:        # numel==1 (如 PReLU 單一斜率) → std() dof<=0；跳過、不影響多樣性
                            p.add_(torch.randn_like(p) * (p.detach().std() * self._init_perturb))
                            #! 同步 Lookahead 慢權重錨：暖啟動載回的 Ranger slow_buffer 是「未擾動的共同錨」，
                            #  Ranger 每 k 步 (slow←slow+α(p−slow); p←slow) 會把權重拉回它 → 吃掉 init_perturb
                            #  的多樣性 → uncertainty 邊跑邊塌。把 slow_buffer 一併設成擾動後的權重，讓各成員
                            #  從自己的擾動點出發、不被拉回共同錨。(無 state 的參數如 head_rad 由 Ranger 首步
                            #  lazy-init slow_buffer=擾動後 p，本就正確 → 只需處理已暖啟動 optimizer state 的 trunk。)
                            st = m.optimizer.state.get(p)
                            if st is not None and "slow_buffer" in st:
                                st["slow_buffer"].copy_(p.detach())


###* EnsembleMLPSurrogate — 工廠：建 K 個 MLPSurrogate 成員、包成 EnsembleSurrogate ###
#? zoo 名字 "ensemble"。簽名與 MLPSurrogate 對齊 (build_surrogate 共用呼叫)，多 ensemble_size /
#? init_perturb 兩個架構參數 (對應 YAML surrogate.ensemble_size / surrogate.init_perturb)。
def EnsembleMLPSurrogate(checkpoint, in_dim, response_shape, hidden=(2048, 1024, 512, 128, 64),
                         lr=0.001, min_loss=0.1, max_epoch=20000, rad_response=None, rad_n_basis=16,
                         ensemble_size=5, init_perturb=0.02):
    members = [
        MLPSurrogate(checkpoint, in_dim, response_shape, hidden=hidden, lr=lr,
                     min_loss=min_loss, max_epoch=max_epoch, rad_response=rad_response, rad_n_basis=rad_n_basis)
        for _ in range(int(ensemble_size))
    ]
    return EnsembleSurrogate(members, init_perturb=init_perturb)

