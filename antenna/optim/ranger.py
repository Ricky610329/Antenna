# Ranger deep learning optimizer - RAdam + Lookahead + Gradient Centralization, combined into one optimizer.

# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
# and/or
# https://github.com/lessw2020/Best-Deep-Learning-Optimizers

# Ranger has now been used to capture 12 records on the FastAI leaderboard.

# This version = 20.4.11

# Credits:
# Gradient Centralization --> https://arxiv.org/abs/2004.01461v2 (a new optimization technique for DNNs), github:  https://github.com/Yonghongwei/Gradient-Centralization
# RAdam -->  https://github.com/LiyuanLucasLiu/RAdam
# Lookahead --> rewritten by lessw2020, but big thanks to Github @LonePatient and @RWightman for ideas from their code.
# Lookahead paper --> MZhang,G Hinton  https://arxiv.org/abs/1907.08610

# summary of changes:
# 4/11/20 - add gradient centralization option.  Set new testing benchmark for accuracy with it, toggle with use_gc flag at init.
# full code integration with all updates at param level instead of group, moves slow weights into state dict (from generic weights),
# supports group learning rates (thanks @SHolderbach), fixes sporadic load from saved model issues.
# changes 8/31/19 - fix references to *self*.N_sma_threshold;
# changed eps to 1e-5 as better default than 1e-8.

###* ============================================================
###* Ranger 優化器 — 移植自 lessw2020/Ranger-Deep-Learning-Optimizer
###*
###* 組成三層：
###*   1. RAdam  (Rectified Adam)
###*      ─ 標準 Adam 在訓練初期因二階矩估計樣本數不足，方差極不穩定。
###*        RAdam 根據當前步數推算「簡單移動平均長度 N_sma」，
###*        當 N_sma > 閾值 (預設 5) 時才啟用方差修正縮放，
###*        否則退化為帶偏差修正的 SGD + 動量，避免初期跑飛。
###*   2. Lookahead
###*      ─ 每 k 步做一次「慢權重回看插值」：
###*        slow ← slow + α × (fast − slow)
###*        相當於在更新軌跡上取加權平均，壓低震盪，對超參數更不敏感。
###*   3. Gradient Centralization (GC)
###*      ─ 梯度減去其空間均值，讓每個濾波器/神經元的梯度零均值化，
###*        穩定梯度幅度，加快收斂並略具正則效果。
###*
###* 為何 SM（代理模型 MLPSurrogate / UNetSM）訓練選 Ranger：
###*   ─ 天線反向設計的訓練集規模偏小（小批量）且輸入雜訊較大，
###*     Adam 初期方差不穩定問題在此更為顯著；RAdam 修正此缺陷。
###*   ─ Lookahead 的慢權重機制對雜訊梯度有天然平滑效果，
###*     不需大量調超參數即可獲得穩定收斂，適合快速實驗迭代。
###*   ─ GC 對 UNet 卷積層梯度零均值化，可緩解深層特徵圖梯度漂移。
###* ============================================================

import math
import torch
from torch.optim.optimizer import Optimizer, required


class Ranger(Optimizer):
    """Ranger 優化器：RAdam + Lookahead + Gradient Centralization 的三合一實作。

    核心思路：
      - RAdam 在訓練初期保守（退回 SGD+動量），等二階矩估計足夠可靠後
        再轉為自適應更新，避免 Adam 初期學習率過大導致的不穩定。
      - Lookahead 在 RAdam（快權重）之上維護一組慢權重，每 k 步插值一次，
        相當於在最優解周圍做「回看探索」，顯著降低震盪。
      - GC 對維度 > gc_gradient_threshold 的梯度做空間均值去除，
        使梯度更均勻、收斂更穩定。
    """

    def __init__(self, params, lr=1e-3,                       # lr
                 alpha=0.5, k=6, N_sma_threshhold=5,           # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=0,  # Adam options
                 # Gradient centralization on or off, applied to conv layers only or conv + fc layers
                 use_gc=True, gc_conv_only=False
                 ):
        """初始化 Ranger 優化器。

        Args:
            params: 模型參數（同 PyTorch 標準 Optimizer）。
            lr: 快權重的學習率；Lookahead 慢權重插值速率由 alpha 控制，兩者分工不同。
            alpha: Lookahead 插值係數（0~1）。
                   alpha=0.5 表示慢權重每次走向快權重一半的距離；
                   值越大慢權重跟得越緊，平滑效果越弱。
            k: Lookahead 同步週期（步數）。
                每隔 k 步才做一次慢權重插值；k 越大探索空間越廣，
                但每次插值跳躍也越大。預設 6 為作者建議值。
            N_sma_threshhold: RAdam 啟用方差修正的最低 N_sma 門檻。
                              N_sma < 閾值時退化為 SGD+動量，避免初期方差爆炸。
            betas: (beta1, beta2)，分別為一階矩（動量）與二階矩的衰減係數。
                   beta1=0.95 比常見的 0.90 在作者測試中表現更好。
            eps: 數值穩定項，防止除以零；預設 1e-5 比 Adam 的 1e-8 更保守。
            weight_decay: L2 正則化係數（參數衰減）。
            use_gc: 是否啟用 Gradient Centralization。
            gc_conv_only: True → GC 僅作用於卷積層（grad.dim() > 3）；
                          False → 同時作用於全連接層（grad.dim() > 1）。
        """

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # parameter comments:
        # beta1 (momentum) of .95 seems to work better than .90...
        # N_sma_threshold of 5 seems better in testing than 4.
        # In both cases, worth testing on your dataset (.90 vs .95, 4 vs 5) to make sure which works best for you.

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]
        #? radam_buffer：快取最近 10 個不同步數對應的 (step, N_sma, step_size)。
        #? 因為同一 step 下所有參數共用相同的 N_sma/step_size，
        #? 使用 step % 10 作為索引，避免每個參數重複計算，節省開銷。

        # gc on or off
        self.use_gc = use_gc

        # level of gradient centralization
        self.gc_gradient_threshold = 3 if gc_conv_only else 1
        #? gc_conv_only=True  → threshold=3，只對 4D 及以上張量（卷積核 OIHW）做 GC；
        #? gc_conv_only=False → threshold=1，對 2D 及以上張量（含全連接 weight）都做 GC。

        # print(
        #     f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        # if (self.use_gc and self.gc_gradient_threshold == 1):
        #     print(f"GC applied to both conv and fc layers")
        # elif (self.use_gc and self.gc_gradient_threshold == 3):
        #     print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        # print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        """執行一步參數更新。

        整體流程（每個參數）：
          1. [GC]       梯度零均值化（可選）
          2. [RAdam]    更新一、二階矩指數移動平均
          3. [RAdam]    計算 N_sma，決定是否啟用方差修正縮放
          4. [L2]       Weight decay（可選）
          5. [RAdam]    根據 N_sma 選擇自適應或動量更新模式
          6. [Lookahead] 每 k 步做一次慢權重插值同步
        """
        loss = None
        # note - below is commented out b/c I have other work that passes back the loss as a float, and thus not a callable closure.
        # Uncomment if you need to use the actual closure...

        # if closure is not None:
        #loss = closure()

        # Evaluate averages and grad, update param tensors
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:  # if first time to run...init dictionary with our desired entries
                    # if self.first_run_check==0:
                    # self.first_run_check=1
                    #print("Initializing slow buffer...should not see this at load from saved model!")
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)      # 一階矩（動量）初始化為 0
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)   # 二階矩（方差估計）初始化為 0

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                    #? slow_buffer 儲存 Lookahead 的「慢權重」，初始值與快權重相同。
                    #? 將其放入 state dict 而非全域屬性，可正確支援 group 學習率
                    #? 以及從 checkpoint 載入時的狀態還原。

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                if grad.dim() > self.gc_gradient_threshold:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                #? Gradient Centralization（GC）：
                #? 對每個輸出通道（dim=0）以外的所有維度計算梯度均值，
                #? 再從梯度中減去該均值，使每個濾波器的梯度向量零均值化。
                #? 幾何意義：把梯度投影到超平面上，去除整體偏移分量，
                #? 讓不同通道的更新幅度更一致，有助於縮短收斂時間並具正則效果。

                state['step'] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # compute mean moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                #? 標準 Adam EMA 更新：
                #?   exp_avg    = β1 × exp_avg    + (1-β1) × grad        （一階矩）
                #?   exp_avg_sq = β2 × exp_avg_sq + (1-β2) × grad²       （二階矩）
                #? 此時尚未做偏差修正（bias correction），偏差修正折入後面的 step_size。

                buffered = self.radam_buffer[int(state['step'] % 10)]
                #? 用 step % 10 索引快取，若此步已算過則直接取結果，
                #? 避免同一 step 的多個參數重複執行下面的 sqrt/pow 計算。

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    #? N_sma_max：二階矩 EMA 在無限步時的「等效簡單移動平均長度」上界，
                    #? 由 β2 決定，β2=0.999 → N_sma_max ≈ 1999。
                    N_sma = N_sma_max - 2 * \
                        state['step'] * beta2_t / (1 - beta2_t)
                    #? N_sma：當前步數下二階矩的「有效樣本數」估計。
                    #? 訓練初期 step 小、beta2_t 接近 1，N_sma 遠小於 N_sma_max，
                    #? 代表方差估計的樣本不足，此時不宜使用自適應縮放。
                    #? 隨步數增加，N_sma 逐漸趨近 N_sma_max，估計才可靠。
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                            N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        #? RAdam 方差修正縮放因子（N_sma > 閾值時啟用）：
                        #?   分子 sqrt(...) 項 = 對二階矩方差的修正係數，
                        #?     (1-β2^t) 為偏差修正，其餘項修正有限樣本造成的方差偏移。
                        #?   除以 (1-β1^t) = 一階矩偏差修正（移回分母形式融入 step_size）。
                        #? 整體使 Adam 更新量在訓練初期不因方差估計不準而過大。
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                        #? N_sma 不足時，方差估計不可信，退化為 SGD + 動量：
                        #? 僅做一階矩偏差修正，不使用二階矩除法，保守更新。
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(p_data_fp32,
                                     alpha=-group['weight_decay'] * group['lr'])

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom,
                                         value=-step_size * group['lr'])
                    #? 自適應模式（RAdam 完整版）：
                    #?   p ← p - lr × step_size × exp_avg / (√exp_avg_sq + eps)
                    #? step_size 已包含方差修正與一階矩偏差修正，
                    #? denom 提供逐元素自適應縮放（類 Adam）。
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * group['lr'])
                    #? 退化模式（SGD + 動量）：
                    #?   p ← p - lr × step_size × exp_avg
                    #? 不除以二階矩，避免初期方差估計不穩定造成步長暴增。

                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    #? Lookahead 同步條件：每 k 個 RAdam 步才執行一次慢權重插值。
                    #? 在參數層級（而非 group 層級）執行，可正確支援不同 group 的學習率。

                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    #? Lookahead 插值：slow ← slow + α × (fast − slow)
                    #? 等價於：slow ← (1-α) × slow + α × fast
                    #? 慢權重沿快權重方向移動 α 比例，α=0.5 即取中點。
                    #? 這讓優化器在損失面上做「短程衝刺（fast）+ 長程修正（slow）」，
                    #? 對超參數敏感度低，在小資料集上尤為穩健。

                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)
                    #? 將慢權重同步回快權重，讓下一輪 RAdam 從插值後的位置重新出發，
                    #? 防止快權重越走越遠而慢權重追趕不及。

        return loss
