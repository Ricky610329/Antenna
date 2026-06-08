# -*- coding: utf-8 -*-
"""
================================================================================
train_single.py — 單埠 (Single-Port) 微帶貼片天線「反向設計」訓練腳本
================================================================================

【這個檔案在做什麼？】
    這是一個「反向設計 (Inverse Design)」的閉迴路訓練腳本。
    傳統流程是：先畫好天線結構 → 丟進 HFSS 模擬 → 看電磁響應。
    這裡反過來：先指定「想要的目標響應」(S11 要在 28GHz 凹下去、Gain 要拉高)，
    再讓一個神經網路「生成」能達成該響應的金屬像素圖樣 (pattern)。

【三個核心角色】 (詳見 antenna/ 各模組)
    1. GEN  生成器  (SigmoidGEN)：  目標響應 ─▶ 25x25 二元像素 pattern
                                    透過 STE 可微分二值化輸出 0/1 圖樣。
    2. SM   代理模型 (OldSM/HFSSNet)：pattern ─▶ 預測響應 (S11, Gain)
                                    一個「可微分」且「快」的網路，用來模仿昂貴的 HFSS。
    3. SIM  模擬器  (SinglePortSimulator)：pattern ─▶ 真實響應 (Ground Truth)
                                    透過 COM 驅動 Ansys HFSS 做真正的電磁模擬，慢但準確。

【為什麼要 SM？】
    HFSS 模擬不可微分 (無法 backward)，且每次要好幾分鐘。
    所以引入 SM 作為 HFSS 的「可微分替身」：
      - GEN 的梯度可以經由 SM 反向傳播 (GEN → pattern → SM → loss)，
        如此 GEN 才能被訓練，而不必對 HFSS 求導。
      - 每跑一次真實 HFSS，就用 (pattern, 真實響應) 線上 (online) 訓練 SM，
        讓 SM 在「當前 GEN 常產生的圖樣」附近越來越貼近真實 HFSS。

【每個 epoch 的閉迴路流程】
    (A) 早停判斷：若 real_loss 連續 patience 次沒進步 → 回滾 (rollback) GEN 到歷史最佳，
        並用 online_dataset 重訓 SM (試圖跳出局部最佳)。
    (B) GEN 依目標響應生成 pattern，並疊上固定的饋電金屬塊 lower。
    (C) 去重：若該 pattern 先前模擬過 → 直接取快取結果 (jump)，省下昂貴的 HFSS。
    (D) 否則跑 HFSS → 得真實響應與 real_loss → 用此單筆資料線上訓練 SM。
    (E) 更新歷史最小 loss / best epoch。
    (F) 更新 GEN：pattern 經 SM 得預測響應，算 loss (含 pattern 正則化) → backward → step。
    (G) 存檔、畫圖、記錄到 TEMP (可斷點續跑)。

【與 train_dual.py 的差異】
    本檔為「單埠」：1 個 port，響應為 S11 + Gain，損失用 custom_loss_minmax，
    只疊一塊 lower 饋電塊。train_dual.py 為「雙埠」：S11 + S21 + S22，用 interval_loss，
    疊 upper + lower 兩塊。

--------------------------------------------------------------------------------
Created on Wed May  8 16:38:05 2024
@author: user
"""
# antenna.utils 匯入了 config(全域設定)、Path、Figure、Record、Complete、
# connect_network_drive、get_local_ip、ROOTDIR、DATASET_PATH、logger、time 等工具。
from antenna.utils import *
config.device = "cpu"   # 強制使用 CPU (HFSS 模擬本身在 Windows COM 端跑，GEN/SM 很小，CPU 即可)

import numpy as np
import torch
# antenna 套件根：AntennaPattern(像素圖樣)、AntennaResponse(頻率響應)、MultiResponses、
#                 get_result_path、logger 等都在這裡。
from antenna import *
# functions.py：饋電連通度指標、自適應週期排程器、兩種「圖樣連通性」正則化損失。
from antenna.functions import FeedReachability, AdaptiveCyclicalScheduler, GapClosingLoss, SpectralConnectivityLoss
# models.py：Models(模型管理外殼，含存讀檔/換label/step)、OldGEN/SigmoidGEN(生成器)。
from antenna.models import (
    Models, OldGEN, SigmoidGEN
)
# patch：SinglePortSimulator(單埠 HFSS 模擬器)、custom_loss_minmax(單埠用的目標損失函數)。
from antenna.patch import (
    SinglePortSimulator, custom_loss_minmax
)
# smodels.py：OldSM(代理模型工廠，回傳包好的 SurrogateModel，內部是 HFSSNet)。
from antenna.smodels import OldSM
# data.py：DataManager(可持久化、可去重、可當 PyTorch Dataset 的資料集容器)。
from antenna.utils.data import DataManager
# from antenna.functions import mirror, mutate
torch.autograd.set_detect_anomaly(True)  # 開啟自動微分異常偵測 (除錯用，會變慢)

###* Basic Config ###
# MultiConfig：一組「具名實驗設定」。實際採用哪一組由命令列參數 sys.argv[1] 決定
# (例如 `python train_single.py 3` 會選用 '3' 這組，啟用 total_variation_loss=0.01)。
# 每組可覆寫：實驗名稱、各種正則化損失權重、排程器 on_plateau 策略等。
MULTICONFIG = MultiConfig(
    {
        '1': {
            'name': "[Patch-Single-{device}-{hash_id}] pixel_base_1"
        },
        '2': {
            'name': "[Patch-Single-{device}-{hash_id}] pixel_base_2"
        },

        #* 換不同 Base
        '3': {
            'name': "[Patch-Single-{device}-{hash_id}] pixel_base_1_total_variation_loss_01",
            'KuoHung': 'KuoHung-1',         # 指定 SM 暖身用的學長(KuoHung)參考圖樣
            "total_variation_loss": 0.01    # 啟用 Total Variation 損失 (抑制破碎圖樣)
        },
        '4': {
            'name': "[Patch-Single-{device}-{hash_id}] pixel_base_2_total_variation_loss_01",
            'KuoHung': 'KuoHung-2',
            "total_variation_loss": 0.01
        },

        #* on_plateau (排程器停滯時的重啟策略：linear=線性爬升 / peak=直接跳峰值)
        '5': {
            'name': "[Patch-Single-{device}-{hash_id}] pixel_base_on_plateau_linear",
            "on_plateau": "linear"
        },
        '6': {
            'name': "[Patch-Single-{device}-{hash_id}] pixel_base_on_plateau_peak",
            "on_plateau": "peak"
        },
        '7': {
            'name': "[Patch-Single-{device}-{hash_id}] pixel_base_linear_tv50",
            "total_variation_loss": 50,
            "on_plateau": "linear"
        },
        '8': {
            'name': "[Patch-Single-{device}-{hash_id}] pixel_base_linear_tv100",
            "island_suppression_loss": 100, # 啟用孤島抑制損失 (消除孤立噪點/孔洞)
            "on_plateau": "linear"
        },
        '9': {
            'name': "[Patch-Single-{device}-{hash_id}] pixel_base_linear_is100",
            "island_suppression_loss": 100,
        },
        '10': {
            'name': "[Patch-Single-{device}-{hash_id}] pixel_base_linear_is1",
            "island_suppression_loss": 1,
        },
    }
)
# 掛載 NAS 網路磁碟 (T:)，模擬結果與資料集都存到實驗室的 NAS。
connect_network_drive("T:", r"\\140.123.106.219\temp", "user", "ailab120") #save to nas
# check resume and setup save path
# error reminder
# 建立/取得本次實驗的結果資料夾。回傳 (結果路徑, 此資料夾是否已存在)。
#   - CONTINUE_RUN=True 代表資料夾已存在 → 視為「斷點續跑」。
#   - generate_code=__file__：把本腳本原始碼複製一份到結果夾 (實驗可重現)。
#   - enable_exception_handler=True：掛上全域例外處理 (出錯會寄 email 通知)。
RESULT_PATH, CONTINUE_RUN = get_result_path(
    MULTICONFIG('name', "[Patch-Single-{device}-{hash_id}] pixel_base"),
    rootdir = ROOTDIR, generate_code = __file__, enable_exception_handler = True
)

# DATASET_PATH = Path(r"T:\碩二_吳維文's\Patch Antenna\Experiment\dataset")

# 預訓練 SM 權重檔路徑：若存在，就直接載入當作 SM 的起點 (省去離線預訓練)。
SM_PRETRAIN_MODEL_PATH = DATASET_PATH.joinpath('old_sm.pth')
# TEMP：本次訓練的「持久化狀態 + 時序記錄器」(Record)。
#   - 以類似 DataFrame 的方式逐 epoch 記錄 real_loss/fake_loss/min_loss/pattern 緩存等。
#   - load=True：若續跑則載回先前進度，是斷點續跑的關鍵。
TEMP = Record("temp", rootdir=RESULT_PATH, load=True)
# sys.excepthook = global_exception_handlerc

path_pic = RESULT_PATH.joinpath("pic").not_exist_create()              # 每 epoch 結果圖輸出夾
path_checkpoint = RESULT_PATH.joinpath("checkpoint").not_exist_create()# 模型權重檔輸出夾
# 離線資料集：SM 的「預訓練」資料 (大量 pattern→真實響應 的歷史資料，含鏡像增強)。
data_manager = DataManager("patch_single_mirror", rootdir=DATASET_PATH)
# 線上資料集：本次訓練「邊跑邊收集」的 (pattern, 真實響應)，rollback 時用來重訓 SM。
online_dataset = DataManager("online", rootdir=RESULT_PATH)

# 把選定那組 MULTICONFIG 的鍵值合併進全域 config，並補上一些訓練超參數。
config.update(MULTICONFIG.get_label_data())
config['Name'] = RESULT_PATH.stem
config['File'] = __file__
config.setWarning()             # filterwarnings('ignore')：關掉警告訊息
config.epochs = 1000            # 主訓練迴圈總 epoch 數
config.lr = 0.005               # GEN 的 Adam 初始學習率
config.checkpoint_save_path = path_checkpoint

config['patience'] = 10         # 早停/回滾的耐心值 (real_loss 連續幾次沒進步就 rollback)
config['mutation_rate'] = 0.001 # (目前未啟用) pattern 突變比例
config['HFSS.lr'] = 0.001       # SM 的學習率 (OldSM 內部 Ranger 優化器使用)
config['HFSS.min_loss'] = 0.1   # SM 單筆訓練 train_one_data 的收斂門檻 (預設值)
config['HFSS.max_epoch'] = 20000# SM 單筆訓練的最大迭代上限 (預設值)

###* Set Antemma Pattern ###
# 設定設計區為 25x25 的像素網格 (座標 (x1,x2,y1,y2)=(0,25,0,25))。
AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
# lower：固定不變的「饋電金屬塊」。一塊 5x5 的全金屬 (torch.ones)，
#        放在座標 (x:10~15, y:20~25)，即網格底部中央 (單埠饋入點所在)。
#        每個 epoch 都會把 GEN 生成的圖樣疊上這塊 lower，確保有饋電連接。
lower = AntennaPattern(torch.ones((5, 5)), (10, 15, 20, 25))

# 建立單埠 HFSS 模擬器，並註冊給 AntennaPattern (之後 pattern.simulate() 會用它)。
simulator = SinglePortSimulator(
    record_path = RESULT_PATH,
)
AntennaPattern.register_simulator(simulator)


###* Set Antenna Response ###
# 註冊兩條要計算 loss 的響應曲線：S11 與 Gain；x 軸用 'n257' → 24~32GHz 取 17 點 (28GHz 為中心)。
AntennaResponse.registerLabels('S11', 'Gain', x = 'n257')
x = AntennaResponse.x()   # x = np.linspace(24, 32, 17)，後續畫圖的橫軸 (頻率)

#? S11 S22 -> high low high (-1.25, -12)
# 設計 S11 目標曲線：registerTargetResponse(side=0, center=-10, width=(5,0,7,0,5))。
#   width=(5,0,7,0,5) 表示 [5點維持side, 0點過渡, 7點維持center, 0點過渡, 5點維持side]，共17點。
#   形狀為「兩端高(0dB) / 中央低(-10dB)」→ 在 28GHz 形成凹陷，代表良好匹配/低反射。
returnloss = AntennaResponse.registerTargetResponse(0, -10, (5, 0, 7, 0, 5), label="S11")
# returnloss_upper = AntennaResponse.registerTargetResponse(0, -10, (4, 2, 5, 2, 4), label="returnloss_upper")
# returnloss_lower = AntennaResponse.registerTargetResponse(-2.5, -50, (3, 4, 3, 4, 3), label="returnloss_lower")

# 為 S11 註冊損失函數 custom_loss_minmax，method='low'：
#   只在「目標最低點 (中央 -10dB)」處，當預測值「高於」目標時才罰 (要求 S11 夠低即可，更低不罰)。
AntennaResponse.registerLossHook(custom_loss_minmax, label = "S11", target=returnloss, method='low')

#? Gain -> low high low (-2, -19.5) (0, -25)
# 設計 Gain 目標曲線：side=-19, center=4 → 「兩端低 / 中央高」，要求 28GHz 處增益高。
gain = AntennaResponse.registerTargetResponse(-19, 4, (5, 0, 7, 0, 5), label="Gain")
# gain_upper = AntennaResponse.registerTargetResponse(-17, 0, (1, 2, 11, 2, 1), label="gain_upper")
# gain_lower = AntennaResponse.registerTargetResponse(-22, -3, (4, 2, 5, 2, 4), label="gain_lower")

# 為 Gain 註冊損失，method='high'：只在「目標最高點 (中央 +4dB)」處，
#   當預測「低於」目標時才罰 (要求增益夠高即可，更高不罰)。
AntennaResponse.registerLossHook(custom_loss_minmax, label = "Gain", target=gain, method='high')

# 畫出兩條目標響應曲線並存檔 (Figure 是 matplotlib 的 with-context 包裝)。
with Figure('Target Response', (1, 2), rootdir=RESULT_PATH, save=True, size=(18*2, 9*2), default_axes_title_size=50, default_tick_size=40) as fig:
    fig.addAll()

    fig[0].set_title('S11')
    fig[0].plot(x, returnloss.cpu().detach().numpy(), color='red', marker="o")
    # fig[0].plot(x, returnloss_upper.cpu(), color='blue', marker="o")
    # fig[0].plot(x, returnloss_lower.cpu(), color='blue', marker="o")
    fig[0].grid(True)
    # fig[0].set_ylim(-13, 1)

    fig[1].set_title('Gain')
    fig[1].plot(x,gain.cpu().detach().numpy(), color='red', marker="o")
    # fig[1].plot(x, gain_upper.cpu(), color='blue', marker="o")
    # fig[1].plot(x, gain_lower.cpu(), color='blue', marker="o")
    fig[1].grid(True)


# ── 建立 GEN (生成器) 及其優化器、排程器 ───────────────────────────────────────
# SigmoidGEN：MLP(目標響應 → logits) 後接 STE 可微分二值化，輸出 0/1 的 25x25 pattern。
model = SigmoidGEN()
# Adam，betas=(0.5, 0.999)：較低的一階動量 (0.5) 是 GAN 類訓練常用設定，較不易震盪。
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=config.lr, betas=(0.5, 0.999)
)
# 自適應週期排程器：融合 OneCycle(暖身) + CosineAnnealingWarmRestarts(週期退火) + ReduceLROnPlateau。
#   同時調整「學習率 lr」與「二值化溫度 tau」(tau 越小越接近硬 0/1)。
#   停滯 (patience 步無改善) 時依 on_plateau 策略強制重啟，幫助跳出局部最佳。
scheduler = AdaptiveCyclicalScheduler(
    optimizer,
    T_0=100,                # 增加初始週期長度
    T_mult=1,               # 暫時關閉週期長度增加，讓每個週期條件一致
    lr_max=0.005,           # 稍微降低最大學習率
    lr_min=1e-6,            # 0.0001
    temp_max=4.0,           # 稍微降低最高溫度
    temp_min=0.1,
    warmup_ratio=0.2,       # 增加暖身時間
    patience=25,            # 顯著增加耐心
    factor=0.7,
    mode='min',
    on_plateau = MULTICONFIG("on_plateau", "linear") # TODO
)
# Models：把 model/optimizer/scheduler/criterion 包成統一外殼，
#   提供 .change(label)(換存檔名)、.save()/.load()、.step()、.requires_grad() 等。
#   name 含 "{label}" → 不同 epoch 會存成不同檔名 (用於 rollback 載回任一歷史 epoch)。
generator = Models(
    name = "generator_{label}",
    rootdir = path_checkpoint,
    model = model,
    optimizer = optimizer,
    scheduler = scheduler,
    criterion=custom_loss_minmax
)

# ── 建立 SM (代理模型) ─────────────────────────────────────────────────────────
# OldSM：學長版本，內部是 HFSSNet (純 MLP: 625像素 → (2,17)響應)，
#        優化器 Ranger、損失 MSE、排程器 ReduceLROnPlateau。
smodel = OldSM(checkpoint=config.checkpoint_save_path)

###* 斷點續跑 ###
if CONTINUE_RUN and ('epoch' in TEMP):
    # 情況 1：續跑。把 generator 換到上次的 epoch 標籤並載回權重，SM 也載回。
    generator.change(TEMP('epoch'), load=True)
    smodel.load()
elif SM_PRETRAIN_MODEL_PATH.exists():
    # 情況 2：首次跑，但已有預訓練 SM 權重 → 直接載入。
    smodel.pre_load_model(SM_PRETRAIN_MODEL_PATH)

    # from antenna.utils.data import Data
    # data_result = Data(
    #     name = MULTICONFIG("KuoHung", 'KuoHung-1'),
    #     rootdir = r"\\140.123.106.219\temp\碩二_吳維文's\Patch Antenna\Experiment\result\[Test][37] KuoHung Pattern"
    # )
    # 再用一個已知良好的「學長(KuoHung)參考圖樣」對 SM 做單筆暖身微調，
    # 讓 SM 在訓練一開始就對「好圖樣」附近有較準的預測。
    from KuoHung import KuoHung as _kh
    KuoHung, response = _kh.load(MULTICONFIG("KuoHung", '1'))

    smodel.train_one_data(AntennaPattern(KuoHung).series, response, min_loss=0.001, max_epoch=1e4)
else:
    # 情況 3：首次跑且無預訓練權重 → 用離線資料集 data_manager 從頭預訓練 SM，並存檔。
    with Figure('Pre Train', (1, 1), rootdir=RESULT_PATH, save=True, default_axes_title_size=50, default_tick_size=40, requires_grad=True) as fig:
        fig.addAll()
        fig[0].plot(smodel.train_by_datas(data_manager))  # 回傳每 epoch 平均 loss，畫成收斂曲線
        smodel.save()



# Optimizer setting
# optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)
# optimizer = torch.optim.RMSprop(params=model.parameters(), lr=init_lr)

# 把關鍵設定 (響應定義、模型結構、優化器、SM) 寫進 config.json 存檔 (實驗紀錄/可重現)。
config['AntennaResponse'] = AntennaResponse.to_str()
config['Generator'] = model
config['optimizer'] = optimizer
config['SurrogateModel'] = smodel
config.save(rootdir=RESULT_PATH)

###* Training ###
epoch = TEMP('epoch', 0) # 總訓練次數 (續跑時從 TEMP 取回上次的 epoch)
current_epoch = 0   # 斷掉後的訓練次數 (本次行程內的計數)
jump = 0 # 跳躍次數 (pattern 重複，不重複模擬)
skip = 0
simulator.open()    # 連線到 HFSS (啟動 COM、取得 Desktop 物件)
r_feed = FeedReachability.single_feed()             # 單埠饋電連通度指標 (饋電點是否連到同一金屬塊)
spectral_connectivity_loss = SpectralConnectivityLoss()  # 圖論連通性損失 (拉普拉斯 Fiedler 值)
gap_closing_loss = GapClosingLoss()                 # 形態學閉運算損失 (填補裂縫)
while epoch < config.epochs + 1:
    start = time()          # 計時本 epoch 耗時
    epoch += 1
    current_epoch += 1
    generator.change(epoch) # 把 generator 的存檔標籤切換為當前 epoch (之後 save 會存成此檔名)

    simulator.start(epoch)  # HFSS：建立並切換到本 epoch 的新專案/設計

    logger.info(f"Start {epoch} of {config.epochs}")

    generator.requires_grad(True, train=True)  # 解凍 GEN 參數並切到 train() 模式
    generator.optimizer.zero_grad() # adjust_lr(optimizer, epoch, init_lr) # 清空梯度

    TEMP['tau'] = 0
    # ── (A) 早停 / 回滾判斷 ───────────────────────────────────────────────────
    # early_stop：若 real_loss 最近 patience 次都沒比之前的最佳更好 → 觸發回滾。
    if  TEMP.early_stop('real_loss', config['patience']):
        ###* Rollback ###
        # 找出「歷史 real_loss == 目前 min_loss」的那個 epoch，把 GEN 載回該最佳狀態
        # (save=True：先存目前狀態；load=True：再載回最佳 epoch 的權重)。
        generator.change(
            TEMP.find('real_loss', TEMP('min_loss', float('inf')), 'epoch'),
            save=True, load=True
        )

        # 用本次累積的線上資料集重訓 SM (比每 epoch 的單筆訓練更充分，幫 SM 校正)。
        smodel.train_by_datas(online_dataset)

        ###* 生成 pattern 並儲存於 buffer ###
        #? target response -> 生成模型 -> pattern
        # 回滾後，重新用「目標響應」生成一張新的 pattern。
        output_element = AntennaPattern(
            generator(AntennaResponse.target.concat())
        )

        ###* Mutation ###
        TEMP['mutation'] = TEMP('min_loss')
        # output_element = output_element.mutate(config['mutation_rate'])  # (突變目前停用)
        skip = 0

    else:
        # 未觸發回滾：正常生成 pattern。
        output_element = AntennaPattern(
            generator(AntennaResponse.target.concat())
        )
        TEMP['mutation'] = 0
        skip += 1
    output_element = output_element + lower  # ── (B) 疊上固定饋電金屬塊 lower ──

    ###* 檢查 pattern 是否重複，不重複模擬 ###
    # ── (C) 去重：~output_element 是 merge 後 detach 到 CPU 的張量 (可比對)。
    #         若這張 pattern 沒在 patch_pattern_buf 緩存裡 → 才真的跑 HFSS。
    if 'patch_pattern_buf' not in TEMP or TEMP.index('patch_pattern_buf', ~output_element) is None:
        #* 未重複，進行HFSS模擬
        # ── (D) 真實模擬 ──────────────────────────────────────────────────────
        output_result = output_element.simulate()       # HFSS → MultiResponses(真實 S11/Gain)
        real_loss = output_result.criterion()           # 真實響應 vs 目標 的 loss (Ground Truth 指標)
        stack_output_result = output_result.stack()      # 疊成張量 [2, 17]，供 SM 訓練/繪圖
        # 用這「單筆真實資料」線上訓練 SM，使 SM 更貼近 HFSS。
        sm_loss = smodel.train_one_data(output_element.series, stack_output_result)
        smodel.save()

        TEMP['real_loss'] = real_loss.item()    # 儲存 HFSS結果 的 loss
        # 若本筆 real_loss 優於歷史平均，視為「好資料」收進線上資料集 (供日後重訓 SM)。
        if TEMP('real_loss') < TEMP.average('real_loss'):
            online_dataset.add_and_save([~output_element, stack_output_result])

        jump = 0

    else:
        #* 重複，直接使用之前的結果
        # ── (C') 命中快取：直接取回先前模擬過的響應與 real_loss，省下一次 HFSS。
        stack_output_result, real_loss = TEMP.find(
            'patch_pattern_buf', ~output_element, ('patch_result_buf', 'real_loss')
        )
        sm_loss = []
        TEMP['real_loss'] = real_loss
        jump = jump + 1
    TEMP['real_loss_average'] = TEMP.average('real_loss')  # 記錄到目前為止的 real_loss 平均

    ###* 更新 loss 的最小值 ###
    #? de: 更新最小loss的次數
    # ── (E) 更新歷史最佳。de = 距離上次刷新最佳已過幾個 epoch (用於圖標題標 'best')。
    min_loss = TEMP('min_loss', float('inf'))
    if TEMP('real_loss') <= min_loss:
        min_loss = TEMP('real_loss')
        TEMP['de'] = 0                  # 本 epoch 刷新最佳 → de 歸零

        config['best_epoch'] = epoch
        config.save(rootdir=RESULT_PATH)
    else:
        min_loss = min_loss
        TEMP.add('de', 1, default = 0)  # 沒刷新 → de += 1
    TEMP["min_loss"] = min_loss

    ###*  儲存HFSS的輸入與輸出，再訓練代理模型並儲存 ###
    # 把本 epoch 的 pattern / 結果 / 饋電連通率寫入緩存，供下一輪去重與繪圖。
    TEMP['patch_pattern_buf'] = ~output_element
    TEMP['patch_result_buf'] = stack_output_result
    TEMP['r_feed'] = r_feed(~output_element)    # 計算饋電連通度 R_feed (0~1，越高越「可製造/連通」)


    ###* 權重全部凍結 ###
    # for name, para in model_HFSS.named_parameters():
    #     para.requires_grad_(False)

    ###* 更新GEN ###
    #? target response -> 生成模型 -> pattern -> 代理模型 -> predicted response
    #? calculate loss (target response, predicted response)
    #? update optimizer
    # ── (F) 用「可微分的 SM」當 HFSS 替身，反向傳播更新 GEN ───────────────────
    # output_element = model(AntennaResponse.merge_target_responses())
    response = smodel(output_element.series)    # pattern → SM → 預測響應 (保留梯度，連回 GEN)
    loss = (
        response.criterion()                    # 主損失：SM 預測響應 vs 目標響應
        # 以下為 pattern 的「可製造性」正則化 (權重由 MULTICONFIG 決定，預設 0=不啟用)：
        + output_element.total_variation_loss(MULTICONFIG("total_variation_loss", 0))        # 抑制破碎
        + output_element.island_suppression_loss(MULTICONFIG("island_suppression_loss", 0))  # 消孤島
        + MULTICONFIG("spectral_connectivity_loss", 0) * spectral_connectivity_loss.forward(output_element.size_converter(output_shape="B, 1, H, W"))  # 圖論連通
        + MULTICONFIG("gap_closing_loss", 0) * gap_closing_loss.forward(output_element.size_converter(output_shape="B, 1, H, W"))                      # 填補裂縫
    )
    loss.backward()  # 梯度經由 SM 流回 GEN (HFSS 不可微，故借道 SM)
    # generator.step：optimizer.step() 更新 GEN 權重；scheduler.step(real_loss) 依「真實 loss」調 lr/tau。
    generator.step(scheduler_param=real_loss)
    generator.model.eval()  # 切回 eval (下一輪生成前再 train())

    TEMP['fake_loss'] = loss.item() # 儲存 GEN 與 代理模型 的 loss (相對於「真實」real_loss 稱 fake)

    ###* 儲存模型 ###
    generator.save()  # ── (G) 存檔 (存成 generator_{epoch}.pth，rollback 時可載回) ──

    exe_time = simulator.end()  # 結束本 epoch 的 HFSS 專案 (儲存/刪除設計/關專案)，回傳耗時
    simulator.clean()           # 清理舊專案檔，只保留最近數個

    TEMP['epoch'] = epoch
    TEMP['time'] = round(time()-start, 1)
    TEMP.save(f"{epoch} times")  # 把 TEMP 狀態寫入磁碟 (斷點續跑的存檔點)

    # ── 繪製本 epoch 的 2x3 結果總覽圖 (重複命中 jump>0 時不存圖以省 IO) ──────────
    with Figure(
        f"Result {epoch} {'best' if TEMP('de') == 0 else ''}",  # de==0 表本 epoch 為歷史最佳
        nrowcol = (2,3),
        rootdir = path_pic,
        save = False if jump > 0 else True,
        size = (18*2, 9*2),
        default_axes_title_size = 20
    ) as fig:
        pattern_ax = fig.index(-1)
        # output_element.plot(pattern_ax)
        r_feed.plot(pattern_ax)         # 子圖1：pattern 與饋電連通區 (綠色=連通到饋電點的金屬)

        s11_ax = fig.index(-1)          # 子圖2：S11 (藍=模擬, 藍虛線=目標)
        s11_ax.plot(x,stack_output_result[0].cpu(), color='blue')
        s11_ax.plot(x,returnloss.cpu(), color='blue', linestyle='--')
        # s11_ax.plot(x,returnloss_upper.cpu(), color='red')
        # s11_ax.plot(x, returnloss_lower.cpu(), color='red')
        s11_ax.set_title('S11', fontsize=20)
        # s11_ax.set_ylim(-15,1)

        gain_ax = fig.index(-1)         # 子圖3：Gain (藍=模擬, 藍虛線=目標)
        gain_ax.plot(x,stack_output_result[1].cpu(), color='blue')
        gain_ax.plot(x,gain.cpu(), color='blue', linestyle='--')
        # gain_ax.plot(x,gain_upper.cpu(), color='red')
        # gain_ax.plot(x, gain_upper.cpu(), color='red')
        gain_ax.set_title('Gain', fontsize=20)
        # gain_ax.set_ylim(-20,1)

        scheduler_ax = fig.index(-1)    # 子圖4：排程器的 lr 與 tau 走勢
        generator.scheduler.plot(scheduler_ax)

        loss_ax = fig.index(-1)         # 子圖5：各種 loss 曲線
        loss_ax.plot(TEMP['real_loss'], color='red', label='real_loss')
        loss_ax.plot(TEMP['fake_loss'], color='purple', label='fake_loss', alpha=0.8)
        # loss_ax.plot(TEMP['mutation'], label='mutation')
        loss_ax.plot(TEMP['min_loss'], label='min_loss')
        loss_ax.plot(TEMP['real_loss_average'], label='real_loss_average')
        loss_ax.legend()
        loss_ax.set_title(f"Loss Curve (Current: {TEMP('real_loss', ''):.2f})", fontsize=20)

        # sm_loss_ax = fig.index(-1)
        # sm_loss_ax.set_title('sm_loss', fontsize=20)
        # sm_loss_ax.plot(sm_loss)

        index_ax = fig.index(-1)        # 子圖6：饋電連通率 R_feed (左軸) 與每 epoch 耗時 (右軸)
        r_feed_ax = index_ax
        time_ax = r_feed_ax.twinx()
        p1, = r_feed_ax.plot(TEMP['r_feed'], color='tab:blue', label=f"{r_feed.r_feed_str} (Avg. {TEMP.average('r_feed'):.2f})")
        p2, = time_ax.plot(TEMP['time'], color='tab:orange', label=f"Time (s) (Avg. {TEMP.average('time'):.2f})")
        r_feed_ax.set_ylabel(r_feed.r_feed_str, color='tab:blue')
        time_ax.set_ylabel('Time (s)', color='tab:orange')
        r_feed_ax.tick_params(axis='y', labelcolor='tab:blue')
        time_ax.tick_params(axis='y', labelcolor='tab:orange')
        r_feed_ax.legend(handles=[p1, p2])
        index_ax.set_title(f"Index E{TEMP('epoch')}", fontsize=20)

    logger.info(f"End {epoch} of {config.epochs}, Loss: {TEMP('real_loss'):4f}, Time: {exe_time} s, jump: {jump}")




# 訓練結束：印出/寄送「歷史最小 real_loss」的完成通知 (send_email=True 會寄 email)。
Complete(f"Training Finished! (Min Loss: {TEMP.custom('real_loss', min)})", **config, send_email=True)
