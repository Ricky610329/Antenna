# -*- coding: utf-8 -*-
"""
================================================================================
train_dual.py — 雙埠 (Dual-Port) 微帶貼片天線「反向設計」訓練腳本
================================================================================

【這個檔案在做什麼？】
    與 train_single.py 相同，是「反向設計 (Inverse Design)」的閉迴路訓練腳本：
    先指定想要的目標響應，再讓生成器 (GEN) 產生能達成該響應的金屬像素圖樣 (pattern)。
    差別在於本檔針對「雙埠」天線 (兩個 feed/port)，響應從 (S11, Gain) 改為
    (S11, S21, S22)——即兩個埠各自的反射 (S11/S22) 與兩埠之間的耦合/傳輸 (S21)。

【三個核心角色】 (同 single 版，詳見 antenna/ 各模組)
    1. GEN  生成器  (SigmoidGEN)：       目標響應 ─▶ 25x25 二元 pattern (STE 可微分二值化)
    2. SM   代理模型 (OldSM/HFSSNet)：    pattern ─▶ 預測響應 (S11/S21/S22)，HFSS 的可微分替身
    3. SIM  模擬器  (DualPortSimulator)： pattern ─▶ 真實響應 (Ground Truth)，COM 驅動 Ansys HFSS

【閉迴路訓練流程】 (同 single)
    (A) early_stop → rollback GEN 並重訓 SM；(B) 生成 pattern 疊上 upper+lower；
    (C) 去重；(D) 跑 HFSS 得真實響應 → 線上訓練 SM；(E) 更新最佳；
    (F) 借道 SM 反向傳播更新 GEN；(G) 存檔/繪圖/記錄。

【與 train_single.py 的主要差異】
    1. 模擬器：DualPortSimulator (兩個 lumped port)。
    2. 響應：S11 + S21 + S22 (single 是 S11 + Gain)。
    3. 損失：interval_loss (要求預測落在「目標 ± 容差」區間內)，
             而非 single 的 custom_loss_minmax；另備有 custom_loss_r / custom_loss_g。
    4. 饋電塊：同時疊 upper (頂部中央) 與 lower (底部中央) 兩塊金屬 (對應兩個 port)。
    5. 回滾重訓 SM 時，會用 online_dataset.filter(upper=平均loss) 只挑「較好的樣本」訓練。
    6. SM 未預訓練時：本檔會「先預訓練 SM 然後 exit()」，需再執行一次才會進入主訓練迴圈。

--------------------------------------------------------------------------------
Created on Wed May  8 16:38:05 2024
@author: user
"""
# antenna.utils：config(全域設定)、Path、Figure、Record、Complete、ROOTDIR、DATASET_PATH、logger、time…
from antenna.utils import *
config.device = "cpu"   # 強制使用 CPU (HFSS 在 COM 端跑，GEN/SM 很小，CPU 足夠)

import numpy as np
import torch
# antenna 根：AntennaPattern、AntennaResponse、MultiResponses、get_result_path…
from antenna import *
# functions.py：饋電連通度、自適應排程器、兩種圖樣連通性正則化損失。
from antenna.functions import FeedReachability, AdaptiveCyclicalScheduler, GapClosingLoss, SpectralConnectivityLoss
# models.py：Models(模型外殼)、OldGEN/SigmoidGEN(生成器)。
from antenna.models import (
    Models, OldGEN, SigmoidGEN
)
# patch：DualPortSimulator(雙埠 HFSS 模擬器)、interval_loss(區間損失)、
#        custom_loss_r(反射損失)、custom_loss_g(增益/耦合損失)。
from antenna.patch import (
    DualPortSimulator, interval_loss, custom_loss_r, custom_loss_g
)
# smodels.py：OldSM(代理模型工廠)。
from antenna.smodels import OldSM
# data.py：DataManager(資料集容器)、dynamic_loss_filter(依 loss 區間過濾資料的函數)。
from antenna.utils.data import DataManager, dynamic_loss_filter
# from antenna.functions import mirror, mutate
torch.autograd.set_detect_anomaly(True)  # 自動微分異常偵測 (除錯用)
#%%
###* Basic Config ###
# 一組具名實驗設定；採用哪組由命令列 sys.argv[1] 指定 (例如 `python train_dual.py 1`)。
# 各組可覆寫實驗名稱與正則化權重 (total_variation_loss / island_suppression / spectral_connectivity_loss…)。
MULTICONFIG = MultiConfig(
    {
        '1': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_oldloss_tv100",
            'total_variation_loss' : 100   # 啟用 Total Variation 損失 (抑制破碎圖樣)
        },
        '2': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_tv1",
            'total_variation_loss' : 1
        },
        '3': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_oldloss_is100",
            'island_suppression': 100      # 註：主迴圈讀的鍵是 island_suppression_loss，此鍵名不符故不生效
        },
        '4': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_intervalloss5_tv100_dlfavg",
            'total_variation_loss' : 100
        },
        '5': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_intervalloss5_isrelu9",
            'island_suppression' : 100,
            'relu': 0.9
        },
        '6': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_intervalloss5_dlfavg",
        },
        '7': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_dlfavg",
        },
        '8': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_intervalloss1_dlfavg",
        },
        '9': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_base_linear_sc0_0005",
            "spectral_connectivity_loss": 0.0005,   # 啟用圖論連通性損失 (極小權重)
        },
    }
)
connect_network_drive("T:", r"\\140.123.106.219\temp", "user", "ailab120")  # 掛載 NAS (T:)
# 建立/取得本次實驗結果夾。回傳 (結果路徑, 是否已存在=是否續跑)；同時複製原始碼、掛例外處理。
RESULT_PATH, CONTINUE_RUN = get_result_path(
    MULTICONFIG('name', "[Patch-Dual-{device}-{hash_id}] pixel_base"),
    rootdir = ROOTDIR, generate_code = __file__, enable_exception_handler = True
)

# DATASET_PATH = Path(r"T:\碩二_吳維文's\Patch Antenna\Experiment\dataset")

SM_PRETRAIN_MODEL_PATH = DATASET_PATH.joinpath('patch_dual.pth') #TODO  # 雙埠 SM 預訓練權重
# TEMP：本次訓練的持久化狀態 + 時序記錄器 (斷點續跑核心)。
TEMP = Record("temp", rootdir=RESULT_PATH, load=True)
# sys.excepthook = global_exception_handlerc

path_pic = RESULT_PATH.joinpath("pic").not_exist_create()               # 每 epoch 結果圖
path_checkpoint = RESULT_PATH.joinpath("checkpoint").not_exist_create() # 模型權重檔
data_manager = DataManager("patch_dual", rootdir=DATASET_PATH) # TODO   # SM 離線預訓練資料集
online_dataset = DataManager("online", rootdir=RESULT_PATH)             # 線上收集的 (pattern, 真實響應)

# 把選定那組 MULTICONFIG 合併進 config，並補上訓練超參數。
config.update(MULTICONFIG.get_label_data())
config['Name'] = RESULT_PATH.stem
config['File'] = __file__
config.setWarning()
config.epochs = 1000        # 主訓練迴圈總 epoch
config.lr = 0.005           # GEN 的 Adam 初始學習率
config.checkpoint_save_path = path_checkpoint

config['patience'] = 10         # 早停/回滾耐心值
config['mutation_rate'] = 0.001 # (未啟用) 突變比例
config['HFSS.lr'] = 0.001       # SM 學習率
config['HFSS.min_loss'] = 0.1   # SM 單筆訓練收斂門檻 (預設)
config['HFSS.max_epoch'] = 20000# SM 單筆訓練迭代上限 (預設)

###* Set Antemma Pattern ###
# 25x25 設計網格。
AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
# 雙埠：兩塊固定饋電金屬塊 (各 5x5 全金屬)。
#   upper：頂部中央 (x:10~15, y:0~5)  → 對應 port 2
#   lower：底部中央 (x:10~15, y:20~25)→ 對應 port 1
# 每 epoch 都把 GEN 生成的圖樣同時疊上 upper 與 lower。
upper = AntennaPattern(torch.ones((5, 5)), (10, 15, 0, 5))
lower = AntennaPattern(torch.ones((5, 5)), (10, 15, 20, 25))

# 建立雙埠 HFSS 模擬器並註冊給 AntennaPattern。
simulator = DualPortSimulator(
    record_path = RESULT_PATH,
)
AntennaPattern.register_simulator(simulator)


###* Set Antenna Response ###
# 註冊三條要算 loss 的響應：S11、S21、S22；x 軸 'n257' → 24~32GHz 取 17 點。
AntennaResponse.registerLabels('S11', 'S21', 'S22', x = 'n257')
x = AntennaResponse.x()

#? S11 S22 -> high low high (-1.25, -12) -15
# S11 與 S22 共用同一條目標曲線：side=-1.25, center=-12, width=(4,2,5,2,4) → 兩端高/中央低 (匹配良好)。
# (returnloss 變數被覆寫兩次，但 registerTargetResponse 內部已分別存進 S11 / S22 兩個 label。)
returnloss = AntennaResponse.registerTargetResponse(-1.25, -12, (4, 2, 5, 2, 4), label="S11")
returnloss = AntennaResponse.registerTargetResponse(-1.25, -12, (4, 2, 5, 2, 4), label="S22")
# 下面兩條僅供繪圖參考的「上下緣」曲線 (未註冊損失)。
returnloss_upper = AntennaResponse.registerTargetResponse(0, -10, (4, 2, 5, 2, 4), label="returnloss_upper")
returnloss_lower = AntennaResponse.registerTargetResponse(-2.5, -20, (3, 4, 3, 4, 3), label="returnloss_lower")

# lower_response=-5
# 為 S11/S22 註冊 interval_loss：要求預測落在「目標 ± 容差」區間 [target-1, target+1] 內；
#   在區間內 loss=0，超出才依超出量計罰 (比 minmax 更柔性，允許小幅偏離)。
AntennaResponse.registerLossHook(interval_loss, label = "S11", lower_response=-1,upper_response=1, target=returnloss) # method='low'
AntennaResponse.registerLossHook(interval_loss, label = "S22", lower_response=-1,upper_response=1, target=returnloss)

# AntennaResponse.registerLossHook(custom_loss_r, label = "S11", target=returnloss) # method='low'
# AntennaResponse.registerLossHook(custom_loss_r, label = "S22", target=returnloss)

#? Gain -> low high low (-2, -19.5) (0, -25)
# S21 (兩埠間傳輸/耦合) 目標：side=-20, center=-3, width=(3,0,11,0,3)。
gain = AntennaResponse.registerTargetResponse(-20, -3, (3, 0, 11, 0, 3), label="S21")
# 僅供繪圖參考的上下緣。
gain_upper = AntennaResponse.registerTargetResponse(-17, 0, (1, 2, 11, 2, 1), label="gain_upper")
gain_lower = AntennaResponse.registerTargetResponse(-22, -3, (4, 2, 5, 2, 4), label="gain_lower")

# upper_response=5
# 為 S21 註冊 interval_loss (同樣是區間容差形式)。
AntennaResponse.registerLossHook(interval_loss, label = "S21", lower_response=-1,upper_response=1, target=gain) # method='high'

# AntennaResponse.registerLossHook(custom_loss_g, label = "S21", target=gain) # method='high'

# 畫出目標響應 (左:S11&S22 與上下緣, 右:S21 與上下緣) 並存檔。
with Figure('Target Response', (1, 2), rootdir=RESULT_PATH, save=True, size=(18*2, 9*2), default_axes_title_size=50, default_tick_size=40) as fig:
    fig.addAll()

    fig[0].set_title('S11 & S22')
    fig[0].plot(x, returnloss.cpu().detach().numpy(), color='red', marker="o")
    fig[0].plot(x, returnloss_upper.cpu(), color='blue', marker="o")
    fig[0].plot(x, returnloss_lower.cpu(), color='blue', marker="o")
    fig[0].grid(True)
    # fig[0].set_ylim(-13, 1)

    fig[1].set_title('S21')
    fig[1].plot(x,gain.cpu().detach().numpy(), color='red', marker="o")
    fig[1].plot(x, gain_upper.cpu(), color='blue', marker="o")
    fig[1].plot(x, gain_lower.cpu(), color='blue', marker="o")
    fig[1].grid(True)

# ── 建立 GEN、優化器、排程器 (同 single；注意此處 Models 未傳 criterion) ──────────
model = SigmoidGEN()    # MLP(目標響應→logits) + STE 二值化 → 0/1 的 25x25 pattern
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=config.lr, betas=(0.5, 0.999)   # betas 一階動量 0.5 (GAN 類常用)
)
# 自適應週期排程器：同時調 lr 與二值化溫度 tau；停滯時依 on_plateau 策略強制重啟。
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
    on_plateau = MULTICONFIG("on_plateau", "linear")
)
generator = Models(
    name = "generator_{label}",     # 含 {label} → 不同 epoch 存不同檔名 (供 rollback 載回)
    rootdir = path_checkpoint,
    model = model,
    optimizer = optimizer,
    scheduler = scheduler,
)

# ── 建立 SM (代理模型) ─────────────────────────────────────────────────────────
smodel = OldSM(checkpoint=config.checkpoint_save_path)  # HFSSNet: pattern(625) → 響應(3,17)

###* 斷點續跑 ###
if CONTINUE_RUN and ('epoch' in TEMP):
    # 情況 1：續跑 → 載回 generator 與 SM 的上次狀態。
    generator.change(TEMP('epoch'), load=True)
    smodel.load()
elif SM_PRETRAIN_MODEL_PATH.exists():
    # 情況 2：已有預訓練 SM 權重 → 直接載入 (雙埠版未再做 KuoHung 暖身)。
    smodel.pre_load_model(SM_PRETRAIN_MODEL_PATH)

    # from antenna.utils.data import Data
    # data_result = Data(
    #     name = MULTICONFIG("KuoHung", 'KuoHung-1'),
    #     rootdir = r"\\140.123.106.219\temp\碩二_吳維文's\Patch Antenna\Experiment\result\[Test][37] KuoHung Pattern"
    # )
    # KuoHung, response = data_result.load()

    # smodel.train_one_data(AntennaPattern(KuoHung).series, response, min_loss=0.001, max_epoch=1e4)
else:
    # 情況 3：首次跑且無預訓練權重 → 用離線資料集預訓練 SM、存檔後「直接 exit()」。
    #         亦即：本檔需先跑一次以產生 SM 權重，之後再跑一次才會進入下方主訓練迴圈。
    with Figure('Pre Train', (1, 1), rootdir=RESULT_PATH, save=True, default_axes_title_size=50, default_tick_size=40, requires_grad=True) as fig:
        fig.addAll()
        fig[0].plot(smodel.train_by_datas(data_manager))
        smodel.save()
        exit()



# Optimizer setting
# optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)
# optimizer = torch.optim.RMSprop(params=model.parameters(), lr=init_lr)

# 把關鍵設定寫進 config.json 存檔。
config['AntennaResponse'] = AntennaResponse.to_str()
config['Generator'] = model
config['optimizer'] = optimizer
config['SurrogateModel'] = smodel
config.save(rootdir=RESULT_PATH)

###* Training ###
epoch = TEMP('epoch', 0) # 總訓練次數 (續跑時取回)
current_epoch = 0   # 斷掉後的訓練次數 (本行程內計數)
jump = 0 # 跳躍次數 (pattern 重複，不重複模擬)
skip = 0
simulator.open()    # 連線 HFSS
r_feed = FeedReachability.dual_feed()                    # 雙埠饋電連通度 (兩饋電點需連到同一金屬塊)
spectral_connectivity_loss = SpectralConnectivityLoss()  # 圖論連通性損失
gap_closing_loss = GapClosingLoss()                      # 形態學閉運算損失 (填補裂縫)
while epoch < config.epochs + 1:
    start = time()
    epoch += 1
    current_epoch += 1
    generator.change(epoch)     # 切換 generator 存檔標籤為當前 epoch

    simulator.start(epoch)      # HFSS 建立本 epoch 新專案/設計
    logger.info(f"Start {epoch} of {config.epochs}")

    generator.requires_grad(True, train=True)  # 解凍 GEN、切 train()
    generator.optimizer.zero_grad() # adjust_lr(optimizer, epoch, init_lr)  # 清梯度

    TEMP['tau'] = 0
    # ── (A) 早停 / 回滾 ───────────────────────────────────────────────────────
    if  TEMP.early_stop('real_loss', config['patience']):
        ###* Rollback ###
        # real_loss 連續 patience 次沒進步 → 把 GEN 載回歷史最佳 epoch。
        generator.change(
            TEMP.find('real_loss', TEMP('min_loss', float('inf')), 'epoch'),
            save=True, load=True
        )

        # smodel.train_by_datas(
        #     online_dataset.filter(
        #         dynamic_loss_filter, lower=smaller or float('inf'), upper=bigger or float('-inf')
        #     )
        # )
        # 重訓 SM，但只用「loss <= 目前平均」的較佳樣本 (filter(upper=平均))，避免被壞樣本帶偏。
        smodel.train_by_datas(online_dataset.filter(upper=TEMP.average('real_loss')))


        ###* 生成 pattern 並儲存於 buffer ###
        #? target response -> 生成模型 -> pattern
        output_element = AntennaPattern(
            generator(AntennaResponse.target.concat())
        )

        ###* Mutation ###
        TEMP['mutation'] = TEMP('min_loss')
        # output_element = output_element.mutate(config['mutation_rate'])  # (突變停用)
        skip = 0

    else:
        # 正常生成 pattern。
        output_element = AntennaPattern(
            generator(AntennaResponse.target.concat())
        )
        TEMP['mutation'] = 0
        skip += 1
    output_element = output_element + lower + upper  # ── (B) 疊上 lower + upper 兩塊饋電金屬 ──

    ###* 檢查 pattern 是否重複，不重複模擬 ###
    # ── (C) 去重：沒在緩存裡才真的跑 HFSS。
    if 'patch_pattern_buf' not in TEMP or TEMP.index('patch_pattern_buf', ~output_element) is None:
        #* 未重複，進行HFSS模擬
        # ── (D) 真實模擬 ──────────────────────────────────────────────────────
        output_result = output_element.simulate()       # HFSS → MultiResponses(S11/S21/S22)
        real_loss = output_result.criterion()           # 真實響應 vs 目標 的 loss (Ground Truth)
        stack_output_result = output_result.stack()      # 疊成張量 [3, 17]
        sm_loss = smodel.train_one_data(output_element.series, stack_output_result)  # 單筆線上訓練 SM
        smodel.save()

        TEMP['real_loss'] = real_loss.item()    # 儲存 HFSS結果 的 loss
        # if TEMP('real_loss') < TEMP.average('real_loss'):
        # 雙埠版：不論好壞，每筆真實資料都收進線上資料集 (single 版只在優於平均時才收)。
        online_dataset.add_and_save([~output_element, stack_output_result])

        jump = 0

    else:
        #* 重複，直接使用之前的結果
        # ── (C') 命中快取：取回先前結果，省一次 HFSS。
        stack_output_result, real_loss = TEMP.find(
            'patch_pattern_buf', ~output_element, ('patch_result_buf', 'real_loss')
        )
        sm_loss = []
        TEMP['real_loss'] = real_loss
        jump = jump + 1
    TEMP['real_loss_average'] = TEMP.average('real_loss')

    ###* 更新 loss 的最小值 ###
    #? de: 更新最小loss的次數
    # ── (E) 更新歷史最佳；de = 距上次刷新最佳的 epoch 數。
    min_loss = TEMP('min_loss', float('inf'))
    if TEMP('real_loss') <= min_loss:
        min_loss = TEMP('real_loss')
        TEMP['de'] = 0

        config['best_epoch'] = epoch
        config.save(rootdir=RESULT_PATH)
    else:
        min_loss = min_loss
        TEMP.add('de', 1, default = 0)
    TEMP["min_loss"] = min_loss

    ###*  儲存HFSS的輸入與輸出，再訓練代理模型並儲存 ###
    # 寫入緩存供下一輪去重與繪圖。
    TEMP['patch_pattern_buf'] = ~output_element
    TEMP['patch_result_buf'] = stack_output_result
    TEMP['r_feed'] = r_feed(~output_element)    # 雙埠饋電連通度 R_feed


    ###* 權重全部凍結 ###
    # for name, para in model_HFSS.named_parameters():
    #     para.requires_grad_(False)

    ###* 更新GEN ###
    #? target response -> 生成模型 -> pattern -> 代理模型 -> predicted response
    #? calculate loss (target response, predicted response)
    #? update optimizer
    # ── (F) 借道可微分 SM 反向傳播更新 GEN ──────────────────────────────────────
    # output_element = model(AntennaResponse.merge_target_responses())
    response = smodel(output_element.series)    # pattern → SM → 預測響應 (保留梯度連回 GEN)
    loss = (
        response.criterion()                    # 主損失：SM 預測響應 vs 目標 (interval_loss 加總)
        # pattern 可製造性正則化 (權重由 MULTICONFIG 決定，預設 0)：
        + output_element.total_variation_loss(MULTICONFIG("total_variation_loss", 0))        # 抑制破碎
        + output_element.island_suppression_loss(MULTICONFIG("island_suppression_loss", 0))  # 消孤島
        + MULTICONFIG("spectral_connectivity_loss", 0) * spectral_connectivity_loss.forward(output_element.size_converter(output_shape="B, 1, H, W"))  # 圖論連通
        + MULTICONFIG("gap_closing_loss", 0) * gap_closing_loss.forward(output_element.size_converter(output_shape="B, 1, H, W"))                      # 填補裂縫
    )
    loss.backward()                             # 梯度經 SM 流回 GEN
    generator.step(scheduler_param=real_loss)   # 更新 GEN 權重；排程器依「真實 loss」調 lr/tau
    generator.model.eval()

    TEMP['fake_loss'] = loss.item() # 儲存 GEN 與 代理模型 的 loss
    # TEMP['bigger'] = (TEMP.average('real_loss') + TEMP.custom('real_loss', max)) / 2
    # TEMP['smaller'] = (TEMP.average('real_loss') + TEMP.custom('real_loss', min)) / 2

    ###* 儲存模型 ###
    generator.save()    # ── (G) 存檔 (generator_{epoch}.pth) ──

    exe_time = simulator.end()  # 結束本 epoch HFSS 專案，回傳耗時
    simulator.clean()           # 清理舊專案檔

    TEMP['epoch'] = epoch
    TEMP['time'] = round(time()-start, 1)
    TEMP.save(f"{epoch} times")  # 寫入 TEMP 存檔點 (斷點續跑)

    # ── 繪製本 epoch 的 2x3 結果總覽圖 ────────────────────────────────────────
    with Figure(
        f"Result {epoch} {'best' if TEMP('de') == 0 else ''}",
        nrowcol = (2,3),
        rootdir = path_pic,
        save = False if jump > 0 else True,     # 重複命中時不存圖
        size = (18*2, 9*2),
        default_axes_title_size = 20
    ) as fig:
        ###* pattern
        pattern_fig = fig.index(1)              # 子圖1：pattern 與饋電連通區
        # output_element.plot(pattern_fig)
        r_feed.plot(pattern_fig)


        ###* returnloss
        returnloss_fig = fig.index(2, title='Return Loss')  # 子圖2：S11 (與 S22) 對目標
        returnloss_fig.plot(x, returnloss.cpu(), color='red', linestyle='--', label='target')
        returnloss_fig.plot(
            x, stack_output_result[0].cpu(), color='blue',
            label=AntennaResponse.target.labels[0]
        )
        if len(stack_output_result)==3:         # 三條響應時，第 3 條 (S22) 也畫上
            returnloss_fig.plot(
                x, stack_output_result[2].cpu(),
                label=AntennaResponse.target.labels[2]
            )
        # returnloss_fig.plot(x,returnloss_upper.cpu(), color='red')
        # returnloss_fig.plot(x, returnloss_lower.cpu(), color='red')
        # returnloss_fig.set_ylim(-15,1)v
        returnloss_fig.legend()

        ###* gain
        gain_fig = fig.index(3, title="S21")    # 子圖3：S21 (耦合/傳輸) 對目標
        gain_fig.plot(x, gain.cpu(), color='red', linestyle='--', label='target')
        gain_fig.plot(
            x, stack_output_result[1].cpu(), color='blue',
            label=AntennaResponse.target.labels[1]
        )
        # gain_fig.plot(x,gain_upper.cpu(), color='red')
        # gain_fig.plot(x, gain_upper.cpu(), color='red')
        # gain_fig.set_ylim(-20,1)
        gain_fig.legend()

        ###* Scheduler Parameters (LR And Temperature)
        scheduler_fig = fig.index(-1)           # 子圖4：lr 與 tau 走勢
        generator.scheduler.plot(scheduler_fig)

        ###* The Loss Of Generator
        gen_loss_fig = fig.index(-1, title=f"Generator Loss ({TEMP('real_loss', '')})")  # 子圖5：各 loss 曲線
        gen_loss_fig.plot(TEMP['real_loss'], color='red', label='real_loss')
        gen_loss_fig.plot(TEMP['fake_loss'], color='purple', label='fake_loss', alpha=0.8)
        # gen_loss_fig.plot(TEMP['mutation'], label='mutation')
        gen_loss_fig.plot(TEMP['min_loss'], label='min_loss')
        gen_loss_fig.plot(TEMP['real_loss_average'], label='real_loss_average')
        gen_loss_fig.legend()

        ###* The Loss Of Surrogate Model
        # sm_loss_fig = fig.index(-1, title="Surrogate Model Loss")
        # sm_loss_fig.plot(sm_loss)

        index_ax = fig.index(-1)                # 子圖6：R_feed (左軸) 與每 epoch 耗時 (右軸)
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


# 訓練結束：印出/寄送「歷史最小 real_loss」的完成通知。
Complete(f"Training Finished! (Min Loss: {TEMP.custom('real_loss', min)})", **config, send_email=True)
