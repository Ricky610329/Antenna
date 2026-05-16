# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:38:05 2024

@author: user
"""
from antenna.utils import *
config.device = "cpu"

import numpy as np
import torch
from antenna import *
from antenna.functions import FeedReachability, AdaptiveCyclicalScheduler, GapClosingLoss, SpectralConnectivityLoss
from antenna.models import (
    Models, OldGEN, SigmoidGEN
)
from antenna.patch import (
    DualPortSimulator, interval_loss, custom_loss_r, custom_loss_g
)
from antenna.smodels import OldSM
from antenna.utils.data import DataManager, dynamic_loss_filter
# from antenna.functions import mirror, mutate
torch.autograd.set_detect_anomaly(True)
#%% 
###* Basic Config ###
MULTICONFIG = MultiConfig(
    {
        '1': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_oldloss_tv100",
            'total_variation_loss' : 100
        },
        '2': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_tv1",
            'total_variation_loss' : 1
        },
        '3': {
            'name': "[Patch-Dual-{device}-{hash_id}] pixel_oldloss_is100",
            'island_suppression': 100
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
            "spectral_connectivity_loss": 0.0005,
        },
    }
)
connect_network_drive("T:", r"\\140.123.106.219\temp", "user", "ailab120")
RESULT_PATH, CONTINUE_RUN = get_result_path(
    MULTICONFIG('name', "[Patch-Dual-{device}-{hash_id}] pixel_base"), 
    rootdir = ROOTDIR, generate_code = __file__, enable_exception_handler = True
)

# DATASET_PATH = Path(r"T:\碩二_吳維文's\Patch Antenna\Experiment\dataset")

SM_PRETRAIN_MODEL_PATH = DATASET_PATH.joinpath('patch_dual.pth') #TODO
TEMP = Record("temp", rootdir=RESULT_PATH, load=True)
# sys.excepthook = global_exception_handlerc

path_pic = RESULT_PATH.joinpath("pic").not_exist_create()
path_checkpoint = RESULT_PATH.joinpath("checkpoint").not_exist_create()
data_manager = DataManager("patch_dual", rootdir=DATASET_PATH) # TODO
online_dataset = DataManager("online", rootdir=RESULT_PATH)

config.update(MULTICONFIG.get_label_data())
config['Name'] = RESULT_PATH.stem
config['File'] = __file__
config.setWarning()
config.epochs = 1000
config.lr = 0.005
config.checkpoint_save_path = path_checkpoint

config['patience'] = 10
config['mutation_rate'] = 0.001
config['HFSS.lr'] = 0.001
config['HFSS.min_loss'] = 0.1
config['HFSS.max_epoch'] = 20000

###* Set Antemma Pattern ###
AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
upper = AntennaPattern(torch.ones((5, 5)), (10, 15, 0, 5))
lower = AntennaPattern(torch.ones((5, 5)), (10, 15, 20, 25))

simulator = DualPortSimulator(
    record_path = RESULT_PATH,
)
AntennaPattern.register_simulator(simulator)


###* Set Antenna Response ###
AntennaResponse.registerLabels('S11', 'S21', 'S22', x = 'n257')
x = AntennaResponse.x()

#? S11 S22 -> high low high (-1.25, -12) -15
returnloss = AntennaResponse.registerTargetResponse(-1.25, -12, (4, 2, 5, 2, 4), label="S11")
returnloss = AntennaResponse.registerTargetResponse(-1.25, -12, (4, 2, 5, 2, 4), label="S22")
returnloss_upper = AntennaResponse.registerTargetResponse(0, -10, (4, 2, 5, 2, 4), label="returnloss_upper")
returnloss_lower = AntennaResponse.registerTargetResponse(-2.5, -20, (3, 4, 3, 4, 3), label="returnloss_lower")

# lower_response=-5
AntennaResponse.registerLossHook(interval_loss, label = "S11", lower_response=-1,upper_response=1, target=returnloss) # method='low'
AntennaResponse.registerLossHook(interval_loss, label = "S22", lower_response=-1,upper_response=1, target=returnloss)

# AntennaResponse.registerLossHook(custom_loss_r, label = "S11", target=returnloss) # method='low'
# AntennaResponse.registerLossHook(custom_loss_r, label = "S22", target=returnloss)

#? Gain -> low high low (-2, -19.5) (0, -25)
gain = AntennaResponse.registerTargetResponse(-20, -3, (3, 0, 11, 0, 3), label="S21")
gain_upper = AntennaResponse.registerTargetResponse(-17, 0, (1, 2, 11, 2, 1), label="gain_upper")
gain_lower = AntennaResponse.registerTargetResponse(-22, -3, (4, 2, 5, 2, 4), label="gain_lower")

# upper_response=5
AntennaResponse.registerLossHook(interval_loss, label = "S21", lower_response=-1,upper_response=1, target=gain) # method='high'

# AntennaResponse.registerLossHook(custom_loss_g, label = "S21", target=gain) # method='high'

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

model = SigmoidGEN()
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=config.lr, betas=(0.5, 0.999)
)
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
    name = "generator_{label}",
    rootdir = path_checkpoint,
    model = model,
    optimizer = optimizer,
    scheduler = scheduler,
)

smodel = OldSM(checkpoint=config.checkpoint_save_path)

###* 斷點續跑 ###
if CONTINUE_RUN and ('epoch' in TEMP):
    generator.change(TEMP('epoch'), load=True)
    smodel.load()
elif SM_PRETRAIN_MODEL_PATH.exists():
    smodel.pre_load_model(SM_PRETRAIN_MODEL_PATH)

    # from antenna.utils.data import Data
    # data_result = Data(
    #     name = MULTICONFIG("KuoHung", 'KuoHung-1'), 
    #     rootdir = r"\\140.123.106.219\temp\碩二_吳維文's\Patch Antenna\Experiment\result\[Test][37] KuoHung Pattern"
    # )
    # KuoHung, response = data_result.load()

    # smodel.train_one_data(AntennaPattern(KuoHung).series, response, min_loss=0.001, max_epoch=1e4)
else:
    with Figure('Pre Train', (1, 1), rootdir=RESULT_PATH, save=True, default_axes_title_size=50, default_tick_size=40, requires_grad=True) as fig:
        fig.addAll()
        fig[0].plot(smodel.train_by_datas(data_manager))
        smodel.save()
        exit()
    


# Optimizer setting
# optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)
# optimizer = torch.optim.RMSprop(params=model.parameters(), lr=init_lr)

config['AntennaResponse'] = AntennaResponse.to_str()
config['Generator'] = model
config['optimizer'] = optimizer
config['SurrogateModel'] = smodel
config.save(rootdir=RESULT_PATH)

###* Training ###
epoch = TEMP('epoch', 0) # 總訓練次數
current_epoch = 0   # 斷掉後的訓練次數
jump = 0 # 跳躍次數 (pattern 重複，不重複模擬)
skip = 0
simulator.open()
r_feed = FeedReachability.dual_feed()
spectral_connectivity_loss = SpectralConnectivityLoss()
gap_closing_loss = GapClosingLoss()
while epoch < config.epochs + 1:
    start = time()
    epoch += 1
    current_epoch += 1
    generator.change(epoch)

    simulator.start(epoch)
    logger.info(f"Start {epoch} of {config.epochs}")

    generator.requires_grad(True, train=True)
    generator.optimizer.zero_grad() # adjust_lr(optimizer, epoch, init_lr)
    
    TEMP['tau'] = 0
    if  TEMP.early_stop('real_loss', config['patience']):
        ###* Rollback ###
        generator.change(
            TEMP.find('real_loss', TEMP('min_loss', float('inf')), 'epoch'), 
            save=True, load=True
        )

        # smodel.train_by_datas(
        #     online_dataset.filter(
        #         dynamic_loss_filter, lower=smaller or float('inf'), upper=bigger or float('-inf')
        #     )
        # )
        smodel.train_by_datas(online_dataset.filter(upper=TEMP.average('real_loss')))
    

        ###* 生成 pattern 並儲存於 buffer ###
        #? target response -> 生成模型 -> pattern
        output_element = AntennaPattern(
            generator(AntennaResponse.target.concat())
        ) 

        ###* Mutation ###
        TEMP['mutation'] = TEMP('min_loss')
        # output_element = output_element.mutate(config['mutation_rate'])
        skip = 0

    else:
        output_element = AntennaPattern(
            generator(AntennaResponse.target.concat())
        ) 
        TEMP['mutation'] = 0
        skip += 1
    output_element = output_element + lower + upper

    ###* 檢查 pattern 是否重複，不重複模擬 ###
    if 'patch_pattern_buf' not in TEMP or TEMP.index('patch_pattern_buf', ~output_element) is None:
        #* 未重複，進行HFSS模擬
        output_result = output_element.simulate()
        real_loss = output_result.criterion()
        stack_output_result = output_result.stack()
        sm_loss = smodel.train_one_data(output_element.series, stack_output_result)
        smodel.save()

        TEMP['real_loss'] = real_loss.item()    # 儲存 HFSS結果 的 loss
        # if TEMP('real_loss') < TEMP.average('real_loss'):
        online_dataset.add_and_save([~output_element, stack_output_result])
            
        jump = 0

    else:
        #* 重複，直接使用之前的結果
        stack_output_result, real_loss = TEMP.find(
            'patch_pattern_buf', ~output_element, ('patch_result_buf', 'real_loss')
        )
        sm_loss = []
        TEMP['real_loss'] = real_loss
        jump = jump + 1
    TEMP['real_loss_average'] = TEMP.average('real_loss')

    ###* 更新 loss 的最小值 ###
    #? de: 更新最小loss的次數
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
    TEMP['patch_pattern_buf'] = ~output_element
    TEMP['patch_result_buf'] = stack_output_result
    TEMP['r_feed'] = r_feed(~output_element)
    

    ###* 權重全部凍結 ###
    # for name, para in model_HFSS.named_parameters():
    #     para.requires_grad_(False)

    ###* 更新GEN ###
    #? target response -> 生成模型 -> pattern -> 代理模型 -> predicted response
    #? calculate loss (target response, predicted response)
    #? update optimizer
    # output_element = model(AntennaResponse.merge_target_responses())
    response = smodel(output_element.series)
    loss = (
        response.criterion()
        + output_element.total_variation_loss(MULTICONFIG("total_variation_loss", 0))
        + output_element.island_suppression_loss(MULTICONFIG("island_suppression_loss", 0))
        + MULTICONFIG("spectral_connectivity_loss", 0) * spectral_connectivity_loss.forward(output_element.size_converter(output_shape="B, 1, H, W"))
        + MULTICONFIG("gap_closing_loss", 0) * gap_closing_loss.forward(output_element.size_converter(output_shape="B, 1, H, W"))
    )
    loss.backward()
    generator.step(scheduler_param=real_loss)
    generator.model.eval()
    
    TEMP['fake_loss'] = loss.item() # 儲存 GEN 與 代理模型 的 loss
    # TEMP['bigger'] = (TEMP.average('real_loss') + TEMP.custom('real_loss', max)) / 2
    # TEMP['smaller'] = (TEMP.average('real_loss') + TEMP.custom('real_loss', min)) / 2

    ###* 儲存模型 ###
    generator.save()

    exe_time = simulator.end()
    simulator.clean()

    TEMP['epoch'] = epoch
    TEMP['time'] = round(time()-start, 1)
    TEMP.save(f"{epoch} times")

    with Figure(
        f"Result {epoch} {'best' if TEMP('de') == 0 else ''}",
        nrowcol = (2,3), 
        rootdir = path_pic, 
        save = False if jump > 0 else True, 
        size = (18*2, 9*2),
        default_axes_title_size = 20
    ) as fig:
        ###* pattern
        pattern_fig = fig.index(1)
        # output_element.plot(pattern_fig)
        r_feed.plot(pattern_fig)


        ###* returnloss
        returnloss_fig = fig.index(2, title='Return Loss')
        returnloss_fig.plot(x, returnloss.cpu(), color='red', linestyle='--', label='target')
        returnloss_fig.plot(
            x, stack_output_result[0].cpu(), color='blue', 
            label=AntennaResponse.target.labels[0]
        )
        if len(stack_output_result)==3:
            returnloss_fig.plot(
                x, stack_output_result[2].cpu(), 
                label=AntennaResponse.target.labels[2]
            )
        # returnloss_fig.plot(x,returnloss_upper.cpu(), color='red')
        # returnloss_fig.plot(x, returnloss_lower.cpu(), color='red')
        # returnloss_fig.set_ylim(-15,1)v
        returnloss_fig.legend()

        ###* gain
        gain_fig = fig.index(3, title="S21")
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
        scheduler_fig = fig.index(-1)
        generator.scheduler.plot(scheduler_fig)  

        ###* The Loss Of Generator
        gen_loss_fig = fig.index(-1, title=f"Generator Loss ({TEMP('real_loss', '')})")
        gen_loss_fig.plot(TEMP['real_loss'], color='red', label='real_loss')
        gen_loss_fig.plot(TEMP['fake_loss'], color='purple', label='fake_loss', alpha=0.8)
        # gen_loss_fig.plot(TEMP['mutation'], label='mutation')
        gen_loss_fig.plot(TEMP['min_loss'], label='min_loss')
        gen_loss_fig.plot(TEMP['real_loss_average'], label='real_loss_average')
        gen_loss_fig.legend()

        ###* The Loss Of Surrogate Model
        # sm_loss_fig = fig.index(-1, title="Surrogate Model Loss")
        # sm_loss_fig.plot(sm_loss)

        index_ax = fig.index(-1)
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


Complete(f"Training Finished! (Min Loss: {TEMP.custom('real_loss', min)})", **config, send_email=True)
