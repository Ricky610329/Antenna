# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:38:05 2024

@author: user
"""
from antenna.utils import *
config.device = "cpu"

import torch.nn as nn
import numpy as np
import torch
import argparse
from antenna import *

from antenna.models import (
    SPGEN, HFSSNet
)
from antenna.patch import (
    SinglePortSimulator, custom_loss_minmax
)
from antenna.smodels import OldSM
from script.process_files import FileProcessor
# from antenna.functions import mirror, mutate
torch.autograd.set_detect_anomaly(True)
#%% 
###* Basic Config ###
connect_network_drive("T:", r"\\140.123.106.219\temp", "user", "ailab120")
RESULT_PATH, is_connect_run = get_result_path('[Patch Single][{device}] selection16_rollback_-3', rootdir=ROOTDIR)
TEMP = Record("temp", rootdir=RESULT_PATH, load=True)
# sys.excepthook = global_exception_handler

path_pic = RESULT_PATH.joinpath("pic").not_exist_create()
path_checkpoint = RESULT_PATH.joinpath("checkpoint").not_exist_create()


config['Name'] = RESULT_PATH.stem
config['File'] = __file__
config.setWarning()
config.epochs = 1000
config.lr = 0.005
config.checkpoint_save_path = path_checkpoint

config['patience'] = 100
config['mutation_rate'] = 0.001
config['HFSS.lr'] = 0.001
config['HFSS.min_loss'] = 0.1
config['HFSS.max_epoch'] = 20000

logger.info(f"The results will be saved in {RESULT_PATH.absolute()} (Continue: {is_connect_run}, CUDA: {torch.cuda.is_available()})")

###* Set Antemma Pattern ###
AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
lower = AntennaPattern(torch.ones((5, 5)), (10, 15, 20, 25))

simulator = SinglePortSimulator(
    record_path = RESULT_PATH,
)
AntennaPattern.register_simulator(simulator)


###* Set Antenna Response ###
AntennaResponse.registerLabels('S11', 'Gain', x = 'n257')
x = AntennaResponse.x()

#? S11 S22 -> high low high (-1.25, -12)
returnloss = AntennaResponse.registerTargetResponse(0, -10, (5, 0, 7, 0, 5), label="S11")
# returnloss_upper = AntennaResponse.registerTargetResponse(0, -10, (4, 2, 5, 2, 4), label="returnloss_upper")
# returnloss_lower = AntennaResponse.registerTargetResponse(-2.5, -50, (3, 4, 3, 4, 3), label="returnloss_lower")

AntennaResponse.registerLossHook(custom_loss_minmax, label = "S11", target=returnloss, method='low')

#? Gain -> low high low (-2, -19.5) (0, -25)
gain = AntennaResponse.registerTargetResponse(-19, 4, (5, 0, 7, 0, 5), label="Gain")
# gain_upper = AntennaResponse.registerTargetResponse(-17, 0, (1, 2, 11, 2, 1), label="gain_upper")
# gain_lower = AntennaResponse.registerTargetResponse(-22, -3, (4, 2, 5, 2, 4), label="gain_lower")

AntennaResponse.registerLossHook(custom_loss_minmax, label = "Gain", target=gain, method='high')

with Figure('Target Response', (1, 2), rootdir=RESULT_PATH, save=True, size=(18*2, 9*2)) as fig:
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

###*  初始化神經網絡模型 ###
pattern_table = (
    #* 粗
    [
        [0, 1, 1, 1, 0], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]
    ], 
    [
        [0, 1, 1, 1, 0], 
        [0, 1, 1, 1, 0], 
        [0, 1, 1, 1, 0], 
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0]
    ], 
    [
        [0, 0, 0, 0, 0], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0]
    ], 
    [
        [1, 1, 0, 0, 0], 
        [1, 1, 1, 0, 0], 
        [0, 1, 1, 1, 0], 
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]
    ], 
    [
        [0, 0, 0, 1, 1], 
        [0, 0, 1, 1, 1], 
        [0, 1, 1, 1, 0], 
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0]
    ], 
    [
        [1, 1, 0, 1, 1], 
        [1, 1, 1, 1, 1], 
        [0, 1, 1, 1, 0], 
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1]
    ], 
    [
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 0, 1, 1], 
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ], 
    #* 細
    [
        [0, 0, 1, 0, 0], 
        [0, 0, 1, 0, 0], 
        [0, 0, 1, 0, 0], 
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ], 
    [
        [0, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0], 
        [1, 1, 1, 1, 1], 
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], 
    [
        [1, 0, 0, 0, 0], 
        [0, 1, 0, 0, 0], 
        [0, 0, 1, 0, 0], 
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ], 
    [
        [0, 0, 0, 0, 1], 
        [0, 0, 0, 1, 0], 
        [0, 0, 1, 0, 0], 
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0]
    ], 
    [
        [1, 0, 0, 0, 1], 
        [0, 1, 0, 1, 0], 
        [0, 0, 1, 0, 0], 
        [0, 1, 0, 1, 0],
        [1, 0, 0, 0, 1]
    ], 
    [
        [1, 1, 1, 1, 1], 
        [1, 0, 0, 0, 1], 
        [1, 0, 0, 0, 1], 
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ],
    [ # 螺旋
        [1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ],
    [ # 點狀
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0]
    ],
    [ # 圓點
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
    ]

)
model = SPGEN(pattern_table, 25)
model.save((2,8), RESULT_PATH)
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=config.lr, betas=(0.5, 0.999)
)
smodel = OldSM()

###* 斷點續跑 ###
if is_connect_run and ('epoch' in TEMP):
    last_model = path_checkpoint.joinpath(f"gen_model_{TEMP('epoch')}.pth")
    Antenna_checkpoint_loaded = last_model.load_torch()
    model.load_state_dict(Antenna_checkpoint_loaded['state_dict'])
    optimizer.load_state_dict(Antenna_checkpoint_loaded['optimizer'])
    smodel.load(config.checkpoint_save_path)


# Optimizer setting
# optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)
# optimizer = torch.optim.RMSprop(params=model.parameters(), lr=init_lr)

config['AntennaResponse'] = AntennaResponse.to_str()
config['Generator'] = model
config['optimizer'] = optimizer
config['SurrogateModel'] = smodel
config.save(rootdir=RESULT_PATH)

FileProcessor(
    output_dir = RESULT_PATH,
    project_name=config['Name'],
    generated_by=config['File']
).run()

###* Training ###
epoch = TEMP('epoch', 0) # 總訓練次數
current_epoch = 0   # 斷掉後的訓練次數
jump = 0 # 跳躍次數 (pattern 重複，不重複模擬)
skip = 0
while epoch < config.epochs + 1:

    epoch += 1
    current_epoch += 1

    if current_epoch % 15 == 0 or current_epoch == 1:
        simulator.reopen()

    simulator.start(epoch)
    logger.info(f"Start {epoch} of {config.epochs}")

    model.train()
    optimizer.zero_grad() # adjust_lr(optimizer, epoch, init_lr)

    if  TEMP.early_stop('real_loss', config['patience']) and skip > config['patience']:
        ###* Rollback ###
        _epoch = TEMP.find('real_loss', TEMP('min_loss', float('inf')), 'epoch')
        best_model = path_checkpoint.joinpath(f"gen_model_{_epoch}.pth")
        Antenna_checkpoint_loaded = best_model.load_torch()
        model.load_state_dict(Antenna_checkpoint_loaded['state_dict'])
        optimizer.load_state_dict(Antenna_checkpoint_loaded['optimizer'])

        ###* 生成 pattern 並儲存於 buffer ###
        #? target response -> 生成模型 -> pattern
        output_element = AntennaPattern(
            model()
        ) 

        ###* Mutation ###
        TEMP['mutation'] = 0 # TEMP('min_loss')
        # output_element = output_element.mutate(config['mutation_rate'])
        skip = 0

    else:
        output_element = AntennaPattern(
            model()
        ) 
        TEMP['mutation'] = 0
        skip += 1
    output_element = output_element + lower

    ###* 檢查 pattern 是否重複，不重複模擬 ###
    if 'patch_pattern_buf' not in TEMP or TEMP.index('patch_pattern_buf', ~output_element) is None:
        #* 未重複，進行HFSS模擬
        output_result = output_element.simulate()
        real_loss = output_result.criterion()
        stack_output_result = output_result.stack()

        sm_loss = smodel.train(output_element.series, stack_output_result)
        smodel.save(path_checkpoint)

        TEMP['real_loss'] = real_loss.item()    # 儲存 HFSS結果 的 loss

        jump = 0
        
    else:
        #* 重複，直接使用之前的結果
        stack_output_result, real_loss = TEMP.find(
            'patch_pattern_buf', ~output_element, ('patch_result_buf', 'real_loss')
        )
        sm_loss = []
        TEMP['real_loss'] = real_loss
        jump = jump + 1

    ###* 更新 loss 的最小值 ###
    #? de: 更新最小loss的次數
    min_loss = TEMP('min_loss', float('inf'))
    if TEMP('real_loss') <= min_loss:
        min_loss = TEMP('real_loss')
        TEMP['de'] = 0
    else:
        min_loss = min_loss
        TEMP.add('de', 1, default = 0)
    TEMP["min_loss"] = min_loss

    ###*  儲存HFSS的輸入與輸出，再訓練代理模型並儲存 ###
    TEMP['patch_pattern_buf'] = ~output_element
    TEMP['patch_result_buf'] = stack_output_result
    

    ###* 權重全部凍結 ###
    # for name, para in model_HFSS.named_parameters():
    #     para.requires_grad_(False)

    ###* 更新GEN ###
    #? target response -> 生成模型 -> pattern -> 代理模型 -> predicted response
    #? calculate loss (target response, predicted response)
    #? update optimizer
    # output_element = model(AntennaResponse.merge_target_responses())
    response = smodel(output_element.series)
    loss = response.criterion()
    loss.backward()
    optimizer.step()
    model.eval()
    TEMP['fake_loss'] = loss.item() # 儲存 GEN 與 代理模型 的 loss

    ###* 儲存模型 ###
    gen_checkpoint = {
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(gen_checkpoint, path_checkpoint.joinpath(f"gen_model_{epoch}.pth"))

    with Figure(f"Result {epoch} {'best' if TEMP('de') == 0 else ''}",(2,3), rootdir=path_pic, save=False if jump > 0 else True, size=(18*2, 9*2)) as fig:
        fig.addAll()

        fig[0].plot(x,stack_output_result[0].cpu(), color='blue')
        fig[0].plot(x,returnloss.cpu(), color='blue', linestyle='--')
        # fig[0].plot(x,returnloss_upper.cpu(), color='red')
        # fig[0].plot(x, returnloss_lower.cpu(), color='red')
        fig[0].set_title('S11', fontsize=20)
        # fig[0].set_ylim(-15,1)

        fig[1].plot(x,stack_output_result[1].cpu(), color='blue')
        fig[1].plot(x,gain.cpu(), color='blue', linestyle='--')
        # fig[1].plot(x,gain_upper.cpu(), color='red')
        # fig[1].plot(x, gain_upper.cpu(), color='red')
        fig[1].set_title('Gain', fontsize=20)
        # fig[1].set_ylim(-20,1)


        fig[2].plot(TEMP['min_loss'], label='min_loss')
        fig[2].set_title("Min Loss", fontsize=20)

        fig[3].plot(TEMP['real_loss'], color='red', label='real_loss')
        fig[3].plot(TEMP['fake_loss'], color='purple', label='fake_loss', alpha=0.8)
        fig[3].plot(TEMP['mutation'], label='mutation')
        # fig[3].plot(TEMP['min_loss'], label='min_loss')
        fig[3].legend()
        fig[3].set_title("Loss Curve", fontsize=20)

        fig[4].set_title('sm_loss', fontsize=20)
        fig[4].plot(sm_loss)

        fig[5].set_title('Pattern', fontsize=20)
        output_element.plot(fig[5])

    
    exe_time = simulator.end()
    logger.info(f"End {epoch} of {config.epochs}, Loss: {TEMP('real_loss'):4f}, Time: {exe_time} s, jump: {jump}")

    TEMP['epoch'] = epoch
    TEMP.save(f"{epoch} times")

logger.info(f"Training Finished! (Min Loss: {TEMP.custom('real_loss', min)})")

#%%
simulator.save()
simulator.quit()