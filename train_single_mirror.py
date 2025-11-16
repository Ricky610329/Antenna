# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:38:05 2024

@author: user
"""
from typing import TypedDict
from antenna.utils import *
config.device = "cpu"

import torch.nn as nn
import numpy as np
import torch
from antenna import *
from script.process_files import FileProcessor

from antenna.models import (
    Models, OldGEN, MirrorCVAE
)
from antenna.patch import (
    SinglePortSimulator, custom_loss_minmax
)
from antenna.smodels import OldSM
from antenna.functions import mirror
from antenna.utils.data import DataManager
# torch.autograd.set_detect_anomaly(True)

#%% 
###* Basic Config ###
connect_network_drive("T:", r"\\140.123.106.219\temp", "user", "ailab120")
RESULT_PATH, CONTINUE_RUN = get_result_path(
    "[Patch Single][{device}] MirrorCVAE_dual", 
    rootdir = ROOTDIR, generate_code = __file__, enable_exception_handler = True
)
SM_PRETRAIN_MODEL_PATH = DATASET_PATH.joinpath('old_sm.pth')
TEMP = Record("temp", rootdir=RESULT_PATH, load=True)

path_pic = RESULT_PATH.joinpath("pic").not_exist_create()
path_checkpoint = RESULT_PATH.joinpath("checkpoint").not_exist_create()
data_manager = DataManager("patch_single_mirror", rootdir=DATASET_PATH)
online_dataset = DataManager("online", rootdir=RESULT_PATH)

config['Name'] = RESULT_PATH.stem
config['File'] = __file__
config.setWarning()
config.epochs = 1000
config.lr = 0.003
config.checkpoint_save_path = path_checkpoint

config['patience'] = 10
config['mutation_rate'] = 0.005
config['HFSS.lr'] = 0.001
config['HFSS.min_loss'] = 0.1
config['HFSS.max_epoch'] = 10000

###* Set Antemma Pattern ###
AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
upper = AntennaPattern(torch.ones((5, 5)), (10, 15, 0, 5))
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
smodel = OldSM(checkpoint=config.checkpoint_save_path)
model = MirrorCVAE(128, smodel=smodel, lower_pattern=lower)
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=config.lr, betas=(0.5, 0.999)
)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, 
    base_lr=0.0001,      # 學習率下界
    max_lr=config.lr,    # 學習率上界 (使用您config中的值)
    step_size_up=25,     # 從下界到上界所需的 epoch 數
    mode='triangular2',  # 模式: triangular2 會在每個循環後將 max_lr 減半
    cycle_momentum=True  # 對於Adam等帶動量的優化器，建議開啟
)
generator = Models(
    name = "generator_{label}",
    rootdir = path_checkpoint,
    model = model,
    optimizer = optimizer,
    scheduler = scheduler
)

###* 斷點續跑 ###
if CONTINUE_RUN and ('epoch' in TEMP):
    generator.change(TEMP('epoch'), load=True)
    smodel.load()
elif SM_PRETRAIN_MODEL_PATH.exists():
    smodel.pre_load_model(SM_PRETRAIN_MODEL_PATH)
else:
    with Figure('Pre Train', (1, 1), rootdir=RESULT_PATH, save=True, default_axes_title_size=50, default_tick_size=40, requires_grad=True) as fig:
        fig.addAll()
        fig[0].plot(smodel.train_by_datas(data_manager))
        smodel.save()


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
while epoch < config.epochs + 1:

    epoch += 1
    current_epoch += 1
    generator.change(epoch)
    if current_epoch % 15 == 0 or current_epoch == 1:
        simulator.reopen()

    logger.info(f"Start {epoch} of {config.epochs}")

    generator.requires_grad(True, train=True)
    generator.optimizer.zero_grad() # adjust_lr(optimizer, epoch, init_lr)
    

    if  TEMP.early_stop('real_loss', config['patience']) : # and skip > config['patience']
        ###* Rollback ###
        generator.change(
            TEMP.find('real_loss', TEMP('min_loss', float('inf')), 'epoch'), 
            save=True, load=True
        )
        smodel.train_by_datas(online_dataset)

        ###* 生成 pattern 並儲存於 buffer ###
        #? target response -> 生成模型 -> pattern
        all_results = generator(AntennaResponse.target.concat())

        ###* Mutation ###
        TEMP['mutation'] = TEMP('min_loss')
        # output_element = output_element.mutate(config['mutation_rate'])
        skip = 0

    else:
        all_results = generator(AntennaResponse.target.concat())
        TEMP['mutation'] = 0
        skip += 1
    

    # 訓練完所有 pattern 後，統一儲存一次 smodel
    smodel.save()

    # 從結果中找出最好的那一個
    best_result:ResultType = all_results[0]
    worst_result:ResultType = all_results[-1]
    
    best_pattern = best_result['pattern']
    worst_pattern = worst_result['pattern']

    #* Worst Result
    simulator.start(f"{epoch}_worst")
    worst_pattern.binarization_()
    output_result = worst_pattern.simulate()
    worst_result_loss =  output_result.criterion().item()
    worst_result['sm_loss'] = smodel.train_one_data(worst_pattern.series, output_result.stack())
    worst_result['real_loss'] = worst_result_loss
    worst_result['real_result'] = output_result
    exe_time = simulator.end()
    worst_result['time'] = exe_time

    
    #* Best Result
    simulator.start(f"{epoch}_best")
    best_pattern.binarization_()
    output_result = best_pattern.simulate()
    real_loss =  output_result.criterion()
    best_result['sm_loss'] = smodel.train_one_data(best_pattern.series, output_result.stack())
    best_result['real_loss'] = real_loss.item()
    best_result['real_result'] = output_result

    online_dataset.add_and_save([best_pattern.series, output_result.stack()])
    exe_time = simulator.end()
    best_result['time'] = exe_time

    TEMP['real_loss'] = real_loss.item()

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
    # TEMP['patch_pattern_buf'] = ~output_element
    # TEMP['patch_result_buf'] = stack_output_result
    

    ###* 權重全部凍結 ###
    # for name, para in model_HFSS.named_parameters():
    #     para.requires_grad_(False)

    ###* 更新GEN ###
    #? target response -> 生成模型 -> pattern -> 代理模型 -> predicted response
    #? calculate loss (target response, predicted response)
    #? update optimizer
    # output_element = model(AntennaResponse.merge_target_responses())
    response = smodel(best_pattern.series)
    loss = response.criterion()
    loss.backward()
    optimizer.step()
    model.eval()
    TEMP['fake_loss'] = loss.item() # 儲存 GEN 與 代理模型 的 loss

    with Figure(f"Result {epoch} {'best' if TEMP('de') == 0 else ''}", save=False if jump > 0 else True, rootdir=path_pic, nrowcol=(4,9), size=(18*2, 9*2),  default_tick_size = 14) as  fig:
            fig_original = fig.index(1)
            best_pattern.plot(fig_original)
            fig_original.set_title('Generators', fontsize=20)

            fig_response_s11 = fig.index(1+9)
            fig_response_s11.set_title(f"S11", fontsize=20)
            fig_response_s11.plot(x, ~output_result['S11'], label='Real')
            fig_response_s11.plot(x,returnloss.cpu(), linestyle='--')
            fig_response_s11.plot(x,~response[0], label='Fake')
            fig_response_s11.legend()

            fig_response_gain = fig.index(1+18)
            fig_response_gain.set_title(f"Gain", fontsize=20)
            fig_response_gain.plot(x, ~output_result['Gain'], label='Real')
            fig_response_gain.plot(x,gain.cpu(), linestyle='--')
            fig_response_gain.plot(x,~response[1], label='Fake')
            fig_response_gain.legend()

            fig_loss = fig.index(1+27)
            fig_loss.plot(TEMP['real_loss'], color='red', label='real_loss')
            fig_loss.plot(TEMP['fake_loss'], color='purple', label='fake_loss', alpha=0.8)
            # fig_loss.plot(TEMP['mutation'], label='mutation')
            fig_loss.set_title('Loss Curve', fontsize=20)
            fig_loss.plot(TEMP["min_loss"], label='min_loss')
            fig_loss.legend()

            for n, result in enumerate(all_results, 2):
                is_best:bool = result['is_best']
                time_str:str = f"No simulation" if result['time'] is None else f"time: {int(result['time'])}"
                loss_title = f"{result['fake_loss']:.2f}" if result['real_loss'] is None else f"{result['fake_loss']:.2f} -> {result['real_loss']:.2f}"

                fig_index_pattern = fig.index(n)
                fig_index_response_s11 = fig.index(n+9)
                fig_index_response_gain = fig.index(n+18)
                fig_index_loss = fig.index(n+27)

                result['pattern'].plot(fig_index_pattern)

                fig_index_pattern.set_title(f"P{n-1} ({time_str})", fontsize=20)
                fig_index_response_s11.set_title(f"S11 {'(best)' if is_best else ''}", fontsize=20)
                fig_index_response_gain.set_title(f"Gain {'(best)' if is_best else ''}", fontsize=20)
                fig_index_loss.set_title(f"Loss {loss_title}", fontsize=20)

                fig_index_response_s11.plot(x, ~result['fake_result']['S11'],  color='blue', label='fake')
                fig_index_response_s11.plot(x,returnloss.cpu(), color='blue', linestyle='--')

                fig_index_response_gain.plot(x, ~result['fake_result']['Gain'],  color='red', label='fake')
                fig_index_response_gain.plot(x,gain.cpu(), color='red', linestyle='--')

                if result['real_result'] is not None:
                    fig_index_response_s11.plot(x, ~result['real_result']['S11'],  color='green', label='real')
                    fig_index_response_gain.plot(x, ~result['real_result']['Gain'],  color='green', label='real')


                    fig_index_response_s11.legend()
                    fig_index_response_gain.legend()

                fig_index_loss.plot(result['sm_loss'])

    generator.save()

    exe_time = 0
    logger.info(f"End {epoch} of {config.epochs}, Loss: {TEMP('real_loss'):4f}, Time: {sum([r['time'] for r in all_results])/60} m, jump: {jump}")

    TEMP['epoch'] = epoch
    TEMP.save(f"{epoch} times")

logger.info(f"Training Finished! (Min Loss: {TEMP.custom('real_loss', min)})")
