# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:38:05 2024

@author: user
"""
from antenna.utils import *
config.device = "cpu"

from torch.autograd import Variable
from torch.autograd import Function
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import argparse
from antenna import *

from antenna.models import (
    SPGEN, HFSSNet
)
from antenna.patch import (
    DualPortSimulator, custom_loss_g, custom_loss_r
)
from antenna.functions import SpecialSM
#%% 
###* Basic Config ###
RESULT_PATH, is_connect_run = get_result_path('1751885534')
TEMP = Record("temp", rootdir=RESULT_PATH, load=True)

path_pic = RESULT_PATH.joinpath("pic").not_exist_create()
path_checkpoint = RESULT_PATH.joinpath("checkpoint").not_exist_create()

# sys.excepthook = global_exception_handler
config['Name'] = RESULT_PATH.stem
config['File'] = __file__
config.setWarning()
config.epochs = 1000
config.lr = 1e-2
config.checkpoint_save_path = path_checkpoint

config['HFSS.lr'] = 0.0005
config['HFSS.min_loss'] = 0.00005
config['HFSS.max_epoch'] = 2000

logger.info(f"The results will be saved in {RESULT_PATH.absolute()} (Continue: {is_connect_run}, CUDA: {torch.cuda.is_available()})")

    
# %%
off_buf = []
on_buf = []
pilotLoss = []
pilotAcc = []
pilot_val_Loss = []
pilot_val_Acc = []


# %%

pixel_row = 25
pixel_column = 25

# 定義神經網絡的結構參數
output_size = pixel_row*pixel_column
jump = 0
rd_lr_cnt = 0

# sys.excepthook = global_exception_handler # Catch Global Error


# %%
#* Set Antemma Pattern
AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
upper = AntennaPattern(torch.ones((5, 5)), (10, 15, 0, 5))
lower = AntennaPattern(torch.ones((5, 5)), (10, 15, 20, 25))
pattern_table = (
    np.zeros((5, 5)),
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
)
simulator = DualPortSimulator(
    record_path = RESULT_PATH,
)
AntennaPattern.register_simulator(simulator)


#%% 
#* Set Antenna Response
AntennaResponse.registerLabels('S11', 'S21', 'S22', x = 'n257')

# S11 S22 -> high low high
returnloss = AntennaResponse.registerTargetResponse(-1.25, -15, (4, 2, 5, 2, 4), label="S11")
returnloss = AntennaResponse.registerTargetResponse(-1.25, -15, (4, 2, 5, 2, 4), label="S22")
returnloss_upper = AntennaResponse.registerTargetResponse(0, -10, (4, 2, 5, 2, 4), label="returnloss_upper")
returnloss_lower = AntennaResponse.registerTargetResponse(-2.5, -50, (3, 4, 3, 4, 3), label="returnloss_lower")

AntennaResponse.registerLossHook(custom_loss_r, label = "S11")
AntennaResponse.registerLossHook(custom_loss_r, label = "S22")

# S21 -> low high low
gain = AntennaResponse.registerTargetResponse(-19, 0, (3, 0, 11, 0, 3), label="S21")
gain_upper = AntennaResponse.registerTargetResponse(-17, 0, (2, 3, 7, 3, 2), label="gain_upper")
gain_lower = AntennaResponse.registerTargetResponse(-22, -3, (4, 2, 5, 2, 4), label="gain_lower")

AntennaResponse.registerLossHook(custom_loss_g, label = "S21")

x = AntennaResponse.x()

with Figure('Target Response', (1, 2), rootdir=RESULT_PATH, save=True, size=(18*2, 9*2)) as fig:
    fig.addAll()
    
    fig[0].set_title('S11 & S22')
    fig[0].plot(x, returnloss.detach().numpy(), color='red', marker="o")
    fig[0].plot(x, returnloss_upper, color='blue', marker="o")
    fig[0].plot(x, returnloss_lower, color='blue', marker="o")
    fig[0].grid(True)
    # fig[0].set_ylim(-13, 1)
    
    fig[1].set_title('S21')
    fig[1].plot(x,gain.detach().numpy(), color='red', marker="o")
    fig[1].plot(x, gain_upper, color='blue', marker="o")
    fig[1].plot(x, gain_lower, color='blue', marker="o")
    fig[1].grid(True)

#%% 初始化神經網絡模型
model = SPGEN(pattern_table, 25)
optimizer = torch.optim.Adam(
    params=model.parameters(), lr=config.lr, betas=(0.5, 0.999)
)
ssm = SpecialSM()

if is_connect_run:
    last_model = path_checkpoint.joinpath(f"Antenna_Pattern_model_{TEMP('epoch')-1}.pth")
    Antenna_checkpoint_loaded = last_model.load_torch()
    model.load_state_dict(Antenna_checkpoint_loaded['state_dict'])
    optimizer.load_state_dict(Antenna_checkpoint_loaded['optimizer'])
    
    # model_name = path_checkpoint.joinpath(f"GEN_model_{TEMP('epoch')-1}.pth")

# Optimizer setting
# optimizer = torch.optim.Adam(params=model.parameters(), lr=init_lr)
# optimizer = torch.optim.RMSprop(params=model.parameters(), lr=init_lr)



criterion = nn.MSELoss()


config['AntennaResponse'] = AntennaResponse.to_str()
config['Generator'] = model
config['optimizer'] = optimizer
config['SurrogateModel'] = ssm
config.save(rootdir=RESULT_PATH)

# 訓練過程
epoch = TEMP('epoch', 0)
while epoch < config.epochs + 1:

    epoch += 1
    TEMP.add('pt', 1, default = 0)

    if (TEMP('pt', 0) % 15 == 0):
        simulator.reopen()

  
    
    simulator.start(TEMP('count', 0))
    logger.info(f"Start {epoch} of {config.epochs}")

    model.train()


    # adjust_lr(optimizer, epoch, init_lr)
    optimizer.zero_grad()

    training_loss = 0.0

    TEMP['cnt'] = TEMP('count', 0)


    # output_element = model()
    # output_element = AntennaPattern(output_element.reshape(-1)) + upper + lower
    output_element = AntennaPattern(model()) + upper + lower


    output_element_1 = (~output_element).numpy()

    plt_element = output_element_1.reshape(pixel_column, pixel_row)

    with Figure(f"Element_{TEMP('pt')-1}", save=True, rootdir=path_pic) as  fig:
        fig.addAll()
        output_element.plot(fig[0])


    if ((output_element_1 ==  TEMP('output_element_buf')).all()):
        output_result = output_result_buf

        if rd_lr_cnt > 29:
            lr = init_lr*10
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
            init_lr = lr
            rd_lr_cnt = -1

        rd_lr_cnt = rd_lr_cnt + 1
    else:
        output_result = output_element.simulate()
        output_result_buf = output_result

        # train HFSS model
        # output_result_1 = output_result.reshape(3, 17)

        # 將資料存至buf
        TEMP['patch_pattern_buf'] = output_element_1
        TEMP['patch_result_buf'] = stack([ n.response for n in output_result.values()])

        
        # 算real_loss
        # S11_buf = output_result[:17]
        # S21_buf = output_result[17:34]
        # S22_buf = output_result[34:]

        
        rd_lr_cnt = 0

        sm_loss = ssm.train(TEMP.end('patch_pattern_buf'))

    # model_HFSS = model_name.load_torch()
    ssm.model.eval()
    ssm.save(path_checkpoint)
    ###* 權重全部凍結 ###
    # for name, para in model_HFSS.named_parameters():
    #     para.requires_grad_(False)

    # 得到 pattern
    output_element = model()

    output_element = output_element.reshape(-1) #  + OnesBuffer

    output_element = torch.clamp(output_element, min=0.0, max=1.0)

    # 得到結果
    response = ssm(output_element)
    loss = AntennaResponse.multi_responses_to_loss(response)
    # response_l = response.reshape(1, 51)

    # #=====Count Loss=====


    # s11 = AntennaResponse(response_l[0][:17])
    # s21 = AntennaResponse(response_l[0][17:34])
    # s22 = AntennaResponse(response_l[0][34:])

    # loss_s11 = s11.criterion('S11')
    # loss_s21 = s21.criterion('S21')
    # loss_s22 = s22.criterion('S22')
    
    # loss = loss_s11 + loss_s21 + loss_s22


    loss.backward()
    optimizer.step()


    MPGN_checkpoint = {
        'model':model,
        'state_dict':model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    
    
    torch.save(MPGN_checkpoint, path_checkpoint.joinpath(f"Antenna_Pattern_model_{epoch}.pth"))
    
    #========真實Loss=================
    loss_real = AntennaResponse.multi_responses_to_loss(output_result)
    # real_loss_s11 = output_result['S11'].criterion('S11')
    # real_loss_s21 = output_result['S21'].criterion('S21')
    # real_loss_s22 = output_result['S22'].criterion('S22')

    # loss_real = real_loss_s11 + real_loss_s21 + real_loss_s22


    training_loss = loss_real
    pilotLoss.append(loss_real.detach().numpy())

    
    TEMP['pilotLoss'] = pilotLoss   # np.save(path_save_data.joinpath("pilotLoss.npy"), pilotLoss)
    TEMP['fake_loss_record'] = loss.item() # np.save(path_save_data.joinpath("fake_loss_record.npy"), fake_loss_record)

    # print(f'Epoch [{epoch}/{args.MPGN_epoch}], Loss: {loss_real:.4f}')
    # print('Loss:', loss_real)
    
    with Figure(f"LossCurve_{TEMP('pt')-1}", save=True, rootdir=path_pic) as fig:
        fig.addAll()
        fig[0].plot(pilotLoss, color='red', label='real_loss')
        fig[0].plot(TEMP['fake_loss_record'], color='purple', label='fake_loss', alpha=0.8)
        fig[0].legend()
        fig[0].set_title("Loss Curve")

    
    
#    #%%
    
    with Figure(f"Response {epoch}",(1,3), rootdir=path_pic, save=True) as fig:
        fig.addAll()
        fig.fig.set_size_inches(18*2, 9*2)

        fig[0].plot(x,output_result['S11'].response, color='blue')
        fig[0].plot(x,returnloss, color='blue', linestyle='--')
        fig[0].plot(x,returnloss_upper, color='red')
        fig[0].plot(x, returnloss_lower, color='red')
        fig[0].set_title('S11 Response', fontsize=20)
        fig[0].set_ylim(-13,1)


        fig[1].plot(x,output_result['S21'].response, color='blue')
        fig[1].plot(x,gain, color='blue', linestyle='--')
        fig[1].plot(x,gain_upper, color='red')
        fig[1].plot(x, gain_lower, color='red')
        fig[1].set_title('S21 Response', fontsize=20)
        fig[1].set_ylim(-22,1)


        fig[2].plot(x,output_result['S22'].response, color='blue')
        fig[2].plot(x,returnloss, color='blue', linestyle='--')
        fig[2].plot(x,returnloss_upper, color='red')
        fig[2].plot(x, returnloss_lower, color='red')
        fig[2].set_title('S22 Response', fontsize=20)
        fig[2].set_ylim(-13,1)

    model.eval()

    

    if ((output_element_1 == TEMP('output_element_buf')).all()):
        jump = jump + 1
    else:
        TEMP['output_element_buf'] = output_element_1
        # _TEMP["output_element_buf"] = output_element_buf # np.save(path_save_data.joinpath("output_element_buf.npy"), output_element_buf)
        if loss_real <= TEMP('min_loss', float('inf')):
            min_loss = loss_real.item()

            de = TEMP('de', 0)
            count = TEMP('count', 0) + jump
            count += 1
        else:
            min_loss = TEMP('min_loss', float('inf'))
            de = TEMP('de', 0) + 1
            count = TEMP('count', 0) + jump
            count += 1

        jump = 0

        exe_time = simulator.end()
        logger.info(f"End {epoch} of {config.epochs}, Loss: {loss_real:4f}, Time: {exe_time} s")

        TEMP['count'] = count
        TEMP['de'] = de     #  np.save(path_save_data.joinpath("de.npy"), de)
        TEMP['epoch'] = epoch
        TEMP["min_loss"] = min_loss    # np.save(path_save_data.joinpath("min_loss.npy"), min_loss.detach().numpy())

        TEMP.save(f"{epoch} times")
        

logger.info(f"Training Finished! (Min Loss: {np.min(pilotLoss)})")

#%%
simulator.save()
simulator.quit()

