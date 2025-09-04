from antenna.utils import *
config.device = "cuda:0"

import torch.nn as nn
import numpy as np
import torch

#* Import Antenna Kits
from antenna import *
from antenna.models import (
    SPGEN, HFSSNet
)
from antenna.ris import (
    RISSimulator, custom_loss
)
from antenna.ranger import Ranger
from antenna.functions import OldSM, train_HFSS_model

#%% 
###* Basic Config ###
RESULT_PATH, is_connect_run = get_result_path('')
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

config['HFSS.lr'] = 0.001
config['HFSS.min_loss'] = 0.00005
config['HFSS.max_epoch'] = 2000
config.response_size = (1, 361)

logger.info(f"The results will be saved in {RESULT_PATH.absolute()} (Continue: {is_connect_run}, CUDA: {torch.cuda.is_available()})")

# %%
#* Set Antemma Pattern
AntennaPattern.setDefaultCoordinate((0, 40, 0, 40))
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
simulator = RISSimulator(
    40,
)
AntennaPattern.register_simulator(simulator)
#%% Set Antenna Response
x = AntennaResponse.x_ris
# S11 S22 -> high low high
returnloss = AntennaResponse.registerTargetResponse(0, -20, (190,15,25,15,116))

AntennaResponse.registerLossHook(custom_loss)

with Figure('Target Response', rootdir=path_pic, show=False, size=(18*2, 9*2)) as fig:
    fig.addAll()
    
    fig[0].set_title('S11 & S22')
    fig[0].plot(x, returnloss.cpu().detach().numpy(), color='red', marker="o")
    fig[0].grid(True)

#%%
###* Generator Model ###
tqdm_main = trange(1, config.epochs+1, desc="loss: NaN")
model = SPGEN( # Response (Target) -> Pattern
    pattern_table
)
optimizer = Ranger(
    params=model.parameters(), lr=config.lr
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
)

###* Gradient Estimator Model ###
# tqdm_ge = trange(500)
osm = OldSM()

config['Generator'] = model
config['optimizer'] = optimizer
config['SurrogateModel'] = osm
config.save(rootdir=RESULT_PATH)

#%%
best_loss = float('inf')
def update(i, n):
    tqdm_main.set_postfix(
        SM=f"{i}/{n} (Loss: {osm.loss:.2f})"
    )
osm.progress_callback = update
best_model = model.state_dict()

###* Main Training Loop ###
for epoch in tqdm_main:
    ###* Main Model ###
    # model.load_state_dict(best_model)
    model.train()
    optimizer.zero_grad()
    
    pattern = AntennaPattern(model())
    response_label = pattern.simulate(no_grad=True)['response']
    
    TEMP['real_loss'] = response_label.criterion().item()
    TEMP['real_response_buffer'] = response_label.response
    TEMP['pattern_buffer'] = pattern.series
    TEMP['Pattern'] = pattern.merge()
    
    ###* Gradient Estimator Model ###
    sm_loss = osm.train(TEMP.end('pattern_buffer'), TEMP.end('real_response_buffer'))
    osm.save(path_checkpoint)
    response = AntennaResponse(osm(pattern.series))

    ###* Update Main Model ###
    loss = response.criterion()
    loss.backward()
    optimizer.step()

    TEMP['Response'] = response.response.cpu().detach().numpy()


    
    
    
    tqdm_main.set_description(f"loss: {loss.item():.4f}")
    
    ###* Record Loss ###
    TEMP['loss'] = loss.item()
    if loss.item() < best_loss:
        logger.info(f"Update loss: {loss.item()} -> {best_loss}")
        best_loss = loss.item()
        best_model = model.state_dict()
        # scheduler.step(best_loss)


    
    ###* Save ploting Result ###
    with Figure(f"All Patern {epoch}", (2, 2), save=True, figsize=(22,22), rootdir=path_pic) as fig:
        fig.fig.suptitle(f'File -> {Path(__file__).stem}', fontsize=24)

        ax1 = fig.index(1)
        ax2 = fig.index(2)
        ax3 = fig.index(3)
        ax4 = fig.index(4)
        
        ax1.set_title("Loss")
        ax1.plot(TEMP['loss'], label = 'fake_loss')
        ax1.plot(TEMP['real_loss'], label = 'real_loss')
        ax1.set_ylim(-0.5, 100)
        ax1.legend()
       
        ax2.set_title("SM Loss")
        ax2.plot(sm_loss)

        ax3.set_title("Response")
        ax3.plot(TEMP['Response'][-1], color='blue', label='Simulation')
        ax3.plot(AntennaResponse.getTargetResponse().cpu().detach(), color='red', label='Target')

        ax4.set_title("Pattern")
        ax4.imshow(TEMP['Pattern'][-1].cpu().detach(), cmap='gray', vmin=0, vmax=1)
        
        ax3.legend()