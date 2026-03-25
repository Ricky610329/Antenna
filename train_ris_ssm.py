from antenna.utils import *

config.device = "cpu"

import numpy as np
import torch
import torch.nn as nn

# * Import Antenna Kits
from antenna import *
from antenna.functions import SpecialSM
from antenna.models import SPGEN, HFSSNet
from antenna.ranger import Ranger
from antenna.ris import RISSimulator, custom_loss

# %%
###* Basic Config ###
RESULT_PATH, is_connect_run = get_result_path("1751791116")
TEMP = Record("temp", rootdir=RESULT_PATH, load=True)

path_pic = RESULT_PATH.joinpath("pic").not_exist_create()
path_checkpoint = RESULT_PATH.joinpath("checkpoint").not_exist_create()

# sys.excepthook = global_exception_handler
config["Name"] = RESULT_PATH.stem
config["File"] = __file__
config.setWarning()
config.epochs = 1000
config.lr = 1e-2
config.checkpoint_save_path = path_checkpoint

config["HFSS.lr"] = 0.0005
config["HFSS.min_loss"] = 0.00005
config["HFSS.max_epoch"] = 2000
config.response_size = (1, 361)

logger.info(
    f"The results will be saved in {RESULT_PATH.absolute()} (Continue: {is_connect_run}, CUDA: {torch.cuda.is_available()})"
)

# %%
# * Set Antemma Pattern
AntennaPattern.setDefaultCoordinate((0, 40, 0, 40))
upper = AntennaPattern(torch.ones((5, 5)), (10, 15, 0, 5))
lower = AntennaPattern(torch.ones((5, 5)), (10, 15, 20, 25))
pattern_table = (
    np.zeros((5, 5)),
    [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [0, 1, 1, 1, 0]],
    [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]],
    [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
    [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]],
    [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
    [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1]],
    [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
)
simulator = RISSimulator(
    40,
)
AntennaPattern.register_simulator(simulator)
# %% Set Antenna Response
x = AntennaResponse.x_ris
# S11 S22 -> high low high
returnloss = AntennaResponse.registerTargetResponse(0, -20, (190, 15, 25, 15, 116))

AntennaResponse.registerLossHook(custom_loss)

with Figure("Target Response", rootdir=path_pic, show=False, size=(18 * 2, 9 * 2)) as fig:
    fig.addAll()

    fig[0].set_title("S11 & S22")
    fig[0].plot(x, returnloss.detach().numpy(), color="red", marker="o")
    fig[0].grid(True)

# %%
###* Generator Model ###
tqdm_main = trange(1, config.epochs + 1, desc="loss: NaN")
model = SPGEN(  # Response (Target) -> Pattern
    pattern_table
)
optimizer = Ranger(params=model.parameters(), lr=config.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

###* Gradient Estimator Model ###
# tqdm_ge = trange(500)
ssm = SpecialSM()

if is_connect_run:
    last_model = path_checkpoint.joinpath(f"Antenna_Pattern_model_{TEMP('epoch') - 1}.pth")
    Antenna_checkpoint_loaded = last_model.load_torch()
    model.load_state_dict(Antenna_checkpoint_loaded["state_dict"])
    optimizer.load_state_dict(Antenna_checkpoint_loaded["optimizer"])
    ssm.load(config.checkpoint_save_path)
    model_name = path_checkpoint.joinpath(f"GEN_model_{TEMP('epoch') - 1}.pth")

config["HFSS Simulator"] = simulator
config["Generator"] = model
config["optimizer"] = optimizer
config["SurrogateModel"] = ssm
config.save(rootdir=RESULT_PATH)

# %%
best_loss = float("inf")


def update(i, n):
    tqdm_main.set_postfix(main=f"{i}/{n} (Loss: {ssm.loss})")


ssm.progress_callback = update
best_model = model.state_dict()

###* Main Training Loop ###
for epoch in tqdm_main:
    ###* Main Model ###
    # model.load_state_dict(best_model)
    model.train()
    optimizer.zero_grad()

    pattern = AntennaPattern(model())
    response_label = pattern.simulate(no_grad=1)["response"]
    TEMP["real_loss"] = response_label.criterion().item()

    ###* Gradient Estimator Model ###
    sm_loss = ssm.train(pattern.series)

    response = AntennaResponse(ssm(pattern.series))

    ###* Update Main Model ###
    loss = response.criterion()
    loss.backward()
    optimizer.step()

    TEMP["Response"] = response.response.cpu().detach().numpy()
    TEMP["Pattern"] = pattern.merge()

    tqdm_main.set_description(f"loss: {loss.item():.4f}")

    ###* Record Loss ###
    TEMP["loss"] = loss.item()
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model = model.state_dict()
        # scheduler.step(best_loss)

    ###* Save ploting Result ###
    with Figure(f"All Patern {epoch}", (2, 2), save=True, figsize=(22, 22), rootdir=path_pic) as fig:
        fig.fig.suptitle(f"File -> {Path(__file__).stem}", fontsize=24)

        ax1 = fig.index(1)
        ax2 = fig.index(2)
        ax3 = fig.index(3)
        ax4 = fig.index(4)

        ax1.set_title("Loss")
        ax1.plot(TEMP["loss"], label="fake_loss")
        ax1.plot(TEMP["real_loss"], label="real_loss")
        ax1.legend()

        ax2.set_title("SM Loss")
        ax2.plot(sm_loss)

        ax3.set_title("Response")
        ax3.plot(TEMP["Response"][-1], color="blue", label="Simulation")
        ax3.plot(AntennaResponse.getTargetResponse().cpu().detach(), color="red", label="Target")

        ax4.set_title("Pattern")
        ax4.imshow(TEMP["Pattern"][-1].cpu().detach(), cmap="gray", vmin=0, vmax=1)

        ax3.legend()
# %%
