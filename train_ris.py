"""RIS 訓練腳本（使用 GumbelSigmoidGEN 生成器）。"""

from antenna.utils import *

config.device = "cuda:0"

import torch

from antenna import *
from antenna.models import GumbelSigmoidGEN
from antenna.ris import RISSimulator, custom_loss
from antenna.smodels import OldSM

torch.autograd.set_detect_anomaly(True)

###* Basic Config ###
RESULT_PATH, is_connect_run = get_result_path("test_GumbelSigmoidGEN")
TEMP = Record("temp", rootdir=RESULT_PATH, load=True)

path_pic = RESULT_PATH.joinpath("pic").not_exist_create()
path_checkpoint = RESULT_PATH.joinpath("checkpoint").not_exist_create()


config["Name"] = RESULT_PATH.stem
config["File"] = __file__
config.setWarning()
config.epochs = 1000
config.lr = 0.001
config.checkpoint_save_path = path_checkpoint

config["patience"] = 200
config["mutation_rate"] = 0.001
config["HFSS.lr"] = 0.001
config["HFSS.min_loss"] = 0.1
config["HFSS.max_epoch"] = 20000

logger.info(
    f"The results will be saved in {RESULT_PATH.absolute()} (Continue: {is_connect_run}, CUDA: {torch.cuda.is_available()})"
)

###* Set Antemma Pattern ###
AntennaPattern.setDefaultCoordinate((0, 40, 0, 40))
simulator = RISSimulator(
    40,
)
AntennaPattern.register_simulator(simulator)


###* Set Antenna Response ###
AntennaResponse.registerLabels("response", x="ris")
x = AntennaResponse.x()

# ? 目標響應：high low high (-1.25, -12)
returnloss = AntennaResponse.registerTargetResponse(0, -20, (190, 15, 25, 15, 116))

AntennaResponse.registerLossHook(custom_loss)

with Figure("Target Response", (1, 1), rootdir=RESULT_PATH, save=True, size=(18 * 2, 9 * 2)) as fig:
    fig.addAll()

    fig[0].set_title("S11 & S22")
    fig[0].plot(x, returnloss.cpu().detach().numpy(), color="red", marker="o")
    fig[0].grid(True)

###* 初始化神經網絡模型 ###
model = GumbelSigmoidGEN()
optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)
smodel = OldSM()

###* 斷點續跑 ###
if is_connect_run and ("epoch" in TEMP):
    last_model = path_checkpoint.joinpath(f"gen_model_{TEMP('epoch')}.pth")
    Antenna_checkpoint_loaded = last_model.load_torch()
    model.load_state_dict(Antenna_checkpoint_loaded["state_dict"])
    optimizer.load_state_dict(Antenna_checkpoint_loaded["optimizer"])
    scheduler.load_state_dict(Antenna_checkpoint_loaded["scheduler"])
    smodel.load(config.checkpoint_save_path)


config["AntennaResponse"] = AntennaResponse.to_str()
config["Generator"] = model
config["optimizer"] = optimizer
config["scheduler"] = scheduler
config["SurrogateModel"] = smodel
config.save(rootdir=RESULT_PATH)

###* Training ###
epoch = TEMP("epoch", 0)  # 總訓練次數
current_epoch = 0  # 斷掉後的訓練次數
jump = 0  # 跳躍次數 (pattern 重複，不重複模擬)
skip = 0
while epoch < config.epochs + 1:
    epoch += 1
    current_epoch += 1

    logger.info(f"Start {epoch} of {config.epochs}")

    model.train()
    optimizer.zero_grad()  # adjust_lr(optimizer, epoch, init_lr)

    if TEMP.early_stop("real_loss", config["patience"]) and skip > config["patience"] or jump > 2:
        ###* Rollback ###
        ###* 生成 pattern 並儲存於 buffer ###
        # ? target response -> 生成模型 -> pattern
        output_element = AntennaPattern(model(AntennaResponse.target.concat()))

        ###* Mutation ###
        TEMP["mutation"] = TEMP("min_loss")
        skip = 0

    else:
        output_element = AntennaPattern(model(AntennaResponse.target.concat()))
        TEMP["mutation"] = 0
        skip += 1

    output_element_bi = model.binarize()

    ###* 檢查 pattern 是否重複，不重複模擬 ###
    if "patch_pattern_buf" not in TEMP or TEMP.index("patch_pattern_buf", ~output_element) is None:
        # * 未重複，進行HFSS模擬
        output_result = output_element_bi.simulate()
        real_loss = output_result.criterion()
        stack_output_result = output_result.stack()

        sm_loss = smodel.train(output_element_bi.series, stack_output_result)
        smodel.save(path_checkpoint)

        TEMP["real_loss"] = real_loss.item()  # 儲存 HFSS結果 的 loss

        jump = 0

    else:
        # * 重複，直接使用之前的結果
        stack_output_result, real_loss = TEMP.find(
            "patch_pattern_buf", ~output_element, ("patch_result_buf", "real_loss")
        )
        sm_loss = []
        TEMP["real_loss"] = real_loss
        jump = jump + 1

    ###* 更新 loss 的最小值 ###
    # ? de: 更新最小loss的次數
    min_loss = TEMP("min_loss", float("inf"))
    if TEMP("real_loss") <= min_loss:
        min_loss = TEMP("real_loss")
        TEMP["de"] = 0
    else:
        min_loss = min_loss
        TEMP.add("de", 1, default=0)
    TEMP["min_loss"] = min_loss

    ###* 儲存 HFSS 的輸入與輸出，再訓練代理模型並儲存 ###
    TEMP["patch_pattern_buf"] = ~output_element
    TEMP["patch_result_buf"] = stack_output_result

    ###* 更新 GEN ###
    # ? target response -> 生成模型 -> pattern -> 代理模型 -> predicted response
    # ? 計算 loss(target response, predicted response) 並更新 optimizer
    response = smodel(output_element.series)
    loss = response.criterion()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    model.eval()
    TEMP["fake_loss"] = loss.item()  # 儲存 GEN 與 代理模型 的 loss

    ###* 儲存模型 ###
    gen_checkpoint = {
        "model": model,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(gen_checkpoint, path_checkpoint.joinpath(f"gen_model_{epoch}.pth"))

    with Figure(
        f"Result {epoch} {'best' if TEMP('de') == 0 else ''}",
        (2, 4),
        rootdir=path_pic,
        save=False if jump > 0 else True,
        size=(18 * 2, 9 * 2),
    ) as fig:
        fig_response = fig.index(-1)
        fig_response.plot(x, stack_output_result[0].cpu(), color="blue")
        fig_response.plot(x, returnloss.cpu(), color="blue", linestyle="--")
        fig_response.set_title("Response", fontsize=20)
        fig_response.set_ylim(-25, 1)

        fig_pattern = fig.index(-1)
        output_element.plot(fig_pattern)
        fig_pattern.set_title("Pattern", fontsize=20)

        fig_pattern_binarize = fig.index(-1)
        output_element_bi.plot(fig_pattern_binarize)
        fig_pattern_binarize.set_title("Pattern", fontsize=20)

        fig_tau = fig.index(-1)
        fig_tau.plot(model.tau_history)
        fig_tau.set_title(f"Tau {model.tau_history[-1]:.2f}", fontsize=20)

        fig_loss = fig.index(-1)
        fig_loss.plot(TEMP["real_loss"], color="red", label="real_loss")
        fig_loss.plot(TEMP["fake_loss"], color="purple", label="fake_loss", alpha=0.8)
        fig_loss.plot(TEMP["mutation"], label="mutation")
        fig_loss.legend()
        fig_loss.set_title(f"Loss Curve (R{TEMP('real_loss'):.2f}, F{TEMP('fake_loss'):.2f})", fontsize=20)

        fig_sm_loss = fig.index(-1)
        fig_sm_loss.set_title("SM Loss", fontsize=20)
        fig_sm_loss.plot(sm_loss)

        fig_min_loss = fig.index(-1)
        fig_min_loss.set_title(f"Min Loss {TEMP('min_loss'):.2f}", fontsize=20)
        fig_min_loss.plot(TEMP["min_loss"])

    logger.info(f"End {epoch} of {config.epochs}, Loss: {TEMP('real_loss'):4f}, jump: {jump}")

    TEMP["epoch"] = epoch
    TEMP.save(f"{epoch} times")

logger.info(f"Training Finished! (Min Loss: {TEMP.custom('real_loss', min)})")
