"""Patch Single Port 訓練腳本 — selection 版（使用 SPGEN 從預定義 pattern 選擇）。"""

from antenna.utils import *

config.device = "cpu"

import numpy as np
import torch

from antenna import *
from antenna.functions import AdaptiveCyclicalScheduler
from antenna.models import SPGEN, Models
from antenna.patch import SinglePortSimulator, custom_loss_minmax
from antenna.smodels import OldSM
from antenna.utils.data import DataManager

torch.autograd.set_detect_anomaly(True)

###* Basic Config ###
connect_default_drive()
RESULT_PATH, CONTINUE_RUN = get_result_path(
    "[Patch Single][{device}] selection_gumbel_sinkhorn_AdaptiveCyclicalScheduler_topdata_freeze",
    rootdir=ROOTDIR,
    generate_code=__file__,
    enable_exception_handler=True,
)

SM_PRETRAIN_MODEL_PATH = DATASET_PATH.joinpath("old_sm.pth")
TEMP = Record("temp", rootdir=RESULT_PATH, load=True)

path_pic = RESULT_PATH.joinpath("pic").not_exist_create()
path_checkpoint = RESULT_PATH.joinpath("checkpoint").not_exist_create()
data_manager = DataManager("patch_single_mirror", rootdir=DATASET_PATH)
online_dataset = DataManager("online", rootdir=RESULT_PATH)


config["Name"] = RESULT_PATH.stem
config["File"] = __file__
config.setWarning()
config.epochs = 1000
config.lr = 0.005
config.checkpoint_save_path = path_checkpoint

config["patience"] = 5  # 50
config["mutation_rate"] = 0.001
config["HFSS.lr"] = 0.001
config["HFSS.min_loss"] = 0.1
config["HFSS.max_epoch"] = 20000

###* Set Antemma Pattern ###
AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
lower = AntennaPattern(torch.ones((5, 5)), (10, 15, 20, 25))

simulator = SinglePortSimulator(
    record_path=RESULT_PATH,
)
AntennaPattern.register_simulator(simulator)


###* Set Antenna Response ###
AntennaResponse.registerLabels("S11", "Gain", x="n257")
x = AntennaResponse.x()

# ? S11 目標：high low high (-1.25, -12)
returnloss = AntennaResponse.registerTargetResponse(0, -10, (5, 0, 7, 0, 5), label="S11")
AntennaResponse.registerLossHook(custom_loss_minmax, label="S11", target=returnloss, method="low")

# ? Gain 目標：low high low (-2, -19.5) (0, -25)
gain = AntennaResponse.registerTargetResponse(-19, 4, (5, 0, 7, 0, 5), label="Gain")
AntennaResponse.registerLossHook(custom_loss_minmax, label="Gain", target=gain, method="high")

with Figure(
    "Target Response",
    (1, 2),
    rootdir=RESULT_PATH,
    save=True,
    size=(18 * 2, 9 * 2),
    default_axes_title_size=50,
    default_tick_size=40,
) as fig:
    fig.addAll()

    fig[0].set_title("S11")
    fig[0].plot(x, returnloss.cpu().detach().numpy(), color="red", marker="o")
    fig[0].grid(True)

    fig[1].set_title("Gain")
    fig[1].plot(x, gain.cpu().detach().numpy(), color="red", marker="o")
    fig[1].grid(True)

###* 初始化神經網絡模型 ###


# --- 旋轉輔助函式 ---
def get_rotations(base_pattern):
    """為給定 pattern 產生 4 個旋轉（0, 90, 180, 270 度）。"""
    base = np.array(base_pattern)
    rotations = [np.rot90(base, k=i).tolist() for i in range(4)]
    return rotations


pattern_table = {
    # ---------------------------------------------------------------
    # 1. 線條 Patterns
    # ---------------------------------------------------------------
    "H-Line (Thin)": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    "H-Line (Bold)": [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]],
    "V-Line (Thin)": [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
    "V-Line (Bold)": [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]],
    "Diag (\\, Bold)": [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 0, 1, 1]],
    "Diag (/, Bold)": [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]],
    # ---------------------------------------------------------------
    # 2. 十字 Patterns
    # ---------------------------------------------------------------
    "+ (Thin)": [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
    "+ (Bold)": [[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]],
    "X (Bold)": [  # 叉叉
        [1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
    ],
    # ---------------------------------------------------------------
    # 3. 圓點 Patterns
    # ---------------------------------------------------------------
    "Dot": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    "Circle": [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]],
    # ---------------------------------------------------------------
    # 4. T 型 Patterns（含 4 個旋轉方向）
    # ---------------------------------------------------------------
    **{
        f"T (Thin, Rot {i * 90})": p
        for i, p in enumerate(
            get_rotations([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]])
        )
    },
    **{
        f"T (Bold, Rot {i * 90})": p
        for i, p in enumerate(
            get_rotations([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]])
        )
    },
    # ---------------------------------------------------------------
    # 5. L 型 Patterns（含 4 個旋轉方向）
    # ---------------------------------------------------------------
    **{
        f"L (Thin, Rot {i * 90})": p
        for i, p in enumerate(
            get_rotations([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]])
        )
    },
    **{
        f"L (Bold, Rot {i * 90})": p
        for i, p in enumerate(
            get_rotations([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]])
        )
    },
    # ---------------------------------------------------------------
    # 6. U 型 Patterns（含 4 個旋轉方向）
    # ---------------------------------------------------------------
    **{
        f"U (Thin, Rot {i * 90})": p
        for i, p in enumerate(
            get_rotations([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        )
    },
    **{
        f"U (Bold, Rot {i * 90})": p
        for i, p in enumerate(
            get_rotations([[1, 1, 0, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
        )
    },
}

model = SPGEN(list(pattern_table.values()), 25, hard=True)
model.save((5, 7), RESULT_PATH, pattern_dict=pattern_table)
optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, betas=(0.5, 0.999))
scheduler = AdaptiveCyclicalScheduler(
    optimizer,
    T_0=100,
    T_mult=2,
    lr_max=0.001,
    lr_min=0.00001,
    temp_max=2.0,
    temp_min=0.1,
    warmup_ratio=0.2,
    patience=50,
    factor=0.7,
    mode="min",
)
generator = Models(
    name="generator_{label}", rootdir=path_checkpoint, model=model, optimizer=optimizer, scheduler=scheduler
)

smodel = OldSM(checkpoint=config.checkpoint_save_path)

###* 斷點續跑 ###
if CONTINUE_RUN and ("epoch" in TEMP):
    generator.change(TEMP("epoch"), load=True)
    smodel.load()
elif SM_PRETRAIN_MODEL_PATH.exists():
    smodel.pre_load_model(SM_PRETRAIN_MODEL_PATH)
else:
    with Figure(
        "Pre Train",
        (1, 1),
        rootdir=RESULT_PATH,
        save=True,
        default_axes_title_size=50,
        default_tick_size=40,
        requires_grad=True,
    ) as fig:
        fig.addAll()
        fig[0].plot(smodel.train_by_datas(data_manager))
        smodel.save()

config["AntennaResponse"] = AntennaResponse.to_str()
config["Generator"] = model
config["optimizer"] = optimizer
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
    generator.change(epoch)
    if current_epoch % 15 == 0 or current_epoch == 1:
        simulator.reopen()

    simulator.start(epoch)
    logger.info(f"Start {epoch} of {config.epochs}")

    generator.model.train()
    generator.optimizer.zero_grad()  # adjust_lr(optimizer, epoch, init_lr)

    TEMP["tau"] = generator.scheduler.get_temp()
    if TEMP.early_stop("real_loss", config["patience"]):  # and skip > config['patience']
        ###* Rollback ###
        generator.change(TEMP.find("real_loss", TEMP("min_loss", float("inf")), "epoch"), save=True, load=True)

        smodel.train_by_datas(online_dataset)

        ###* 生成 pattern 並儲存於 buffer ###
        # ? target response -> 生成模型 -> pattern
        output_element = AntennaPattern(generator(TEMP("tau")))

        ###* Mutation ###
        TEMP["mutation"] = TEMP("min_loss")
        skip = 0

    else:
        output_element = AntennaPattern(generator(TEMP("tau")))
        TEMP["mutation"] = 0
        skip += 1
    output_element = output_element + lower

    ###* 檢查 pattern 是否重複，不重複模擬 ###
    if "patch_pattern_buf" not in TEMP or TEMP.index("patch_pattern_buf", ~output_element) is None:
        # * 未重複，進行HFSS模擬
        output_result = output_element.simulate()
        real_loss = output_result.criterion()
        stack_output_result = output_result.stack()
        sm_loss = smodel.train_one_data(output_element.series, stack_output_result)
        smodel.save()

        TEMP["real_loss"] = real_loss.item()  # 儲存 HFSS結果 的 loss
        if TEMP("real_loss") < TEMP.average("real_loss"):
            online_dataset.add_and_save([~output_element, stack_output_result])
        jump = 0

    else:
        # * 重複，直接使用之前的結果
        stack_output_result, real_loss = TEMP.find(
            "patch_pattern_buf", ~output_element, ("patch_result_buf", "real_loss")
        )
        sm_loss = []
        TEMP["real_loss"] = real_loss
        jump = jump + 1
    TEMP["real_loss_average"] = TEMP.average("real_loss")

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

    ###* 權重全部凍結 ###
    smodel.requires_grad(False, train=False)

    ###* 更新 GEN ###
    # ? target response -> 生成模型 -> pattern -> 代理模型 -> predicted response
    # ? 計算 loss(target response, predicted response) 並更新 optimizer
    output_element = AntennaPattern(generator(TEMP("tau")))
    response = smodel(output_element.series)
    loss = response.criterion()
    loss.backward()
    generator.step(scheduler_param=real_loss)
    generator.model.eval()
    TEMP["lr"] = generator.optimizer.param_groups[0]["lr"]
    TEMP["fake_loss"] = loss.item()  # 儲存 GEN 與 代理模型 的 loss

    ###* 儲存模型 ###
    generator.save()

    with Figure(
        f"Result {epoch} {'best' if TEMP('de') == 0 else ''}",
        nrowcol=(2, 3),
        rootdir=path_pic,
        save=False if jump > 0 else True,
        size=(18 * 2, 9 * 2),
        default_axes_title_size=20,
    ) as fig:
        fig.addAll()

        fig[0].plot(x, stack_output_result[0].cpu(), color="blue")
        fig[0].plot(x, returnloss.cpu(), color="blue", linestyle="--")
        fig[0].set_title("S11", fontsize=20)

        fig[1].plot(x, stack_output_result[1].cpu(), color="blue")
        fig[1].plot(x, gain.cpu(), color="blue", linestyle="--")
        fig[1].set_title("Gain", fontsize=20)

        generator.scheduler.plot(fig[2])

        fig[3].plot(TEMP["real_loss"], color="red", label="real_loss")
        fig[3].plot(TEMP["fake_loss"], color="purple", label="fake_loss", alpha=0.8)
        fig[3].plot(TEMP["mutation"], label="mutation")
        fig[3].plot(TEMP["min_loss"], label="min_loss")
        fig[3].plot(TEMP["real_loss_average"], label="real_loss_average")
        fig[3].legend()
        fig[3].set_title("Loss Curve", fontsize=20)

        fig[4].set_title("sm_loss", fontsize=20)
        fig[4].plot(sm_loss)

        output_element.plot(fig[5])

    exe_time = simulator.end()
    logger.info(f"End {epoch} of {config.epochs}, Loss: {TEMP('real_loss'):4f}, Time: {exe_time} s, jump: {jump}")

    TEMP["epoch"] = epoch
    TEMP.save(f"{epoch} times")

logger.info(f"Training Finished! (Min Loss: {TEMP.custom('real_loss', min)})")
