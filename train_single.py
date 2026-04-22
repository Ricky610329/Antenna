"""Patch Single Port 訓練腳本（使用 SigmoidGEN 生成器）。"""

from antenna.utils import *

config.device = "cpu"

import torch

from antenna import *
from antenna.functions import AdaptiveCyclicalScheduler, FeedReachability, GapClosingLoss, SpectralConnectivityLoss
from antenna.models import Models, SigmoidGEN
from antenna.patch import SinglePortSimulator, custom_loss_minmax
from antenna.smodels import OldSM
from antenna.utils.data import DataManager

torch.autograd.set_detect_anomaly(True)

###* Basic Config ###
MULTICONFIG = MultiConfig(
    {
        "1": {"name": "[Patch-Single-{device}-{hash_id}] pixel_base_1"},
        "2": {"name": "[Patch-Single-{device}-{hash_id}] pixel_base_2"},
        # * 換不同 Base
        "3": {
            "name": "[Patch-Single-{device}-{hash_id}] pixel_base_1_total_variation_loss_01",
            "KuoHung": "KuoHung-1",
            "total_variation_loss": 0.01,
        },
        "4": {
            "name": "[Patch-Single-{device}-{hash_id}] pixel_base_2_total_variation_loss_01",
            "KuoHung": "KuoHung-2",
            "total_variation_loss": 0.01,
        },
        # * on_plateau
        "5": {"name": "[Patch-Single-{device}-{hash_id}] pixel_base_on_plateau_linear", "on_plateau": "linear"},
        "6": {"name": "[Patch-Single-{device}-{hash_id}] pixel_base_on_plateau_peak", "on_plateau": "peak"},
        "7": {
            "name": "[Patch-Single-{device}-{hash_id}] pixel_base_linear_tv50",
            "total_variation_loss": 50,
            "on_plateau": "linear",
        },
        "8": {
            "name": "[Patch-Single-{device}-{hash_id}] pixel_base_linear_tv100",
            "island_suppression_loss": 100,
            "on_plateau": "linear",
        },
        "9": {
            "name": "[Patch-Single-{device}-{hash_id}] pixel_base_linear_is100",
            "island_suppression_loss": 100,
        },
        "10": {
            "name": "[Patch-Single-{device}-{hash_id}] pixel_base_linear_is1",
            "island_suppression_loss": 1,
        },
    }
)
connect_default_drive()
RESULT_PATH, CONTINUE_RUN = get_result_path(
    MULTICONFIG("name", "[Patch-Single-{device}-{hash_id}] pixel_base"),
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

config.update(MULTICONFIG.get_label_data())
config["Name"] = RESULT_PATH.stem
config["File"] = __file__
config.setWarning()
config.epochs = 1000
config.lr = 0.005
config.checkpoint_save_path = path_checkpoint

config["patience"] = 10
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


model = SigmoidGEN()
optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, betas=(0.5, 0.999))
scheduler = AdaptiveCyclicalScheduler(
    optimizer,
    T_0=100,  # 增加初始週期長度
    T_mult=1,  # 暫時關閉週期長度增加，讓每個週期條件一致
    lr_max=0.005,  # 稍微降低最大學習率
    lr_min=1e-6,  # 0.0001
    temp_max=4.0,  # 稍微降低最高溫度
    temp_min=0.1,
    warmup_ratio=0.2,  # 增加暖身時間
    patience=25,  # 顯著增加耐心
    factor=0.7,
    mode="min",
    on_plateau=MULTICONFIG("on_plateau", "linear"),  # TODO
)
generator = Models(
    name="generator_{label}",
    rootdir=path_checkpoint,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=custom_loss_minmax,
)

smodel = OldSM(checkpoint=config.checkpoint_save_path)

###* 斷點續跑 ###
if CONTINUE_RUN and ("epoch" in TEMP):
    generator.change(TEMP("epoch"), load=True)
    smodel.load()
elif SM_PRETRAIN_MODEL_PATH.exists():
    smodel.pre_load_model(SM_PRETRAIN_MODEL_PATH)

    from KuoHung import KuoHung as _kh

    KuoHung, response = _kh.load(MULTICONFIG("KuoHung", "1"))

    smodel.train_one_data(AntennaPattern(KuoHung).series, response, min_loss=0.001, max_epoch=1e4)
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
simulator.open()
r_feed = FeedReachability.single_feed()
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
    generator.optimizer.zero_grad()  # adjust_lr(optimizer, epoch, init_lr)

    TEMP["tau"] = 0
    if TEMP.early_stop("real_loss", config["patience"]):
        ###* Rollback ###
        generator.change(TEMP.find("real_loss", TEMP("min_loss", float("inf")), "epoch"), save=True, load=True)

        smodel.train_by_datas(online_dataset)

        ###* 生成 pattern 並儲存於 buffer ###
        # ? target response -> 生成模型 -> pattern
        output_element = AntennaPattern(generator(AntennaResponse.target.concat()))

        ###* Mutation ###
        TEMP["mutation"] = TEMP("min_loss")
        skip = 0

    else:
        output_element = AntennaPattern(generator(AntennaResponse.target.concat()))
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

        config["best_epoch"] = epoch
        config.save(rootdir=RESULT_PATH)
    else:
        min_loss = min_loss
        TEMP.add("de", 1, default=0)
    TEMP["min_loss"] = min_loss

    ###*  儲存HFSS的輸入與輸出，再訓練代理模型並儲存 ###
    TEMP["patch_pattern_buf"] = ~output_element
    TEMP["patch_result_buf"] = stack_output_result
    TEMP["r_feed"] = r_feed(~output_element)

    ###* 更新 GEN ###
    # ? target response -> 生成模型 -> pattern -> 代理模型 -> predicted response
    # ? 計算 loss(target response, predicted response) 並更新 optimizer
    response = smodel(output_element.series)
    loss = (
        response.criterion()
        + output_element.total_variation_loss(MULTICONFIG("total_variation_loss", 0))
        + output_element.island_suppression_loss(MULTICONFIG("island_suppression_loss", 0))
        + MULTICONFIG("spectral_connectivity_loss", 0)
        * spectral_connectivity_loss.forward(output_element.size_converter(output_shape="B, 1, H, W"))
        + MULTICONFIG("gap_closing_loss", 0)
        * gap_closing_loss.forward(output_element.size_converter(output_shape="B, 1, H, W"))
    )
    loss.backward()
    generator.step(scheduler_param=real_loss)
    generator.model.eval()

    TEMP["fake_loss"] = loss.item()  # 儲存 GEN 與 代理模型 的 loss

    ###* 儲存模型 ###
    generator.save()

    exe_time = simulator.end()
    simulator.clean()

    TEMP["epoch"] = epoch
    TEMP["time"] = round(time() - start, 1)
    TEMP.save(f"{epoch} times")

    with Figure(
        f"Result {epoch} {'best' if TEMP('de') == 0 else ''}",
        nrowcol=(2, 3),
        rootdir=path_pic,
        save=False if jump > 0 else True,
        size=(18 * 2, 9 * 2),
        default_axes_title_size=20,
    ) as fig:
        pattern_ax = fig.index(-1)
        r_feed.plot(pattern_ax)

        s11_ax = fig.index(-1)
        s11_ax.plot(x, stack_output_result[0].cpu(), color="blue")
        s11_ax.plot(x, returnloss.cpu(), color="blue", linestyle="--")
        s11_ax.set_title("S11", fontsize=20)

        gain_ax = fig.index(-1)
        gain_ax.plot(x, stack_output_result[1].cpu(), color="blue")
        gain_ax.plot(x, gain.cpu(), color="blue", linestyle="--")
        gain_ax.set_title("Gain", fontsize=20)

        scheduler_ax = fig.index(-1)
        generator.scheduler.plot(scheduler_ax)

        loss_ax = fig.index(-1)
        loss_ax.plot(TEMP["real_loss"], color="red", label="real_loss")
        loss_ax.plot(TEMP["fake_loss"], color="purple", label="fake_loss", alpha=0.8)
        loss_ax.plot(TEMP["min_loss"], label="min_loss")
        loss_ax.plot(TEMP["real_loss_average"], label="real_loss_average")
        loss_ax.legend()
        loss_ax.set_title(f"Loss Curve (Current: {TEMP('real_loss', ''):.2f})", fontsize=20)

        index_ax = fig.index(-1)
        r_feed_ax = index_ax
        time_ax = r_feed_ax.twinx()
        (p1,) = r_feed_ax.plot(
            TEMP["r_feed"], color="tab:blue", label=f"{r_feed.r_feed_str} (Avg. {TEMP.average('r_feed'):.2f})"
        )
        (p2,) = time_ax.plot(TEMP["time"], color="tab:orange", label=f"Time (s) (Avg. {TEMP.average('time'):.2f})")
        r_feed_ax.set_ylabel(r_feed.r_feed_str, color="tab:blue")
        time_ax.set_ylabel("Time (s)", color="tab:orange")
        r_feed_ax.tick_params(axis="y", labelcolor="tab:blue")
        time_ax.tick_params(axis="y", labelcolor="tab:orange")
        r_feed_ax.legend(handles=[p1, p2])
        index_ax.set_title(f"Index E{TEMP('epoch')}", fontsize=20)

    logger.info(f"End {epoch} of {config.epochs}, Loss: {TEMP('real_loss'):4f}, Time: {exe_time} s, jump: {jump}")


Complete(f"Training Finished! (Min Loss: {TEMP.custom('real_loss', min)})", **config, send_email=True)
