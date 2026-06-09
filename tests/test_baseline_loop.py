"""
端到端「行為基準」(config 驅動版)：單一 _run_loop(cfg) 同時支援單埠/雙埠。

這是「整合 train_single/dual 成 config 驅動」的測試先行 (TDD)：
- 共用迴圈 _run_loop(cfg) 用 LoopCfg 驅動「單/雙埠的差異」(模擬器響應、饋電塊、
  r_feed、響應註冊)，證明同一套核心能重現兩條路徑的行為。
- single 的 golden (golden_loop.json) 必須與重構前完全一致 → 回歸保證。
- dual 的 golden (golden_loop_dual.json) 首次跑自動產生。

simulator 重點：mock 實作完整 COM 生命週期 (open/start/end/clean) 並記錄呼叫次數，
測試會斷言生命週期被正確呼叫 (open 一次、start/end/clean 每 epoch)。

確定性：每條路徑各自 torch.manual_seed(0) 起手 (CPU、無 dropout)，互不影響。
高 patience → 不觸發 rollback (其 train_by_datas 的 DataLoader shuffle 會引入非確定性)。
"""
import os
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from antenna.utils import config
from antenna import AntennaPattern, AntennaResponse, TargetResponse
from antenna.models import Models, SigmoidGEN
from antenna.smodels import OldSM
from antenna.functions import (
    FeedReachability, AdaptiveCyclicalScheduler,
    SpectralConnectivityLoss, GapClosingLoss,
)
from antenna.patch import custom_loss_minmax, interval_loss
from antenna.utils.data import DataManager
from antenna.utils.utils import Record

EPOCHS = 6
PATIENCE = 50
TV_W = 0.01
SC_W = 0.0005


# ===== 全域響應註冊 (單/雙埠不同；切換前先 reset target 避免殘留污染) =====

def _setup_single_responses():
    AntennaResponse.target = TargetResponse()
    AntennaResponse.registerLabels("S11", "Gain", x="n257")
    s11 = AntennaResponse.registerTargetResponse(0, -10, (5, 0, 7, 0, 5), label="S11")
    AntennaResponse.registerLossHook(custom_loss_minmax, label="S11", target=s11, method="low")
    gain = AntennaResponse.registerTargetResponse(-19, 4, (5, 0, 7, 0, 5), label="Gain")
    AntennaResponse.registerLossHook(custom_loss_minmax, label="Gain", target=gain, method="high")


def _setup_dual_responses():
    AntennaResponse.target = TargetResponse()
    AntennaResponse.registerLabels("S11", "S21", "S22", x="n257")
    rl = AntennaResponse.registerTargetResponse(-1.25, -12, (4, 2, 5, 2, 4), label="S11")
    AntennaResponse.registerTargetResponse(-1.25, -12, (4, 2, 5, 2, 4), label="S22")
    AntennaResponse.registerLossHook(interval_loss, label="S11", lower_response=-1, upper_response=1, target=rl)
    AntennaResponse.registerLossHook(interval_loss, label="S22", lower_response=-1, upper_response=1, target=rl)
    g = AntennaResponse.registerTargetResponse(-20, -3, (3, 0, 11, 0, 3), label="S21")
    AntennaResponse.registerLossHook(interval_loss, label="S21", lower_response=-1, upper_response=1, target=g)


# ===== Mock 模擬器：確定性響應 + 記錄 COM 生命週期呼叫 =====

class _MockSim:
    def __init__(self, labels):
        self.labels = labels
        self.calls = {"open": 0, "start": 0, "end": 0, "clean": 0}

    # --- HFSS COM 生命週期 (mock 為 no-op，但記錄被呼叫次數) ---
    def open(self): self.calls["open"] += 1
    def start(self, num): self.calls["start"] += 1
    def end(self, *a, **k): self.calls["end"] += 1; return 0
    def clean(self, *a, **k): self.calls["clean"] += 1
    def restart(self, **k): pass

    # --- 模擬：響應只取決於 pattern 填充率 (確定性、無隨機) ---
    def __call__(self, pattern, **kw):
        fr = pattern.float().mean()
        x = torch.linspace(0, 1, 17)
        bump = torch.exp(-((x - 0.5) ** 2) / 0.05)
        out = {}
        for lbl in self.labels:
            if lbl in ("S11", "S22"):
                out[lbl] = -12.0 * bump * fr               # 反射型：中央凹
            else:                                          # S21 / Gain
                out[lbl] = 4.0 * bump * fr - 19.0 * (1.0 - fr)
        return out


@dataclass
class LoopCfg:
    name: str
    setup_responses: object   # callable() -> None
    sim_labels: tuple
    feeds: tuple              # ((shape, coord), ...) 用來建固定饋電金屬塊
    make_r_feed: object       # callable() -> FeedReachability
    golden_file: str


def single_cfg():
    return LoopCfg(
        name="single",
        setup_responses=_setup_single_responses,
        sim_labels=("S11", "Gain"),
        feeds=(((5, 5), (10, 15, 20, 25)),),                 # 只有 lower
        make_r_feed=FeedReachability.single_feed,
        golden_file="golden_loop.json",
    )


def dual_cfg():
    return LoopCfg(
        name="dual",
        setup_responses=_setup_dual_responses,
        sim_labels=("S11", "S21", "S22"),
        feeds=(((5, 5), (10, 15, 20, 25)), ((5, 5), (10, 15, 0, 5))),  # lower + upper
        make_r_feed=FeedReachability.dual_feed,
        golden_file="golden_loop_dual.json",
    )


def _run_loop(cfg, tmpdir):
    config.device = "cpu"
    config.lr = 0.005
    config["HFSS.lr"] = 0.001
    config["HFSS.min_loss"] = 0.5
    config["HFSS.max_epoch"] = 30
    config.checkpoint_save_path = Path(tmpdir) / "ckpt"

    cfg.setup_responses()                 # 全域響應 (reset + 註冊)；須在建 GEN/SM 前，尺寸才正確

    torch.manual_seed(0)

    sim = _MockSim(cfg.sim_labels)
    AntennaPattern.register_simulator(sim)
    sim.open()                            # 生命週期：開一次

    feeds = [AntennaPattern(torch.ones(shape), coord) for (shape, coord) in cfg.feeds]

    model = SigmoidGEN()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.5, 0.999))
    scheduler = AdaptiveCyclicalScheduler(
        optimizer, T_0=100, T_mult=1, lr_max=0.005, lr_min=1e-6,
        temp_max=4.0, temp_min=0.1, warmup_ratio=0.2, patience=25,
        factor=0.7, mode="min", on_plateau="linear",
    )
    generator = Models(name="g_{label}", rootdir=config.checkpoint_save_path,
                       model=model, optimizer=optimizer, scheduler=scheduler,
                       criterion=custom_loss_minmax)
    smodel = OldSM(checkpoint=config.checkpoint_save_path)

    online = DataManager(f"online_{cfg.name}", rootdir=str(tmpdir), verbose=False)
    TEMP = Record(f"temp_{cfg.name}", rootdir=str(tmpdir))
    r_feed = cfg.make_r_feed()
    sc = SpectralConnectivityLoss()
    gc = GapClosingLoss()

    series = {k: [] for k in ["real_loss", "min_loss", "fake_loss", "r_feed", "tau", "lr"]}

    epoch = 0
    while epoch < EPOCHS:
        epoch += 1
        generator.change(epoch)
        sim.start(epoch)                  # 生命週期：每 epoch
        generator.requires_grad(True, train=True)
        generator.optimizer.zero_grad()

        output_element = AntennaPattern(
            generator(AntennaResponse.target.concat(), tau=generator.scheduler.get_temp())
        )
        for f in feeds:                   # 疊固定饋電塊 (single: lower；dual: lower+upper)
            output_element = output_element + f

        if "patch_pattern_buf" not in TEMP or TEMP.index("patch_pattern_buf", ~output_element) is None:
            result = output_element.simulate()
            real_loss = result.criterion()
            stack = result.stack()
            smodel.train_one_data(output_element.series, stack, verbose=False)
            TEMP["real_loss"] = real_loss.item()
            if TEMP("real_loss") < TEMP.average("real_loss"):
                online.add_and_save([~output_element, stack])
        else:
            stack, rl = TEMP.find("patch_pattern_buf", ~output_element,
                                  ("patch_result_buf", "real_loss"))
            real_loss = rl
            TEMP["real_loss"] = rl

        TEMP["real_loss_average"] = TEMP.average("real_loss")

        min_loss = TEMP("min_loss", float("inf"))
        if TEMP("real_loss") <= min_loss:
            min_loss = TEMP("real_loss")
            TEMP["de"] = 0
        else:
            TEMP.add("de", 1, default=0)
        TEMP["min_loss"] = min_loss

        TEMP["patch_pattern_buf"] = ~output_element
        TEMP["patch_result_buf"] = stack
        TEMP["r_feed"] = r_feed(~output_element)

        response = smodel(output_element.series)
        loss = (
            response.criterion()
            + output_element.total_variation_loss(TV_W)
            + output_element.island_suppression_loss(0.0)
            + SC_W * sc.forward(output_element.size_converter(output_shape="B, 1, H, W"))
            + 0.0 * gc.forward(output_element.size_converter(output_shape="B, 1, H, W"))
        )
        loss.backward()
        generator.step(scheduler_param=real_loss)
        generator.model.eval()
        TEMP["fake_loss"] = loss.item()
        TEMP["epoch"] = epoch

        sim.end()                         # 生命週期：每 epoch
        sim.clean()

        series["real_loss"].append(float(TEMP("real_loss")))
        series["min_loss"].append(float(TEMP("min_loss")))
        series["fake_loss"].append(float(TEMP("fake_loss")))
        series["r_feed"].append(float(TEMP("r_feed")))
        series["tau"].append(float(generator.scheduler.get_temp()))
        series["lr"].append(float(optimizer.param_groups[0]["lr"]))

    return series, sim.calls


def _compare_golden(series, golden_file):
    path = os.path.join(os.path.dirname(__file__), golden_file)
    assert len(series["real_loss"]) == EPOCHS
    for k, vals in series.items():
        assert all(v == v for v in vals), f"{k} 含 NaN"
    if os.path.exists(path):
        golden = json.load(open(path, encoding="utf-8"))
        for key, gvals in golden.items():
            assert key in series and len(series[key]) == len(gvals), f"{golden_file} {key} 長度不符"
            for i, (a, b) in enumerate(zip(series[key], gvals)):
                assert abs(a - b) <= 1e-4, (
                    f"[golden drift] {golden_file} {key}[{i}]: 現在={a:.8g} vs 基準={b:.8g}"
                )
    else:
        json.dump(series, open(path, "w", encoding="utf-8"), indent=2)


@pytest.fixture
def _restore_single():
    """dual 測試後把全域響應還原成 single，避免污染後續 (單埠) 測試。"""
    yield
    _setup_single_responses()


def test_baseline_single(tmp_path):
    series, calls = _run_loop(single_cfg(), tmp_path)
    _compare_golden(series, "golden_loop.json")
    assert calls["open"] == 1
    assert calls["start"] == EPOCHS
    assert calls["end"] == EPOCHS
    assert calls["clean"] == EPOCHS


def test_baseline_dual(tmp_path, _restore_single):
    series, calls = _run_loop(dual_cfg(), tmp_path)
    _compare_golden(series, "golden_loop_dual.json")
    assert calls["open"] == 1
    assert calls["start"] == EPOCHS
    assert calls["end"] == EPOCHS
