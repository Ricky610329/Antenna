"""
端到端「行為基準」：用 Mock 模擬器跑精簡但忠實的訓練迴圈，捕捉關鍵數列當 golden。

這是大重構 (Pipeline / Trainer / RollbackPolicy 抽取 + tau 去耦合) 的驗證基石：
之後每個重構步驟都重跑本測試，比對數列是否與 golden_loop.json 一致。

刻意設計：
- Mock 模擬器：給 pattern 回「確定性、可重現」的假響應，取代慢且需 Windows COM 的 HFSS。
- 高 patience：讓 rollback 不觸發。rollback 會呼叫 smodel.train_by_datas()，其 DataLoader
  (shuffle=True) 會引入非確定性；故主迴圈基準避開它，rollback 原語另以單元測試涵蓋。
- 固定種子：torch.manual_seed(0) → 全程可重現 (CPU、無 dropout)。
- 對齊 train_single.py 的迴圈骨架 (generate → +lower → 去重 → simulate → SM 單筆訓練 →
  更新最佳 → SM 預測+正則化 loss → backward → scheduler step)。
"""
import os
import json
from pathlib import Path

import torch

from antenna.utils import config
from antenna import AntennaPattern, AntennaResponse
from antenna.models import Models, SigmoidGEN
from antenna.smodels import OldSM
from antenna.functions import (
    FeedReachability, AdaptiveCyclicalScheduler,
    SpectralConnectivityLoss, GapClosingLoss,
)
from antenna.patch import custom_loss_minmax
from antenna.utils.data import DataManager
from antenna.utils.utils import Record

_GOLDEN_LOOP = os.path.join(os.path.dirname(__file__), "golden_loop.json")

EPOCHS = 6
PATIENCE = 50          # 高到不會觸發 rollback
TV_W = 0.01            # 固定的正則化權重 (讓基準涵蓋 loss 組裝)
SC_W = 0.0005


class _MockSimulator:
    """確定性假模擬器：響應只取決於 pattern 的填充率，無隨機性。"""
    def __call__(self, pattern, **kw):
        fr = pattern.float().mean()
        x = torch.linspace(0, 1, 17)
        bump = torch.exp(-((x - 0.5) ** 2) / 0.05)   # 中央凸起
        s11 = -12.0 * bump * fr
        gain = 4.0 * bump * fr - 19.0 * (1.0 - fr)
        return {"S11": s11, "Gain": gain}

    def restart(self, **kw):   # 容錯介面 (mock 不會用到)
        pass


def _run_baseline(tmpdir):
    config.device = "cpu"
    config.lr = 0.005
    config["HFSS.lr"] = 0.001
    config["HFSS.min_loss"] = 0.5      # 提高門檻 + 限縮 max_epoch → 測試快
    config["HFSS.max_epoch"] = 30
    config.checkpoint_save_path = Path(tmpdir) / "ckpt"

    torch.manual_seed(0)

    AntennaPattern.register_simulator(_MockSimulator())
    lower = AntennaPattern(torch.ones((5, 5)), (10, 15, 20, 25))

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

    online = DataManager("online_baseline", rootdir=str(tmpdir), verbose=False)
    TEMP = Record("temp_baseline", rootdir=str(tmpdir))
    r_feed = FeedReachability.single_feed()
    sc = SpectralConnectivityLoss()
    gc = GapClosingLoss()

    series = {k: [] for k in ["real_loss", "min_loss", "fake_loss", "r_feed", "tau", "lr"]}

    epoch = 0
    while epoch < EPOCHS:
        epoch += 1
        generator.change(epoch)
        generator.requires_grad(True, train=True)
        generator.optimizer.zero_grad()

        # (高 patience → early_stop 永遠 False，不走 rollback 分支)
        output_element = AntennaPattern(generator(AntennaResponse.target.concat()))
        output_element = output_element + lower

        # 去重
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

        # 更新最佳
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

        # 更新 GEN (借道 SM)
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

        series["real_loss"].append(float(TEMP("real_loss")))
        series["min_loss"].append(float(TEMP("min_loss")))
        series["fake_loss"].append(float(TEMP("fake_loss")))
        series["r_feed"].append(float(TEMP("r_feed")))
        series["tau"].append(float(AntennaPattern.tau))
        series["lr"].append(float(optimizer.param_groups[0]["lr"]))

    return series


def test_baseline_loop(tmp_path):
    series = _run_baseline(tmp_path)

    # 基本健全性
    assert len(series["real_loss"]) == EPOCHS
    for k, vals in series.items():
        assert all(isinstance(v, float) for v in vals)
        assert all(v == v for v in vals), f"{k} 含 NaN"   # NaN 檢查

    # golden 比對 (第一次跑自動產生)
    if os.path.exists(_GOLDEN_LOOP):
        golden = json.load(open(_GOLDEN_LOOP, encoding="utf-8"))
        for key, gvals in golden.items():
            assert key in series, f"缺少數列 {key}"
            assert len(series[key]) == len(gvals), f"{key} 長度不符"
            for i, (a, b) in enumerate(zip(series[key], gvals)):
                assert abs(a - b) <= 1e-4, (
                    f"[loop golden drift] {key}[{i}]: 現在={a:.8g} vs 基準={b:.8g}"
                )
    else:
        json.dump(series, open(_GOLDEN_LOOP, "w", encoding="utf-8"), indent=2)
