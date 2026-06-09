"""
antenna/training.py — 由「外部 YAML config」驅動的單/雙埠共用訓練核心。

設計（與 docs/refactor-proposal.html 一致）：
- 一個 YAML = 一組實驗的完整設定 (一檔一實驗)。
- YAML 只放「純量/資料」+ 一個 `port: single|dual` 選擇器；
  「結構性」元件 (模擬器 class、損失函式、饋電塊座標、FeedReachability) 由 port
  在 code 裡解析 (PORT_SPECS)，不放 YAML。
- run_training() 是單/雙埠共用的訓練迴圈；模擬器以參數注入 (production 用真實
  Single/DualPortSimulator，測試用 mock)，故可被 golden 測試驗證行為不變。

對應關係：
    port=single → SinglePortSimulator + custom_loss_minmax + [lower]        + single_feed + S11/Gain
    port=dual   → DualPortSimulator   + interval_loss      + [lower, upper] + dual_feed   + S11/S21/S22
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

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


# ── port → 結構性元件 (不放 YAML，由 code 解析) ──────────────────────────────
#! register_order 與 labels 刻意不同 (dual)：labels 決定 criterion/SM 的標籤順序，
#! 但 registerTargetResponse 的「註冊順序」決定 target.concat()(GEN 輸入向量)的排列。
#! train_dual.py 原始註冊順序是 S11→S22→S21 (S22 在 S21 前)，為保留行為必須照舊。
PORT_SPECS = {
    "single": dict(
        labels=["S11", "Gain"],
        register_order=["S11", "Gain"],
        feeds=[((5, 5), (10, 15, 20, 25))],               # 只有 lower (底部中央)
        make_r_feed=FeedReachability.single_feed,
    ),
    "dual": dict(
        labels=["S11", "S21", "S22"],
        register_order=["S11", "S22", "S21"],             # ← GEN 輸入向量排列 (與 train_dual 一致)
        feeds=[((5, 5), (10, 15, 20, 25)), ((5, 5), (10, 15, 0, 5))],  # lower + upper
        make_r_feed=FeedReachability.dual_feed,
    ),
}


@dataclass
class TrainConfig:
    """一組實驗的設定 (從 YAML 載入)。"""
    name: str
    port: str                                       # "single" | "dual"
    epochs: int = 1000
    lr: float = 0.005
    patience: int = 10
    loss: dict = field(default_factory=dict)        # total_variation / island_suppression / spectral_connectivity / gap_closing
    hfss: dict = field(default_factory=dict)        # lr / min_loss / max_epoch (代理模型線上訓練)
    scheduler: dict = field(default_factory=dict)   # on_plateau
    targets: dict = field(default_factory=dict)     # {label: {side, center, width, method|interval}}

    def __post_init__(self):
        if self.port not in PORT_SPECS:
            raise ValueError(f"port 必須是 {list(PORT_SPECS)}，但得到 {self.port!r}")
        need = set(PORT_SPECS[self.port]["labels"])
        missing = need - set(self.targets)
        if missing:
            raise ValueError(f"port={self.port} 缺少 targets: {sorted(missing)}")


def load_config(path) -> TrainConfig:
    """讀取 YAML → TrainConfig。"""
    import yaml
    data = yaml.safe_load(open(path, encoding="utf-8"))
    allowed = set(TrainConfig.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"YAML 含未知欄位: {sorted(unknown)} (允許: {sorted(allowed)})")
    return TrainConfig(**data)


def setup_responses(cfg: TrainConfig):
    """依 port + cfg.targets 註冊響應標籤、目標曲線與損失 hook (全域狀態)。"""
    spec = PORT_SPECS[cfg.port]
    AntennaResponse.target = TargetResponse()                  # reset，避免殘留污染
    AntennaResponse.registerLabels(*spec["labels"], x="n257")  # labels 順序 (criterion/SM)
    for label in spec["register_order"]:                       # 註冊順序 (決定 GEN 輸入 concat 排列)
        t = cfg.targets[label]
        resp = AntennaResponse.registerTargetResponse(
            t["side"], t["center"], tuple(t["width"]), label=label
        )
        if cfg.port == "single":
            AntennaResponse.registerLossHook(
                custom_loss_minmax, label=label, target=resp, method=t["method"]
            )
        else:  # dual
            lo, hi = t.get("interval", [-1, 1])
            AntennaResponse.registerLossHook(
                interval_loss, label=label, lower_response=lo, upper_response=hi, target=resp
            )


def build_feeds(cfg: TrainConfig):
    """依 port 建固定饋電金屬塊 (AntennaPattern)。"""
    return [AntennaPattern(torch.ones(shape), coord)
            for (shape, coord) in PORT_SPECS[cfg.port]["feeds"]]


def run_training(
    cfg: TrainConfig,
    *,
    simulator,                                  # 注入：production 真實模擬器 / 測試 mock
    record_path,                                # 結果根目錄 (checkpoint/online/temp)
    seed: Optional[int] = None,                 # 設種子以求可重現 (測試用)
    max_epochs: Optional[int] = None,           # 覆寫 cfg.epochs (測試用)
    patience: Optional[int] = None,             # 覆寫 cfg.patience (測試用)
    on_epoch: Optional[Callable[[int, dict], None]] = None,  # 每 epoch 回呼 (繪圖/記錄/捕捉)
) -> Record:
    """單/雙埠共用訓練迴圈。回傳 TEMP(Record)。"""
    record_path = Path(record_path)
    config.device = "cpu"
    config.lr = cfg.lr
    config["HFSS.lr"] = cfg.hfss.get("lr", 0.001)
    config["HFSS.min_loss"] = cfg.hfss.get("min_loss", 0.1)
    config["HFSS.max_epoch"] = cfg.hfss.get("max_epoch", 20000)
    config.checkpoint_save_path = record_path / "checkpoint"

    setup_responses(cfg)                        # 須在建 GEN/SM 前 (尺寸才正確)
    if seed is not None:
        torch.manual_seed(seed)

    AntennaPattern.register_simulator(simulator)
    simulator.open()
    feeds = build_feeds(cfg)

    model = SigmoidGEN()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    scheduler = AdaptiveCyclicalScheduler(
        optimizer, T_0=100, T_mult=1, lr_max=cfg.lr, lr_min=1e-6,
        temp_max=4.0, temp_min=0.1, warmup_ratio=0.2, patience=25,
        factor=0.7, mode="min", on_plateau=cfg.scheduler.get("on_plateau", "linear"),
    )
    generator = Models(name="generator_{label}", rootdir=config.checkpoint_save_path,
                       model=model, optimizer=optimizer, scheduler=scheduler,
                       criterion=custom_loss_minmax)
    smodel = OldSM(checkpoint=config.checkpoint_save_path)

    online = DataManager("online", rootdir=str(record_path), verbose=False)
    TEMP = Record("temp", rootdir=str(record_path))
    r_feed = PORT_SPECS[cfg.port]["make_r_feed"]()
    sc = SpectralConnectivityLoss()
    gc = GapClosingLoss()

    tv_w = cfg.loss.get("total_variation", 0.0)
    is_w = cfg.loss.get("island_suppression", 0.0)
    sc_w = cfg.loss.get("spectral_connectivity", 0.0)
    gap_w = cfg.loss.get("gap_closing", 0.0)

    epochs = max_epochs if max_epochs is not None else cfg.epochs
    pat = patience if patience is not None else cfg.patience

    epoch = 0
    while epoch < epochs:
        epoch += 1
        generator.change(epoch)
        simulator.start(epoch)
        generator.requires_grad(True, train=True)
        generator.optimizer.zero_grad()

        # 早停 → 回滾 (測試用高 patience 不觸發)
        if TEMP.early_stop("real_loss", pat):
            generator.change(
                TEMP.find("real_loss", TEMP("min_loss", float("inf")), "epoch"),
                save=True, load=True,
            )
            smodel.train_by_datas(online)
            output_element = AntennaPattern(
                generator(AntennaResponse.target.concat(), tau=generator.scheduler.get_temp())
            )
        else:
            output_element = AntennaPattern(
                generator(AntennaResponse.target.concat(), tau=generator.scheduler.get_temp())
            )
        for f in feeds:
            output_element = output_element + f

        # 去重：沒模擬過才跑 (mock/HFSS)
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

        # 更新 GEN (借道可微分 SM)
        response = smodel(output_element.series)
        loss = (
            response.criterion()
            + output_element.total_variation_loss(tv_w)
            + output_element.island_suppression_loss(is_w)
            + sc_w * sc.forward(output_element.size_converter(output_shape="B, 1, H, W"))
            + gap_w * gc.forward(output_element.size_converter(output_shape="B, 1, H, W"))
        )
        loss.backward()
        generator.step(scheduler_param=real_loss)
        generator.model.eval()
        TEMP["fake_loss"] = loss.item()
        TEMP["epoch"] = epoch

        simulator.end()
        simulator.clean()

        if on_epoch is not None:
            on_epoch(epoch, dict(
                real_loss=float(TEMP("real_loss")),
                min_loss=float(TEMP("min_loss")),
                fake_loss=float(TEMP("fake_loss")),
                r_feed=float(TEMP("r_feed")),
                tau=float(generator.scheduler.get_temp()),
                lr=float(optimizer.param_groups[0]["lr"]),
            ))

    return TEMP
