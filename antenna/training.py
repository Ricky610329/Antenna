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
from antenna import zoo
from antenna.models import Models
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
    generator: dict = field(default_factory=dict)   # GEN：zoo 名字 (字串) 或 {name, hidden, pretrained}
    surrogate: dict = field(default_factory=dict)   # SM：{name, hidden, pretrained, offline_dataset, warmup}
    targets: dict = field(default_factory=dict)     # {label: {side, center, width, method|interval}}

    def __post_init__(self):
        if self.port not in PORT_SPECS:
            raise ValueError(f"port 必須是 {list(PORT_SPECS)}，但得到 {self.port!r}")
        need = set(PORT_SPECS[self.port]["labels"])
        missing = need - set(self.targets)
        if missing:
            raise ValueError(f"port={self.port} 缺少 targets: {sorted(missing)}")
        if isinstance(self.generator, str):     # 簡寫 generator: sigmoid → {name: sigmoid}
            self.generator = {"name": self.generator}


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


# ── 模型建構：名字查 antenna/zoo.py，維度在此推好傳入 (模型不碰全域註冊狀態)。
#    架構參數 (如 hidden) 直接透傳給模型；載入欄位 (pretrained 等) 不屬於架構，先剔除。
def _arch_params(section: dict, loading_keys=("name", "pretrained", "offline_dataset", "warmup")):
    params = {k: v for k, v in section.items() if k not in loading_keys}
    if "hidden" in params:
        params["hidden"] = tuple(params["hidden"])
    return params


def build_generator(cfg: TrainConfig):
    """依 cfg.generator (zoo 名字 + 架構參數) 建生成器 GEN。未指定 → sigmoid 預設。"""
    name = cfg.generator.get("name", "sigmoid")
    if name not in zoo.GENERATORS:
        raise ValueError(f"未知的 generator {name!r}，可用: {sorted(zoo.GENERATORS)} (見 antenna/zoo.py)")
    return zoo.GENERATORS[name](
        AntennaResponse.size(flatten=True), AntennaPattern.size(flatten=True),
        **_arch_params(cfg.generator),
    )


def build_surrogate(cfg: TrainConfig, checkpoint):
    """依 cfg.surrogate (zoo 名字 + 架構參數) 建代理模型 SM。未指定 → mlp 預設。"""
    name = cfg.surrogate.get("name", "mlp")
    if name not in zoo.SURROGATES:
        raise ValueError(f"未知的 surrogate {name!r}，可用: {sorted(zoo.SURROGATES)} (見 antenna/zoo.py)")
    return zoo.SURROGATES[name](
        checkpoint, AntennaPattern.size(flatten=True), AntennaResponse.size(),
        **_arch_params(cfg.surrogate),
    )


def run_training(
    cfg: TrainConfig,
    *,
    simulator,                                  # 注入：production 真實模擬器 / 測試 mock
    record_path,                                # 結果根目錄 (checkpoint/online/temp)
    seed: Optional[int] = None,                 # 設種子以求可重現 (測試用)
    max_epochs: Optional[int] = None,           # 覆寫 cfg.epochs (測試用)
    patience: Optional[int] = None,             # 覆寫 cfg.patience (測試用)
    on_epoch: Optional[Callable[[int, dict], None]] = None,  # 每 epoch 回呼 (繪圖/記錄/捕捉)
    continue_run: bool = False,                 # 結果夾已存在 → 嘗試斷點續跑
    sm_pretrained_path: Optional[str] = None,   # 預訓練 SM 權重檔 (L1)
    gen_pretrained_path: Optional[str] = None,  # 預載入 GEN 權重檔 (L2 暖啟動)
    offline_dataset=None,                       # 離線資料集 (無預訓練檔時用來預訓練 SM)
    warmup=None,                                # KuoHung 暖身：(pattern, response) 對 SM 做單筆訓練
) -> Record:
    """單/雙埠共用訓練迴圈。回傳 TEMP(Record)。"""
    record_path = Path(record_path)
    config.device = "cpu"
    config.lr = cfg.lr
    config["HFSS.lr"] = cfg.hfss.get("lr", 0.001)
    config["HFSS.min_loss"] = cfg.hfss.get("min_loss", 0.1)
    config["HFSS.max_epoch"] = cfg.hfss.get("max_epoch", 20000)
    config.checkpoint_save_path = record_path / "checkpoint"
    (record_path / "checkpoint").mkdir(parents=True, exist_ok=True)   # GEN/SM 權重存放處

    setup_responses(cfg)                        # 須在建 GEN/SM 前 (尺寸才正確)
    if seed is not None:
        torch.manual_seed(seed)

    AntennaPattern.register_simulator(simulator)
    simulator.open()
    feeds = build_feeds(cfg)

    model = build_generator(cfg)                # 架構查 zoo (cfg.generator 的名字 + 參數)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    scheduler = AdaptiveCyclicalScheduler(
        optimizer, T_0=100, T_mult=1, lr_max=cfg.lr, lr_min=1e-6,
        temp_max=4.0, temp_min=0.1, warmup_ratio=0.2, patience=25,
        factor=0.7, mode="min", on_plateau=cfg.scheduler.get("on_plateau", "linear"),
    )
    generator = Models(name="generator_{label}", rootdir=config.checkpoint_save_path,
                       model=model, optimizer=optimizer, scheduler=scheduler,
                       criterion=custom_loss_minmax)
    smodel = build_surrogate(cfg, config.checkpoint_save_path)   # 架構由 cfg.surrogate (type/hidden) 決定

    online = DataManager("online", rootdir=str(record_path), verbose=False)
    TEMP = Record("temp", rootdir=str(record_path))
    r_feed = PORT_SPECS[cfg.port]["make_r_feed"]()
    sc = SpectralConnectivityLoss()
    gc = GapClosingLoss()

    tv_w = cfg.loss.get("total_variation", 0.0)
    is_w = cfg.loss.get("island_suppression", 0.0)
    sc_w = cfg.loss.get("spectral_connectivity", 0.0)
    gap_w = cfg.loss.get("gap_closing", 0.0)

    # 模型載入 (續跑 / GEN 預載入 / SM 預訓練 / KuoHung 暖身)；回傳續跑起始 epoch
    start_epoch = prepare_models(
        cfg, generator, smodel, TEMP,
        continue_run=continue_run,
        gen_pretrained_path=gen_pretrained_path,
        sm_pretrained_path=sm_pretrained_path,
        offline_dataset=offline_dataset,
        warmup=warmup,
    )

    epochs = max_epochs if max_epochs is not None else cfg.epochs
    pat = patience if patience is not None else cfg.patience

    epoch = start_epoch
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

        # 生成：模型只出 logits；STE 二值化是管線的固定一步，tau 由 ACP 單獨控制
        logits = generator(AntennaResponse.target.concat())
        output_element = AntennaPattern(
            AntennaPattern.binarization(logits, generator.scheduler.get_temp())
        )
        for f in feeds:
            output_element = output_element + f

        # 去重：沒模擬過才跑 (mock/HFSS)
        if "patch_pattern_buf" not in TEMP or TEMP.index("patch_pattern_buf", ~output_element) is None:
            result = output_element.simulate()
            real_loss = result.criterion()
            stack = result.stack()
            smodel.train_one_data(output_element.series, stack, verbose=False)
            smodel.save()                       # 存 SM (斷點續跑 / rollback 重訓基礎)
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

        generator.save()                       # 存 GEN (供 rollback 載回 / 斷點續跑)

        simulator.end()
        simulator.clean()

        TEMP["epoch"] = epoch
        TEMP.save(f"{epoch} times")            # 斷點續跑檢查點

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


def prepare_models(cfg, generator, smodel, TEMP, *, continue_run=False,
                   gen_pretrained_path=None, sm_pretrained_path=None,
                   offline_dataset=None, warmup=None) -> int:
    """GEN/SM 的「載入策略」(模組化、由 config 的 generator/surrogate 區段指定)。
    回傳續跑起始 epoch。

    優先序：
      (1) 斷點續跑：結果夾已存在且 TEMP 有 epoch → 載回 GEN/SM，從上次 epoch 續跑 (其餘略過)。
      (2) GEN 預載入 (L2 暖啟動)：gen_pretrained_path 存在 → 載入 GEN 權重。
      (3) SM 載入 (L1)：sm_pretrained_path 存在 → 直接載入；否則若有離線資料集 → 從頭預訓練。
      (4) KuoHung 暖身：warmup 是可呼叫物 warmup(smodel) → 對 SM 做單筆暖身訓練
          (補齊舊 single 3/4 行為；資料與收斂門檻由呼叫端綁定，prepare_models 不耦合 KuoHung)。
    皆無 → GEN/SM 從隨機權重開始 (純靠線上學習)。
    """
    # (1) 斷點續跑：載回 GEN+SM，後續載入策略全部略過
    if continue_run and ("epoch" in TEMP):
        last = TEMP("epoch")
        generator.change(last, load=True)
        smodel.load()
        return int(last)
    # (2) GEN 預載入 (暖啟動)
    if gen_pretrained_path is not None and Path(gen_pretrained_path).exists():
        generator.pre_load_model(gen_pretrained_path)
    # (3) SM 載入：預訓練檔 > 離線預訓練
    if sm_pretrained_path is not None and Path(sm_pretrained_path).exists():
        smodel.pre_load_model(sm_pretrained_path)
    elif offline_dataset is not None and len(offline_dataset) > 0:
        smodel.train_by_datas(offline_dataset)
    # (4) KuoHung 暖身：呼叫端綁好的 warmup(smodel)，對 SM 做單筆暖身訓練
    if warmup is not None:
        warmup(smodel)
    return 0


def build_simulator(cfg: "TrainConfig", record_path):
    """依 port 建真實 HFSS 模擬器 (production 用；測試以 mock 注入)。"""
    from antenna.patch import SinglePortSimulator, DualPortSimulator
    cls = {"single": SinglePortSimulator, "dual": DualPortSimulator}[cfg.port]
    return cls(record_path=record_path)
