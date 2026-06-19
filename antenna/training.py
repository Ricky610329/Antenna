"""
antenna/training.py — 由「外部 YAML config」驅動的單/雙埠共用訓練核心。

設計：
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
from time import time
from typing import Optional, Callable

import torch

from antenna.utils import config, logger
from antenna import AntennaPattern, AntennaResponse, TargetResponse
from antenna import zoo
from antenna.models import Models
from antenna.losses import FeedReachability, SpectralConnectivityLoss, GapClosingLoss
from antenna.optim import AdaptiveCyclicalScheduler
from antenna.losses import custom_loss_minmax, interval_loss, beam_coverage_loss
from antenna.utils.store import SampleStore
from antenna.utils.runstate import RunState


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


#! 各區段允許的鍵 (白名單)。鍵打錯「必須報錯」而不是默默用預設值 ——
#! 歷史教訓：舊 dual 的 island_suppression 鍵名打錯，正則化默默沒開、實驗白跑。
SECTION_KEYS = {
    "loss":      {"total_variation", "island_suppression", "spectral_connectivity", "gap_closing"},
    "sm_train":  {"lr", "min_loss", "max_epoch"},
    "scheduler": {"on_plateau", "T_0", "T_mult", "lr_min", "temp_max", "temp_min",
                  "warmup_ratio", "patience", "factor"},
    "generator": {"name", "hidden", "pretrained"},
    "surrogate": {"name", "hidden", "pretrained", "offline_dataset", "warmup"},
    "radiation": {"enable", "weight", "window_deg", "floor_db", "boresight_weight", "warmup_epochs", "n_theta"},
}
TARGET_KEYS = {"side", "center", "width", "method", "interval"}


@dataclass
class TrainConfig:
    """一組實驗的設定 (從 YAML 載入)。"""
    name: str
    port: str                                       # "single" | "dual"
    epochs: int = 1000
    lr: float = 0.005
    patience: int = 10
    seed: Optional[int] = None                      # 設了 → torch.manual_seed (可重現性)
    loss: dict = field(default_factory=dict)        # 見 SECTION_KEYS["loss"]
    sm_train: dict = field(default_factory=dict)    # lr / min_loss / max_epoch (代理模型線上訓練)
    scheduler: dict = field(default_factory=dict)   # ACP 超參數 (見 SECTION_KEYS["scheduler"])
    generator: dict = field(default_factory=dict)   # GEN：zoo 名字 (字串) 或 {name, hidden, pretrained}
    surrogate: dict = field(default_factory=dict)   # SM：{name, hidden, pretrained, offline_dataset, warmup}
    targets: dict = field(default_factory=dict)     # {label: {side, center, width, method|interval}}
    radiation: dict = field(default_factory=dict)   # 方向圖 (選用，預設 off)：見 SECTION_KEYS["radiation"]

    def __post_init__(self):
        if self.port not in PORT_SPECS:
            raise ValueError(f"port 必須是 {list(PORT_SPECS)}，但得到 {self.port!r}")
        need = set(PORT_SPECS[self.port]["labels"])
        missing = need - set(self.targets)
        if missing:
            raise ValueError(f"port={self.port} 缺少 targets: {sorted(missing)}")
        if isinstance(self.generator, str):     # 簡寫 generator: sigmoid → {name: sigmoid}
            self.generator = {"name": self.generator}
        # 區段鍵白名單驗證 (打錯鍵 → 明確報錯，不默默吃預設)
        for section, allowed in SECTION_KEYS.items():
            unknown = set(getattr(self, section)) - allowed
            if unknown:
                raise ValueError(f"{section} 區段含未知鍵: {sorted(unknown)} (允許: {sorted(allowed)})")
        for label, t in self.targets.items():
            unknown = set(t) - TARGET_KEYS
            if unknown:
                raise ValueError(f"targets.{label} 含未知鍵: {sorted(unknown)} (允許: {sorted(TARGET_KEYS)})")


def load_config(path) -> TrainConfig:
    """讀取 YAML → TrainConfig。"""
    import yaml
    data = yaml.safe_load(open(path, encoding="utf-8"))
    allowed = set(TrainConfig.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"YAML 含未知欄位: {sorted(unknown)} (允許: {sorted(allowed)})")
    return TrainConfig(**data)


def setup_responses(cfg: TrainConfig) -> TargetResponse:
    """依 port + cfg.targets 建一組「全新的」響應規格 (spec) 並安裝。

    建構過程不碰全域狀態，最後以 AntennaResponse.use(spec) 原子安裝；
    回傳 spec 實例 —— 訓練端後續的維度/GEN 輸入請拿著它讀，不要讀類別狀態。
    """
    port = PORT_SPECS[cfg.port]
    spec = TargetResponse(labels=port["labels"], x="n257")     # labels 順序 (criterion/SM 對齊)
    for label in port["register_order"]:                       # 加入順序 (決定 GEN 輸入 concat 排列)
        t = cfg.targets[label]
        resp = spec(t["side"], t["center"], tuple(t["width"]), label=label, add=True)
        if cfg.port == "single":
            spec.register_loss_fn(label, custom_loss_minmax, target=resp, method=t["method"])
        else:  # dual
            lo, hi = t.get("interval", [-1, 1])
            spec.register_loss_fn(label, interval_loss, lower_response=lo, upper_response=hi, target=resp)
    return AntennaResponse.use(spec)


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


def build_scheduler(cfg: TrainConfig, optimizer):
    """依 cfg.scheduler 建 ACP (AdaptiveCyclicalScheduler)。
    ACP 是論文核心機制，超參數可在 YAML 調整；預設值 = 原 train_single/dual 的設定。
    lr_max 與 cfg.lr 綁定 (週期高點 = 訓練 lr)；mode 固定 min (監控 sim_loss)。"""
    s = cfg.scheduler
    return AdaptiveCyclicalScheduler(
        optimizer,
        T_0=s.get("T_0", 100), T_mult=s.get("T_mult", 1),
        lr_max=cfg.lr, lr_min=s.get("lr_min", 1e-6),
        temp_max=s.get("temp_max", 4.0), temp_min=s.get("temp_min", 0.1),
        warmup_ratio=s.get("warmup_ratio", 0.2),
        patience=s.get("patience", 25), factor=s.get("factor", 0.7),
        mode="min", on_plateau=s.get("on_plateau", "linear"),
    )


def build_generator(cfg: TrainConfig, spec: TargetResponse):
    """依 cfg.generator (zoo 名字 + 架構參數) 建生成器 GEN。未指定 → sigmoid 預設。

    響應維度從傳入的 spec 讀 (顯式資料流)，不讀 AntennaResponse 類別狀態。"""
    name = cfg.generator.get("name", "sigmoid")
    if name not in zoo.GENERATORS:
        raise ValueError(f"未知的 generator {name!r}，可用: {sorted(zoo.GENERATORS)} (見 antenna/zoo.py)")
    return zoo.GENERATORS[name](
        spec.size(flatten=True), AntennaPattern.size(flatten=True),
        **_arch_params(cfg.generator),
    )


def build_surrogate(cfg: TrainConfig, checkpoint, spec: TargetResponse):
    """依 cfg.surrogate (zoo 名字 + 架構參數) 建代理模型 SM。未指定 → mlp 預設。

    YAML 的 sm_train 區段 (lr / 單筆訓練門檻) 顯式傳入 SM；響應維度從 spec 讀。"""
    name = cfg.surrogate.get("name", "mlp")
    if name not in zoo.SURROGATES:
        raise ValueError(f"未知的 surrogate {name!r}，可用: {sorted(zoo.SURROGATES)} (見 antenna/zoo.py)")
    #? 方向圖開啟 → 多建一個方向圖頭 (n_phi=2: phi0/phi90；n_theta 由 config，預設 181=3D 球 step2°)。
    #  關閉時 rad_response=None → SM 與原樣完全相同 (golden 零漂移)。
    rad_response = (2, cfg.radiation.get("n_theta", 181)) if cfg.radiation.get("enable", False) else None
    return zoo.SURROGATES[name](
        checkpoint, AntennaPattern.size(flatten=True), spec.size(),
        lr=cfg.sm_train.get("lr", 0.001),
        min_loss=cfg.sm_train.get("min_loss", 0.1),
        max_epoch=cfg.sm_train.get("max_epoch", 20000),
        rad_response=rad_response,
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
    verbose: bool = True,                       # SM 訓練進度條 + 慢步驟 log (測試關閉)
) -> RunState:
    """單/雙埠共用訓練迴圈。回傳 RunState (metrics.csv + patterns/ 的訓練狀態)。"""
    record_path = Path(record_path)
    config.device = "cpu"
    config.checkpoint_save_path = record_path / "checkpoint"
    (record_path / "checkpoint").mkdir(parents=True, exist_ok=True)   # GEN/SM 權重存放處

    spec = setup_responses(cfg)                 # 建立並安裝響應規格；後續維度/GEN 輸入都從 spec 讀
    seed = seed if seed is not None else cfg.seed   # 參數優先，否則吃 YAML 的 seed (可重現性)
    if seed is not None:
        torch.manual_seed(seed)

    AntennaPattern.register_simulator(simulator)
    if verbose: logger.info("啟動模擬器 (HFSS COM)…")
    simulator.open()
    feeds = build_feeds(cfg)

    model = build_generator(cfg, spec)          # 架構查 zoo；維度從 spec (顯式，不依賴安裝時序)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    scheduler = build_scheduler(cfg, optimizer)
    generator = Models(name="generator_{label}", rootdir=config.checkpoint_save_path,
                       model=model, optimizer=optimizer, scheduler=scheduler,
                       criterion=custom_loss_minmax)
    smodel = build_surrogate(cfg, config.checkpoint_save_path, spec)

    #? online 樣本庫用新格式 (一筆一檔，見 antenna/utils/store.py)：append O(1)、去重免維護。
    #! 舊 run 的 online.data (單一 pickle) 不會帶進來 —— 續跑時 rollback 從新樣本重新累積
    #! (SM checkpoint 不受影響；train_by_datas 對空資料集是 no-op)。
    online = SampleStore(record_path / "online", verbose=False)
    #? 訓練狀態走「結果夾即資料庫」：metrics.csv (純量) + patterns/ (模擬快取)。
    #? 取代舊 temp.record —— 那是每 epoch 全量重寫的單一 pickle (最後的 O(n²) NAS 寫入者)。
    state = RunState(record_path, verbose=verbose)
    r_feed = PORT_SPECS[cfg.port]["make_r_feed"]()
    sc = SpectralConnectivityLoss()
    gc = GapClosingLoss()

    tv_w = cfg.loss.get("total_variation", 0.0)
    is_w = cfg.loss.get("island_suppression", 0.0)
    sc_w = cfg.loss.get("spectral_connectivity", 0.0)
    gap_w = cfg.loss.get("gap_closing", 0.0)

    # 方向圖 (選用，預設 off)：開啟時 SM 已建方向圖頭、simulator 為 SinglePortRadSimulator。
    # 全程用 rad_on 閘住；rad_on=False → 下方所有方向圖分支不執行，行為與原樣完全相同 (golden 零漂移)。
    rad_on = cfg.radiation.get("enable", False)
    rad_w = cfg.radiation.get("weight", 1.0)
    rad_window = cfg.radiation.get("window_deg", 55.0)
    rad_floor = cfg.radiation.get("floor_db", 3.0)
    rad_bore_w = cfg.radiation.get("boresight_weight", 1.0)
    rad_warmup = cfg.radiation.get("warmup_epochs", 0)
    rad_theta = None        # 從第一筆方向圖資料擷取 (角度網格固定，整個 run 不變)

    # 模型載入 (續跑 / GEN 預載入 / SM 預訓練 / KuoHung 暖身)；回傳續跑起始 epoch
    start_epoch = prepare_models(
        cfg, generator, smodel, state,
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
        epoch_t0 = time()
        generator.change(epoch)
        simulator.start(epoch)
        generator.requires_grad(True, train=True)
        generator.optimizer.zero_grad()
        rad_real_this_epoch = None      # 本 epoch 真實方向圖 (僅非快取分支會填；供監控疊圖)

        # 早停 → 回滾 (測試用高 patience 不觸發)
        if state.early_stop("sim_loss", pat):
            if verbose: logger.info(f"[{epoch}] sim_loss 連續 {pat} 次未改善 → 回滾至最佳 epoch、重訓 SM (online {len(online)} 筆)")
            generator.change(state.best_epoch("sim_loss"), save=True, load=True)
            smodel.train_by_datas(online, verbose=verbose)

        # 生成：模型只出 logits；STE 二值化是管線的固定一步，tau 由 ACP 單獨控制
        logits = generator(spec.concat())
        output_element = AntennaPattern(
            AntennaPattern.binarization(logits, generator.scheduler.get_temp())
        )
        for f in feeds:
            output_element = output_element + f

        # 去重：patterns/ 的 hash 檔名即「模擬過」快取，沒見過才跑 (mock/HFSS)
        cached = state.lookup(~output_element)
        if cached is None:
            if verbose: logger.info(f"[{epoch}] HFSS 模擬中…")
            result = output_element.simulate()
            sim_loss = result.criterion()
            stack = result.stack()
            #? verbose=True 時顯示 SM 單筆訓練的 tqdm 進度條 (與舊腳本行為一致)
            smodel.train_one_data(output_element.series, stack, verbose=verbose)
            # 方向圖 (選用)：讀 SinglePortRadSimulator.last_radiation，線上訓練方向圖頭 (trunk 不凍)。
            if rad_on:
                rad = getattr(simulator, "last_radiation", None)
                if isinstance(rad, dict) and rad.get("theta") is not None:
                    rad_theta = rad["theta"]
                    rad_stack = torch.stack([rad["phi0"], rad["phi90"]])    # (2, n_theta)
                    rad_real_this_epoch = rad_stack.detach().cpu()          # 監控疊圖用 (真實 HFSS 方向圖)
                    smodel.train_one_data_rad(output_element.series, rad_stack, verbose=verbose)
            smodel.save()                       # 存 SM (斷點續跑 / rollback 重訓基礎)
            state.append("sim_loss", sim_loss.item())
            phash = state.add_pattern(~output_element, stack, sim_loss.item())
            if state.last("sim_loss") < state.average("sim_loss"):
                online.add(~output_element, stack)
        else:
            if verbose: logger.info(f"[{epoch}] pattern 重複 → 取快取結果 (跳過 HFSS)")
            stack, sim_loss, phash = cached
            state.append("sim_loss", sim_loss)

        state.append("sim_loss_avg", state.average("sim_loss"))

        best_loss = state.last("best_loss", float("inf"))
        if state.last("sim_loss") <= best_loss:
            best_loss = state.last("sim_loss")
        state.append("best_loss", best_loss)

        state.append("pattern_hash", phash)
        state.append("r_feed", r_feed(~output_element))

        # 更新 GEN (借道可微分 SM)
        response = smodel(output_element.series)
        loss = (
            response.criterion()
            + output_element.total_variation_loss(tv_w)
            + output_element.island_suppression_loss(is_w)
            + sc_w * sc.forward(output_element.size_converter(output_shape="B, 1, H, W"))
            + gap_w * gc.forward(output_element.size_converter(output_shape="B, 1, H, W"))
        )
        # 方向圖 loss (選用)：經方向圖頭可微 → 加進 GEN loss 一起反傳。
        # 有方向圖資料就先「預測一次」(供監控疊圖)；但 warmup_epochs 內先不讓它進 loss
        # (方向圖頭還在冷啟動，避免雜訊梯度把 G 帶歪)。
        rad_loss_val = 0.0
        rad_snapshot = None
        if rad_on and rad_theta is not None:
            rad_pred = smodel.rad_predict(output_element.series)
            if epoch > rad_warmup:
                rad_loss = beam_coverage_loss(
                    rad_pred, rad_theta,
                    window_deg=rad_window, floor_db=rad_floor, boresight_weight=rad_bore_w,
                )
                loss = loss + rad_w * rad_loss
                rad_loss_val = float(rad_loss)
            rad_snapshot = dict(                        # 監控素材 (參考 pattern 的傳法，交給 monitor 畫)
                theta=rad_theta.detach().cpu(),
                pred=rad_pred.detach().cpu(),           # SM 方向圖頭預測 (2, n_theta)
                real=rad_real_this_epoch,               # 本 epoch 真實 HFSS 方向圖 (快取 epoch 為 None)
                window_deg=rad_window,
                floor_db=rad_floor,
            )
        if rad_on:
            state.append("rad_loss", rad_loss_val)      # 監控用 (只在 rad run 出現此欄)
        loss.backward()
        generator.step(scheduler_param=sim_loss)
        generator.model.eval()
        state.append("gen_loss", loss.item())

        generator.save()                       # 存 GEN (供 rollback 載回 / 斷點續跑)

        simulator.end()
        simulator.clean()

        state.append("epoch", epoch)
        state.append("time", round(time() - epoch_t0, 1))   # 本 epoch 耗時 (HFSS 為主)
        state.save_row()                       # metrics.csv append 一行 (斷點續跑檢查點)

        # on_epoch 收到「本 epoch 快照」：純量指標 + 繪圖素材 (監控端畫 pattern/響應/方向圖用)
        if on_epoch is not None:
            snap = dict(
                sim_loss=float(state.last("sim_loss")),
                best_loss=float(state.last("best_loss")),
                gen_loss=float(state.last("gen_loss")),
                r_feed=float(state.last("r_feed")),
                tau=float(generator.scheduler.get_temp()),
                lr=float(optimizer.param_groups[0]["lr"]),
                time=float(state.last("time")),
                pattern=~output_element,               # 本 epoch 的 pattern (已 detach)
                response=stack.detach().cpu(),         # 本 epoch 的響應 (labels, 點數)
                spec=spec,                             # 響應規格 (labels/x/目標曲線)
                r_feed_painter=r_feed,                 # 饋電連通圖的繪圖器 (plot 最新一筆)
            )
            if rad_on:                                 # 方向圖 (選用)：純量 + 疊圖素材 (monitor 端畫)
                snap["rad_loss"] = rad_loss_val
                if rad_snapshot is not None:
                    snap["radiation"] = rad_snapshot
            on_epoch(epoch, snap)

    return state


def prepare_models(cfg, generator, smodel, state, *, continue_run=False,
                   gen_pretrained_path=None, sm_pretrained_path=None,
                   offline_dataset=None, warmup=None) -> int:
    """GEN/SM 的「載入策略」(模組化、由 config 的 generator/surrogate 區段指定)。
    回傳續跑起始 epoch。

    優先序：
      (1) 斷點續跑：結果夾已存在且 metrics.csv 有 epoch → 載回 GEN/SM，從上次 epoch 續跑 (其餘略過)。
      (2) GEN 預載入 (L2 暖啟動)：gen_pretrained_path 存在 → 載入 GEN 權重。
      (3) SM 載入 (L1)：sm_pretrained_path 存在 → 直接載入；否則若有離線資料集 → 從頭預訓練。
      (4) KuoHung 暖身：warmup 是可呼叫物 warmup(smodel) → 對 SM 做單筆暖身訓練
          (補齊舊 single 3/4 行為；資料與收斂門檻由呼叫端綁定，prepare_models 不耦合 KuoHung)。
    皆無 → GEN/SM 從隨機權重開始 (純靠線上學習)。
    """
    # (1) 斷點續跑：載回 GEN+SM，後續載入策略全部略過
    if continue_run and state.last_epoch > 0:
        last = state.last_epoch
        logger.info(f"斷點續跑：載回 epoch {last} 的 GEN/SM，從 epoch {int(last) + 1} 繼續")
        generator.change(last, load=True)
        smodel.load()
        return int(last)
    # (2) GEN 預載入 (暖啟動)
    if gen_pretrained_path is not None and Path(gen_pretrained_path).exists():
        generator.pre_load_model(gen_pretrained_path)
    # (3) SM 載入：預訓練檔 > 離線預訓練
    if sm_pretrained_path is not None and Path(sm_pretrained_path).exists():
        if cfg.radiation.get("enable", False):
            #? 方向圖版 SM 多了 head_rad、舊 sm.pth 沒有 → strict=False 部分載入共用 trunk/freq
            #  head (缺的 head_rad 維持隨機)，避免退回 elif 在數萬筆上從零預訓練 (HFSS 前先卡死)。
            smodel.pre_load_model(sm_pretrained_path, strict=False)
        else:
            smodel.pre_load_model(sm_pretrained_path)       # 共用路徑：簽名與行為與原樣相同
    elif offline_dataset is not None and len(offline_dataset) > 0:
        smodel.train_by_datas(offline_dataset)
    # (4) KuoHung 暖身：呼叫端綁好的 warmup(smodel)，對 SM 做單筆暖身訓練
    if warmup is not None:
        warmup(smodel)
    return 0


def build_simulator(cfg: "TrainConfig", record_path):
    """依 port 建真實 HFSS 模擬器 (production 用；測試以 mock 注入)。

    單埠 + radiation.enable → 用 SinglePortRadSimulator (求解後多匯出方向圖到 last_radiation)，
    回傳的 S11/Gain dict 與原樣相同；run_training 另從 simulator.last_radiation 取方向圖。
    """
    from antenna.patch import SinglePortSimulator, DualPortSimulator, SinglePortRadSimulator
    if cfg.port == "single" and cfg.radiation.get("enable", False):
        return SinglePortRadSimulator(record_path=record_path)
    cls = {"single": SinglePortSimulator, "dual": DualPortSimulator}[cfg.port]
    return cls(record_path=record_path)
