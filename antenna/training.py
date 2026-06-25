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
from antenna.models import Models, BatchLatentGenerator
from antenna.losses import FeedReachability, SpectralConnectivityLoss, GapClosingLoss
from antenna.optim import AdaptiveCyclicalScheduler
from antenna.losses import custom_loss_minmax, interval_loss, beam_coverage_loss, boundary_loss
from antenna.utils.store import SampleStore
from antenna.utils.replay import ReplayBuffer
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


#! 方向圖 target 的 dB 合理界線：dB(GainTotal) 在深零點可能 -inf/極端負，clamp 防梯度爆炸。
_RAD_DB_FLOOR = -60.0
_RAD_DB_CEIL = 60.0


#! 各區段允許的鍵 (白名單)。鍵打錯「必須報錯」而不是默默用預設值 ——
#! 歷史教訓：舊 dual 的 island_suppression 鍵名打錯，正則化默默沒開、實驗白跑。
SECTION_KEYS = {
    "loss":      {"total_variation", "island_suppression", "spectral_connectivity", "gap_closing", "boundary"},
    "sm_train":  {"lr", "min_loss", "max_epoch", "mode", "newest_steps", "replay_size"},
    "scheduler": {"on_plateau", "T_0", "T_mult", "lr_min", "temp_max", "temp_min",
                  "warmup_ratio", "patience", "factor"},
    "generator": {"name", "hidden", "pretrained", "num_candidates", "sigma", "sigma_min", "scales"},
    "surrogate": {"name", "hidden", "pretrained", "offline_dataset", "warmup"},
    "radiation": {"enable", "weight", "window_deg", "floor_db", "boresight_weight", "flatness_weight",
                  "warmup_epochs", "n_theta", "n_basis", "freeze_trunk", "sm_max_epoch", "sm_min_loss"},
    # 多候選 (batch_latent) 選擇分數：閘門 + 排序。僅 generator: batch_latent 時生效。
    "selection": {"boundary_weight", "feasibility_max"},
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
    selection: dict = field(default_factory=dict)   # 多候選選擇 (選用，僅 batch_latent)：見 SECTION_KEYS["selection"]

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
        rad_n_basis=cfg.radiation.get("n_basis", 16),   # 方向圖頭平滑基底數 (rad 關閉時無頭、此值不生效)
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
    bnd_w = cfg.loss.get("boundary", 0.0)            # boundary/trust-region：拉 G 回 SM 已見分布 (需 replay/dlf 緩衝)

    # SM 線上更新策略：single = 把「最新一筆」擬到收斂 (原樣)；replay = 對最新一筆跑少數步
    # ＋ 從「經驗回放緩衝」再訓一遍 (防 catastrophic forgetting、削掉擬到收斂的暴衝尾巴)。
    # 預設 single → 行為與原樣完全相同 (golden 零漂移)；replay 為 opt-in 改良。
    sm_mode = cfg.sm_train.get("mode", "single")                  # single | replay | dlf
    sm_newest_steps = cfg.sm_train.get("newest_steps", 50)        # replay/dlf：對最新一筆跑幾步 (取代擬到死)
    replay = ReplayBuffer(cfg.sm_train.get("replay_size", 256)) if sm_mode in ("replay", "dlf") else None
    if bnd_w > 0 and replay is None:
        logger.warning("loss.boundary > 0 但 sm_train.mode 非 replay/dlf（無緩衝定義已見分布）→ boundary loss 停用")

    # 多候選 (batch_latent)：同批生成 K 個候選 → 在 SM 上評分選最佳 → 只把選中的那張送昂貴路徑 (HFSS)，
    # 其餘候選僅進「聚合 loss」(mean over K) 一起反傳。multi=False (其他 generator) → 全程走單張原路，
    # 下方多候選分支一概不執行 → 與原樣完全相同 (golden 零漂移)。
    multi = isinstance(generator.model, BatchLatentGenerator)
    sel_bnd_w = cfg.selection.get("boundary_weight", bnd_w)    # 選擇排序的 boundary 權重 λ (省略=沿用訓練 bnd_w)
    sel_feas_max = cfg.selection.get("feasibility_max", None)  # SC 可行性閘門門檻 (None=不開閘門，等真實尺度再設)

    # 方向圖 (選用，預設 off)：開啟時 SM 已建方向圖頭、simulator 為 SinglePortRadSimulator。
    # 全程用 rad_on 閘住；rad_on=False → 下方所有方向圖分支不執行，行為與原樣完全相同 (golden 零漂移)。
    rad_on = cfg.radiation.get("enable", False)
    rad_w = cfg.radiation.get("weight", 1.0)
    rad_window = cfg.radiation.get("window_deg", 55.0)
    rad_floor = cfg.radiation.get("floor_db", 3.0)
    rad_bore_w = cfg.radiation.get("boresight_weight", 1.0)
    rad_flat_w = cfg.radiation.get("flatness_weight", 0.0)   # 選用：主動壓平窗內波形 (0=不啟用，與原樣相同)
    rad_warmup = cfg.radiation.get("warmup_epochs", 0)
    rad_freeze = cfg.radiation.get("freeze_trunk", True)   # 預設凍 trunk：方向圖頭只更新自己，不污染 S11/Gain backbone
    # 方向圖擬合上限 (rad head 凍 trunk 下單層 Linear 擬不到 sm_train.min_loss → 會撞滿 max_epoch、
    # 純浪費時間)。獨立旋鈕讓 rad 用較低上限；預設沿用 sm_train (不設就不改行為)。
    rad_max_epoch = cfg.radiation.get("sm_max_epoch", cfg.sm_train.get("max_epoch", 20000))
    rad_min_loss = cfg.radiation.get("sm_min_loss", cfg.sm_train.get("min_loss", 0.1))
    rad_theta = None        # 從第一筆方向圖資料擷取 (角度網格固定，整個 run 不變)
    #? 順手收集方向圖資料 (零額外 HFSS)：每筆真跑過的 pattern 連同真實方向圖存進 <結果夾>/radiation，
    #  供日後離線重訓 rad head / backbone (Stage 3)。一筆一檔、hash 去重，與 patterns/ 同調。
    rad_store = SampleStore(record_path / "radiation", verbose=False) if rad_on else None

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

    # ── 多候選 (batch_latent) 用的小工具 (閉包讀 smodel/sc/權重/replay/rad_*/state) ──
    #    僅 multi 時呼叫；rad_theta 為 run 中才填的閉包變數，呼叫時讀當前值 (只讀不寫)。
    def _build_candidate(logits_row, tau):
        oe = AntennaPattern(AntennaPattern.binarization(logits_row, tau))
        for f in feeds:
            oe = oe + f
        return oe

    def _candidate_loss(oe, ep):
        #? 現役四 loss：sm_target + SC + boundary + rad (TV/island/gap 為 legacy，多候選分支不接)。
        L = smodel(oe.series).criterion()
        if sc_w:
            L = L + sc_w * sc.forward(oe.size_converter(output_shape="B, 1, H, W"))
        if bnd_w > 0 and replay is not None and len(replay) > 1:
            L = L + bnd_w * boundary_loss(oe.series, replay.patterns())
        if rad_on and rad_theta is not None and ep > rad_warmup:
            rp = smodel.rad_predict(oe.series)
            L = L + rad_w * beam_coverage_loss(
                rp, rad_theta, window_deg=rad_window, floor_db=rad_floor,
                boresight_weight=rad_bore_w, flatness_weight=rad_flat_w,
            )
        return L

    def _select_best(cands):
        #? 選擇分數 (no_grad)：閘門 (SC ≤ feas_max) + 排序 (sm_target + λ·boundary)；優先沒模擬過的候選。
        #  回傳 (k_star, stats)：stats 是監控用的「候選池健康度」(交給 TB 診斷 Z 有沒有賺頭)。
        scores, feas, seen = [], [], []
        with torch.no_grad():
            for c in cands:
                s = float(smodel(c.series).criterion())
                if sel_bnd_w > 0 and replay is not None and len(replay) > 1:
                    s = s + sel_bnd_w * float(boundary_loss(c.series, replay.patterns()))
                scores.append(s)
                feas.append(float(sc.forward(c.size_converter(output_shape="B, 1, H, W"))))
                seen.append(state.lookup(~c) is not None)
        idx = list(range(len(cands)))
        gated = [i for i in idx if sel_feas_max is None or feas[i] <= sel_feas_max] or idx  # 全濾掉→退回全體
        fresh = [i for i in gated if not seen[i]]
        pool = fresh or gated                          # 優先沒見過的；全見過→從 gated 挑
        k_star = min(pool, key=lambda i: scores[i])
        n = len(scores)
        smin, smean = min(scores), sum(scores) / n
        stats = {
            "score_best": smin,                        # 選中候選的分數
            "score_mean": smean,                       # 全候選平均
            "score_spread": smean - smin,              # 分散度：越大=best-of-K 越有賺頭；→0=候選塌縮 (Z 失效)
            "fresh_frac": sum(1 for x in seen if not x) / n,  # 沒模擬過的候選比例 (探索廣度)
        }
        return k_star, stats

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
        sel_stats = None
        if multi:
            # 多候選：同批生成 K 個 → SM 評分選最佳 → output_element = 選中的那張 (走下方昂貴路徑)
            generator.model.anneal_sigma(epoch / epochs)            # σ 隨進度退火 (探索→收斂)
            tau = generator.scheduler.get_temp()
            logits_K = generator(spec.concat())                    # (K, out_dim)
            cands = [_build_candidate(logits_K[k], tau) for k in range(generator.model.K)]
            k_star, sel_stats = _select_best(cands)
            output_element = cands[k_star]
        else:
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
            #? SM 線上更新：single=最新一筆擬到收斂(原樣)；replay=最新少數步＋回放整個緩衝；
            #  dlf=最新少數步＋只訓「菁英子集」(loss ≤ 累計門檻 λ_t)。緩衝一律「全收」(學長論文 §3.5)。
            if sm_mode in ("replay", "dlf"):
                replay.add(~output_element, stack, sim_loss.item())   # 全收 (含 loss；不在寫入端篩)
                #? 對最新一筆跑少數步 (它＝G 此刻的位置，SM 最需要在那準)，max_epoch 上限避免擬到死
                smodel.train_one_data(output_element.series, stack, max_epoch=sm_newest_steps, verbose=verbose)
                if sm_mode == "dlf":
                    #? DLF：λ_t = 累計真實損失歷史平均 (含本筆)；只取 loss ≤ λ_t 的菁英子集訓 SM。
                    #  門檻隨訓練自動收緊 → 前期多樣、後期精準 (論文消融 >50% 改善)。
                    hist = state.series("sim_loss")
                    lam = (sum(hist) + sim_loss.item()) / (len(hist) + 1)
                    elite = replay.elite(lam)
                    if len(elite) > 0:
                        smodel.train_by_datas(elite, epochs=1, verbose=verbose)
                else:                                              # 純 replay：回放整個緩衝
                    smodel.train_by_datas(replay, epochs=1, verbose=verbose)
            else:
                #? verbose=True 時顯示 SM 單筆訓練的 tqdm 進度條 (與舊腳本行為一致)
                smodel.train_one_data(output_element.series, stack, verbose=verbose)
            # 方向圖 (選用)：讀 SinglePortRadSimulator.last_radiation，線上訓練方向圖頭。
            if rad_on:
                rad = getattr(simulator, "last_radiation", None)
                if isinstance(rad, dict) and rad.get("theta") is not None:
                    #? 整個 run 第一次拿到真實 θ 網格 → 用它重建方向圖頭的平滑基底，使預測逐點對齊 θ_i
                    #  (θ 網格整 run 固定；basis 逐欄獨立算 → HFSS 匯出序未排序也對位正確)。
                    if rad_theta is None and hasattr(smodel.model, "set_rad_theta"):
                        smodel.model.set_rad_theta(rad["theta"])
                    rad_theta = rad["theta"]
                    rad_stack = torch.stack([rad["phi0"], rad["phi90"]])    # (2, n_theta)
                    #! 清理方向圖 target：dB(GainTotal) 在深零點可能是 -inf/極端負值，直接進 MSE 會爆梯度。
                    #  剔除非有限值並 clamp 到合理 dB 範圍 (深零點對 beam coverage 無所謂)。
                    rad_stack = torch.nan_to_num(rad_stack, nan=_RAD_DB_FLOOR,
                                                 posinf=_RAD_DB_CEIL, neginf=_RAD_DB_FLOOR
                                                 ).clamp(_RAD_DB_FLOOR, _RAD_DB_CEIL)
                    rad_real_this_epoch = rad_stack.detach().cpu()          # 監控疊圖用 (真實 HFSS 方向圖)
                    if rad_store is not None:
                        #? 存 (pattern, [theta, phi0, phi90]) (3, n_theta)：θ 一起存 → 自我說明、可離線重訓。
                        #  存 clamp 後的 finite 版 (rad_stack 已 nan_to_num+clamp)，直接可訓、不含 -inf。
                        theta_row = torch.as_tensor(rad["theta"], dtype=rad_stack.dtype).reshape(1, -1)
                        rad_store.add(~output_element,
                                      torch.cat([theta_row, rad_stack.detach().cpu()], dim=0))
                    smodel.train_one_data_rad(output_element.series, rad_stack,
                                              min_loss=rad_min_loss, max_epoch=rad_max_epoch,
                                              freeze_trunk=rad_freeze, verbose=verbose)
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
        if multi:
            # 聚合：mean over K 候選的「現役四 loss」(reparam → 最小化雲的 E[loss]，把中心 z* 拉向好區)。
            loss = sum(_candidate_loss(c, epoch) for c in cands) / len(cands)
            # 方向圖監控快照 (用選中的那張；rad loss 已含在聚合 loss 內，這裡只為監控、不重複加)。
            rad_loss_val = 0.0
            rad_snapshot = None
            if rad_on and rad_theta is not None:
                rad_pred = smodel.rad_predict(output_element.series)
                if epoch > rad_warmup:
                    rad_loss_val = float(beam_coverage_loss(
                        rad_pred, rad_theta, window_deg=rad_window, floor_db=rad_floor,
                        boresight_weight=rad_bore_w, flatness_weight=rad_flat_w,
                    ))
                rad_snapshot = dict(
                    theta=rad_theta.detach().cpu(), pred=rad_pred.detach().cpu(),
                    real=rad_real_this_epoch, window_deg=rad_window, floor_db=rad_floor,
                )
        else:
            response = smodel(output_element.series)
            loss = (
                response.criterion()
                + output_element.total_variation_loss(tv_w)
                + output_element.island_suppression_loss(is_w)
                + sc_w * sc.forward(output_element.size_converter(output_shape="B, 1, H, W"))
                + gap_w * gc.forward(output_element.size_converter(output_shape="B, 1, H, W"))
            )
            # boundary/trust-region (選用)：拉 G 回 SM 已見分布，防 G 鑽 SM 盲區、白燒 HFSS 評估。
            # 需 replay/dlf 緩衝定義「已見分布」；bnd_w=0 或無緩衝 → 不加 (golden 零漂移)。
            if bnd_w > 0 and replay is not None and len(replay) > 1:
                loss = loss + bnd_w * boundary_loss(output_element.series, replay.patterns())
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
                        flatness_weight=rad_flat_w,
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
        if multi:                                       # 多候選：σ 退火 + 候選池健康度 → 也落 metrics.csv (離線可分析)
            state.append("sigma", float(generator.model.sigma))
            if sel_stats is not None:
                for _k in ("score_best", "score_mean", "score_spread", "fresh_frac"):
                    state.append(_k, sel_stats[_k])
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
            if multi:                                  # 多候選 (選用)：σ 退火 + 候選池健康度 (診斷 Z 賺頭)
                snap["sigma"] = float(generator.model.sigma)
                if sel_stats is not None:
                    snap.update(sel_stats)             # score_best / score_mean / score_spread / fresh_frac
            on_epoch(epoch, snap)

    return state


def _assert_sm_checkpoint_sane(smodel, epoch, *, max_abs: float = 1e3):
    """續跑守衛：SM 權重若非有限或量級異常（灌爆但非 NaN）→ 明確報錯叫使用者重開。
    踩過的雷：發散訓練把 sm.pth 灌爆（健康 max|w|≈0.3，壞掉 5680），續跑只 `smodel.load()`
    載回這個爛檔、繞過 old_sm.pth 暖啟動，永遠卡在同一個壞 pattern（gen_loss 1e37、撞快取空轉）。
    與 mock 相容：MagicMock 的 named_parameters() 迭代為空 → 守衛自動略過、不擋既有測試。"""
    for name, p in smodel.model.named_parameters():
        if p.numel() == 0:
            continue
        m = float(p.detach().abs().max())
        if not torch.isfinite(p).all() or m > max_abs:
            raise RuntimeError(
                f"續跑的 SM checkpoint 已損壞（epoch {epoch}，參數 '{name}' max|w|={m:.3g} "
                f"> {max_abs:g} 或非有限）。此結果夾的 sm.pth 被先前的發散訓練灌爆，續跑只會繼續壞下去。"
                f"請把該結果夾改名/刪除後重新開始（全新 run 會從 old_sm.pth 暖啟動）。"
            )


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
        _assert_sm_checkpoint_sane(smodel, last)   # 守衛：續跑載到灌爆的 SM → 明確報錯叫重開，別默默撞
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
