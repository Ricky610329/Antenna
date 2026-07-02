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
import math
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
from antenna.losses import (custom_loss_minmax, interval_loss, beam_coverage_loss, boundary_loss,
                            candidate_repulsion, boundary_threshold, worst_margin)
from antenna.utils.store import SampleStore, fingerprint
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
    "loss":      {"total_variation", "island_suppression", "spectral_connectivity", "gap_closing", "boundary",
                  # 信任懲罰 λ_trust：罰「SM 沒把握 (ensemble 分歧大)」的 pattern → 把候選推離 SM 盲區。需 ensemble SM。
                  "uncertainty"},
    "sm_train":  {"lr", "min_loss", "max_epoch", "mode", "newest_steps", "replay_size", "elite_epochs"},
    "scheduler": {"on_plateau", "T_0", "T_mult", "lr_min", "temp_max", "temp_min",
                  "warmup_ratio", "patience", "factor",
                  # boundary-gated warm restart (opt-in)：boundary 當 ACP 探索/固化的依據
                  "boundary_gate", "boundary_kappa", "boundary_recompute_every", "boundary_suppress_cap"},
    "generator": {"name", "hidden", "pretrained", "num_candidates", "sigma", "sigma_min", "scales", "init_scale"},
    "surrogate": {"name", "hidden", "pretrained", "offline_dataset", "warmup",
                  # ensemble SM (surrogate: ensemble)：成員數 + 暖啟動後各成員權重擾動尺度 (製造多樣性)。
                  "ensemble_size", "init_perturb"},
    "radiation": {"enable", "weight", "window_deg", "floor_db", "boresight_weight", "flatness_weight",
                  "warmup_epochs", "n_theta", "n_basis", "freeze_trunk", "sm_max_epoch", "sm_min_loss"},
    # 多候選 (batch_latent/direct) 選擇分數：閘門 + 排序 + 候選排斥 + acquisition 不確定性權重。
    # uncertainty_weight (β)：acquisition 偏好「SM 沒把握」的候選 (主動學習，去修 SM)。需 ensemble SM。
    "selection": {"boundary_weight", "feasibility_max", "diversity_weight", "uncertainty_weight"},
    # 閉迴路信任控制 (TrustController)：gap (SM vs HFSS) → t → 調 tau/λ_trust/κ。enable=False → 靜態。
    "trust": {"enable", "g0", "ema", "t_min", "t_max", "tau_inflate"},
    # 自適應 SM 訓練量 (AdaptiveSMTrainController)：以 held-out fresh HFSS 點量 member0 泛化、自調每輪重訓
    # epoch 數 (沿途快照 member0、下一輪新點評快照取 argmin)。由 sm_train.mode: adaptive 開啟；此區放旋鈕。
    "adaptive": {"snapshots", "epoch_min", "epoch_max", "ema"},
}
TARGET_KEYS = {"side", "center", "width", "method", "interval"}
#! 各 port 目標的「必填」子鍵：缺就在 load_config/建構時報錯，不拖到 setup_responses
#  (在 simulator.open() 啟動 HFSS COM 之後) 才 KeyError、白啟動一次 HFSS。對齊 setup_responses
#  實際「無預設」的解參——single 需 method (custom_loss_minmax)；dual 的 interval 有預設 [-1,1]
#  → 非必填、不強制 (與現行行為一致，避免破壞省略 interval 的舊 config)。
TARGET_REQUIRED = {
    "single": {"side", "center", "width", "method"},
    "dual":   {"side", "center", "width"},
}
#! sm_train.mode 的允許「值」(白名單只驗鍵不驗值；mode 打錯字會靜默退回 single、害 A/B 白跑 →
#  必須額外驗值，比照 island_suppression 鍵打錯的歷史教訓)。
SM_MODES = ("single", "replay", "dlf", "dlf_fit", "refit", "adaptive")


class TrustController:
    """閉迴路信任控制：用「SM 預測 vs 真實 HFSS 的落差 gap」推出單一信任標量 t∈[0,1]，再以 t
    同軸調節「探索力度」的三個致動器——tau 乘子 / 信任懲罰 λ_trust / acquisition κ。

    語意 (見 docs/guided_search_design.md)：t→1 (SM 可信) = 利用 (tau 不放軟、長牽繩、純收割)；
    t→0 (SM 失準) = 探索 (tau 放軟保持可塑、短牽繩拉回可信區、去獵不確定點修 SM)。

    enable=False (Exp1/Exp2) → t 恆 1：tau_mult≡1 (純 ACP)、λ_trust/κ 退化成「靜態 base 值」
      (Exp2 用靜態 λ_trust/κ，不隨 gap 變)。enable=True (Exp3) → 閉迴路、三者隨 t 動。
    enable=False 且 base=0 → 三者皆 0、tau_mult≡1 → 與原樣逐位元相同 (golden 零漂移)。"""

    def __init__(self, *, enable: bool, lambda_trust_base: float, kappa_base: float,
                 g0: float = 1.0, ema: float = 0.3, t_min: float = 0.05, t_max: float = 0.95,
                 tau_inflate: float = 3.0):
        self.enable = bool(enable)
        self.lambda_trust_base = float(lambda_trust_base)   # λ_trust 上限 (= loss.uncertainty)
        self.kappa_base = float(kappa_base)                 # acquisition κ 上限 (= selection.uncertainty_weight)
        self.g0 = float(g0)                                 # 落差參考尺度：gap=g0 → t≈0.37 (失去大半信任)
        self.ema = float(ema)                               # gap 的 EMA 係數 (信任是逐步累積的信念)
        self.t_min, self.t_max = float(t_min), float(t_max)
        self.tau_inflate = float(tau_inflate)               # t→0 時 tau 最多放軟幾倍
        self.gap_ema = None
        self.t = 1.0                                        # 初始全信任 (還沒看到落差)
        #! 誤設護欄：g0≤0 會讓 exp(−gap/g0) 失義；tau_inflate<1 會在 t→0 時「銳化」tau (方向倒轉)。
        if self.g0 <= 0:
            raise ValueError(f"TrustController g0 須 >0 (落差參考尺度)，得到 {g0}")
        if self.tau_inflate < 1.0:
            raise ValueError(f"TrustController tau_inflate 須 ≥1 (t→0 放軟 tau；<1 會反向銳化)，得到 {tau_inflate}")

    def update(self, sm_pred_loss: float, real_loss: float):
        """用本筆真實 HFSS 的 (SM 預測 loss, 真實 loss) 更新 gap_ema → t。enable=False 不更新。"""
        if not self.enable:
            return
        gap = abs(float(sm_pred_loss) - float(real_loss))
        if not math.isfinite(gap):
            #! SM 預測/真實出現 NaN/inf → SM 嚴重失準 → 直接最大探索 (t_min)，且「不」把 NaN 折進
            #  gap_ema (否則 EMA 永遠 NaN、clamp(NaN) 在 CPython 反而回 t_max=高信任，方向完全相反)。
            self.t = self.t_min
            return
        self.gap_ema = gap if self.gap_ema is None else (self.ema * gap + (1.0 - self.ema) * self.gap_ema)
        t = math.exp(-self.gap_ema / self.g0)               # g0>0 由建構子保證
        self.t = max(self.t_min, min(self.t_max, t))

    def tau_mult(self) -> float:
        """tau 乘子：t→1 → 1 (純 ACP 退火)；t→0 → tau_inflate (放軟、保持 pattern 可塑、不亂鎖定)。"""
        return 1.0 + (self.tau_inflate - 1.0) * (1.0 - self.t) if self.enable else 1.0

    def lambda_trust(self) -> float:
        """信任懲罰權重：enable → base·(1−t) (SM 失準才收緊牽繩)；否則靜態 base。"""
        return self.lambda_trust_base * (1.0 - self.t) if self.enable else self.lambda_trust_base

    def kappa(self) -> float:
        """acquisition 不確定性權重：enable → base·(1−t) (SM 失準才去獵不確定點)；否則靜態 base。"""
        return self.kappa_base * (1.0 - self.t) if self.enable else self.kappa_base


class AdaptiveSMTrainController:
    """自適應 SM 訓練量：以「held-out 的 fresh HFSS 點」量 member0 的泛化，自調每輪 SM 重訓的 epoch 數。

    做法 (見 docs/discuss/decisions.md「自適應 SM 訓練量」)：本輪把 member0 訓到 target_epochs()，沿途在
    schedule() 的 epoch 點快照 member0 權重；**下一輪**的新點 (held-out、還沒被這些快照訓過) 逐一評快照
    → 得「訓練量→泛化誤差」→ per-bucket EMA → argmin。argmin 落在本輪最大訓練量點 (邊界) = 還能再多訓 →
    加碼 target；argmin 落在中間 = 訓過頭 → target 移向 argmin。避免「1 epoch 太少」與「逼 min_loss 過擬合」
    兩個極端。enable=False → 惰性 (target_epochs() 回 fallback、schedule() 空、不快照) → 與原樣相同。

    只探測 member0 (各成員同架構同資料、只差 init 擾動 → 曲線代表全體)；快照留記憶體、每輪覆蓋 (零磁碟)。"""

    def __init__(self, *, enable: bool, snapshots: int = 5, epoch_min: int = 1, epoch_max: int = 64,
                 ema: float = 0.3, fallback_epochs: int = 1):
        self.enable = bool(enable)
        self.n = max(2, int(snapshots))                 # 快照點數 (≥2 才有曲線)
        self.epoch_min = max(1, int(epoch_min))
        self.epoch_max = max(self.epoch_min + 1, int(epoch_max))
        self.ema = float(ema)
        self.fallback = max(1, int(fallback_epochs))    # enable=False / 尚無觀測時的訓練量
        self.bucket: dict = {}                          # {epoch(int): EMA(held-out err)}
        #! 誤設護欄：ema 是 EMA 係數，須 ∈ (0,1]。
        if not (0.0 < self.ema <= 1.0):
            raise ValueError(f"adaptive.ema 須 ∈ (0,1]，得到 {ema}")
        #! target 內部存 float、只在對外 (target_epochs/schedule) 取整：整數上做 EMA 會被 round() 吃掉
        #  小增量 (target≤5 時邊界 1.3× 加碼全被 round 回原值 → 低訓練量吸收態；2026-07-02 模擬實證)。
        self.target = self._geomid()                    # 初值取範圍幾何中點 (float)

    def _geomid(self) -> float:
        return math.exp((math.log(self.epoch_min) + math.log(self.epoch_max)) / 2)

    def schedule(self) -> list:
        """本輪要快照 member0 的 epoch 點：[epoch_min, target] 內 log 間距的 n 個 (去重遞增)。enable=False → []。"""
        if not self.enable:
            return []
        hi = math.log(max(self.epoch_min + 1, int(round(self.target))))
        lo = math.log(self.epoch_min)
        pts = {max(1, int(round(math.exp(lo + (hi - lo) * i / (self.n - 1))))) for i in range(self.n)}
        return sorted(pts)

    def observe(self, errors: dict):
        """errors: {epoch: held-out 誤差}（上一輪快照在這一輪新點上的誤差）。per-bucket EMA → 更新 target。"""
        if not self.enable or not errors:
            return
        finite = {int(e): float(v) for e, v in errors.items() if math.isfinite(v)}
        if not finite:
            return
        for e, v in finite.items():
            self.bucket[e] = v if e not in self.bucket else self.ema * v + (1.0 - self.ema) * self.bucket[e]
        #! argmin 只在「本輪有觀測到的桶」裡選 (值仍用跨輪 EMA 平滑)：target 縮小後範圍外的舊桶再也不會
        #  被重新觀測、卻永遠參與 argmin → 一個過期低值就把 target 釘死 (死鎖成因之二，同上模擬實證)。
        best = min(finite, key=lambda e: self.bucket[e])
        top = max(finite)                                # 本輪最大訓練量快照點 (邊界)
        # 邊界(最多訓的贏) → 加碼探更多；否則 target 移向 argmin。再對 target 做 EMA、慢走不震盪 (全程 float)。
        cand = min(float(self.epoch_max), self.target * 1.3) if best >= top else float(best)
        self.target = max(float(self.epoch_min), min(float(self.epoch_max),
                          self.ema * cand + (1.0 - self.ema) * self.target))

    def target_epochs(self) -> int:
        """本輪 SM 重訓的 epoch 數 (enable=False → fallback)。"""
        return int(round(self.target)) if self.enable else self.fallback

    def seed_target(self, value):
        """斷點續跑：用 metrics.csv 最後一筆 sm_train_epochs 續 target。控制器狀態只在記憶體，
        沒這個的話每次重啟 (HFSS 當機很常見) target 都歸零回中點重學。bucket 重學即可、不續。
        value=None/非有限/enable=False → 不動。"""
        if not self.enable or value is None:
            return
        v = float(value)
        if math.isfinite(v):
            self.target = max(float(self.epoch_min), min(float(self.epoch_max), v))

    def probe_stats(self, errors: dict) -> dict:
        """監控用：本輪 held-out 探測曲線的 argmin/最小/最大誤差 (落 csv)。無有限值 → 空 dict。"""
        finite = {int(e): float(v) for e, v in (errors or {}).items() if math.isfinite(v)}
        if not finite:
            return {}
        best = min(finite, key=finite.get)
        return {"probe_argmin": float(best), "probe_min_err": float(finite[best]),
                "probe_max_err": float(max(finite.values()))}


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
    trust: dict = field(default_factory=dict)        # 閉迴路信任控制 (選用，預設 off)：見 SECTION_KEYS["trust"]
    adaptive: dict = field(default_factory=dict)     # 自適應 SM 訓練量 (選用；由 sm_train.mode: adaptive 開)：見 SECTION_KEYS["adaptive"]

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
        required = TARGET_REQUIRED[self.port]
        for label, t in self.targets.items():
            unknown = set(t) - TARGET_KEYS
            if unknown:
                raise ValueError(f"targets.{label} 含未知鍵: {sorted(unknown)} (允許: {sorted(TARGET_KEYS)})")
            #! 必填子鍵缺失 → fail-fast (見 TARGET_REQUIRED)：否則拖到 setup_responses 才 KeyError。
            missing_keys = required - set(t)
            if missing_keys:
                raise ValueError(f"targets.{label} 缺少必填鍵: {sorted(missing_keys)} "
                                 f"(port={self.port} 需 {sorted(required)})")
        #! 驗 sm_train.mode 的「值」(不只鍵)：打錯字 (如 dlffit/Refit) 會靜默退回 single → A/B 白跑。
        mode = self.sm_train.get("mode", "single")
        if mode not in SM_MODES:
            raise ValueError(f"sm_train.mode 未知值: {mode!r} (允許: {list(SM_MODES)})")
        #! adaptive 區段有設但 mode 非 adaptive → 該區段不會生效 → fail-fast (比照 island_suppression 靜默沒開的教訓)。
        if self.adaptive and mode != "adaptive":
            raise ValueError(f"設了 adaptive 區段但 sm_train.mode={mode!r} (非 'adaptive') → 該區段不會生效；"
                             f"要用自適應 SM 訓練量請設 sm_train.mode: adaptive。")


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
        boundary_suppress_cap=s.get("boundary_suppress_cap", 3),   # 連續抑制上限 (僅 boundary_gate 開時生效)
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


def _update_surrogate(sm_mode, smodel, replay, output_element, stack, sim_loss, state,
                      *, newest_steps, elite_epochs, verbose, sm_ctrl=None):
    """SM 線上更新的模式分派 (single/replay/dlf/dlf_fit/refit)——由 run_training 主迴圈原樣抽出、
    行為逐位元不變。回傳本輪 SM 重訓的逐步/逐 epoch loss 清單 sm_fit_hist (監控用；single 為單筆逐步；
    菁英/緩衝為空時回 [])。

      single   = 把「最新一筆」擬到收斂 (學長原始單筆過擬合;反模式,但 golden 基準)。
      replay   = 最新一筆少數步 ＋ 回放整個緩衝 (防遺忘)。
      dlf      = 最新一筆少數步 ＋ elite 子集訓「1 epoch」(現行;經查 = under-trained 的 DLF)。
      dlf_fit  = 全收 ＋ 重過濾 elite ＋ 把 elite「訓到收斂(fit)」,不做單筆 step ＝ 學長原版 DLF。
      refit    = 全收 ＋ 把「整個 buffer(不挑 elite)」訓到 fit ＝「不一定要 elite」版 (對抗洞疫苗)。
    緩衝一律「全收」(學長論文 §3.5)；DLF/dlf_fit 的 λ_t = 累計真實 sim_loss 歷史平均 (含本筆)。
    ⚠ refit 的「整個 buffer」受 replay_size 上限 (FIFO)；計算量 > dlf_fit，配 ensemble 會再 ×K。
    """
    sm_fit_hist = []
    if sm_mode == "dlf_fit":
        replay.add(~output_element, stack, sim_loss.item())   # 全收
        hist = state.series("sim_loss")
        lam = (sum(hist) + sim_loss.item()) / (len(hist) + 1)
        elite = replay.elite(lam)
        if len(elite) > 0:
            sm_fit_hist = smodel.train_by_datas(elite, epochs=elite_epochs,
                                                min_loss=smodel.min_loss, verbose=verbose)
    elif sm_mode == "refit":
        replay.add(~output_element, stack, sim_loss.item())   # 全收
        if len(replay) > 0:
            sm_fit_hist = smodel.train_by_datas(replay, epochs=elite_epochs,
                                                min_loss=smodel.min_loss, verbose=verbose)
    elif sm_mode in ("replay", "dlf"):
        replay.add(~output_element, stack, sim_loss.item())   # 全收 (含 loss；不在寫入端篩)
        #? 對最新一筆跑少數步 (它＝G 此刻的位置，SM 最需要在那準)，max_epoch 上限避免擬到死
        smodel.train_one_data(output_element.series, stack, max_epoch=newest_steps, verbose=verbose)
        if sm_mode == "dlf":
            #? DLF：λ_t = 累計真實損失歷史平均 (含本筆)；只取 loss ≤ λ_t 的菁英子集訓 SM。
            hist = state.series("sim_loss")
            lam = (sum(hist) + sim_loss.item()) / (len(hist) + 1)
            elite = replay.elite(lam)
            if len(elite) > 0:
                sm_fit_hist = smodel.train_by_datas(elite, epochs=1, verbose=verbose)
        else:                                              # 純 replay：回放整個緩衝
            sm_fit_hist = smodel.train_by_datas(replay, epochs=1, verbose=verbose)
    elif sm_mode == "adaptive":
        #? 自適應訓練量 (見 AdaptiveSMTrainController)：全收 → dlf 的 elite(λ_t) 篩選 → 把 elite 訓到
        #  sm_ctrl.target_epochs()（自調、非固定 1），沿途在 sm_ctrl.schedule() 的 epoch 點快照 member0
        #  供下一輪 held-out 探測。early_stop/min_loss 關掉以訓滿、拿齊快照。elite 空 → 這輪不訓、無快照。
        replay.add(~output_element, stack, sim_loss.item())
        hist = state.series("sim_loss")
        lam = (sum(hist) + sim_loss.item()) / (len(hist) + 1)
        elite = replay.elite(lam)
        state.append("elite_n", float(len(elite)))         # 本輪 elite 集大小 (成本=epochs×elite_n 步,解讀 wall-clock 用)
        smodel._probe_snapshots = {}                       # 先清：elite 空/沒訓 → 下一輪無 stale 快照可評
        if len(elite) > 0:
            sm_fit_hist = smodel.train_by_datas(
                elite, epochs=sm_ctrl.target_epochs(), min_loss=None,
                snapshot_epochs=sm_ctrl.schedule(), early_stop=False, verbose=verbose)
    else:
        #? verbose=True 時顯示 SM 單筆訓練的 tqdm 進度條 (與舊腳本行為一致)
        sm_fit_hist = smodel.train_one_data(output_element.series, stack, verbose=verbose)
    return sm_fit_hist


def _radiation_online_step(smodel, simulator, rad_store, output_element, rad_theta,
                           *, min_loss, max_epoch, freeze_trunk, verbose):
    """方向圖頭的線上更新 (rad_on 且 fresh-HFSS epoch 才呼叫)——由 run_training 主迴圈原樣抽出、
    行為逐位元不變。讀 SinglePortRadSimulator.last_radiation、清理 dB target (nan_to_num+clamp 防
    深零點 -inf 爆梯度)、順手存 rad_store (零額外 HFSS)、把方向圖頭訓到收斂。

    回傳 (rad_theta, rad_real_this_epoch, rad_fit_val)：rad_theta 首次拿到真 θ 網格後即固定 (整 run
    不變)；無 last_radiation → 回傳原 rad_theta 與 (None, 0.0)，等同原樣不進方向圖分支。
    """
    rad = getattr(simulator, "last_radiation", None)
    if not (isinstance(rad, dict) and rad.get("theta") is not None):
        return rad_theta, None, 0.0
    #? 整個 run 第一次拿到真實 θ 網格 → 用它重建方向圖頭的平滑基底，使預測逐點對齊 θ_i
    #  (θ 網格整 run 固定；basis 逐欄獨立算 → HFSS 匯出序未排序也對位正確)。
    if rad_theta is None:
        smodel.set_rad_theta(rad["theta"])   # 委派給 SM 介面 (集成則 fan-out 所有成員)
    rad_theta = rad["theta"]
    rad_stack = torch.stack([rad["phi0"], rad["phi90"]])    # (2, n_theta)
    #! 清理方向圖 target：dB(GainTotal) 在深零點可能是 -inf/極端負值，直接進 MSE 會爆梯度。
    rad_stack = torch.nan_to_num(rad_stack, nan=_RAD_DB_FLOOR,
                                 posinf=_RAD_DB_CEIL, neginf=_RAD_DB_FLOOR
                                 ).clamp(_RAD_DB_FLOOR, _RAD_DB_CEIL)
    rad_real_this_epoch = rad_stack.detach().cpu()          # 監控疊圖用 (真實 HFSS 方向圖)
    if rad_store is not None:
        #? 存 (pattern, [theta, phi0, phi90]) (3, n_theta)：θ 一起存 → 自我說明、可離線重訓。
        theta_row = torch.as_tensor(rad["theta"], dtype=rad_stack.dtype).reshape(1, -1)
        rad_store.add(~output_element,
                      torch.cat([theta_row, rad_stack.detach().cpu()], dim=0))
    rad_hist = smodel.train_one_data_rad(output_element.series, rad_stack,
                                         min_loss=min_loss, max_epoch=max_epoch,
                                         freeze_trunk=freeze_trunk, verbose=verbose)
    rad_fit_val = float(rad_hist[-1]) if rad_hist else 0.0   # rad head 收斂後擬合 loss (監控)
    return rad_theta, rad_real_this_epoch, rad_fit_val


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
    max_consecutive_skips: int = 5,             # 連續幾筆 HFSS 失敗 → reopen 重生；reopen 後再連敗 → 中斷+寄信
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

    #? 訓練狀態走「結果夾即資料庫」：metrics.csv (純量) + patterns/ (模擬快取)。
    #  (online 好樣本庫已隨回滾移除 2026-06-28——它只被回滾的重訓讀，回滾拔掉後即死碼。)
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
    unc_w = cfg.loss.get("uncertainty", 0.0)         # 信任懲罰 λ_trust：罰 ensemble 分歧大的 pattern (需 ensemble SM)

    # SM 線上更新策略：
    #   single   = 把「最新一筆」擬到收斂 (學長原始單筆過擬合;反模式,但 golden 基準)。
    #   replay   = 最新一筆少數步 ＋ 回放整個緩衝 (防遺忘)。
    #   dlf      = 最新一筆少數步 ＋ elite 子集訓「1 epoch」(現行;經查 = under-trained 的 DLF)。
    #   dlf_fit  = 全收 ＋ 重過濾 elite ＋ 把 elite「訓到收斂(fit)」,不做單筆 step ＝ 學長原版 DLF。
    #   refit    = 全收 ＋ 把「整個 buffer(不挑 elite)」訓到 fit ＝ 你的「不一定要 elite」版;對抗洞:SM 也學
    #              「爛 pattern 是爛的」→ guidance 會避開、不被騙進洞。dlf_fit(elite) vs refit(all) 即 A/B。
    # 預設 single → 行為與原樣完全相同 (golden 零漂移);其餘為 opt-in。
    sm_mode = cfg.sm_train.get("mode", "single")                  # single | replay | dlf | dlf_fit | refit
    sm_newest_steps = cfg.sm_train.get("newest_steps", 50)        # replay/dlf：對最新一筆跑幾步 (取代擬到死)
    sm_elite_epochs = cfg.sm_train.get("elite_epochs", 50)        # dlf_fit/refit：每輪重訓 epoch 上限 (配 min_loss 訓到 fit)
    replay = ReplayBuffer(cfg.sm_train.get("replay_size", 256)) if sm_mode in ("replay", "dlf", "dlf_fit", "refit", "adaptive") else None
    #? 自適應 SM 訓練量控制器 (mode:adaptive 才啟用；否則惰性 → target_epochs 回 fallback、不快照 → 與原樣相同)。
    sm_ctrl = AdaptiveSMTrainController(
        enable=(sm_mode == "adaptive"),
        snapshots=cfg.adaptive.get("snapshots", 5), epoch_min=cfg.adaptive.get("epoch_min", 1),
        epoch_max=cfg.adaptive.get("epoch_max", 64), ema=cfg.adaptive.get("ema", 0.3),
    )
    sm_ctrl.seed_target(state.last("sm_train_epochs", None))   # 斷點續跑：續上次的 target (見 seed_target)
    prev_snapshots = {}     # 上一輪 member0 權重快照 {epoch: state_dict}，供本輪 held-out 探測 (記憶體、每輪覆蓋)
    if bnd_w > 0 and replay is None:
        logger.warning("loss.boundary > 0 但 sm_train.mode 非 replay/dlf/dlf_fit/refit（無緩衝定義已見分布）→ boundary loss 停用")

    # 多候選 (batch_latent / direct)：同批生成 K 個候選 → 在 SM 上評分選最佳 → 只把選中的那張送昂貴
    # 路徑 (HFSS)，其餘候選僅進「聚合 loss」(mean over K) 一起反傳。multi=False (sigmoid 等單張 G) →
    # 全程走單張原路，下方多候選分支一概不執行 → 與原樣完全相同 (golden 零漂移)。
    #? 偵測用 isinstance(BatchLatent) ∪ is_multi_candidate 旗標 (generator-free 的 direct 走此旗標)：
    #  BatchLatent 仍由 isinstance 命中 → 行為逐位元不變；新多候選 G 用旗標 opt-in、彼此解耦。
    multi = isinstance(generator.model, BatchLatentGenerator) or getattr(generator.model, "is_multi_candidate", False)
    sel_bnd_w = cfg.selection.get("boundary_weight", bnd_w)    # 選擇排序的 boundary 權重 λ (省略=沿用訓練 bnd_w)
    sel_feas_max = cfg.selection.get("feasibility_max", None)  # SC 可行性閘門門檻 (None=不開閘門，等真實尺度再設)
    div_w = cfg.selection.get("diversity_weight", 0.0)         # 候選排斥權重 λ_div (0=關，治 batch_latent 崩塌)
    sel_unc_w = cfg.selection.get("uncertainty_weight", 0.0)   # acquisition κ：偏好 SM 沒把握的候選 (主動學習)

    #? 信任控制器 (Exp2 靜態 / Exp3 閉迴路)：base 權重 = loss.uncertainty (λ_trust) / selection.uncertainty_weight (κ)。
    #  trust.enable=False 且兩 base=0 → tau_mult≡1、λ_trust/κ≡0 → 與原樣逐位元相同 (golden 零漂移)。
    #! 信任懲罰 / acquisition 不確定性需 ensemble SM (有 uncertainty())；單一 SM 用了會在下方 hasattr 處停用。
    has_unc = hasattr(smodel, "uncertainty")
    if (unc_w > 0 or sel_unc_w > 0 or cfg.trust.get("enable", False)) and not has_unc:
        logger.warning("loss.uncertainty / selection.uncertainty_weight / trust.enable 需 surrogate: ensemble "
                       "(單一 SM 無 uncertainty())；信任懲罰與 acquisition 不確定性項停用。")
    trust = TrustController(
        enable=cfg.trust.get("enable", False) and has_unc,
        lambda_trust_base=unc_w if has_unc else 0.0,
        kappa_base=sel_unc_w if has_unc else 0.0,
        g0=cfg.trust.get("g0", 1.0), ema=cfg.trust.get("ema", 0.3),
        t_min=cfg.trust.get("t_min", 0.05), t_max=cfg.trust.get("t_max", 0.95),
        tau_inflate=cfg.trust.get("tau_inflate", 3.0),
    )

    # boundary-gated ACP (opt-in)：boundary 當「探索/固化」的依據——衝出 SM 可信區 (boundary≥τ_b) 就
    # 抑制 ACP 的 plateau warm restart (不加熱、冷卻固化)。需 replay/dlf 緩衝定義已見分布；無緩衝/未開 →
    # 不傳 boundary 給 scheduler → 與現行 ACP 逐位元相同 (golden 安全)。τ_b = κ·replay 典型 NN 間距。
    b_gate = cfg.scheduler.get("boundary_gate", False)
    b_kappa = cfg.scheduler.get("boundary_kappa", 1.5)
    b_recompute = cfg.scheduler.get("boundary_recompute_every", 20)
    _b_tau = {"val": None, "epoch": -10**9}    # τ_b 快取 (每 b_recompute epoch 重算一次，避免每步 O(M²))

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
    #? cfg.patience / patience 參數保留 (config & 測試相容) 但回滾移除後不再生效。

    # ── 多候選 (batch_latent) 用的小工具 (閉包讀 smodel/sc/權重/replay/rad_*/state) ──
    #    僅 multi 時呼叫；rad_theta 為 run 中才填的閉包變數，呼叫時讀當前值 (只讀不寫)。
    def _build_candidate(logits_row, tau):
        oe = AntennaPattern(AntennaPattern.binarization(logits_row, tau))
        for f in feeds:
            oe = oe + f
        return oe

    def _candidate_loss(oe, ep):
        #? 現役 loss：sm_target + SC + boundary + 信任懲罰 + rad (TV/island/gap 為 legacy，多候選分支不接)。
        L = smodel(oe.series).criterion()
        if sc_w:
            L = L + sc_w * sc.forward(oe.size_converter(output_shape="B, 1, H, W"))
        if bnd_w > 0 and replay is not None and len(replay) > 1:
            L = L + bnd_w * boundary_loss(oe.series, replay.patterns())
        #? 信任懲罰 λ_trust·u(x)：把候選推離「SM 自己沒把握 (ensemble 分歧大)」的洞。λ_trust 由 trust
        #  控制器給 (Exp2 靜態 / Exp3 隨 gap 動)；需 ensemble SM。lambda_trust()=0 → 不加 (golden 安全)。
        if trust.lambda_trust() > 0 and has_unc:
            L = L + trust.lambda_trust() * smodel.uncertainty(oe.series)
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
                #? acquisition：減去 κ·u(x) → 偏好「SM 沒把握」的候選 (主動學習：花 HFSS 去修 SM 最不確定處)。
                #  κ 由 trust 控制器給 (Exp2 靜態 / Exp3 SM 失準時才探)；需 ensemble SM。κ()=0 → 純收割 SM 最佳。
                if trust.kappa() > 0 and has_unc:
                    s = s - trust.kappa() * float(smodel.uncertainty(c.series))
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

    def _boundary_signal():
        #? boundary-gated ACP 用：回傳 (boundary_now, τ_b)。boundary_now = 當前 pattern 到最近已見的
        #  MSE 距離；τ_b = κ·replay 典型 NN 間距 (每 b_recompute epoch 重算一次，避免每步 O(M²))。
        #  未開閘門 / 無緩衝 → (None, None) → scheduler 收到 None → 現行 ACP 行為。
        if not (b_gate and replay is not None and len(replay) > 2):
            return None, None
        if _b_tau["val"] is None or (epoch - _b_tau["epoch"]) >= b_recompute:
            with torch.no_grad():
                _b_tau["val"] = boundary_threshold(replay.patterns(), b_kappa)   # κ·replay 典型 NN 間距
            _b_tau["epoch"] = epoch
        with torch.no_grad():
            b_now = float(boundary_loss(output_element.series, replay.patterns()))
        return b_now, _b_tau["val"]

    epoch = start_epoch
    consecutive_skips = 0          # 連續 HFSS 失敗計數 (成功/快取命中即歸零)
    reopened_once = False          # 本波連敗是否已 reopen 過 (再連敗 → 判系統性故障中斷)
    last_stack = None              # 最近一筆成功響應 (skip 的 epoch 拿來佔位給監控)
    prev_pattern = None            # 上一 epoch 的 pattern (算 flips 探索量；首 epoch 無)
    #? 累計真實 HFSS 模擬次數 (cache 命中/skip 不算)——loss 曲線的正確 x 軸。斷點續跑：續舊欄；
    #  舊 run 無此欄 → 用歷史 fresh 標記數 (sm_gap 只在 fresh epoch 落值) 回填,計數不歸零。
    hfss_calls = int(float(state.last("hfss_calls", len(state.series("sm_gap")))))
    prev_best = state.last("best_loss", float("inf"))  # 追 stall：best_loss 上次刷新的值 (續跑沿用載回的最佳)
    stall = 0                      # best_loss 連續幾個 epoch 沒刷新 (停滯偵測)
    while epoch < epochs:
        epoch += 1
        epoch_t0 = time()
        generator.change(epoch)
        simulator.start(epoch)
        generator.requires_grad(True, train=True)
        generator.optimizer.zero_grad()
        rad_real_this_epoch = None      # 本 epoch 真實方向圖 (僅非快取分支會填；供監控疊圖)
        rad_fit_val = 0.0               # rad head 線上擬合最終 loss (僅 fresh-rad epoch 更新；其餘 0)
        sm_gap_val = None               # SM「訓前」對新點誤差 = generalization (僅 fresh-HFSS epoch 算)
        sm_fit_hist = []                # 本 epoch SM 重訓的逐 epoch/step loss (看訓到 fit 沒；僅 fresh)
        probe_stats = {}                # 本 epoch 自適應探測曲線 (argmin/min/max err；僅 adaptive+有 prev 快照)
        #? 回滾 (early_stop → 載回最佳 epoch + 重訓 SM) 已於 2026-06-28 移除：
        #  對「generator-free + K 候選 + 線上更新 SM」不合身——(1)貪婪規則:沒贏過最佳就退回 →
        #  探索性 pattern 拿不到「成為新據點」的權利、卡在第一個山頭;(2)退回舊 generator 卻配當下
        #  變動的 SM、本質矛盾;(3)原實作有 off-by-one(存 step 後狀態) + 覆蓋最佳檔兩個 bug、實際 ≈ no-op。
        #  探索改交給 K 個獨立候選 + SM 引導 (+ trust)；最佳 pattern 仍安全存在 patterns/ (不可變)。

        # 生成：模型只出 logits；STE 二值化是管線的固定一步，tau 由 ACP 控制 (× 信任乘子)。
        #? tau_eff = ACP 的 tau × trust.tau_mult()：閉迴路下 SM 失準 (t↓) → tau 放軟、保持 pattern 可塑、
        #  不在 SM 盲區亂鎖定二值；SM 可信 (t→1) → tau_mult=1 → 純 ACP 退火銳化。enable=False → ×1.0 (golden 安全)。
        tau_eff = generator.scheduler.get_temp() * trust.tau_mult()
        sel_stats = None
        if multi:
            # 多候選：同批生成 K 個 → SM 評分選最佳 → output_element = 選中的那張 (走下方昂貴路徑)
            if hasattr(generator.model, "anneal_sigma"):           # σ 退火僅 batch_latent 有；direct 無 σ → 跳過
                generator.model.anneal_sigma(epoch / epochs)        # σ 隨進度退火 (探索→收斂)
            tau = tau_eff
            logits_K = generator(spec.concat())                    # (K, out_dim)
            cands = [_build_candidate(logits_K[k], tau) for k in range(generator.model.K)]
            k_star, sel_stats = _select_best(cands)
            output_element = cands[k_star]
        else:
            logits = generator(spec.concat())
            output_element = AntennaPattern(
                AntennaPattern.binarization(logits, tau_eff)
            )
            for f in feeds:
                output_element = output_element + f

        # 去重：patterns/ 的 hash 檔名即「模擬過」快取，沒見過才跑 (mock/HFSS)
        cached = state.lookup(~output_element)
        skipped = False
        if cached is None:
            if verbose: logger.info(f"[{epoch}] HFSS 模擬中…")
            try:
                result = output_element.simulate()
            except Exception as e:
                #! 單筆 HFSS 失敗 (常見：病態幾何讓 oEditor.Unite 丟 COM 例外) 不該帶走整個 run。
                #  skip 這筆：計數 → 未到門檻只收半成品專案；連敗到頂先 reopen 重生、再連敗才判系統性故障中斷。
                skipped = True
                consecutive_skips += 1
                logger.warning(
                    f"[{epoch}] HFSS 模擬失敗 (連續第 {consecutive_skips}/{max_consecutive_skips} 次，"
                    f"pattern {fingerprint(~output_element)[:12]})，skip 這一筆：{type(e).__name__}: {e}"
                )
                if consecutive_skips >= max_consecutive_skips:
                    if not reopened_once:
                        #! 連敗到頂：可能是 COM session 退化 → reopen 重生 (kill+重連) 後再給一輪。
                        logger.error(f"連續 {consecutive_skips} 次 HFSS 失敗 → reopen() 重生 HFSS 連線後再試")
                        try:
                            simulator.reopen()
                        except Exception as re:
                            #! reopen 自己也救不回來 (連 GetAppDesktop 都連不上) → 判系統性故障、優雅中斷,
                            #  不讓 raw com_error 半路逃到 excepthook。(open() 已內建重試;到這代表真的起不來。)
                            raise RuntimeError(
                                f"HFSS reopen() 重生失敗 (epoch {epoch})，判定系統性故障、中斷 run："
                                f"{type(re).__name__}: {re}"
                            ) from re
                        reopened_once = True
                        consecutive_skips = 0
                    else:
                        #! 重生後又連敗 → 判定系統性故障 (license/磁碟/碼 bug)，中斷交給 excepthook 寄信。
                        raise RuntimeError(
                            f"HFSS reopen 後仍連續 {max_consecutive_skips} 次失敗 (epoch {epoch})；"
                            f"判定系統性故障、中斷 run。最後錯誤 {type(e).__name__}: {e}"
                        )
                else:
                    try:
                        simulator.end(save_project=False)   # 未到重生門檻：只收這回合半成品 (end 內建逐級容錯)
                    except Exception as ee:
                        logger.warning(f"[{epoch}] 失敗回合收尾也異常，略過: {ee}")
                #? 讓 G 仍對 SM 走一步 (G 梯度來自 SM、與 HFSS 無關)：sim_loss 用 carry-forward 佔位、
                #  只餵 ACP 排程器與 metrics；不寫 pattern 快取、不更新 SM (無真實響應可學)。
                sim_loss = float(state.last("sim_loss", 1.0))
                stack = last_stack
                phash = ""
                state.append("sim_loss", sim_loss)
        if cached is None and not skipped:
            consecutive_skips = 0           # 成功 → 連敗歸零
            reopened_once = False
            hfss_calls += 1                 # 真的燒了一次 HFSS (cache/skip 分支不加)
            sim_loss = result.criterion()
            stack = result.stack()
            last_stack = stack
            #? sm_gap = SM「線上訓練前」對這張新 pattern 的預測誤差 = generalization 訊號 (dlf/dlf_fit/refit
            #  在比的本體)。**一律算、落 csv**(不再只綁 trust);必須訓練前量 (train_one_data 後 SM 已擬合本筆、
            #  gap 被抹平成假 0)。對照 sm_target(訓後,近 0=memorize):兩者落差大 = 在背、沒 generalize。
            with torch.no_grad():
                _sm_pred_pre = float(smodel(output_element.series).criterion())
            sm_gap_val = abs(_sm_pred_pre - sim_loss.item())
            if trust.enable:
                trust.update(_sm_pred_pre, sim_loss.item())   # 閉迴路:同一個訓前 gap 餵控制器
            #? 自適應訓練量：這一輪的新點是 held-out (產生它的那版 SM 沒訓過它) → 拿它評「上一輪的快照」→ observe。
            #  held-out 鐵律：評的是「產生這個點的那段先前訓練」的快照，不是接下來要對它訓練的那段 (否則洩題)。
            if sm_mode == "adaptive" and prev_snapshots:
                probe_errs = {ep: smodel.eval_snapshot(sd, output_element.series, stack)
                              for ep, sd in prev_snapshots.items()}
                sm_ctrl.observe(probe_errs)
                probe_stats = sm_ctrl.probe_stats(probe_errs)
            #? SM 線上更新：模式分派抽成 _update_surrogate (single/replay/dlf/dlf_fit/refit)。
            #  緩衝一律「全收」(學長論文 §3.5)；細節與各模式語意見該函式 docstring。
            sm_fit_hist = _update_surrogate(
                sm_mode, smodel, replay, output_element, stack, sim_loss, state,
                newest_steps=sm_newest_steps, elite_epochs=sm_elite_epochs, sm_ctrl=sm_ctrl, verbose=verbose,
            )
            if sm_mode == "adaptive":              # 這一輪的新快照 → 下一輪的 held-out 探測對象 (記憶體、覆蓋上一輪)
                prev_snapshots = dict(getattr(smodel, "_probe_snapshots", {}))
            # 方向圖 (選用)：讀 last_radiation、清理 dB target、順手存 rad_store、線上訓練方向圖頭。
            #  細節抽成 _radiation_online_step (行為逐位元不變)；rad_theta 首次拿到真 θ 後即固定。
            if rad_on:
                rad_theta, rad_real_this_epoch, rad_fit_val = _radiation_online_step(
                    smodel, simulator, rad_store, output_element, rad_theta,
                    min_loss=rad_min_loss, max_epoch=rad_max_epoch,
                    freeze_trunk=rad_freeze, verbose=verbose,
                )
            smodel.save()                       # 存 SM (斷點續跑 / rollback 重訓基礎)
            state.append("sim_loss", sim_loss.item())
            phash = state.add_pattern(~output_element, stack, sim_loss.item())
        elif cached is not None:
            if verbose: logger.info(f"[{epoch}] pattern 重複 → 取快取結果 (跳過 HFSS)")
            consecutive_skips = 0           # 快取命中 = 非失敗 → 連敗歸零
            reopened_once = False
            stack, sim_loss, phash = cached
            last_stack = stack
            state.append("sim_loss", sim_loss)

        state.append("sim_loss_avg", state.average("sim_loss"))

        best_loss = state.last("best_loss", float("inf"))
        if state.last("sim_loss") <= best_loss:
            best_loss = state.last("sim_loss")
        state.append("best_loss", best_loss)
        if best_loss < prev_best:          # best_loss 有刷新 → 歸零並記新最佳；否則累加 (停滯偵測)
            stall, prev_best = 0, best_loss
        else:
            stall += 1

        state.append("pattern_hash", phash)
        state.append("r_feed", r_feed(~output_element))

        # 審查後優化：GEN 反傳時把 SM 當「凍住的可微中介」——關 SM 梯度，省其無謂反傳 + 防禦
        # (杜絕日後誤讀/誤用 GEN backward 殘留在 SM 上的梯度)。GEN 梯度與 loss 值不變 → golden 零漂移。
        smodel.requires_grad(False)
        # 更新 GEN (借道可微分 SM)
        if multi:
            # 聚合：mean over K 候選的「現役四 loss」(reparam → 最小化雲的 E[loss]，把中心 z* 拉向好區)。
            loss = sum(_candidate_loss(c, epoch) for c in cands) / len(cands)
            if div_w:                                       # 候選排斥 (opt-in)：防候選塌縮 → best-of-K 持續有賺頭
                loss = loss + div_w * candidate_repulsion(logits_K)
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
            # 信任懲罰 λ_trust·u(x) (選用，與多候選分支對稱)：罰 ensemble 分歧大的 pattern。
            # 需 ensemble SM；lambda_trust()=0 → 不加 (golden 零漂移)。
            if trust.lambda_trust() > 0 and has_unc:
                loss = loss + trust.lambda_trust() * smodel.uncertainty(output_element.series)
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
        # loss 分量診斷 (no_grad、不影響訓練 loss)：對「本 epoch 選中的 pattern」拆出各分量落 csv/TB。
        #   sm_target = SM 對目標的預測損失 (對照真實 sim_loss → 看 SM 準不準、是不是 plateau 瓶頸)
        #   sc_loss = 連通性懲罰；bnd_loss = 離已見分布距離 (= boundary 控制訊號本身；replay 才有已見分布)
        with torch.no_grad():
            sm_target_val = float(smodel(output_element.series).criterion())
            sc_loss_val = float(sc.forward(output_element.size_converter(output_shape="B, 1, H, W")))
            bnd_loss_val = (float(boundary_loss(output_element.series, replay.patterns()))
                            if (replay is not None and len(replay) > 1) else None)
            #? SM 不確定性 (ensemble 成員分歧)：信任懲罰/acquisition 的訊號本身，落 csv 供診斷。
            sm_unc_val = float(smodel.uncertainty(output_element.series)) if has_unc else None
            #? worst-margin (in-band S11/Gain dB 餘裕,正=達標)：用「真實 HFSS 響應」stack 算 (非 SM 預測)。
            #  僅 single port (method 型 target);dual 用 interval、餘裕定義不同 → 留空。metal_frac=金屬比例 (崩塌)。
            if stack is not None and cfg.port == "single":
                worst_margin_val, wm_by_label = worst_margin(stack, PORT_SPECS[cfg.port]["labels"], cfg.targets)
            else:
                worst_margin_val, wm_by_label = None, {}    # dual/首個 skip：per-label 餘裕留空
            cur_pattern = (~output_element)                 # 本 epoch pattern (含固定 feed)；算金屬比例與探索量
            metal_frac_val = float(cur_pattern.float().mean())
            flips_val = int((cur_pattern != prev_pattern).sum()) if prev_pattern is not None else None
            #? sm_bias = 真實 sim_loss − SM 對目標預測 (SM 樂觀偏差；僅 fresh，對照 sm_gap 一起看)
            sm_bias_val = (float(state.last("sim_loss")) - sm_target_val) if sm_gap_val is not None else None
        state.append("sm_target", sm_target_val)
        state.append("sc_loss", sc_loss_val)
        if bnd_loss_val is not None:                     # replay/dlf 才有已見分布 (single 模式留空)
            state.append("bnd_loss", bnd_loss_val)
        if sm_unc_val is not None:                       # ensemble SM 才有不確定性 (單一 SM 留空)
            state.append("sm_unc", sm_unc_val)
        if trust.enable:                                 # 閉迴路才落信任標量 t + 驅動它的 gap_ema (開迴路 run 留空)
            state.append("trust_t", float(trust.t))
            #? gap_ema = 驅動 t 的訊號本身 → 落 csv 供正式機調 g0 時稽核控制器有沒有在反應 (None=尚無 fresh HFSS)。
            state.append("gap_ema", float(trust.gap_ema) if trust.gap_ema is not None else 0.0)
        #? debug 訊號 (2026-06-27)：sm_gap/sm_fit_* 僅 fresh-HFSS epoch 有 (留空於 cached/skip)；
        #  worst_margin/metal_frac 每 epoch 都有 (worst_margin 在首個就 skip、stack=None 時留空)。
        if sm_gap_val is not None:
            state.append("sm_gap", sm_gap_val)           # SM 訓前對新點誤差 = generalization (dlf/dlf_fit/refit 在比的)
        if sm_fit_hist:
            state.append("sm_fit_loss", float(sm_fit_hist[-1]))    # SM 重訓收斂後 loss (看訓到 fit 沒)
            state.append("sm_fit_epochs", float(len(sm_fit_hist))) # SM 重訓實跑 epoch/step 數 (dlf=1 vs dlf_fit=N)
        if worst_margin_val is not None:
            state.append("worst_margin", worst_margin_val)
        state.append("metal_frac", metal_frac_val)
        state.append("skipped", 1.0 if skipped else 0.0)
        #? 追蹤訊號：flips(探索量,首 epoch 空)/stall(每 epoch)/sm_bias(fresh)/wm_per-label(single)。
        #  sparse 欄靠 runstate touched-set 在沒 append 的 epoch 真·留空。
        if flips_val is not None:
            state.append("flips", float(flips_val))
        state.append("stall", float(stall))
        state.append("hfss_calls", float(hfss_calls))   # 每 epoch 都落 (dense) → 任何欄都能改用它當 x 軸
        if sm_bias_val is not None:
            state.append("sm_bias", sm_bias_val)
        for _lbl in ("S11", "Gain"):
            if _lbl in wm_by_label:
                state.append(f"wm_{_lbl}", float(wm_by_label[_lbl]))
        prev_pattern = cur_pattern              # 供下一 epoch 算 flips
        if sm_mode == "adaptive":               # 本輪訓練量 (每 epoch) + held-out 探測曲線 (僅有 prev 快照時)
            state.append("sm_train_epochs", float(sm_ctrl.target_epochs()))
            for _k in ("probe_argmin", "probe_min_err", "probe_max_err"):
                if _k in probe_stats:
                    state.append(_k, probe_stats[_k])
        if rad_on:
            state.append("rad_loss", rad_loss_val)      # 監控用 (只在 rad run 出現此欄)
            state.append("rad_fit", rad_fit_val)        # rad head 線上擬合 loss (看方向圖頭收斂沒)
        if multi:                                       # 多候選：σ 退火 + 候選池健康度 → 也落 metrics.csv (離線可分析)
            if hasattr(generator.model, "sigma"):       # σ 僅 batch_latent 有 (direct 無 σ → 不落此欄)
                state.append("sigma", float(generator.model.sigma))
            if sel_stats is not None:
                for _k in ("score_best", "score_mean", "score_spread", "fresh_frac"):
                    state.append(_k, sel_stats[_k])
            with torch.no_grad():                       # 候選相似度 (高=塌縮；排斥項要壓低它)
                state.append("cand_similarity", float(candidate_repulsion(logits_K)))
        loss.backward()
        #? grad_norm = guidance 梯度的總範數 (反傳後、step 前量) → 抓梯度消失/爆炸。max_norm=inf →
        #  只回傳範數、不裁剪 (不改梯度、golden 安全)。
        grad_norm_val = float(torch.nn.utils.clip_grad_norm_(generator.model.parameters(), float("inf")))
        state.append("grad_norm", grad_norm_val)
        #? boundary-gated ACP (opt-in)：把 boundary 訊號 + τ_b 餵給 scheduler，讓它決定 plateau 時
        #  要不要 warm restart (衝出可信區就抑制加熱)。未開閘門 → b_thr=None → 不傳 → 現行 ACP。
        b_now, b_thr = _boundary_signal()
        if b_thr is not None:
            generator.step(scheduler_param=sim_loss, boundary=b_now, boundary_threshold=b_thr)
        else:
            generator.step(scheduler_param=sim_loss)
        smodel.requires_grad(True)                       # 還原 SM 梯度狀態 (上方 GEN 反傳前暫關)
        generator.model.eval()
        state.append("gen_loss", loss.item())
        if b_gate:                                       # 閘門診斷落 csv (b_gate run 每 epoch 一筆；τ_b 未定補 0)
            state.append("boundary_threshold", b_thr if b_thr is not None else 0.0)
            state.append("restart_suppressed", 1.0 if generator.scheduler._last_restart_suppressed else 0.0)

        generator.save()                       # 存 GEN (供 rollback 載回 / 斷點續跑)

        if not skipped:                        # skip 的回合在失敗當下已收尾 (或已 reopen)，這裡不重複 end (避免對 num=None 斷言)
            simulator.end()
            simulator.clean()

        state.append("epoch", epoch)
        state.append("time", round(time() - epoch_t0, 1))   # 本 epoch 耗時 (HFSS 為主)
        state.save_row()                       # metrics.csv append 一行 (斷點續跑檢查點)

        # on_epoch 收到「本 epoch 快照」：純量指標 + 繪圖素材 (監控端畫 pattern/響應/方向圖用)
        if on_epoch is not None and stack is not None:   # 首個 epoch 就 skip 時無可沿用響應 → 該 epoch 不發監控快照
            snap = dict(
                sim_loss=float(state.last("sim_loss")),
                best_loss=float(state.last("best_loss")),
                gen_loss=float(state.last("gen_loss")),
                r_feed=float(state.last("r_feed")),
                tau=float(generator.scheduler.get_temp()),   # ACP 排程 tau (post-step；監控 ACP 退火曲線)
                lr=float(optimizer.param_groups[0]["lr"]),
                time=float(state.last("time")),
                pattern=~output_element,               # 本 epoch 的 pattern (已 detach)
                response=stack.detach().cpu(),         # 本 epoch 的響應 (labels, 點數)
                spec=spec,                             # 響應規格 (labels/x/目標曲線)
                r_feed_painter=r_feed,                 # 饋電連通圖的繪圖器 (plot 最新一筆)
            )
            snap["sm_target"] = sm_target_val          # loss 分量診斷 (對照 sim_loss 看 SM 準不準)
            snap["sc_loss"] = sc_loss_val
            snap["skipped"] = 1.0 if skipped else 0.0
            snap["grad_norm"] = grad_norm_val           # guidance 梯度範數 (消失/爆炸)
            snap["metal_frac"] = metal_frac_val         # 金屬比例 (崩塌偵測)
            snap["stall"] = float(stall)                # best_loss 停滯 epoch 數
            if flips_val is not None:
                snap["flips"] = float(flips_val)        # 探索量 (相鄰 epoch 像素翻轉)
            if sm_bias_val is not None:
                snap["sm_bias"] = sm_bias_val           # SM 樂觀偏差 (fresh)
            for _lbl in ("S11", "Gain"):
                if _lbl in wm_by_label:
                    snap[f"wm_{_lbl}"] = float(wm_by_label[_lbl])
            if sm_mode == "adaptive":                   # 自適應訓練量 + 探測曲線 (監控 TB)
                snap["sm_train_epochs"] = float(sm_ctrl.target_epochs())
                snap.update(probe_stats)
            if worst_margin_val is not None:
                snap["worst_margin"] = worst_margin_val # in-band S11/Gain dB 餘裕 (真目標,正=達標)
            if sm_gap_val is not None:                  # SM 訓前對新點誤差 = generalization (僅 fresh)
                snap["sm_gap"] = sm_gap_val
            if sm_fit_hist:                             # SM 重訓收斂 loss / epoch 數 (看訓到 fit 沒;僅 fresh)
                snap["sm_fit_loss"] = float(sm_fit_hist[-1])
                snap["sm_fit_epochs"] = float(len(sm_fit_hist))
            if bnd_loss_val is not None:
                snap["bnd_loss"] = bnd_loss_val
            if sm_unc_val is not None:                  # ensemble SM 的不確定性 (成員分歧)
                snap["sm_unc"] = sm_unc_val
            if trust.enable:                            # 閉迴路信任標量 t + gap_ema + 實際二值化 tau
                snap["trust_t"] = float(trust.t)
                snap["gap_ema"] = float(trust.gap_ema) if trust.gap_ema is not None else 0.0
                snap["tau_eff"] = float(tau_eff)        # 監控：信任乘子作用後、真正用於二值化的 tau
            if rad_on:                                 # 方向圖 (選用)：純量 + 疊圖素材 (monitor 端畫)
                snap["rad_loss"] = rad_loss_val
                snap["rad_fit"] = rad_fit_val
                if rad_snapshot is not None:
                    snap["radiation"] = rad_snapshot
            if multi:                                  # 多候選 (選用)：σ 退火 + 候選池健康度 (診斷 Z 賺頭)
                if hasattr(generator.model, "sigma"):  # σ 僅 batch_latent 有 (direct 無 σ)
                    snap["sigma"] = float(generator.model.sigma)
                if sel_stats is not None:
                    snap.update(sel_stats)             # score_best / score_mean / score_spread / fresh_frac
                snap["cand_similarity"] = float(state.last("cand_similarity"))   # 候選相似度 (高=塌縮)；已於上方算入 csv
            if b_thr is not None:                      # boundary-gated ACP (選用)：診斷閘門有沒有在作用
                snap["boundary"] = b_now
                snap["boundary_threshold"] = b_thr
                snap["in_trusted"] = 1.0 if b_now < b_thr else 0.0
                snap["restart_suppressed"] = 1.0 if generator.scheduler._last_restart_suppressed else 0.0
            on_epoch(epoch, snap)

    return state


def _assert_sm_checkpoint_sane(smodel, epoch, *, max_abs: float = 1e3):
    """續跑守衛：SM 權重若非有限或量級異常（灌爆但非 NaN）→ 明確報錯叫使用者重開。
    踩過的雷：發散訓練把 sm.pth 灌爆（健康 max|w|≈0.3，壞掉 5680），續跑只 `smodel.load()`
    載回這個爛檔、繞過 old_sm.pth 暖啟動，永遠卡在同一個壞 pattern（gen_loss 1e37、撞快取空轉）。
    與 mock 相容：MagicMock 的 members / named_parameters() 迭代皆預設為空 → 兩條分支都自動略過、
    不擋既有測試 (續跑測試以 mock smodel 走 members 分支、迭代空 → 不誤報)。"""
    #? 集成 SM (有 members) → 逐成員檢查；單一 SM → 檢查 smodel.model (原樣)。
    members = getattr(smodel, "members", None)
    if members is not None:
        named = ((f"m{i}.{n}", p) for i, m in enumerate(members) for n, p in m.model.named_parameters())
    else:
        named = smodel.model.named_parameters()
    for name, p in named:
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
    if gen_pretrained_path is not None:
        if Path(gen_pretrained_path).exists():
            generator.pre_load_model(gen_pretrained_path)
        else:
            #! 缺檔靜默降級是踩過的雷：明確示警 (否則 GEN 默默從隨機起步、暖啟動沒生效卻沒人知)。
            logger.warning(f"generator.pretrained 指定了 {gen_pretrained_path} 但檔案不存在 → "
                           f"GEN 從隨機權重起步 (暖啟動未生效)")
    # (3) SM 載入：預訓練檔 > 離線預訓練
    sm_has_pretrained = sm_pretrained_path is not None and Path(sm_pretrained_path).exists()
    offline_ok = offline_dataset is not None and len(offline_dataset) > 0
    if sm_pretrained_path is not None and not sm_has_pretrained:
        #! 缺檔靜默降級是踩過的雷 (old_sm≈隨機、以為暖啟動其實沒有)：
        #  - 連 offline 後援都沒有 → SM 全隨機、整個 _harvest 受控變因作廢 → fail-fast 直接報錯，不空跑。
        #  - 有 offline 後援 → 大聲示警後降級 (至少不是靜默)。
        if not offline_ok:
            raise FileNotFoundError(
                f"surrogate.pretrained 指定了 {sm_pretrained_path} 但檔案不存在，且無 offline_dataset 後援 → "
                f"SM 會全隨機、實驗無意義。請確認路徑 (常見：sm_harvest.pth 未在此機 DATASET_PATH 下)。"
            )
        logger.warning(f"surrogate.pretrained 指定了 {sm_pretrained_path} 但檔案不存在 → 靜默降級為 offline 重訓 "
                       f"(非預期的暖啟動；要量「好 SM」請先確認檔案存在)")
    if sm_has_pretrained:
        if cfg.radiation.get("enable", False):
            #? 方向圖版 SM 多了 head_rad、舊 sm.pth 沒有 → strict=False 部分載入共用 trunk/freq
            #  head (缺的 head_rad 維持隨機)，避免退回 elif 在數萬筆上從零預訓練 (HFSS 前先卡死)。
            smodel.pre_load_model(sm_pretrained_path, strict=False)
        else:
            smodel.pre_load_model(sm_pretrained_path)       # 共用路徑：簽名與行為與原樣相同
    elif offline_dataset is not None:
        if offline_ok:
            smodel.train_by_datas(offline_dataset)
        else:
            logger.warning("surrogate.offline_dataset 指定了但筆數為 0 → 離線預訓練跳過、SM 從隨機起步 "
                           "(資料夾名是否打錯/掛載失敗?)")
    # (4) KuoHung 暖身：呼叫端綁好的 warmup(smodel)，對 SM 做單筆暖身訓練
    if warmup is not None:
        warmup(smodel)
    #? 重置 SM 線上 lr：offline 預訓練 / 暖身 / strict 暖啟動都可能把 ReduceLROnPlateau 的 lr 砍到地板，
    #  或繼承 checkpoint 塌掉的 lr → 線上 train_one_data 幾乎不更新。把 lr 拉回建構值 (只重置步長與
    #  排程器狀態、保留動量/二階矩 → 不會冷 optimizer 過衝爆 NaN)。continue_run 已於上方 return、不受影響。
    smodel.reset_online_lr()
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
