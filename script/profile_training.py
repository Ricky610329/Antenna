"""
script/profile_training.py — 找訓練的速度瓶頸（真實 HFSS，量到每個環節的實際耗時）。

⚠ 只在「正式機」跑（conda patch + 真 HFSS）。開發機沒 HFSS、跑不起來。

用法（正式機）：
    conda activate patch
    python -m script.profile_training configs/single_sc_rad.yaml --epochs 3

做法：用「真實模擬器」跑幾個 epoch，把計時器掛到閉迴路各環節，量到實際耗時：
    - HFSS 求解+讀回（AntennaPattern.simulate）← 通常的 wall-clock 大頭
    - HFSS 開專案 / 收尾 / 清理（simulator.start / end / clean）
    - SM 單筆擬合 S11/Gain（train_one_data，激進過擬合）
    - SM 單筆擬合 方向圖（train_one_data_rad，rad 才有）
    - SC loss（625x625 特徵分解）、SM 推論
停用 pattern 快取 → 每 epoch 都跑完整路徑、各環節可公平比較。

輸出各環節「總耗時 + 每次 + 佔比」，一眼看出瓶頸在哪、之後要加速哪裡。
比較技巧：跑 single_base / single_sc / single_sc_rad / *_mirror，看各功能各加多少。
"""
import argparse
import tempfile
import time
from collections import defaultdict

from antenna.utils import config, DATASET_PATH
config.device = "cpu"

from antenna import AntennaPattern
from antenna.training import load_config, run_training, build_simulator
from antenna.models.surrogates import SurrogateModel
from antenna.losses import SpectralConnectivityLoss, GapClosingLoss
from antenna.utils.runstate import RunState


TIMES = defaultdict(lambda: {"t": 0.0, "n": 0})   # 各環節：累計耗時 + 呼叫次數
ITERS = defaultdict(list)                          # 單筆擬合的內層迭代數 (看是否撞 max_epoch)


def _wrap(target, attr, label, *, grab_iters=False):
    """把 target.attr 包一層計時器 (累進 TIMES[label])；grab_iters 另記單筆擬合的迭代數。"""
    orig = getattr(target, attr)

    def wrapper(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            TIMES[label]["t"] += time.perf_counter() - t0
            TIMES[label]["n"] += 1
            if grab_iters and a:
                try:
                    ITERS[label].append(int(a[0].record("epoch", 0)))   # a[0]=self (SM)
                except Exception:
                    pass

    setattr(target, attr, wrapper)


def main():
    ap = argparse.ArgumentParser(description="訓練速度瓶頸 profiling (真實 HFSS，正式機跑)")
    ap.add_argument("config", help="YAML config 路徑")
    ap.add_argument("--epochs", type=int, default=3, help="量測幾個 epoch (預設 3；HFSS 慢，別設太大)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.epochs = args.epochs
    cfg.patience = 10 ** 9                                # 高 patience → 不觸發 rollback (避免干擾量測)
    AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))   # 與 train.py 一致

    record_path = tempfile.mkdtemp()                     # 暫存結果夾 (HFSS 專案/CSV 寫這；絕對路徑)
    sim = build_simulator(cfg, record_path)              # 真實 HFSS 模擬器 (single/dual/rad 由 cfg 決定)
    simcls = type(sim)

    # 暖啟動 SM (old_sm.pth)：與正式訓練一致，train_one_data 迭代數才真實
    sm_pre = None
    rel = cfg.surrogate.get("pretrained")
    if rel:
        p = DATASET_PATH.joinpath(rel)
        if p.exists():
            sm_pre = str(p)

    # 掛計時器到各環節 (含 HFSS)
    _wrap(AntennaPattern, "simulate", "HFSS 求解+讀回 (simulate)")
    _wrap(simcls, "start", "HFSS 開專案 (start)")
    _wrap(simcls, "end", "HFSS 收尾 (end)")
    _wrap(simcls, "clean", "HFSS 清理 (clean)")
    _wrap(SurrogateModel, "train_one_data", "SM 單筆擬合 (S11/Gain)", grab_iters=True)
    _wrap(SurrogateModel, "train_one_data_rad", "SM 單筆擬合 (方向圖)", grab_iters=True)
    _wrap(SurrogateModel, "__call__", "SM 推論 (GEN loss)")
    _wrap(SpectralConnectivityLoss, "forward", "SC loss (特徵分解)")
    _wrap(GapClosingLoss, "forward", "gap loss")
    RunState.lookup = lambda self, key: None             # 停用 pattern 快取 → 每 epoch 跑完整路徑

    print(f"profiling {cfg.name} ({args.epochs} epochs, 真實 HFSS, SM 暖啟動={'是' if sm_pre else '否'}) …")
    t0 = time.perf_counter()
    run_training(cfg, simulator=sim, record_path=record_path, seed=0,
                 sm_pretrained_path=sm_pre, verbose=False)
    total = time.perf_counter() - t0

    # ── 報表 ────────────────────────────────────────────────────────────────
    measured = sum(v["t"] for v in TIMES.values())
    rows = sorted(TIMES.items(), key=lambda kv: -kv[1]["t"])
    print("\n" + "=" * 70)
    print(f"  訓練速度瓶頸 profile — {cfg.name}  ({args.epochs} epochs, 真實 HFSS)")
    print("=" * 70)
    print(f"  {'環節':<22}{'總計(s)':>9}{'次數':>5}{'每次(s)':>9}{'佔比':>8}")
    print("  " + "-" * 60)
    for label, v in rows:
        pct = 100 * v["t"] / measured if measured else 0.0
        print(f"  {label:<22}{v['t']:>9.2f}{v['n']:>5}{v['t']/max(v['n'],1):>9.2f}{pct:>7.1f}%")
    print("  " + "-" * 60)
    print(f"  {'量測小計':<22}{measured:>9.2f}")
    print(f"  {'整圈 wall':<22}{total:>9.2f}   (每 epoch ≈ {total/args.epochs:.1f}s)")
    for label, its in ITERS.items():
        if its:
            mx = cfg.sm_train.get("max_epoch", 20000)
            print(f"  · {label} 內層迭代數/epoch: {its}  (上限 {mx}{'，撞上限！' if max(its) >= mx else ''})")
    print("\n  佔比最高那行就是瓶頸。HFSS 類環節是外部(難加速)；SM 單筆擬合 / SC 是 compute(可加速)。")
    print("  註：已停用 pattern 快取 → 每 epoch 都實跑(正式訓練重複 pattern 會走快取、更快)。")


if __name__ == "__main__":
    main()
