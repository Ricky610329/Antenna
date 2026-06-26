"""
script/check_harvest_consistency.py — 抽驗「harvest 離線資料」是否與「現在的 HFSS」對得上。

動機 (暖啟動的前提驗證)：harvest_single/dual 是從學長舊資料收割的；sm_harvest.pth 把這些舊
response 當 ground truth 訓練。若當年的頻率網格 / 增益定義 (Realized vs Total) / 底板 / HFSS 版本
與現在「任一處」不同 → response shape 仍對得上、但數值系統性偏 → sm_harvest「對舊 HFSS 準、對
現在偏」，宣稱的 val MSE 名不符實，暖啟動可能有害卻無人察覺 (最典型的靜默 bug)。

本支：從 harvest 隨機抽 N 筆，用「現在的」HFSS 重跑，比對 harvest 存的 response vs 重跑的 response。
只燒 N 次 HFSS (分鐘級)，把「整個 harvest 暖啟動到底有沒有用」從信心問題變成數據問題。

判讀：
  - 中位 MSE ≈ train_sm_offline 宣稱的 val MSE (~13 量級) → harvest 與現在 HFSS 對得上、暖啟動可信。
  - 中位 MSE 顯著更大 (接近 old_sm「≈隨機」的 ~35-38) → 不一致，sm_harvest 須用「現在 HFSS 重收的
    資料」重訓，否則 _harvest 系列的「好 SM 天花板」結論不可信。
  - MSE 巨大且雜亂 (非系統性偏移) → 先懷疑 response 格式 / label 順序不符，而非 HFSS 漂移。

用法 (正式機，需 Ansys HFSS；開發機不可跑)：
    python -m script.check_harvest_consistency --dataset harvest_single --n 15
    python -m script.check_harvest_consistency --dataset harvest_dual --n 15 --config configs/dual_base.yaml
"""
import argparse
from pathlib import Path

import torch

from antenna.utils import config, DATASET_PATH, connect_network_drive, logger
config.device = "cpu"

from antenna import AntennaPattern
from antenna.training import load_config, setup_responses, build_simulator
from antenna.utils.store import SampleStore


def _bin(p) -> torch.Tensor:
    """任意輸入 → (25,25) 純 0/1 (模擬器要求二值；與 collect_radiation 同正規形)。"""
    return (torch.as_tensor(p).float().reshape(25, 25) > 0.5).float()


def main():
    ap = argparse.ArgumentParser(description="抽驗 harvest 與現在 HFSS 的一致性 (需正式機 HFSS)")
    ap.add_argument("--dataset", default="harvest_single", help="harvest SampleStore 名 (DATASET_PATH 下)")
    ap.add_argument("--n", type=int, default=15, help="抽幾筆重模擬 (每筆一發 HFSS)")
    ap.add_argument("--config", default="configs/single_base.yaml", help="決定 port / 安裝 spec")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", default="_harvest_check", help="HFSS 模擬暫存輸出根目錄")
    args = ap.parse_args()

    connect_network_drive("T:", r"\\140.123.106.219\temp", "user", "ailab120")
    AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))
    cfg = load_config(args.config)
    spec = setup_responses(cfg)                        # 安裝 spec → result.stack() 用此 label 順序
    labels = list(spec.labels)

    store = SampleStore(DATASET_PATH.joinpath(args.dataset), verbose=False)
    if len(store) == 0:
        raise SystemExit(f"{args.dataset} 是空的 (DATASET_PATH={DATASET_PATH})")
    torch.manual_seed(args.seed)
    idxs = torch.randperm(len(store))[: args.n].tolist()
    logger.info(f"抽 {len(idxs)} / {len(store)} 筆 {args.dataset}，用現在的 {cfg.port} HFSS 重模擬比對…")

    sim = build_simulator(cfg, str(Path(args.record).resolve()))
    AntennaPattern.register_simulator(sim)
    sim.open()
    rows = []
    try:
        for k, i in enumerate(idxs):
            x, y_harvest = store[i]
            y_harvest = torch.as_tensor(y_harvest).float()
            oe = AntennaPattern(_bin(x))               # harvest pattern = 完整幾何 (含饋電；與 collect_radiation 同慣例)
            sim.start(k)
            try:
                y_now = oe.simulate().stack().float().reshape(y_harvest.shape)
            finally:
                try: sim.end()
                except Exception: pass
                try: sim.clean()
                except Exception: pass
            mse = float(((y_now - y_harvest) ** 2).mean())
            per_ch = ((y_now - y_harvest) ** 2).mean(dim=-1)        # 逐 label MSE
            rows.append(mse)
            ch = ", ".join(f"{labels[c]}={float(per_ch[c]):.2f}" for c in range(len(labels)))
            print(f"[{k + 1}/{len(idxs)}] store#{i}  MSE={mse:.2f}  ({ch})", flush=True)
    finally:
        if hasattr(sim, "quit"):
            try: sim.quit()
            except Exception: pass

    if not rows:
        raise SystemExit("沒有成功重模擬任何一筆 (HFSS 全失敗?)")
    rows.sort()
    med = rows[len(rows) // 2]
    print("\n" + "=" * 60)
    print(f"抽驗 {len(rows)} 筆 | MSE 中位={med:.2f}  最小={rows[0]:.2f}  最大={rows[-1]:.2f}")
    print("判讀：")
    print("  中位 ≈ train_sm_offline 的 val MSE (~13)         → harvest 可信、暖啟動有效")
    print("  中位 顯著更大 (接近隨機 ~35-38)                   → 不一致，sm_harvest 須用現在 HFSS 重收重訓")
    print("  MSE 巨大且雜亂 (非系統性偏移)                     → 先查 response 格式 / label 順序")
    print("=" * 60)


if __name__ == "__main__":
    main()
