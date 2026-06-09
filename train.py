# -*- coding: utf-8 -*-
"""
train.py — 由「外部 YAML config」驅動的訓練入口 (取代 train_single.py / train_dual.py)。

用法：
    conda activate patch
    python train.py configs/single_base.yaml
    python train.py configs/dual_base.yaml

- port (single/dual)、超參數、目標響應、模型載入策略都在 YAML 內指定。
- 本檔只負責 production「外殼」：掛 NAS、結果路徑/斷點偵測、建真實模擬器、
  解析模型載入設定、呼叫 antenna.training.run_training，最後寄完成通知。
- 核心訓練迴圈在 antenna.training.run_training (有 golden 測試把關)。
"""
import sys

from antenna.utils import *        # config, connect_network_drive, ROOTDIR, DATASET_PATH, logger, Complete
config.device = "cpu"

import torch
from antenna import AntennaPattern, get_result_path
from antenna.training import load_config, build_simulator, run_training
from antenna.utils.data import DataManager

torch.autograd.set_detect_anomaly(True)


def resolve_surrogate(cfg):
    """把 config 的 surrogate 區段解析成 (預訓練權重路徑, 離線 DataManager)。
    模型載入完全由 config 指定：未設則回 (None, None) → SM 從隨機權重起步。"""
    pretrained = cfg.surrogate.get("pretrained")
    offline_name = cfg.surrogate.get("offline_dataset")
    pretrained_path = str(DATASET_PATH.joinpath(pretrained)) if pretrained else None
    offline = DataManager(offline_name, rootdir=DATASET_PATH) if offline_name else None
    return pretrained_path, offline


def main(yaml_path):
    cfg = load_config(yaml_path)
    logger.info(f"Loaded config: {cfg.name} (port={cfg.port})")

    connect_network_drive("T:", r"\\140.123.106.219\temp", "user", "ailab120")
    RESULT_PATH, CONTINUE_RUN = get_result_path(
        f"[Patch-{cfg.port}-{{device}}-{{hash_id}}] {cfg.name}",
        rootdir=ROOTDIR, generate_code=__file__, enable_exception_handler=True,
    )
    AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))

    simulator = build_simulator(cfg, RESULT_PATH)
    pretrained_path, offline = resolve_surrogate(cfg)

    def on_epoch(epoch, m):
        logger.info(
            f"[{epoch}/{cfg.epochs}] real={m['real_loss']:.4f} min={m['min_loss']:.4f} "
            f"fake={m['fake_loss']:.4f} r_feed={m['r_feed']:.2%} tau={m['tau']:.3f}"
        )

    TEMP = run_training(
        cfg,
        simulator=simulator,
        record_path=RESULT_PATH,
        continue_run=CONTINUE_RUN,
        pretrained_sm_path=pretrained_path,
        offline_dataset=offline,
        on_epoch=on_epoch,
    )

    Complete(
        f"Training Finished! ({cfg.name}, Min Loss: {TEMP.custom('real_loss', min)})",
        name=cfg.name, send_email=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python train.py configs/<experiment>.yaml")
        raise SystemExit(1)
    main(sys.argv[1])
