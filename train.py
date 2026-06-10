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

from antenna.utils import Complete, DATASET_PATH, ROOTDIR, config, connect_network_drive, logger
config.device = "cpu"

import torch
from antenna import AntennaPattern, get_result_path
from antenna.monitor import TrainingMonitor
from antenna.training import load_config, build_simulator, run_training
from antenna.utils.data import DataManager
from antenna.utils.store import SampleStore

torch.autograd.set_detect_anomaly(True)


def load_dataset(name):
    """載入離線資料集：資料夾 = 新格式 (SampleStore)；否則走舊 DataManager (單一 pickle)。
    NAS 上舊資料集正式轉換後 (script/convert_dataset.py)，這裡會自動切到新格式。"""
    path = DATASET_PATH.joinpath(name)
    if path.is_dir():
        return SampleStore(path)
    return DataManager(name, rootdir=DATASET_PATH)


def resolve_models(cfg):
    """把 config 的 generator/surrogate 區段解析成 run_training 的模型載入參數。
    路徑相對 DATASET_PATH；未設的項目回 None → 對應模型從隨機權重起步。

    回傳 dict：
      - gen_pretrained_path : GEN 預載入權重 (generator.pretrained)        [L2 暖啟動]
      - sm_pretrained_path  : SM 預訓練權重    (surrogate.pretrained)        [L1]
      - offline_dataset     : 離線預訓練資料集 (surrogate.offline_dataset)
      - warmup              : KuoHung 暖身 callable (surrogate.warmup) 或 None
    """
    def _path(rel):
        return str(DATASET_PATH.joinpath(rel)) if rel else None

    warmup = None
    warmup_name = cfg.surrogate.get("warmup")
    if warmup_name:
        from KuoHung import KuoHung          # lazy：僅暖身時才需要 (讀 NAS 上的參考圖樣)
        pattern, response = KuoHung.load(str(warmup_name))
        series = AntennaPattern(pattern).series
        #! 沿用舊 single 3/4 的暖身門檻 (比 hfss.min_loss 預設更緊，max_epoch=1e4)
        warmup = lambda sm: sm.train_one_data(series, response, min_loss=0.001, max_epoch=int(1e4))

    return dict(
        gen_pretrained_path=_path(cfg.generator.get("pretrained")),
        sm_pretrained_path=_path(cfg.surrogate.get("pretrained")),
        offline_dataset=(load_dataset(cfg.surrogate["offline_dataset"])
                         if cfg.surrogate.get("offline_dataset") else None),
        warmup=warmup,
    )


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
    model_kwargs = resolve_models(cfg)

    #* 監控：寫 TB 事件檔到 <結果夾>/tb (多機實驗在任一台跑
    #*   tensorboard --logdir "T:\...\result" 即可並排監看全部)
    monitor = TrainingMonitor(RESULT_PATH / "tb")

    def on_epoch(epoch, m):
        logger.info(
            f"[{epoch}/{cfg.epochs}] real={m['real_loss']:.4f} min={m['min_loss']:.4f} "
            f"fake={m['fake_loss']:.4f} r_feed={m['r_feed']:.2%} tau={m['tau']:.3f} "
            f"({m['time']:.0f}s)"
        )
        monitor.on_epoch(epoch, m)

    TEMP = run_training(
        cfg,
        simulator=simulator,
        record_path=RESULT_PATH,
        continue_run=CONTINUE_RUN,
        on_epoch=on_epoch,
        **model_kwargs,             # gen_pretrained_path / sm_pretrained_path / offline_dataset / warmup
    )

    monitor.summary(TEMP, RESULT_PATH)   # 結尾總覽圖 summary.png (不依賴 TB)
    monitor.close()

    Complete(
        f"Training Finished! ({cfg.name}, Min Loss: {TEMP.custom('real_loss', min)})",
        name=cfg.name, send_email=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python train.py configs/<experiment>.yaml")
        raise SystemExit(1)
    main(sys.argv[1])
