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
import json
import os
import shutil
import socket
import sys
import tempfile
from datetime import datetime

from antenna.utils import Complete, DATASET_PATH, ROOTDIR, config, connect_network_drive, logger
config.device = "cpu"

import torch
from antenna import AntennaPattern, get_result_path
from antenna.utils.monitor import TrainingMonitor, launch_tensorboard, seed_local_tb
from antenna.utils.web import get_local_ip
from antenna.training import load_config, build_simulator, run_training
from antenna.utils.store import SampleStore

torch.autograd.set_detect_anomaly(True)


def load_dataset(name):
    """載入離線資料集：資料夾 = 新格式 (SampleStore)；否則走舊 DataManager (單一 pickle)。
    NAS 上舊資料集正式轉換後 (script/convert_dataset.py)，這裡會自動切到新格式。"""
    path = DATASET_PATH.joinpath(name)
    if path.is_dir():
        return SampleStore(path)
    from antenna.legacy import DataManager   # lazy：僅當指到舊 .dataset 單檔時才需要 (核心不依賴 legacy)
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
        from script.kuohung import KuoHung   # lazy：僅暖身時才需要 (讀 NAS 上的參考圖樣)
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


def write_status(result_path, **fields):
    """運行管理心跳：<結果夾>/status.json (state/機器/epoch/updated_at)。
    與「監控」(TB) 是兩件事 —— 這個回答的是「哪個 run 還活著、跑到哪、在哪台機器」。"""
    fields.update(machine=socket.gethostname(),
                  updated_at=datetime.now().isoformat(timespec="seconds"))
    (result_path / "status.json").write_text(
        json.dumps(fields, ensure_ascii=False, indent=2), encoding="utf-8")


def _local_tb_dir(run_stem):
    """本機暫存的 tb 事件檔目錄 (本地碟)，給 TB tail。

    為何要本機一份：TB 在 SMB(網路碟) 上 tail「正在被寫入」的事件檔，撞到半截 flush 就會
    停讀、畫面卡在某個 epoch 不再前進 (實測 .37 卡 22)。讓 TB tail「本機」檔＝穩定不卡。
    NAS 那份由 monitor 的 mirror_dir 同步雙寫 (即時備份、沒人 tail → 不會卡)。
    以 run 名為 key、各機器各自獨立。"""
    d = os.path.join(tempfile.gettempdir(), "antenna_tb", run_stem)
    os.makedirs(d, exist_ok=True)
    return d


def main(yaml_path):
    cfg = load_config(yaml_path)
    logger.info(f"Loaded config: {cfg.name} (port={cfg.port})")

    connect_network_drive("T:", r"\\140.123.106.219\temp", "user", "ailab120")
    RESULT_PATH, CONTINUE_RUN = get_result_path(
        f"[Patch-{cfg.port}-{{device}}-{{hash_id}}] {cfg.name}",
        rootdir=ROOTDIR, generate_code=__file__, enable_exception_handler=True,
    )
    shutil.copy(yaml_path, RESULT_PATH / "config.yaml")   # 設定快照：結果夾自我說明
    AntennaPattern.setDefaultCoordinate((0, 25, 0, 25))

    simulator = build_simulator(cfg, RESULT_PATH)
    model_kwargs = resolve_models(cfg)

    #* 監控：tb 事件檔「雙寫」——本機一份 (給 TB tail、穩定不卡) + NAS 一份 (即時備份)。
    #*   為何：TB 在 SMB 上 tail 正在寫入的事件檔會卡住 (撞半截 flush 後停讀)。讓 TB tail 本機
    #*   ＝不卡；NAS 那份同步雙寫、沒人 tail → 不卡、且隨時可從 NAS 看實驗過程 (討論/備份)。
    LOCAL_TB = _local_tb_dir(RESULT_PATH.stem)
    seed_local_tb(LOCAL_TB, RESULT_PATH / "tb")   # 續跑/重開機：把 NAS 既有進度同步到本機 → TB 看得到之前跑的結果
    monitor = TrainingMonitor(LOCAL_TB, mirror_dir=RESULT_PATH / "tb")
    monitor.log_config(open(yaml_path, encoding="utf-8").read())   # config 原文進 TB Text 分頁

    #* 開訓即自動起監控面板 (TensorBoard 指向本機 tb) + 印可點連結 → 不必另開 terminal 追蹤。
    tb_port = launch_tensorboard(LOCAL_TB)
    if tb_port:
        ip = get_local_ip()
        logger.success("─" * 56)
        logger.success("  監控面板已就緒，點連結看即時狀況 (免另開 terminal)：")
        logger.success(f"    本機 / RDP 內 → http://localhost:{tb_port}")
        logger.success(f"    從你的電腦看  → http://{ip}:{tb_port}")
        logger.success(f"    本次 run：{RESULT_PATH.stem}")
        logger.success("─" * 56)

    def on_epoch(epoch, m):
        logger.info(
            f"[{epoch}/{cfg.epochs}] sim={m['sim_loss']:.4f} best={m['best_loss']:.4f} "
            f"gen={m['gen_loss']:.4f} r_feed={m['r_feed']:.2%} tau={m['tau']:.3f} "
            f"({m['time']:.0f}s)"
        )
        monitor.on_epoch(epoch, m)
        write_status(RESULT_PATH, state="running", epoch=epoch, total=cfg.epochs,
                     sim_loss=m["sim_loss"], best_loss=m["best_loss"])

    write_status(RESULT_PATH, state="running", epoch=0, total=cfg.epochs)
    try:
        state = run_training(
            cfg,
            simulator=simulator,
            record_path=RESULT_PATH,
            continue_run=CONTINUE_RUN,
            on_epoch=on_epoch,
            **model_kwargs,         # gen_pretrained_path / sm_pretrained_path / offline_dataset / warmup
        )
    except BaseException:
        write_status(RESULT_PATH, state="crashed")
        monitor.close()                       # flush 兩份 writer (本機 + NAS 即時備份)
        raise

    monitor.summary(state, RESULT_PATH)   # 結尾總覽圖 summary.png (不依賴 TB)
    monitor.close()                       # flush/close 兩份 writer；NAS 那份已是即時備份、不需回拷
    best_loss = min(state.series("sim_loss"))
    write_status(RESULT_PATH, state="finished", epoch=state.last_epoch,
                 total=cfg.epochs, best_loss=best_loss)

    Complete(
        f"Training Finished! ({cfg.name}, Best Loss: {best_loss})",
        name=cfg.name, send_email=True,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python train.py configs/<experiment>.yaml")
        raise SystemExit(1)
    main(sys.argv[1])
