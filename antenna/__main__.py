"""
CLI 入口點: python -m antenna [command] [hydra overrides]

使用範例::

    # 使用預設設定訓練
    python -m antenna train

    # 使用特定實驗設定
    python -m antenna train +experiment=train_single

    # 覆蓋參數
    python -m antenna train +experiment=train_single epochs=2000 environment.device=cuda:0

    # Hydra multirun
    python -m antenna train -m +experiment=train_single,train_dual
"""

import sys


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "train":
        sys.argv = [sys.argv[0]] + sys.argv[2:]  # 移除子命令讓 Hydra 解析
        _train()
    elif command == "help" or command == "--help":
        print(
            "使用方式: python -m antenna <command> [options]\n"
            "\n"
            "Commands:\n"
            "  train    執行訓練（使用 Hydra 設定）\n"
            "\n"
            "範例:\n"
            "  python -m antenna train +experiment=train_single\n"
            "  python -m antenna train +experiment=train_dual epochs=2000\n"
        )
    else:
        print(f"未知命令: {command}，使用 'python -m antenna help' 查看使用方式。")
        sys.exit(1)


def _train():
    import hydra
    from omegaconf import DictConfig

    from antenna.configs.schema import register_configs

    register_configs()

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def train_main(cfg: DictConfig) -> None:
        from loguru import logger
        from omegaconf import OmegaConf

        logger.info(f"設定:\n{OmegaConf.to_yaml(cfg)}")
        logger.info(f"實驗名稱: {cfg.experiment_name}")
        logger.info(f"模型: {cfg.model}")
        logger.info(f"模擬器: {cfg.simulator}")
        logger.info(f"Epochs: {cfg.epochs}")

        # TODO: Phase 3 後續 — 實作完整 Trainer class
        # from antenna.training.trainer import Trainer
        # trainer = Trainer(cfg)
        # trainer.run()

        logger.warning("Trainer 尚未實作，目前僅載入設定。請使用原始 train_*.py 腳本或等待後續更新。")

    train_main()


if __name__ == "__main__":
    main()
