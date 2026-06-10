"""
antenna/monitor.py — 訓練監控 (TensorBoard) + 結尾總覽圖。

多機多實驗的監看方式 (取代 app.py 自渲染)：
    每台訓練機: train.py 把指標/圖寫到 <結果夾>/tb/ (每 epoch 幾 KB)
    任何機器:   tensorboard --logdir "T:\\...\\result"  → 瀏覽器並排比較所有實驗

設計：
- 只掛在 train.py 的 on_epoch hook 上，核心 run_training 不知道 TB 的存在。
- tensorboard 未安裝 → 警告並停用監控，「絕不」阻擋訓練 (HFSS 時間很貴)。
- 訓練結束另存 summary.png 進結果夾：結果夾不開 TB 也能自我說明。
- 舊腳本逐 epoch 2x3 圖的內容對照：loss/lr/tau/r_feed/耗時 → Scalars；
  pattern+饋電連通、響應 vs 目標 → Images (有 epoch 滑桿)。
"""
import matplotlib
matplotlib.use("Agg")               # 訓練程序無頭繪圖 (不開視窗)
import matplotlib.pyplot as plt
from loguru import logger


class TrainingMonitor:
    """接 on_epoch(epoch, m) 的 TB 監視器。m = run_training 給的「epoch 快照」。"""

    def __init__(self, logdir, image_every: int = 1):
        """
        Args:
            logdir: TB 事件檔目錄 (慣例: <結果夾>/tb)。
            image_every: 每幾個 epoch 記一次圖 (純量每 epoch 都記)。
        """
        self.image_every = image_every
        self._spec = None               # 從第一次 on_epoch 的快照取得 (供結尾總覽圖用)
        self._targets_logged = False
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(str(logdir))
        except ImportError:
            logger.warning("tensorboard 未安裝 → TB 監控停用 (pip install tensorboard)；訓練照常進行")
            self.writer = None

    # ── 每 epoch ────────────────────────────────────────────────────────────
    def on_epoch(self, epoch: int, m: dict):
        self._spec = m.get("spec", self._spec)
        if self.writer is None:
            return
        for group, keys in (("loss", ("real_loss", "fake_loss", "min_loss")),
                            ("sched", ("lr", "tau")),
                            ("index", ("r_feed", "time"))):
            for k in keys:
                if k in m:
                    self.writer.add_scalar(f"{group}/{k}", m[k], epoch)

        if not self._targets_logged and self._spec is not None:
            self.writer.add_figure("target/curves", self._target_figure(self._spec))
            self._targets_logged = True

        if epoch % self.image_every == 0:
            if m.get("r_feed_painter") is not None:
                fig, ax = plt.subplots(figsize=(5, 5))
                m["r_feed_painter"].plot(ax)        # pattern + 饋電連通區 (綠=連通金屬)
                self.writer.add_figure("pattern/feed_reachability", fig, epoch)
            if m.get("response") is not None and self._spec is not None:
                self.writer.add_figure(
                    "response/sim_vs_target",
                    self._response_figure(self._spec, m["response"]), epoch,
                )
        self.writer.flush()             # 每 epoch 都 flush：TB 端即時可見 (epoch 以分鐘計，開銷可忽略)

    # ── 結尾總覽圖 (存進結果夾，與 TB 無關、無 TB 也會產生) ───────────────────
    def summary(self, TEMP, save_dir):
        """依 TEMP 的完整歷史畫總覽圖 → <save_dir>/summary.png。
        排版：上排 = 最佳 pattern + 各響應 vs 目標 (依 label 數伸縮)；下排 = loss / r_feed·time / 摘要。"""
        spec = self._spec
        losses = TEMP["real_loss"]
        best = losses.index(min(losses))
        labels = list(spec.labels)
        cols = max(3, 1 + len(labels))
        fig, axes = plt.subplots(2, cols, figsize=(6 * cols, 9))

        axes[0][0].imshow(TEMP["patch_pattern_buf"][best])   # ~pattern 是 merge 後的 2D 圖
        axes[0][0].set_title(f"Best Pattern (epoch {best + 1})")

        x = spec.x()
        response = TEMP["patch_result_buf"][best]
        for n, (label, sim) in enumerate(zip(labels, response), start=1):
            ax = axes[0][n]
            ax.plot(x, sim.detach().cpu(), color="blue", label="Simulation")
            ax.plot(x, spec[label].response.detach().cpu(), "b--", label="Target")
            ax.set_title(label); ax.legend()
        for n in range(1 + len(labels), cols):
            axes[0][n].axis("off")

        ax_loss = axes[1][0]
        ax_loss.plot(TEMP["real_loss"], color="red", label="real_loss")
        ax_loss.plot(TEMP["fake_loss"], color="purple", label="fake_loss", alpha=0.8)
        ax_loss.plot(TEMP["min_loss"], label="min_loss")
        ax_loss.set_title("Loss Curve"); ax_loss.legend()

        ax_idx = axes[1][1]
        ax_idx.plot(TEMP["r_feed"], color="tab:blue", label="r_feed")
        ax_t = ax_idx.twinx()
        ax_t.plot(TEMP["time"], color="tab:orange", label="time (s)")
        ax_idx.set_title("R_feed / Time"); ax_idx.legend(loc="upper left"); ax_t.legend(loc="upper right")

        axes[1][2].axis("off")
        axes[1][2].text(0.05, 0.6, f"best epoch: {best + 1}\nmin real_loss: {min(losses):.4f}\n"
                                   f"epochs: {len(losses)}", fontsize=14, family="monospace")
        for n in range(3, cols):
            axes[1][n].axis("off")

        fig.tight_layout()
        path = str(save_dir) + "/summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"總覽圖已存：{path}")

    def close(self):
        if self.writer is not None:
            self.writer.close()

    # ── 內部繪圖 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _target_figure(spec):
        x = spec.x()
        labels = list(spec.labels)
        fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 4))
        for ax, label in zip(axes if len(labels) > 1 else [axes], labels):
            ax.plot(x, spec[label].response.detach().cpu(), color="red", marker="o")
            ax.set_title(f"Target {label}")
        fig.tight_layout()
        return fig

    @staticmethod
    def _response_figure(spec, response):
        x = spec.x()
        labels = list(spec.labels)
        fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 4))
        for ax, label, sim in zip(axes if len(labels) > 1 else [axes], labels, response):
            ax.plot(x, sim, color="blue", label="Simulation")
            ax.plot(x, spec[label].response.detach().cpu(), "b--", label="Target")
            ax.set_title(label); ax.legend()
        fig.tight_layout()
        return fig
