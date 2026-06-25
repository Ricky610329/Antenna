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
import importlib.util
import os
import shutil
import socket
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")               # 訓練程序無頭繪圖 (不開視窗)
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


def _port_in_use(port: int) -> bool:
    """該 port 是否已有人服務 (多半是先前已起的 TensorBoard)。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def launch_tensorboard(logdir, port: int = 6006, max_scan: int = 20):
    """背景啟動 TensorBoard 指向 logdir；回傳「實際使用的 port」(int)，失敗回 None。

    供 train.py 開訓時自動起面板 + 印連結 (使用者不必另開 terminal 跑 tensorboard)。
    - **不重用別人佔住的 port**：6006 被占用多半是「同機前一個 run 的 TB、指向別的 logdir」，
      直接重用會讓連結顯示到舊 run、看起來像卡住。改為從 port 往後找第一個空 port，起
      「這個 run 自己的」TB → 印出的連結保證對到本次 run。回傳那個 port 供 train.py 組連結。
    - 未安裝 tensorboard / 掃描範圍全被占 / 啟動失敗 → 記 warning 回 None，「絕不」阻擋訓練。
    """
    if importlib.util.find_spec("tensorboard") is None:
        logger.warning("未安裝 tensorboard → 無法自動起監控面板 (pip install tensorboard)")
        return None
    for p in range(port, port + max_scan):
        if _port_in_use(p):
            continue                                   # 這個 port 已有人 (多半別的 run 的 TB) → 換下一個
        try:
            subprocess.Popen(
                [sys.executable, "-m", "tensorboard.main",
                 "--logdir", str(logdir), "--port", str(p), "--bind_all"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return p
        except Exception as e:
            logger.warning(f"TensorBoard 自動啟動失敗 (可手動 tensorboard --logdir): {e}")
            return None
    logger.warning(f"port {port}~{port + max_scan - 1} 都被占用 → 沒起新的 TB 面板 (可手動指定 port)")
    return None


def seed_local_tb(local_dir, source_dir):
    """開訓時把 source_dir(NAS 既有事件檔) 同步到 local_dir(本機，給 TB tail)。

    為何需要：續跑/重開機後本機暫存可能沒有之前的事件檔 → TB(tail 本機) 會看不到舊進度
    (現行 run 看不到之前跑的結果)。NAS 是歷史源頭：**清空 local 再從 NAS 整包複製** →
    TB 看得到續跑前的全部進度；『清空再複製』也避免本機殘留舊檔與 NAS mirror 重複計步。
    fault-tolerant：失敗只 warning、不影響訓練 (頂多 TB 看不到舊進度)。
    """
    try:
        if os.path.isdir(local_dir):
            shutil.rmtree(local_dir)
        os.makedirs(local_dir, exist_ok=True)
        if os.path.isdir(str(source_dir)):
            shutil.copytree(str(source_dir), local_dir, dirs_exist_ok=True)
    except Exception as e:
        logger.warning(f"本機 tb 同步 NAS 既有進度失敗（不影響訓練，TB 可能看不到舊進度）：{e}")
        os.makedirs(local_dir, exist_ok=True)


class _FanoutWriter:
    """把 add_*/flush/close 同步分送給多個 SummaryWriter。

    用途：tb 事件檔「雙寫」—— 本機一份 (給 TB tail、穩定不卡) + NAS 一份 (即時備份、可隨時
    從 NAS 看/討論)。NAS 那份沒人 tail → 不會有 SMB tailing 卡住問題。
    - **容錯**：某份 writer 寫入爆掉 (多半 NAS 備份斷線) → 只停用那份、其他照常，絕不拋例外
      拖垮訓練 (監控是外掛、不該污染核心)。
    - add_figure 統一 close=False 後自行 plt.close → 同一張圖能寫進多份 writer、又不漏記憶體。
    """

    def __init__(self, writers):
        self._writers = [w for w in writers if w is not None]

    def _each(self, fn):
        for w in list(self._writers):
            try:
                fn(w)
            except Exception as e:
                logger.warning(f"TB writer 寫入失敗、停用該份 (多半 NAS 備份斷線，不影響訓練)：{e}")
                self._writers.remove(w)

    def add_scalar(self, tag, value, global_step=None):
        self._each(lambda w: w.add_scalar(tag, value, global_step))

    def add_text(self, tag, text):
        self._each(lambda w: w.add_text(tag, text))

    def add_figure(self, tag, figure, global_step=None):
        self._each(lambda w: w.add_figure(tag, figure, global_step, close=False))
        plt.close(figure)

    def flush(self):
        self._each(lambda w: w.flush())

    def close(self):
        self._each(lambda w: w.close())


class TrainingMonitor:
    """接 on_epoch(epoch, m) 的 TB 監視器。m = run_training 給的「epoch 快照」。"""

    def __init__(self, logdir, image_every: int = 1, mirror_dir=None):
        """
        Args:
            logdir: TB 事件檔目錄 (本機；給 TB tail)。
            image_every: 每幾個 epoch 記一次圖 (純量每 epoch 都記)。
            mirror_dir: 若給定 → 事件檔「同時」再寫一份到這 (慣例: NAS 的 <結果夾>/tb)，
                當即時備份。本機那份給 TB tail；NAS 這份沒人 tail → 不會 SMB 卡住。
        """
        self.image_every = image_every
        self._spec = None               # 從第一次 on_epoch 的快照取得 (供結尾總覽圖用)
        self._last_radiation = None     # 最近一次方向圖快照 (供結尾總覽圖疊方向圖；無方向圖實驗恆 None)
        self._targets_logged = False
        try:
            from torch.utils.tensorboard import SummaryWriter
            writers = [SummaryWriter(str(logdir))]
            if mirror_dir is not None:
                try:
                    writers.append(SummaryWriter(str(mirror_dir)))   # NAS 即時備份份
                except Exception as e:
                    logger.warning(f"NAS 即時備份 writer 建立失敗（只寫本機，不影響訓練）：{e}")
            self.writer = _FanoutWriter(writers)
        except ImportError:
            logger.warning("tensorboard 未安裝 → TB 監控停用 (pip install tensorboard)；訓練照常進行")
            self.writer = None

    # ── 一次性：實驗設定 ──────────────────────────────────────────────────────
    def log_config(self, text: str):
        """把實驗 config (YAML 原文) 記進 TB 的 Text 分頁 —— 點開任何 run 即見其完整設定。"""
        if self.writer is not None:
            self.writer.add_text("config", f"```yaml\n{text}\n```")
            self.writer.flush()

    # ── 每 epoch ────────────────────────────────────────────────────────────
    def on_epoch(self, epoch: int, m: dict):
        self._spec = m.get("spec", self._spec)
        if m.get("radiation") is not None:
            self._last_radiation = m["radiation"]   # 供結尾 summary 疊方向圖 (即使無 TB 也記)
        if self.writer is None:
            return
        for group, keys in (("loss", ("sim_loss", "gen_loss", "best_loss", "rad_loss")),
                            ("sched", ("lr", "tau", "sigma")),
                            ("index", ("r_feed", "time")),
                            # 多候選 (batch_latent) 才有：候選池健康度 → 診斷 Z 探索是否有賺頭
                            # (score_spread→0 = 候選塌縮、Z 失效；fresh_frac 低 = 探索停滯)。
                            ("select", ("score_best", "score_mean", "score_spread", "fresh_frac")),
                            # boundary-gated ACP 才有：boundary vs τ_b、是否在可信區、是否抑制了 restart
                            # (restart_suppressed 在 boundary 高時觸發 = 閘門有作用)。
                            ("acp", ("boundary", "boundary_threshold", "in_trusted", "restart_suppressed"))):
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
            if m.get("radiation") is not None:          # 方向圖 (選用)：gain vs 角度，比照 pattern 存圖
                self.writer.add_figure(
                    "radiation/gain_vs_angle",
                    self._radiation_figure(m["radiation"]), epoch,
                )
        self.writer.flush()             # 每 epoch 都 flush：TB 端即時可見 (epoch 以分鐘計，開銷可忽略)

    # ── 結尾總覽圖 (存進結果夾，與 TB 無關、無 TB 也會產生) ───────────────────
    def summary(self, state, save_dir):
        """依 RunState 的完整歷史畫總覽圖 → <save_dir>/summary.png。
        排版：上排 = 最佳 pattern + 各響應 vs 目標 (依 label 數伸縮)；下排 = loss / r_feed·time / 摘要。"""
        spec = self._spec
        rad = getattr(self, "_last_radiation", None)     # 有方向圖實驗才畫；否則佈局與原本相同
        extra = 1 if rad is not None else 0
        losses = state.series("sim_loss")
        best_epoch = state.best_epoch("sim_loss")
        pattern, response = state.pattern_at(best_epoch)
        labels = list(spec.labels)
        cols = max(3, 1 + len(labels) + extra)
        fig, axes = plt.subplots(2, cols, figsize=(6 * cols, 9))

        axes[0][0].imshow(pattern)                       # ~pattern 是 merge 後的 2D 圖
        axes[0][0].set_title(f"Best Pattern (epoch {best_epoch})")

        x = spec.x()
        for n, (label, sim) in enumerate(zip(labels, response), start=1):
            ax = axes[0][n]
            ax.plot(x, sim, color="blue", label="Simulation")
            ax.plot(x, spec[label].response.detach().cpu(), "b--", label="Target")
            ax.set_title(label); ax.legend()
        off_start = 1 + len(labels)
        if rad is not None:                              # 方向圖 (最近一筆，非 best epoch)
            self._draw_radiation_summary(axes[0][off_start], rad)
            off_start += 1
        for n in range(off_start, cols):
            axes[0][n].axis("off")

        ax_loss = axes[1][0]
        ax_loss.plot(losses, color="red", label="sim_loss")
        ax_loss.plot(state.series("gen_loss"), color="purple", label="gen_loss", alpha=0.8)
        ax_loss.plot(state.series("best_loss"), label="best_loss")
        ax_loss.set_title("Loss Curve"); ax_loss.legend()

        ax_idx = axes[1][1]
        ax_idx.plot(state.series("r_feed"), color="tab:blue", label="r_feed")
        ax_t = ax_idx.twinx()
        ax_t.plot(state.series("time"), color="tab:orange", label="time (s)")
        ax_idx.set_title("R_feed / Time"); ax_idx.legend(loc="upper left"); ax_t.legend(loc="upper right")

        axes[1][2].axis("off")
        axes[1][2].text(0.05, 0.6, f"best epoch: {best_epoch}\nbest sim_loss: {min(losses):.4f}\n"
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

    @staticmethod
    def _radiation_figure(rad: dict):
        """方向圖 gain vs theta (phi0/phi90 各一圖)，仿「響應 vs 目標」風格：
        SM 預測 = 藍實線 (Simulation)、HFSS 真實 = 藍虛線 (Target；即 rad head 要學的標籤)。
        疊 ±window 與 G(0°)−floor_db 線 (主波束覆蓋約束，一眼看達標沒)。"""
        theta = np.asarray(rad["theta"])
        order = np.argsort(theta)                        # 依 theta 排序再畫 (防 HFSS 匯出順序造成鋸齒)
        theta = theta[order]
        pred, real = rad.get("pred"), rad.get("real")
        window = float(rad.get("window_deg", 55.0))
        floor = float(rad.get("floor_db", 3.0))
        bore = int(np.argmin(np.abs(theta)))             # boresight = |θ| 最小的取樣點 (≈0°)
        names = ("phi 0° (E-plane)", "phi 90° (H-plane)")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for i, ax in enumerate(axes):
            g0 = None
            if pred is not None:
                p = np.asarray(pred[i])[order]
                ax.plot(theta, p, color="blue", label="SM (pred)"); g0 = p[bore]
            if real is not None:
                r = np.asarray(real[i])[order]
                ax.plot(theta, r, "b--", label="HFSS (real/target)"); g0 = r[bore]
            ax.axvline(-window, color="gray", ls=":", alpha=0.6)
            ax.axvline(window, color="gray", ls=":", alpha=0.6)
            if g0 is not None:
                ax.axhline(g0 - floor, color="red", ls="--", alpha=0.5, label=f"G0-{floor:g}dB")
            ax.set_title(names[i]); ax.set_xlabel("theta (deg)"); ax.set_ylabel("gain (dB)")
            ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    @staticmethod
    def _draw_radiation_summary(ax, rad: dict):
        """summary.png 用的方向圖小格：phi0/phi90 (優先用真實，沒有才用預測) + window/floor 線。"""
        theta = np.asarray(rad["theta"])
        order = np.argsort(theta); theta = theta[order]   # 依 theta 排序再畫
        src = rad.get("real") if rad.get("real") is not None else rad.get("pred")
        window = float(rad.get("window_deg", 55.0))
        floor = float(rad.get("floor_db", 3.0))
        bore = int(np.argmin(np.abs(theta)))
        for i, name in enumerate(("phi0", "phi90")):
            ax.plot(theta, np.asarray(src[i])[order], label=name)
        g0 = float(np.asarray(src[0])[order][bore])
        ax.axvline(-window, color="gray", ls=":"); ax.axvline(window, color="gray", ls=":")
        ax.axhline(g0 - floor, color="red", ls="--", alpha=0.7)
        ax.set_title("Radiation @28GHz"); ax.set_xlabel("theta (deg)"); ax.set_ylabel("gain (dB)")
        ax.legend()
