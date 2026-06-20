"""
TrainingMonitor (TB 監控 + 結尾總覽圖) 的測試。

CI 沒裝 tensorboard → 重點驗證：(1) 降級不擋訓練 (2) 記錄內容正確 (fake writer)
(3) summary.png 不依賴 TB 也能產生。
"""
import sys

import torch

from antenna import TargetResponse
import antenna.utils.monitor as monitor_mod
from antenna.utils.monitor import TrainingMonitor, launch_tensorboard
from antenna.utils.runstate import RunState


def _spec():
    spec = TargetResponse(labels=("S11", "Gain"), x="n257")
    spec(0, -10, (5, 0, 7, 0, 5), label="S11", add=True)
    spec(-19, 4, (5, 0, 7, 0, 5), label="Gain", add=True)
    return spec


class _FakeWriter:
    def __init__(self): self.scalars = []; self.figures = []; self.texts = []
    def add_scalar(self, tag, v, step): self.scalars.append(tag)
    def add_figure(self, tag, fig, step=None): self.figures.append(tag)
    def add_text(self, tag, text): self.texts.append((tag, text))
    def flush(self): pass
    def close(self): pass


class _FakePainter:
    def plot(self, ax): ax.imshow(torch.zeros(25, 25))


def _metrics(spec):
    return dict(sim_loss=1.0, best_loss=1.0, gen_loss=2.0, r_feed=0.5, tau=1.0,
                lr=0.005, time=3.0, pattern=torch.zeros(25, 25),
                response=torch.zeros(2, 17), spec=spec, r_feed_painter=_FakePainter())


def _bare_monitor(writer, spec=None):
    """繞過 __init__ (不碰真 TB)，直接注入 fake writer。"""
    mon = TrainingMonitor.__new__(TrainingMonitor)
    mon.writer = writer; mon.image_every = 1
    mon._spec = spec; mon._targets_logged = spec is not None
    return mon


def test_disabled_without_tensorboard(tmp_path, monkeypatch):
    """tensorboard 不存在 → 警告降級，on_epoch 完全無痛 (絕不擋訓練)。"""
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    mon = TrainingMonitor(tmp_path / "tb")
    assert mon.writer is None
    mon.log_config("port: single")
    mon.on_epoch(1, _metrics(_spec()))
    mon.close()


def test_scalars_and_figures_logged():
    """純量分組 (loss/sched/index) 與三類圖 (target/pattern/response) 都有記。"""
    mon = _bare_monitor(_FakeWriter())
    mon.on_epoch(1, _metrics(_spec()))
    for tag in ("loss/sim_loss", "loss/gen_loss", "loss/best_loss",
                "sched/lr", "sched/tau", "index/r_feed", "index/time"):
        assert tag in mon.writer.scalars
    for tag in ("target/curves", "pattern/feed_reachability", "response/sim_vs_target"):
        assert tag in mon.writer.figures


def test_config_logged_as_yaml_text():
    """config 原文以 markdown code block 記進 Text 分頁。"""
    mon = _bare_monitor(_FakeWriter())
    mon.log_config("port: single\nlr: 0.005")
    (tag, text), = mon.writer.texts
    assert tag == "config" and "port: single" in text and text.startswith("```yaml")


def test_target_curves_logged_once():
    mon = _bare_monitor(_FakeWriter())
    m = _metrics(_spec())
    mon.on_epoch(1, m); mon.on_epoch(2, m)
    assert mon.writer.figures.count("target/curves") == 1


def test_image_every_throttles():
    """image_every=2 → 偶數 epoch 才記圖，純量每 epoch 都記。"""
    mon = _bare_monitor(_FakeWriter())
    mon.image_every = 2
    m = _metrics(_spec())
    mon.on_epoch(1, m); mon.on_epoch(2, m)
    assert mon.writer.figures.count("response/sim_vs_target") == 1
    assert mon.writer.scalars.count("loss/sim_loss") == 2


def test_summary_png(tmp_path):
    """結尾總覽圖：不依賴 TB (writer=None) 也能從 RunState 歷史產生。"""
    mon = _bare_monitor(None, spec=_spec())
    state = RunState(tmp_path, verbose=False)
    for e in range(1, 4):
        loss = 4.0 - e
        h = state.add_pattern(torch.rand(25, 25), torch.rand(2, 17), loss)
        for k, v in dict(epoch=e, sim_loss=loss, gen_loss=2.0, best_loss=loss,
                         sim_loss_avg=loss, r_feed=0.5, time=3.0, pattern_hash=h).items():
            state.append(k, v)
        state.save_row()
    mon.summary(state, tmp_path)
    assert (tmp_path / "summary.png").exists()


# ── 方向圖監控 (選用) ───────────────────────────────────────────────────────
def _rad_metrics(spec):
    m = _metrics(spec)
    m["rad_loss"] = 0.3
    m["radiation"] = dict(theta=torch.linspace(-90, 90, 9),
                          pred=torch.zeros(2, 9), real=torch.zeros(2, 9),
                          window_deg=55, floor_db=3)
    return m


def test_radiation_scalar_and_figure_logged():
    """方向圖實驗：rad_loss 純量 + radiation/gain_vs_angle 圖都有記。"""
    mon = _bare_monitor(_FakeWriter())
    mon.on_epoch(1, _rad_metrics(_spec()))
    assert "loss/rad_loss" in mon.writer.scalars
    assert "radiation/gain_vs_angle" in mon.writer.figures


def test_no_radiation_keys_when_absent():
    """非方向圖實驗：不該出現 rad_loss / radiation 圖 (回歸保護)。"""
    mon = _bare_monitor(_FakeWriter())
    mon.on_epoch(1, _metrics(_spec()))
    assert "loss/rad_loss" not in mon.writer.scalars
    assert "radiation/gain_vs_angle" not in mon.writer.figures


# ── 自動起監控面板 (launch_tensorboard) ─────────────────────────────────────
def test_launch_tensorboard_reuses_existing(monkeypatch):
    """port 已被占用 (先前 TB 在跑) → 不重複啟動、直接回 True。"""
    monkeypatch.setattr(monitor_mod, "_port_in_use", lambda port: True)
    calls = []
    monkeypatch.setattr(monitor_mod.subprocess, "Popen", lambda *a, **k: calls.append(1))
    assert launch_tensorboard("/x", 6006) is True
    assert not calls                                   # 沒重複起 TB


def test_launch_tensorboard_starts_when_free(monkeypatch):
    """port 空 + 有裝 tensorboard → Popen 起一個、回 True。"""
    monkeypatch.setattr(monitor_mod, "_port_in_use", lambda port: False)
    monkeypatch.setattr(monitor_mod.importlib.util, "find_spec", lambda name: object())
    calls = []
    monkeypatch.setattr(monitor_mod.subprocess, "Popen", lambda *a, **k: calls.append((a, k)))
    assert launch_tensorboard("/x", 6006) is True
    assert len(calls) == 1


def test_launch_tensorboard_nonfatal_without_tb(monkeypatch):
    """沒裝 tensorboard → 回 False、不報錯 (絕不擋訓練)。"""
    monkeypatch.setattr(monitor_mod, "_port_in_use", lambda port: False)
    monkeypatch.setattr(monitor_mod.importlib.util, "find_spec", lambda name: None)
    assert launch_tensorboard("/x", 6006) is False


def test_summary_png_with_radiation(tmp_path):
    """summary 疊方向圖：_last_radiation 有值時也能產生 summary.png。"""
    mon = _bare_monitor(None, spec=_spec())
    mon._last_radiation = dict(theta=torch.linspace(-90, 90, 9),
                               pred=torch.zeros(2, 9), real=torch.zeros(2, 9),
                               window_deg=55, floor_db=3)
    state = RunState(tmp_path, verbose=False)
    for e in range(1, 4):
        loss = 4.0 - e
        h = state.add_pattern(torch.rand(25, 25), torch.rand(2, 17), loss)
        for k, v in dict(epoch=e, sim_loss=loss, gen_loss=2.0, best_loss=loss,
                         sim_loss_avg=loss, r_feed=0.5, time=3.0, pattern_hash=h).items():
            state.append(k, v)
        state.save_row()
    mon.summary(state, tmp_path)
    assert (tmp_path / "summary.png").exists()
