"""
HFSS 單筆模擬失敗的容錯（回歸）。

對應 2026-06-25 正式機事故：G 探索到病態幾何 → `oEditor.Unite` 丟 COM 例外
（single_port.py:343）→ 例外一路冒到全域 excepthook → 寄信 → 整個 run 死、需人工重啟。

期望行為（antenna/training.py run_training）：
  1. 單筆 HFSS 失敗 → skip 該筆、記 warning、不中斷；G 仍對 SM 走一步（carry-forward sim_loss）。
  2. 連續失敗到 max_consecutive_skips → reopen() 重生 HFSS 連線後再試（不立刻中斷）。
  3. reopen 後仍連續失敗到門檻 → raise（判系統性故障，交給 excepthook 寄信），避免靜默空轉。
"""
import os

import pytest

from antenna.training import load_config, run_training
from test_baseline_loop import _MockSim, FIX


class _FlakySim(_MockSim):
    """在指定 epoch（或全程）讓 __call__ 丟例外，模擬 HFSS 幾何/COM 失敗。

    其餘行為（open/start/end/clean、確定性響應）沿用 _MockSim；額外提供 reopen() 與
    fail_count，讓測試可斷言「容錯路徑」確實被走到。
    """
    def __init__(self, labels, *, fail_epochs=(), always_fail=False):
        super().__init__(labels)
        self.fail_epochs = set(fail_epochs)
        self.always_fail = always_fail
        self.fail_count = 0
        self.calls["reopen"] = 0
        self._epoch = 0

    def start(self, num):
        super().start(num)
        self._epoch = num          # 記住本回合編號，__call__ 依它決定要不要失敗

    def reopen(self, *a, **k):
        self.calls["reopen"] += 1

    def __call__(self, pattern, **kw):
        if self.always_fail or self._epoch in self.fail_epochs:
            self.fail_count += 1
            raise RuntimeError(f"mock HFSS Unite COM error @epoch {self._epoch}")
        return super().__call__(pattern, **kw)


def _run(tmp_path, sim, **kw):
    """跑 single_test.yaml，捕捉每 epoch (epoch, sim_loss, gen_loss) 快照。"""
    cfg = load_config(os.path.join(FIX, "single_test.yaml"))
    rows = []
    run_training(cfg, simulator=sim, record_path=tmp_path, seed=0,
                 on_epoch=lambda e, m: rows.append((e, m["sim_loss"], m["gen_loss"])),
                 verbose=False, **kw)
    return rows


def test_single_failure_skipped_run_continues(tmp_path):
    """中途某 epoch HFSS 失敗 → skip 該筆，run 不崩、其餘 epoch 照跑、G 仍前進。"""
    sim = _FlakySim(("S11", "Gain"), fail_epochs={3})
    rows = _run(tmp_path, sim)
    by_epoch = {e: (sl, gl) for e, sl, gl in rows}

    # 該 epoch 失敗、未觸發 training 層 reopen（1 次 skip < 預設門檻 5）。
    # fail_count==2：pattern.py simulate() 會先 restart 重跑一次，重跑仍失敗才落到 training 層 skip。
    assert sim.fail_count == 2
    assert sim.calls["reopen"] == 0
    # 全 6 epoch 都有快照：epoch 3 用 carry-forward 響應佔位，仍照常監控（不缺洞）
    assert set(by_epoch) == {1, 2, 3, 4, 5, 6}
    # epoch 3 的 sim_loss = carry-forward 上一筆（epoch 2）；G 仍對 SM 走一步 → gen_loss 有限（非 NaN）
    assert by_epoch[3][0] == by_epoch[2][0]
    assert by_epoch[3][1] == by_epoch[3][1]
    # end 每 epoch 各收一次（skip 的 epoch 在 except 內已收尾、迴圈尾不重複呼叫）
    assert sim.calls["start"] == 6
    assert sim.calls["end"] == 6
    # skip 旗標落 csv：epoch 3 = 1（失敗跳過）、其餘 = 0（debug 時一眼看出哪些 epoch 沒真跑 HFSS）
    import csv
    with open(tmp_path / "metrics.csv", newline="", encoding="utf-8") as f:
        crows = {int(r["epoch"]): r for r in csv.DictReader(f)}
    assert crows[3]["skipped"] == "1.0"
    assert crows[2]["skipped"] == "0.0" and crows[4]["skipped"] == "0.0"


def test_consecutive_failures_reopen_then_abort(tmp_path):
    """連敗到門檻 → reopen 重生一次；reopen 後又連敗到門檻 → 中斷（RuntimeError，交 excepthook 寄信）。"""
    sim = _FlakySim(("S11", "Gain"), always_fail=True)
    with pytest.raises(RuntimeError, match="系統性故障"):
        _run(tmp_path, sim, max_consecutive_skips=3, max_epochs=10)

    # 連敗 3（epoch 1-3）→ reopen 一次；reopen 後又連敗 3（epoch 4-6）→ 中斷。
    # fail_count==12：6 個失敗 epoch，每個都被 pattern.py 重跑一次（原始 + restart 重跑）。
    assert sim.calls["reopen"] == 1
    assert sim.fail_count == 12
