"""
tests/test_simulator_open_retry.py — open() 對「HFSS RPC server 未就緒」的重試韌性。

根因回歸:剛 kill 掉舊 ansysedt 後,新 ansysedt 的 COM/RPC server 要數秒才起得來;過去單發
GetAppDesktop 會撞 com_error(-2147023174 RPC 伺服器無法使用)並逃到 excepthook 帶走整個 run。
open() 現在會「kill 殘行程 → 等 → 重試」。這裡 mock _dispatch/sleep/kill,只驗純 Python 重試邏輯,
不需真 HFSS。
"""
import pytest

pytest.importorskip("win32com")          # dev 無 pywin32 → 跳過(prod / CI 有才跑)

from antenna.patch import patch_simulator as ps


class _Sim(ps.PatchSimulator):
    """最小具體子類:__call__ 是 abstract,給個 no-op 讓基類可實例化。"""
    def __call__(self, *a, **k):
        raise NotImplementedError


class _FakeDesktop:
    def RestoreWindow(self):
        pass


class _FakeApp:
    def GetAppDesktop(self):
        return _FakeDesktop()


def _make_sim(tmp_path, calls):
    sim = _Sim(record_path=str(tmp_path), HFSS_sab_path="x.sab", pixel_count=25)
    sim.kill = lambda: calls.__setitem__("kill", calls["kill"] + 1)   # 別真的 taskkill ansysedt
    return sim


def test_open_retries_until_rpc_ready(tmp_path, monkeypatch):
    """前 2 次連線失敗(RPC 未就緒)→ 第 3 次成功;期間 kill 2 次、拿到連線。"""
    calls = {"dispatch": 0, "kill": 0}

    def fake_dispatch(_name):
        calls["dispatch"] += 1
        if calls["dispatch"] <= 2:
            raise Exception("RPC server unavailable")   # 模擬新 ansysedt 還沒起來
        return _FakeApp()

    monkeypatch.setattr(ps, "_dispatch", fake_dispatch)
    monkeypatch.setattr(ps, "sleep", lambda *a: None)   # 測試不真的等
    sim = _make_sim(tmp_path, calls)

    sim.open(attempts=6, wait=0)
    assert calls["dispatch"] == 3        # 失敗 2 + 成功 1
    assert calls["kill"] == 2            # 每次失敗 kill 一次殘行程
    assert hasattr(sim, "oDesktop")      # 成功拿到連線


def test_open_gives_up_after_attempts(tmp_path, monkeypatch):
    """一直連不上 → 試滿 attempts 次才拋最後例外(交給上層容錯),不無限迴圈。"""
    calls = {"dispatch": 0, "kill": 0}

    def fake_dispatch(_name):
        calls["dispatch"] += 1
        raise Exception("RPC server unavailable")

    monkeypatch.setattr(ps, "_dispatch", fake_dispatch)
    monkeypatch.setattr(ps, "sleep", lambda *a: None)
    sim = _make_sim(tmp_path, calls)

    with pytest.raises(Exception, match="RPC server unavailable"):
        sim.open(attempts=3, wait=0)
    assert calls["dispatch"] == 3
    assert calls["kill"] == 3            # 每次失敗都 kill,最後才放手
