"""HFSS 凍結的最後一層：殺完 HFSS 仍不返回 → 行程退出 → r81v2_worker 自動重啟（2026-10-15 218/216 卡死）。"""
import threading

from script import dedust, r81v2_worker


def test_exit_when_call_never_returns():
    codes = []
    dedust._exit_if_still_hung(threading.Event(), "測試", grace=0.01, exit_fn=codes.append)
    assert codes == [dedust.HANG_EXIT]


def test_no_exit_when_call_returned():
    done, codes = threading.Event(), []
    done.set()
    dedust._exit_if_still_hung(done, "測試", grace=0.01, exit_fn=codes.append)
    assert codes == []


def test_worker_restarts_only_on_hang_exit():
    rcs, kills = [r81v2_worker.HANG_EXIT, r81v2_worker.HANG_EXIT, 3], []
    calls = []

    def call(cmd):
        calls.append(cmd)
        return rcs.pop(0)

    rc = r81v2_worker.main([], call=call, kill=lambda: kills.append(1), sleep=lambda s: None)
    assert rc == 3 and len(calls) == 3 and len(kills) == 2


def test_hang_exit_codes_agree():
    assert r81v2_worker.HANG_EXIT == dedust.HANG_EXIT
