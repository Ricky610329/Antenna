"""
tests/test_scripts.py — status.py / analyze.py 的純函式測試（不碰 NAS）。

只測可離線驗的純邏輯（機器名解析、欄位過濾、cosine 基底、中位差）;掃 NAS 的部分靠實跑驗。
"""
import numpy as np

from script.status import _machine, _num, _liveness
from script.analyze import _cos_basis, _mad


def test_machine_parse():
    assert _machine("[Patch-single-216-2c121f] pixel_single_r3_explore") == "216"
    assert _machine("[Patch-single-37-e6a4f4] x") == "37"
    assert _machine("no-pattern-here") == "?"


def test_num_filters_empty_and_nan():
    rows = [{"a": "1.5"}, {"a": ""}, {"a": "nan"}, {"a": "2"}, {"b": "9"}]
    assert _num(rows, "a") == [1.5, 2.0]      # 空/nan/缺欄都略過


def _lv(**kw):
    base = dict(state="running", advanced=False, elapsed_enough=False, age_min=1.0, tpe_min=5.0)
    base.update(kw)
    return _liveness(**base)


def test_liveness_terminal_states_authoritative():
    """status.json 的 crashed/finished 是權威終態，不會被誤標成『卡住/在跑』。"""
    assert _lv(state="crashed", age_min=1) == ("當機", False)     # 剛當機也是當機（不看新鮮度）
    assert _lv(state="finished") == ("已完成", False)


def test_liveness_advance_is_proof_of_alive():
    """epoch 比上次掃描前進 = 鐵證在跑（即使心跳看似舊）。"""
    assert _lv(advanced=True, age_min=999) == ("在跑", True)


def test_liveness_running_fresh_vs_stuck():
    """宣稱 running：心跳新且未到判定點 → 在跑?；心跳久沒更新 / 隔>1.5ep 沒前進 → 疑卡住（抓硬砍/凍住）。"""
    assert _lv(state="running", age_min=3, tpe_min=5) == ("在跑?", True)
    assert _lv(state="running", age_min=999, tpe_min=5)[0] == "疑卡住"        # 心跳久沒更新
    assert _lv(state="running", elapsed_enough=True, age_min=1)[0] == "疑卡住"  # 該前進卻沒前進


def test_liveness_legacy_run_without_status_json():
    """無 status.json 的舊 run → 純時間啟發式：新→在跑?、舊→停止。"""
    assert _lv(state=None, age_min=1, tpe_min=5) == ("在跑?", True)
    assert _lv(state=None, age_min=999, tpe_min=5) == ("停止", False)


def test_cos_basis_shape_and_k0_constant():
    theta = np.linspace(-180, 180, 181)
    B = _cos_basis(theta, 8)
    assert B.shape == (8, 181)
    assert np.allclose(B[0], 1.0)             # k=0 mode = cos(0) = 常數 1


def test_mad_median_abs_delta():
    assert _mad([1.0, 3.0, 3.0, 6.0]) == 2.0  # 逐差 2,0,3 → 中位 2


# ── run_curve：x 軸綁 hfss_calls（真實模擬次數；使用者定案 2026-07-02） ────────────────


def _fake_run_dir(tmp_path, with_calls: bool):
    """造一個最小 run 夾：config.yaml + metrics.csv + patterns/*.pt (2 顆 pattern、1 次 cache 命中)。"""
    import torch
    import yaml
    targets = {"S11": {"side": 0, "center": -10, "width": [5, 0, 7, 0, 5], "method": "low"},
               "Gain": {"side": -19, "center": 4, "width": [5, 0, 7, 0, 5], "method": "high"}}
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"name": "t", "port": "single", "targets": targets}), encoding="utf-8")
    pat = tmp_path / "patterns"
    pat.mkdir()
    # aaa: S11 全 -12 (margin +2)、Gain 全 5 (margin +1) → worst=+1；bbb: Gain 全 3 → worst=−1
    resp_a = torch.cat([torch.full((1, 17), -12.0), torch.full((1, 17), 5.0)])
    resp_b = torch.cat([torch.full((1, 17), -12.0), torch.full((1, 17), 3.0)])
    torch.save((torch.zeros(25, 25), resp_a, 1.0), pat / "aaa.pt")
    torch.save((torch.zeros(25, 25), resp_b, 2.0), pat / "bbb.pt")
    hdr = "epoch,pattern_hash" + (",hfss_calls" if with_calls else "")
    rows = ["1,aaa,1", "2,bbb,2", "3,aaa,2"] if with_calls else ["1,aaa", "2,bbb", "3,aaa"]
    (tmp_path / "metrics.csv").write_text(hdr + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path


def test_run_curve_uses_hfss_calls_and_merges_cache_hits(tmp_path):
    """有 hfss_calls 欄 → x=真實模擬次數；cache 命中 (同 x) 收斂成一點、best-so-far 正確。"""
    from antenna.utils import config as _config
    _dev = _config.device
    try:
        from script.benchmark_vs_random import run_curve
        xs, best = run_curve(_fake_run_dir(tmp_path, with_calls=True))
        assert xs == [1, 2]                       # 3 列 → 2 個真實模擬 (第 3 列 cache 命中不佔 x)
        assert best[0] == best[1] == 1.0          # best-so-far 保持在 aaa 的 worst=+1
    finally:
        _config.device = _dev                     # 該 script 匯入時把全域 device 設 cpu → 還原,不影響後續測試


def test_resolve_run_prefers_exact_suffix(tmp_path, monkeypatch):
    """回歸 (2026-07-03)：子字串比對讓 `x_dip` 同時命中 `x_dip` 與 `x_dip_explore`、再被 mtime 挑走
    錯的夾 → R3/R4 歸檔兩臂數字相同。修法＝結尾相符優先。"""
    from antenna.utils import config as _config
    _dev = _config.device
    try:
        import script.benchmark_vs_random as bvr
        rd = tmp_path / "result"
        (rd / "[m-1] pixel_x_dip").mkdir(parents=True)
        (rd / "[m-2] pixel_x_dip_explore").mkdir(parents=True)
        (rd / "[m-2] pixel_x_dip_explore" / "newer.txt").write_text("x")   # 讓錯的夾 mtime 較新
        monkeypatch.setattr(bvr, "ROOTDIR", tmp_path)
        assert bvr._resolve_run("x_dip").name.endswith("pixel_x_dip")      # 不被較新的 dip_explore 搶走
        assert bvr._resolve_run("x_dip_explore").name.endswith("pixel_x_dip_explore")
    finally:
        _config.device = _dev


def test_run_curve_falls_back_to_epoch_for_old_runs(tmp_path):
    """舊 run 無 hfss_calls 欄 → 回退 epoch (行為與原樣相同)。"""
    from antenna.utils import config as _config
    _dev = _config.device
    try:
        from script.benchmark_vs_random import run_curve
        xs, best = run_curve(_fake_run_dir(tmp_path, with_calls=False))
        assert xs == [1, 2, 3]
        assert best == [1.0, 1.0, 1.0]
    finally:
        _config.device = _dev
