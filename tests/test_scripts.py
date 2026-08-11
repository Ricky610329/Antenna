"""
tests/test_scripts.py — status.py / analyze.py 的純函式測試（不碰 NAS）。

只測可離線驗的純邏輯（機器名解析、欄位過濾、cosine 基底、中位差）;掃 NAS 的部分靠實跑驗。
"""
import numpy as np

from script.status import _machine, _matches, _num, _liveness
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


def test_liveness_advance_needs_fresh_heartbeat():
    """回歸 (2026-07-05 斷電)：advanced 必須配心跳新鮮才是鐵證——上次快照隔了兩天,
    「epoch 有前進」只證明中間跑過,不證明現在活著（死 31hr 的 run 曾因此被標「在跑」）。"""
    assert _lv(advanced=True, age_min=1) == ("在跑", True)                    # 前進+心跳新 → 鐵證
    assert _lv(advanced=True, age_min=999, state="running")[0] == "疑卡住"    # 前進但心跳古老 → 不算活
    assert _lv(advanced=True, age_min=999, state=None) == ("停止", False)


def test_matches_comma_separated():
    """--match 逗號分隔=任一子字串相符（watchdog 一次盯多個 run）；None=全收。"""
    assert _matches("[m] pixel_single_r5_explore", "single_r5_explore,single_r5_dip_explore")
    assert _matches("[m] pixel_single_r5_dip_explore", "single_r5_explore,single_r5_dip_explore")
    assert not _matches("[m] pixel_single_r4_dip", "single_r5_explore,single_r5_dip_explore")
    assert _matches("anything", None)


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


# ── pattern_anatomy：結構特徵 + variogram 分箱（純函式,不碰 NAS；analysis-01） ────────────


def _feat(p):
    from antenna.utils import config as _config
    _dev = _config.device
    try:
        from script.pattern_anatomy import pattern_features
        return pattern_features(p)
    finally:
        _config.device = _dev


def test_pattern_features_empty_and_full():
    import numpy as np
    f0 = _feat(np.zeros((25, 25)))
    assert f0["n_comp"] == 0 and f0["metal_frac"] == 0.0 and f0["feed_touch"] == 0.0
    f1 = _feat(np.ones((25, 25)))
    assert f1["n_comp"] == 1 and f1["main_frac"] == 1.0 and f1["r_feed"] == 1.0
    assert f1["sym_lr"] == 1.0 and f1["perim_ratio"] == 0.0 and f1["n_holes"] == 0
    assert f1["feed_touch"] == 1.0


def test_pattern_features_two_blocks_and_feed():
    """兩塊分離金屬 (feed 在其中一塊)：n_comp=2、r_feed=feed 塊佔比。「連成一塊算一組」的定義驗證。"""
    import numpy as np
    p = np.zeros((25, 25))
    p[0:5, 0:5] = 1                     # 塊 A：25 px (不含 feed)
    p[20:25, 10:15] = 1                 # 塊 B：25 px,含 feed (24,12)
    f = _feat(p)
    assert f["n_comp"] == 2
    assert f["feed_touch"] == 1.0
    assert abs(f["r_feed"] - 0.5) < 1e-9        # feed 塊 25/50
    assert abs(f["main_frac"] - 0.5) < 1e-9     # 兩塊同大
    assert f["n_holes"] == 0


def test_pattern_features_hole_and_perimeter():
    import numpy as np
    p = np.zeros((25, 25))
    p[10:13, 10:13] = 1
    p[11, 11] = 0                        # 3×3 環,中心挖洞 → 1 個不觸邊的介質組
    f = _feat(p)
    assert f["n_comp"] == 1 and f["n_holes"] == 1
    single = np.zeros((25, 25))
    single[5, 5] = 1                     # 單像素:內部邊界 4 條 / 金屬 1 → perim_ratio=4
    assert abs(_feat(single)["perim_ratio"] - 4.0) < 1e-9


def test_pattern_features_symmetry():
    import numpy as np
    p = np.zeros((25, 25))
    p[3, 5] = 1
    p[3, 19] = 1                         # 5 的鏡像欄 = 24-5 = 19 → 完全對稱
    assert _feat(p)["sym_lr"] == 1.0
    q = np.zeros((25, 25))
    q[0, 0] = 1                          # 鏡像位 (0,24) 是 0 → 兩格不一致
    assert abs(_feat(q)["sym_lr"] - (625 - 2) / 625) < 1e-9


def test_binned_median():
    from antenna.utils import config as _config
    _dev = _config.device
    try:
        from script.pattern_anatomy import binned_median
        med, cnt = binned_median([1, 2, 5, 10], [1.0, 2.0, 3.0, 4.0], [1, 3, 6, 11])
        assert list(cnt) == [2, 1, 1]
        assert med[0] == 1.5 and med[1] == 3.0 and med[2] == 4.0
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


# ── analyze batch 的 dual-port 分支（R57-59 手寫 scratchpad 腳本沉澱,2026-08-11）──────────
#? 合成資料、零 NAS：DATASET_PATH 換 tmp_path、records_dual 換固定 dict（真檔會隨換王變動,
#  拿它當斷言基準＝測試每次破紀錄就紅）。
def _mk_dual_batch(root, store, entries, port="dual"):
    """在 tmp NAS 造一個 dual 店：<store>_input/manifest.json + <store>/results.json。
    entries = [(id, arm, m1, m2, m3, m4, m5, m6, energy)]。"""
    import json
    (root / f"{store}_input").mkdir(parents=True, exist_ok=True)
    (root / store).mkdir(parents=True, exist_ok=True)
    man, res = [], {}
    for pid, arm, m1, m2, m3, m4, m5, m6, en in entries:
        m = {"id": pid, "kind": "dual", "arm": arm}
        if port:
            m["port"] = port
        man.append(m)
        res[pid] = {"wm": [m1, min(m1, m2, m3, m4), m2, min(m1, m2, m3, m4)],
                    "m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5, "m6": m6,
                    "energy_max": en, "time_s": 100.0}
    (root / f"{store}_input" / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    (root / store / "results.json").write_text(json.dumps(res), encoding="utf-8")


_REC_DUAL = {"updated": "TEST", "buffer": None,
             "wm_dual": {"id": "old_wm", "value": -6.04},
             "m3_pass_s21": {"id": None, "value": None},
             "m4_stop_s21": {"id": "old_m4", "value": 5.92}}


def test_wm_dual_ruler_matches_losses():
    """尺一致（硬規範）：analyze 的 `_wm_dual`（吃 results.json 已存分項）必須與
    `antenna.losses.worst_margin_dual`（吃曲線）給同一個數——兩把尺一旦分家,判讀就在說謊。"""
    import torch
    from antenna.losses import worst_margin_dual
    from antenna.training import PORT_SPECS, load_config
    from script.analyze import _wm_dual
    from pathlib import Path as _P
    targets = load_config(_P(__file__).resolve().parents[1] / "configs" / "dual_base.yaml").targets
    labels = PORT_SPECS["dual"]["labels"]
    rng = np.random.default_rng(57)
    for _ in range(20):
        resp = torch.tensor(rng.uniform(-30, 0, size=(3, 17)), dtype=torch.float32)
        wm, per = worst_margin_dual(resp, labels, targets)
        entry = {k: float(per[k]) for k in ("m1", "m2", "m3", "m4", "m5", "m6")}
        assert abs(_wm_dual(entry) - float(wm)) < 1e-4      # m5/m6 在 entry 裡也不得混進 min


def test_wm_dual_returns_none_on_incomplete():
    """分項不全＝不猜（回 None,該筆跳過）——寧可少一筆也不要用半套尺算出假數字。"""
    from script.analyze import _wm_dual
    assert _wm_dual({"m1": -1.0, "m2": -2.0, "m3": -3.0}) is None


def test_med_lo_is_lower_median_desc():
    """中位口徑＝降序下中位（round 檔 §4 歷史數字用這把;np.median 會插值→對不上）。"""
    from script.analyze import _med_lo
    assert _med_lo([-7.0, -8.0, -9.0, -10.0]) == -9.0        # np.median 會給 -8.5
    assert _med_lo([-7.0, -8.0, -9.0]) == -8.0


def test_batch_dispatch_single_when_no_port_key(tmp_path, monkeypatch):
    """port 分派：manifest 無 `port` 鍵（single 歷史夾）→ 必須走 single 分支,一 byte 不變。"""
    import script.analyze as az
    from types import SimpleNamespace
    monkeypatch.setattr(az, "DATASET_PATH", tmp_path)
    _mk_dual_batch(tmp_path, "dedust_r98b1a", [("s98b1_x", "x", -1, -2, -3, -4, -5, -6, 0.9)],
                   port=None)
    called = {}
    monkeypatch.setattr(az, "_batch_single", lambda a: called.setdefault("single", True))
    monkeypatch.setattr(az, "_batch_dual", lambda a: called.setdefault("dual", True))
    az.cmd_batch(SimpleNamespace(round=98, batch=1))
    assert called == {"single": True}


def test_batch_dispatch_dual_and_report(tmp_path, monkeypatch, capsys):
    """dual 分支端到端（三店聚合/臂別表/sel 前緣/紀錄候選+公證指令/m5m6 觸發/→行動）。"""
    import script.analyze as az
    from types import SimpleNamespace
    monkeypatch.setattr(az, "DATASET_PATH", tmp_path)
    monkeypatch.setattr(az, "_records_dual", lambda: _REC_DUAL)
    #  a 臂 4 筆（best -5.00 > 紀錄 -6.04 → wm_dual 候選;降序下中位 = 第 3 高 = -8.00）
    _mk_dual_batch(tmp_path, "dedust_r99b1a", [
        ("d99b1_a_00", "a", -5.0, -6.0, -7.0, -8.0, -20.0, -21.0, 0.88),   # wm -8.00
        ("d99b1_a_01", "a", -5.0, -5.0, -5.0, -5.0, -20.0, -21.0, 0.88),   # wm -5.00 ← best
    ])
    _mk_dual_batch(tmp_path, "dedust_r99b1b", [
        ("d99b1_a_02", "a", -9.0, -9.0, -9.0, -9.0, -20.0, -21.0, 0.88),   # wm -9.00
        ("d99b1_a_03", "a", -6.0, -6.0, -6.0, -6.0, -20.0, -21.0, 0.88),   # wm -6.00
    ])
    #  b 臂:m4 破紀錄（+7.0 > 5.92）+ sel 前緣頭名（sel=min(m3,m4)=-1.0）
    _mk_dual_batch(tmp_path, "dedust_r99b1c", [
        ("d99b1_b_00", "b", -30.0, -30.0, -1.0, 7.0, -20.0, -21.0, 0.90),
        ("d99b1_b_01", "b", -30.0, -30.0, -2.0, 3.0, -20.0, -21.0, 1.004),  # energy 超標
    ])
    az.cmd_batch(SimpleNamespace(round=99, batch=1))
    out = capsys.readouterr().out
    assert "**dual** 收檔判讀（3 夾 6 筆" in out                      # 三店自動聚合
    assert "| a | 4 | -5.00（d99b1_a_01） | -8.00 |" in out          # 臂別:n/best/降序下中位
    assert "energy_max≤1: 5/6 通過" in out and "⚠ 超標 1 筆" in out   # 能量自證
    assert "sel -1.00  d99b1_b_00 [b]" in out                        # sel=min(m3,m4) 前緣
    #  紀錄候選:wm_dual 與 m4 各一件;m3 現任 None → 只列 ○ 行、不印公證指令
    assert "★ d99b1_a_01 [a] wm_dual -5.00 > 現任 -6.04" in out
    assert "★ d99b1_b_00 [b] m4(S21 阻帶) +7.00 > 現任 +5.92" in out
    assert ("select-repeat --source-input dedust_r99b1a_input --id d99b1_a_01 --n 2"
            " --input dedust_r99n1_input") in out
    assert "--store dedust_r99n2 --prio 2" in out                    # 第二件另開 n2,不共用
    assert "○ m3(S21 通帶) 未開帳" in out and "不自動發車" in out
    assert "同機 3/3 bit 級一致" in out                               # 公證口徑提醒
    assert "① 公證候選 2 件" in out and "③ m5/m6 重議: 未觸發" in out


def test_batch_dual_m5m6_trigger_and_incomplete(tmp_path, monkeypatch, capsys):
    """m5/m6 任一樣本 ≥0 → 顯性觸發重議；未收全（error/待跑）→ 顯性警告不靜默。"""
    import json
    import script.analyze as az
    from types import SimpleNamespace
    monkeypatch.setattr(az, "DATASET_PATH", tmp_path)
    monkeypatch.setattr(az, "_records_dual", lambda: _REC_DUAL)
    _mk_dual_batch(tmp_path, "dedust_r99b2a", [
        ("d99b2_a_00", "a", -5.0, -5.0, -5.0, -5.0, +0.3, -21.0, 0.88),   # m5 ≥0 → 觸發
        ("d99b2_a_01", "a", -6.0, -6.0, -6.0, -6.0, -20.0, -21.0, 0.88),
    ])
    res = json.loads((tmp_path / "dedust_r99b2a" / "results.json").read_text(encoding="utf-8"))
    res["d99b2_a_01"] = {"error": "COM 例外", "attempts": 1}              # 一筆 error
    (tmp_path / "dedust_r99b2a" / "results.json").write_text(json.dumps(res), encoding="utf-8")
    az.cmd_batch(SimpleNamespace(round=99, batch=2))
    out = capsys.readouterr().out
    assert "⚠ 未收全 1/2（error 1）" in out
    assert "⚠ **觸發重議**: m5 1 筆 ≥0" in out
    assert "③ m5/m6 重議: ⚠ 觸發" in out


def test_batch_dual_incumbent_is_not_its_own_candidate(tmp_path, monkeypatch, capsys):
    """回歸（2026-08-11 r58b3 實犯）：紀錄以**跨機保守值**入帳時,現任本人在原批的單次值仍高於
    帳面值 → 重跑該批判讀會提示「破自己」,照抄就白燒 2×HFSS 重新公證在位王。
    現任 id 不得進候選;同批第二名若真的超過帳面值,仍要被抓出來。"""
    import script.analyze as az
    from types import SimpleNamespace
    monkeypatch.setattr(az, "DATASET_PATH", tmp_path)
    monkeypatch.setattr(az, "_records_dual", lambda: dict(
        _REC_DUAL, m4_stop_s21={"id": "d99b3_x_king", "value": 5.92}))
    _mk_dual_batch(tmp_path, "dedust_r99b3a", [
        ("d99b3_x_king", "x", -30.0, -30.0, -5.0, 6.67, -20.0, -21.0, 0.9),   # 現任本人（跨機保守 5.92）
        ("d99b3_x_two", "x", -30.0, -30.0, -5.0, 6.10, -20.0, -21.0, 0.9),    # 第二名,真的超過帳面
        ("d99b3_x_low", "x", -30.0, -30.0, -5.0, 1.00, -20.0, -21.0, 0.9),
    ])
    az.cmd_batch(SimpleNamespace(round=99, batch=3))
    out = capsys.readouterr().out
    assert "★ d99b3_x_king" not in out                        # 現任不得自己公證自己
    assert "★ d99b3_x_two [x] m4(S21 阻帶) +6.10 > 現任 +5.92" in out
    assert "另有" not in out.split("-- 紀錄候選")[1].split("--")[0]   # 扣掉現任後只剩 1 筆
    assert "m4(S21 阻帶) +6.67（紀錄 +5.92" in out              # ②前緣仍報原始批 best（不隱藏）
