import json

import numpy as np
import pytest
import torch

from antenna.measurement import frequency_grid, score_response
from script import r81_curve_sm as sm
from script.profiled_batch import pattern_sha256, validate_input
from script.r81_curve_sm import enforce_dual_geometry


@pytest.fixture(scope="module")
def cfg():
    return sm.load_cfg()


@pytest.fixture(scope="module")
def grid(cfg):
    return frequency_grid(cfg.measurement)


def test_grid_is_22_40_37_points(grid):
    assert grid.size == 37 and grid[0] == 22 and grid[-1] == 40


def test_masked_mse_counts_only_supervised_elements():
    pred = torch.zeros(2, 3)
    target = torch.tensor([[1.0, 2.0, 100.0], [3.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    lw = torch.ones(3)
    assert sm.masked_mse(pred, target, mask, lw).item() == pytest.approx((1 + 4 + 9) / 3)
    sw = torch.tensor([1.0, 2.0])
    assert sm.masked_mse(pred, target, mask, lw, sw).item() == pytest.approx((1 + 4 + 2 * 9) / 4)
    lw2 = torch.tensor([0.5, 1.0, 1.0])
    assert sm.masked_mse(pred, target, mask, lw2).item() == pytest.approx((0.5 + 4 + 4.5) / 3)


def test_align_wide_and_history_grids(grid):
    f49 = np.linspace(16, 40, 49)
    y = np.tile(f49, (3, 1))
    out, mask = sm.align(f49, y, grid)
    assert mask.all() and np.allclose(out[1], grid)
    f17 = np.arange(24, 32.25, 0.5)
    out, mask = sm.align(f17, np.tile(f17, (3, 1)), grid)
    covered = (grid >= 24) & (grid <= 32)
    assert (mask[1] == covered).all() and np.allclose(out[1, covered], f17)
    assert (out[:, ~covered] == 0).all()


def test_vectorized_margins_match_score_response(cfg, grid):
    rng = np.random.default_rng(0)
    Y = rng.normal(-10, 8, size=(20, 3, grid.size))
    slices = sm.band_slices(cfg.score_spec, grid)
    vec = sm.margins_vec(Y, slices)
    for y, row in zip(Y, vec):
        ref = score_response(y, sm.LABELS, grid, cfg.score_spec)
        assert np.allclose(row, [ref["margins"][s[0]] for s in slices])
        assert row.min() == pytest.approx(ref["worst_margin"])


def _parents(n=3, family="F"):
    rng = np.random.default_rng(1)
    out = []
    for i in range(n):
        p = enforce_dual_geometry(rng.random((25, 25)) < 0.5)
        out.append({"id": f"p{i}", "pattern": p, "family": family if i < 2 else "G",
                    "source": "wide", "sha": pattern_sha256(p)})
    return out


def test_generate_dedups_against_known_and_itself():
    parents = _parents()
    rng = np.random.default_rng(2)
    P0, _ = sm.generate(parents, set(), 50, rng)
    known = {pattern_sha256(P0[0].reshape(25, 25)), *(p["sha"] for p in parents)}
    P, meta = sm.generate(parents, known, 300, np.random.default_rng(2))
    shas = [m["sha"] for m in meta]
    assert len(set(shas)) == len(shas) == len(P)
    assert not known & set(shas)
    assert all(r[:5, 10:15].all() and r[20:, 10:15].all() for r in P.reshape(-1, 25, 25))
    assert {m["op"] for m in meta} <= {"mutate", "cross", "copy"}


def test_quotas_shift_away_from_exploit_when_quality_bad():
    assert sm.quotas(60, True) == {"exploit": 30, "explore": 18, "random": 12}
    bad = sm.quotas(60, False)
    assert bad["exploit"] == 10 and sum(bad.values()) == 60 and bad["random"] > 12


def test_select_respects_quotas_and_parent_cap():
    parents = _parents()
    P, meta = sm.generate(parents, set(), 400, np.random.default_rng(3))
    rng = np.random.default_rng(4)
    mean = rng.normal(size=len(P))
    std = rng.random(len(P))
    q = sm.quotas(24, True)
    picks = sm.select(P, mean, std, meta, q, rng)
    idx = [i for i, _ in picks]
    assert len(set(idx)) == len(idx) == 24
    for arm, n in q.items():
        assert sum(a == arm for _, a in picks) == n
    for arm in ("exploit", "explore"):
        per = {}
        for i, a in picks:
            if a == arm:
                per[meta[i]["parent"]] = per.get(meta[i]["parent"], 0) + 1
        assert max(per.values()) <= 4
    floor = np.median(mean)
    assert all(mean[i] >= floor for i, a in picks if a == "explore")


def test_written_input_passes_validate_input(tmp_path, cfg):
    parents = _parents()
    P, meta = sm.generate(parents, set(), 30, np.random.default_rng(5))
    n = len(P)
    pred = {"wm_mean": [0.0] * n, "wm_std": [0.1] * n,
            "margins": [{"s21_pass": 0.0}] * n}
    picks = [(i, "random") for i in range(6)]
    out = tmp_path / "r81v2_b01_input"
    rows, sel = sm.write_input(out, cfg, sm.CONFIG.read_text(encoding="utf-8"), P, picks, meta, pred, 1)
    assert validate_input(out, cfg)
    assert rows[0]["id"].startswith("r81v2_b01_000_") and len(sel) == 6
    for f in ("config.yaml", "measurement.json", "score_spec.json", "selection.json"):
        assert (out / f).exists()
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))[0]["selection_arm"] == "random"
    with pytest.raises(SystemExit):
        sm.write_input(out, cfg, "", P, picks, meta, pred, 1)


def test_load_store_and_true_wm_from_fake_store(tmp_path, cfg, grid):
    from antenna.utils.store import fingerprint
    store = tmp_path / "s"
    store.mkdir()
    f49 = np.linspace(16, 40, 49)
    p = torch.tensor(enforce_dual_geometry(np.zeros((25, 25))), dtype=torch.float32)
    y = torch.full((3, 49), -25.0)          # 阻帶比 v2/v3 門檻都深，WM 由通帶 +2 決定
    y[1, (f49 >= 26) & (f49 <= 30)] = -1.0
    name = fingerprint(p, y) + ".pt"
    torch.save((p, y), store / name)
    meas = dict(cfg.measurement, sweep={"start_ghz": 16.0, "stop_ghz": 40.0, "step_ghz": 0.5, "type": "Discrete"})
    (store / "measurement.json").write_text(json.dumps(meas), encoding="utf-8")
    (store / "results.json").write_text(json.dumps({
        "a": {"status": "ok", "sample_file": name, "freqs": f49.tolist(), "lineage_id": "L"},
        "b": {"error": "x"}}), encoding="utf-8")
    assert sm.wide_store_dirs(tmp_path, cfg) == [store]
    rows = sm.load_store(store, grid)
    assert len(rows) == 1 and rows[0]["mask"].all()
    s = sm.true_wm(rows[0], cfg, grid)
    assert s["worst_margin"] == pytest.approx(2.0)


def test_dual_geometry_and_mutation_keep_both_port_pads():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = sm.mutate_pattern(enforce_dual_geometry(rng.random((25, 25)) < 0.5), rng)
        assert p.shape == (25, 25) and p.dtype == bool
        assert p[:5, 10:15].all() and p[20:25, 10:15].all()


def test_holdout_drops_same_family_history(monkeypatch, cfg, grid):
    """回歸（2026-10-13）：holdout 家族在歷史窄頻裡的近親也要排除，不能只排同圖。"""
    rng = np.random.default_rng(0)
    n = len(grid)

    def row(i, fam):
        p = enforce_dual_geometry(rng.random((25, 25)) < 0.5)
        y = np.full((3, n), -10.0, dtype=np.float32)
        y[1] = -5.0 - i
        return {"id": f"w{i}", "store": "s", "pattern": p, "y": y, "mask": np.ones((3, n), bool),
                "sha": pattern_sha256(p), "lineage": fam}

    wide = [row(i, f"F{i % 5}") for i in range(10)]
    hist = {"P": rng.random((15, 625)).astype(np.float32) > 0.5,
            "Y": np.zeros((15, 3, n), np.float32), "M": np.ones((15, 3, n), bool),
            "sha": [f"h{i}" for i in range(15)], "lineage": [f"F{i % 5}" for i in range(15)]}
    seen = {}

    class Fake:
        def predict(self, P):
            return np.full((2, len(P), 3, n), -8.0, np.float32)

    def fake_train(h, rows, **kw):
        seen["hist_lineages"] = set(h["lineage"])
        seen["wide_lineages"] = {r["lineage"] for r in rows}
        return Fake()

    monkeypatch.setattr(sm, "train_ensemble", fake_train)
    rep = sm.holdout_eval(hist, wide, cfg, grid, frac=0.2)
    held = set(rep["holdout_families"])
    assert held and not (held & seen["hist_lineages"])
    assert not (held & seen["wide_lineages"])
    assert rep["n_history_excluded"] == 3 * len(held)


def test_climb_neighbors_are_small_moves_and_best_top_gets_half():
    rng = np.random.default_rng(1)
    tops = [{"id": f"t{k}", "family": "F", "pattern": enforce_dual_geometry(rng.random((25, 25)) < 0.5)}
            for k in range(3)]
    P, meta = sm.climb_candidates(tops, set(), rng, n_double=20)
    for p, m in zip(P[:50], meta[:50]):
        base = tops[int(m["parent"][1])]["pattern"].reshape(-1)
        assert 1 <= int((p.astype(bool) != base).sum()) <= 3
    mean = rng.random(len(P))
    picks = sm.pick_climb(mean, meta, tops, 24)
    n0 = sum(meta[i]["parent"] == "t0" for i, _ in picks)
    assert len(picks) == 24 and n0 == 12 and all(a == "climb" for _, a in picks)


def test_focus_rows_limit_flips_to_selected_rows():
    rng = np.random.default_rng(2)
    allowed = sm.focus_mask("5-9,15-19")
    parent = enforce_dual_geometry(rng.random((25, 25)) < 0.5)
    for _ in range(20):
        child = sm.mutate_pattern(parent, rng, min_flips=1, max_flips=6, allowed=allowed)
        diff = (child != parent).reshape(-1)
        assert not diff[~allowed].any()
    P, _ = sm.climb_candidates([{"id": "t", "family": "F", "pattern": parent}], set(), rng,
                               n_double=10, allowed=allowed)
    assert not ((P.astype(bool) != parent.reshape(-1))[:, ~allowed]).any()


def test_parent_key_violation_prefers_small_total_shortfall():
    one_bad = {"worst_margin": -0.5, "margins": {"a": -0.5, "b": -0.1, "c": +2.0}}    # 總不足 0.6
    all_bad = {"worst_margin": -0.4, "margins": {"a": -0.4, "b": -0.4, "c": -0.4}}    # 總不足 1.2
    assert sm.parent_key(all_bad, "wm") > sm.parent_key(one_bad, "wm")
    assert sm.parent_key(one_bad, "violation") > sm.parent_key(all_bad, "violation")
    assert sm.parent_key(None, "violation") == -1e9


def test_single_climb_top_gets_all_slots():
    rng = np.random.default_rng(3)
    tops = [{"id": "t0", "family": "F", "pattern": enforce_dual_geometry(rng.random((25, 25)) < 0.5)}]
    P, meta = sm.climb_candidates(tops, set(), rng, n_double=5)
    assert len(sm.pick_climb(rng.random(len(P)), meta, tops, 24)) == 24


def test_relaxed_cfg_changes_only_named_thresholds(cfg):
    r = sm.relaxed_cfg(cfg, "s11_match=-8")
    by = {b["name"]: b["threshold"] for b in r.score_spec["bands"]}
    orig = {b["name"]: b["threshold"] for b in cfg.score_spec["bands"]}
    assert by["s11_match"] == -8 and orig["s11_match"] == -10
    assert all(by[k] == orig[k] for k in orig if k != "s11_match")
    assert r.measurement == cfg.measurement
    assert sm.relaxed_cfg(cfg, "") is cfg
    with pytest.raises(SystemExit):
        sm.relaxed_cfg(cfg, "nope=-1")


def test_pads_connected_detects_dc_path_and_cut():
    p = enforce_dual_geometry(np.zeros((25, 25), bool))
    assert not sm.pads_connected(p)
    p[:, 12] = True                                  # 中間一條直通金屬
    assert sm.pads_connected(p)
    p[12, :] = False                                 # 切斷一整列
    assert not sm.pads_connected(p)
    p[12, 13] = True                                 # 與上下兩段只有角接觸，也算相連（保守）
    assert sm.pads_connected(p)


def test_r80_mirror_flip_keeps_symmetry_and_feed():
    from script import r80_sym_climb as sc
    rng = np.random.default_rng(0)
    p = np.zeros((25, 25), bool)
    p[24, 12] = True
    for k in (1, 3, 5):
        q = sc.mirror_flip(p, rng, k)
        assert np.array_equal(q, q[:, ::-1]) and q[24, 12]
    y = np.zeros((1, 2, 17))
    y[0, 0] = -12.0
    y[0, 1] = 5.0
    y[0, 0, 8] = -9.5                                # 28 GHz 的 S11 差 0.5
    assert np.allclose(sc.margins(y), [[-0.5, 1.0]])


def test_load_wide_remeasure_fills_missing_band(tmp_path, cfg):
    """v6 以 11–40 GHz 重量 18–40 設計：同圖去重時要補上第一筆沒量到的頻點（2026-10-15 holdout 空集合）。"""
    from antenna.utils.store import fingerprint
    grid = np.arange(11, 40.01, .5)
    p = torch.tensor(enforce_dual_geometry(np.zeros((25, 25))), dtype=torch.float32)
    for name, lo, val in (("a_first", 18.0, -10.0), ("b_wide", 11.0, -30.0)):
        store = tmp_path / name
        store.mkdir()
        f = np.arange(lo, 40.01, .5)
        y = torch.full((3, f.size), val)
        fn = fingerprint(p, y) + ".pt"
        torch.save((p, y), store / fn)
        meas = dict(cfg.measurement, sweep={"start_ghz": lo, "stop_ghz": 40.0, "step_ghz": 0.5, "type": "Discrete"})
        (store / "measurement.json").write_text(json.dumps(meas), encoding="utf-8")
        (store / "results.json").write_text(json.dumps({
            "x": {"status": "ok", "sample_file": fn, "freqs": f.tolist(), "lineage_id": "L"}}), encoding="utf-8")
    rows = sm.load_wide(tmp_path, cfg, grid)
    assert len(rows) == 1 and rows[0]["mask"].all()
    assert (rows[0]["y"][:, grid >= 18] == -10).all() and (rows[0]["y"][:, grid < 18] == -30).all()


def test_holdout_single_family_splits_by_row(monkeypatch, cfg, grid):
    """回歸（2026-10-16）：v6 真值全是 a0 一家，按家族切會整批進 holdout、train 空、quality_ok 恆 False。"""
    rng = np.random.default_rng(1)
    n = len(grid)
    wide = []
    for i in range(20):
        p = enforce_dual_geometry(rng.random((25, 25)) < 0.5)
        y = np.full((3, n), -10.0, dtype=np.float32)
        y[1] = -5.0 - i
        wide.append({"id": f"w{i}", "store": "s", "pattern": p, "y": y, "mask": np.ones((3, n), bool),
                     "sha": pattern_sha256(p), "lineage": "a0"})
    hist = {"P": np.zeros((0, 625), bool), "Y": np.zeros((0, 3, n), np.float32),
            "M": np.zeros((0, 3, n), bool), "sha": [], "lineage": []}
    seen = {}

    class Fake:
        def predict(self, P):
            return np.broadcast_to(-P.sum(1)[None, :, None, None] / 100, (2, len(P), 3, n)).astype(np.float32)

    def fake_train(h, rows, **kw):
        seen["n_train"] = len(rows)
        return Fake()

    monkeypatch.setattr(sm, "train_ensemble", fake_train)
    rep = sm.holdout_eval(hist, wide, cfg, grid, frac=0.2)
    assert rep["n_holdout"] == 4 and seen["n_train"] == 16 and np.isfinite(rep["wm_mae_const"])


def _two_bodies():
    """上半 0–10 列、下半 12–24 列各一塊實心主體，中間第 11 列整列空。"""
    p = np.zeros((25, 25), bool)
    p[:11, 3:22] = True
    p[12:, 3:22] = True
    return enforce_dual_geometry(p)


def test_push_move_keeps_ports_disconnected_and_moves_boundary():
    rng = np.random.default_rng(0)
    p = _two_bodies()
    moved = 0
    for _ in range(30):
        q = sm.push_move(p, rng, 2)
        assert q is not None and not sm.pads_connected(q)
        assert q[sm.PADS].all()
        top, bot = sm.bodies(q)
        moved += int(top[11:].any() or bot[:11].any())    # 交界離開原本第 11 列
    assert moved > 0


def test_mincut_disconnect_removes_only_the_neck():
    p = _two_bodies()
    p[11, 12] = True                                     # 一格細頸把兩體接起來
    assert sm.pads_connected(p)
    q = sm.mincut_disconnect(p)
    assert not sm.pads_connected(q) and q[sm.PADS].all()
    assert (p & ~q).sum() == 1 and not q[11, 12]


def test_zigzag_slit_always_disconnects_even_diagonally():
    """實心板沿鋸齒切縫挖空後，上下兩埠在 8 連通（角接觸算連）下必斷。"""
    rng = np.random.default_rng(1)
    full = np.ones((25, 25), bool)
    bent = 0
    for _ in range(50):
        s = sm.zigzag_slit(rng, 12, amp=4)
        assert not sm.pads_connected(enforce_dual_geometry(full & ~s))
        bent += int(len(set(np.nonzero(s)[0])) > 1)
    assert bent > 40                                     # 幾乎都不是直線


def test_reseam_moves_straight_seam_and_stays_disconnected():
    rng = np.random.default_rng(2)
    p = _two_bodies()
    moved = 0
    for _ in range(30):
        q = sm.reseam(p, rng, amp=3)
        assert not sm.pads_connected(q) and q[sm.PADS].all()
        moved += int(q[11, 3:22].any())                  # 原本整列空的第 11 列長出金屬
    assert moved > 20


def test_pick_ops_reserves_quota_per_op_with_parent_cap():
    meta = [{"op": op, "parent": par} for op, par in
            [("mutate", "a"), ("push", "a"), ("push", "a"), ("push", "a"), ("push", "b"), ("mincut", "c"), ("mincut", "d")]]
    score = np.array([9, 5, 4, 3, 1, 2, 0], float)
    picks = sm.pick_ops(score, meta, sm.parse_quota("push=3,mincut=1"))
    assert picks == [(1, "push"), (2, "push"), (4, "push"), (5, "mincut")]
