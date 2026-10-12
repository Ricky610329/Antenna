import json

import numpy as np
import pytest
import torch

from antenna.measurement import frequency_grid, score_response
from script import r81_curve_sm as sm
from script.exploration import enforce_dual_geometry, pattern_sha256
from script.profiled_batch import validate_input


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
    y = torch.full((3, 49), -20.0)
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
