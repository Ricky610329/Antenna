import json
from types import SimpleNamespace

import pytest

from script.batch_scope import job_in_scope, validate_scope, validate_worker_scope


def test_legacy_and_scoped_jobs_are_isolated():
    assert job_in_scope({}, None)
    assert not job_in_scope({"scope": "new"}, None)
    assert not job_in_scope({}, "new")
    assert not job_in_scope({"scope": "other"}, "new")
    assert job_in_scope({"scope": "new"}, "new")


def test_scope_requires_no_self_generation():
    with pytest.raises(ValueError, match="selfgen"):
        validate_worker_scope(SimpleNamespace(scope="new", selfgen=12))
    assert validate_worker_scope(SimpleNamespace(scope="new", selfgen=0)) == "new"
    with pytest.raises(ValueError):
        validate_scope("../escape")


def test_worker_filters_before_stale_claim_fail_and_cleanup(tmp_path, monkeypatch):
    from script import dedust as dd
    from antenna.utils import web
    from antenna.utils.utils import Path
    monkeypatch.setattr(dd, "DATASET_PATH", Path(tmp_path))
    monkeypatch.setattr(web, "get_local_ip", lambda: "192.0.2.37")
    monkeypatch.chdir(tmp_path)
    old_work = tmp_path / "_dedust_old"
    old_work.mkdir()
    (old_work / "keep").write_text("owned by old experiment")
    state = tmp_path / "jobs_state"
    state.mkdir()
    fail = state / "old.fail"
    fail.write_text('{"machines":["192.0.2.216"]}')
    claim = state / "old.claim"
    claim.write_text('{"machine":"192.0.2.37"}')
    (tmp_path / "jobs.json").write_text(json.dumps([
        {"input": "old_input", "store": "old", "prio": 1},
    ]))
    monkeypatch.setattr(dd, "run", lambda _: pytest.fail("unrelated run"))
    monkeypatch.setattr(dd, "_probe_check", lambda *args: pytest.fail("unrelated probe"))
    dd.worker(SimpleNamespace(scope="new", selfgen=0, once=True, poll=1, timeout=900, stale=45))
    assert claim.read_text() == '{"machine":"192.0.2.37"}'
    assert fail.read_text() == '{"machines":["192.0.2.216"]}'
    assert (old_work / "keep").exists()


def test_scoped_job_requires_explicit_config_before_queue_creation(tmp_path, monkeypatch):
    from script import dedust as dd
    from antenna.utils.utils import Path
    monkeypatch.setattr(dd, "DATASET_PATH", Path(tmp_path))
    with pytest.raises(ValueError, match="config"):
        dd.jobs_add(SimpleNamespace(scope="new", input="in", store="out", prio=3))
    assert not (tmp_path / "jobs_state").exists()


def test_scoped_job_and_listing(tmp_path, monkeypatch, capsys):
    from script import dedust as dd
    from antenna.utils.utils import Path
    monkeypatch.setattr(dd, "DATASET_PATH", Path(tmp_path))
    for name in ("old", "new"):
        inp = tmp_path / f"{name}_input"
        inp.mkdir()
        (inp / "manifest.json").write_text("[]")
        dd.jobs_add(SimpleNamespace(scope=None if name == "old" else "symmetry_filter_20261007",
                                   input=f"{name}_input", store=name, prio=3,
                                   config="configs/single_base.yaml"))
    capsys.readouterr()
    dd.jobs_ls(SimpleNamespace(scope="symmetry_filter_20261007", all=True))
    output = capsys.readouterr().out
    assert "new:" in output
    assert "old:" not in output
