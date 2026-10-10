from __future__ import annotations

import copy
from pathlib import Path

import pytest

from script import exploration as ex
from script import filter_training
from script import r81_prior_recipe as recipe


def test_builder_changes_only_masked_prior_training_fields(tmp_path):
    base = recipe.load_base_config()
    frozen = copy.deepcopy(base)
    manifest = (tmp_path / "manifest.json").resolve()
    sample_root = (tmp_path / "raw_snapshot").resolve()

    built = recipe.build_recipe(manifest, sample_root, base_config=base)

    assert base == frozen
    for key in ("measurement", "score_spec", "runtime"):
        assert built[key] == frozen[key]
    changed = {
        "training_protocol",
        "pretrain_epochs",
        "epochs",
        "holdout_fraction",
        "filter_prior",
    }
    for key, value in frozen["exploration"].items():
        if key not in changed:
            assert built["exploration"][key] == value
    assert built["exploration"]["training_protocol"] == "filter_masked_prior_v1"
    assert built["exploration"]["pretrain_epochs"] == 30
    assert built["exploration"]["epochs"] == 100
    assert built["exploration"]["holdout_fraction"] == 0.2
    assert built["exploration"]["batch_size"] == 60
    assert built["exploration"]["max_batches"] == 3
    assert built["exploration"]["ensemble_seeds"] == [0, 1]
    assert built["exploration"]["arms"] == {
        "history": 20, "specialist": 20, "random": 20
    }
    assert built["exploration"]["later_arms"] == {
        "best_min_margin": 20, "uncertainty": 20, "random": 20
    }
    assert ex.validate_exploration_config(built) == built
    spec = filter_training._prior_spec(built)
    assert spec["manifest"] == manifest
    assert spec["sample_root"] == sample_root
    assert spec["expected_manifest_sha256"] == recipe.EXPECTED_PRIOR_MANIFEST_SHA256


@pytest.mark.parametrize("which", ["manifest", "sample_root"])
def test_builder_rejects_relative_prior_paths(tmp_path, which):
    manifest = (tmp_path / "manifest.json").resolve()
    sample_root = (tmp_path / "raw_snapshot").resolve()
    if which == "manifest":
        manifest = Path("manifest.json")
    else:
        sample_root = Path("raw_snapshot")
    with pytest.raises(ValueError, match="absolute paths"):
        recipe.build_recipe(manifest, sample_root)


def test_artifact_gate_rejects_wrong_manifest_without_loading_samples(tmp_path):
    manifest = (tmp_path / "manifest.json").resolve()
    sample_root = (tmp_path / "raw_snapshot").resolve()
    manifest.write_text("{}\n", encoding="utf-8")
    sample_root.mkdir()
    built = recipe.build_recipe(manifest, sample_root)
    with pytest.raises(ValueError, match="differs from the frozen R81 prior"):
        recipe.validate_artifact_location(built)


def test_parser_has_only_caller_paths_and_output():
    parser = recipe.make_parser()
    actions = {action.dest for action in parser._actions}
    assert actions == {"help", "prior_manifest", "prior_sample_root", "output"}
    assert recipe.EXPECTED_PRIOR_MANIFEST_SHA256 == (
        "8fcd05614e7851e9c0f7043c00540068df37380423ee4469db0afed90505c777"
    )


def test_cli_exclusive_create_preserves_racing_output(tmp_path, monkeypatch):
    manifest = (tmp_path / "manifest.json").resolve()
    sample_root = (tmp_path / "raw_snapshot").resolve()
    output = (tmp_path / "race" / "recipe.yaml").resolve()
    sentinel = b"concurrent-owner\n"
    original_mkdir = Path.mkdir

    monkeypatch.setattr(recipe, "validate_artifact_location", lambda _config: None)

    def create_concurrent_output(path, *args, **kwargs):
        original_mkdir(path, *args, **kwargs)
        if path == output.parent and not output.exists():
            output.write_bytes(sentinel)

    monkeypatch.setattr(Path, "mkdir", create_concurrent_output)
    with pytest.raises(FileExistsError):
        recipe.main([
            "--prior-manifest", str(manifest),
            "--prior-sample-root", str(sample_root),
            "--output", str(output),
        ])
    assert output.read_bytes() == sentinel
