import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from antenna.measurement import measurement_id, score_spec_id
from antenna.utils.store import fingerprint
from script import profiled_batch as pb
from script.profiled_shards import merge_stores, snapshot_successes, split_bundle


REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "single_r80_symmetry_explore.yaml"


def _canonical(tmp_path, count=48):
    root = tmp_path / "canonical"
    root.mkdir()
    shutil.copy2(CONFIG, root / "config.yaml")
    cfg = pb.load_profile_config(root / "config.yaml")
    pb.atomic_json(root / "measurement.json", cfg.measurement)
    pb.atomic_json(root / "score_spec.json", cfg.score_spec)
    mid, sid = measurement_id(cfg.measurement), score_spec_id(cfg.score_spec)
    rows = []
    for i in range(count):
        pattern = np.zeros((25, 25), dtype=np.float32)
        pattern[24, 12] = 1
        for bit in range(6):
            if i & (1 << bit):
                pattern[bit * 3, bit] = 1
                pattern[bit * 3, 24 - bit] = 1
        name = f"p{i:03d}"
        torch.save(torch.as_tensor(pattern), root / f"{name}.pt")
        rows.append({"id": name, "pattern_file": f"{name}.pt",
                     "pattern_sha256": pb.pattern_sha256(pattern),
                     "port": "single", "measurement_id": mid, "score_spec_id": sid,
                     "geometry_profile": "single25_lr_exact_db100",
                     "selection_arm": ("guided" if i % 2 == 0 else "control"),
                     "prediction": [float(i), float(i) + 0.25],
                     "lineage_id": f"line-{i // 2}"})
    pb.atomic_json(root / "manifest.json", rows)
    pb.validate_input(root, cfg)
    return root


def _complete(input_dir, store_dir):
    cfg = pb.load_profile_config(input_dir / "config.yaml")
    rows = pb.prepare_store(input_dir, store_dir, cfg)
    results = {}
    (store_dir / "rad").mkdir(exist_ok=True)
    theta = torch.arange(-180, 181, 2, dtype=torch.float32)
    for ordinal, row in enumerate(rows):
        pattern = torch.load(input_dir / f"{row['id']}.pt", weights_only=True)
        response = torch.stack((torch.linspace(-15, -5, 17) + ordinal,
                                torch.linspace(0, 5, 17) - ordinal))
        rad = {"theta": theta, "phi0": theta * 0 + ordinal,
               "phi90": theta * 0 - ordinal}
        entry = pb.observation(response, rad, row, pattern, cfg, elapsed=1.0 + ordinal)
        sample = store_dir / entry["sample_file"]
        torch.save((pattern, response), sample)
        rad_path = store_dir / entry["rad_file"]
        torch.save(rad, rad_path)
        assert entry["sample_file"] == fingerprint(pattern, response) + ".pt"
        entry["sample_sha256"] = pb.file_sha256(sample)
        entry["rad_sha256"] = pb.file_sha256(rad_path)
        results[row["id"]] = entry
    pb.atomic_json(store_dir / "results.json", results)
    pb.validate_store(store_dir, require_complete=True)


def test_split_bundle_round_robin_48_into_three_exact_inputs(tmp_path):
    canonical = _canonical(tmp_path)
    rows = pb.read_json(canonical / "manifest.json")
    shards = split_bundle(canonical, tmp_path / "split")

    assert [len(pb.read_json(path / "manifest.json")) for path in shards] == [16, 16, 16]
    assert [[row["id"] for row in pb.read_json(path / "manifest.json")] for path in shards] == [
        [row["id"] for row in rows[offset::3]] for offset in range(3)]
    for path in shards:
        child_rows = pb.read_json(path / "manifest.json")
        assert all(child == rows[int(child["id"][1:])] for child in child_rows)
        pb.validate_input(path, pb.load_profile_config(path / "config.yaml"))
    receipt = pb.read_json(tmp_path / "split" / "split_receipt.json")
    assert receipt["round_robin"] is True and receipt["shard_size"] == 16
    assert receipt["canonical_manifest_sha256"] == pb.file_sha256(canonical / "manifest.json")
    with pytest.raises(FileExistsError):
        split_bundle(canonical, tmp_path / "split")


def test_split_bundle_32_into_two_round_robin_shards(tmp_path):
    canonical = _canonical(tmp_path, count=32)
    rows = pb.read_json(canonical / "manifest.json")

    shards = split_bundle(canonical, tmp_path / "split")

    assert [len(pb.read_json(path / "manifest.json")) for path in shards] == [16, 16]
    assert [[row["id"] for row in pb.read_json(path / "manifest.json")] for path in shards] == [
        [row["id"] for row in rows[offset::2]] for offset in range(2)]


def test_merge_stores_preserves_rows_predictions_artifacts_and_bindings(tmp_path):
    canonical = _canonical(tmp_path, count=6)
    shards = split_bundle(canonical, tmp_path / "split", shard_size=2)
    stores = []
    source_results = {}
    for number, shard in enumerate(shards):
        store = tmp_path / f"store-{number}"
        _complete(shard, store)
        stores.append(store)
        source_results.update(pb.read_json(store / "results.json"))

    merged = merge_stores(canonical, stores, tmp_path / "merged")
    assert pb.read_json(merged / "manifest.json") == pb.read_json(canonical / "manifest.json")
    assert pb.read_json(merged / "results.json") == {
        row["id"]: source_results[row["id"]]
        for row in pb.read_json(canonical / "manifest.json")}
    assert len(pb.validate_store(merged, require_complete=True)) == 6
    bindings = pb.read_json(merged / "source_bindings.json")
    assert list(bindings["items"]) == [f"p{i:03d}" for i in range(6)]
    assert all(Path(item["source_store"]).is_absolute() for item in bindings["items"].values())
    for name, entry in source_results.items():
        binding = bindings["items"][name]
        assert binding["source_sample_sha256"] == entry["sample_sha256"]
        assert pb.file_sha256(merged / entry["sample_file"]) == entry["sample_sha256"]
        assert binding["source_rad_sha256"] == entry["rad_sha256"]
        assert pb.file_sha256(merged / entry["rad_file"]) == entry["rad_sha256"]
    with pytest.raises(FileExistsError):
        merge_stores(canonical, stores, merged)


@pytest.mark.parametrize("mode", ["missing", "overlap", "row-drift"])
def test_merge_rejects_non_exact_child_partition(tmp_path, mode):
    canonical = _canonical(tmp_path, count=4)
    shards = split_bundle(canonical, tmp_path / "split", shard_size=2)
    stores = []
    for number, shard in enumerate(shards):
        store = tmp_path / f"store-{number}"
        _complete(shard, store)
        stores.append(store)

    if mode == "missing":
        stores = stores[:1]
    elif mode == "overlap":
        stores.append(stores[0])
    else:
        rows = pb.read_json(stores[0] / "manifest.json")
        rows[0]["prediction"] = [999.0]
        pb.atomic_json(stores[0] / "manifest.json", rows)

    with pytest.raises((ValueError, FileExistsError), match="canonical union|duplicates|differs"):
        merge_stores(canonical, stores, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_merge_rejects_incomplete_or_tampered_child_before_output(tmp_path):
    canonical = _canonical(tmp_path, count=2)
    shards = split_bundle(canonical, tmp_path / "split", shard_size=1)
    stores = []
    for number, shard in enumerate(shards):
        store = tmp_path / f"store-{number}"
        _complete(shard, store)
        stores.append(store)
    results = pb.read_json(stores[0] / "results.json")
    first = next(iter(results))
    results[first] = {"error": "RPC failure"}
    pb.atomic_json(stores[0] / "results.json", results)

    with pytest.raises(ValueError):
        merge_stores(canonical, stores, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_merge_rejects_distinct_stores_with_overlapping_ids(tmp_path):
    canonical = _canonical(tmp_path, count=4)
    shards = split_bundle(canonical, tmp_path / "split", shard_size=2)
    first = tmp_path / "store-first"
    duplicate = tmp_path / "store-duplicate"
    _complete(shards[0], first)
    shutil.copytree(first, duplicate)

    with pytest.raises(ValueError, match="overlap"):
        merge_stores(canonical, [first, duplicate], tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_merge_rejects_completed_store_with_extra_id(tmp_path):
    canonical = _canonical(tmp_path, count=4)
    shards = split_bundle(canonical, tmp_path / "split", shard_size=2)
    rows = pb.read_json(shards[0] / "manifest.json")
    extra = dict(rows[0], id="p-extra", pattern_file="p-extra.pt")
    shutil.copy2(shards[0] / rows[0]["pattern_file"], shards[0] / extra["pattern_file"])
    pb.atomic_json(shards[0] / "manifest.json", rows + [extra])
    stores = []
    for number, shard in enumerate(shards):
        store = tmp_path / f"store-{number}"
        _complete(shard, store)
        stores.append(store)

    with pytest.raises(ValueError, match="outside canonical"):
        merge_stores(canonical, stores, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_merge_rejects_wrong_measurement_store(tmp_path):
    canonical = _canonical(tmp_path, count=2)
    shards = split_bundle(canonical, tmp_path / "split", shard_size=1)
    stores = []
    for number, shard in enumerate(shards):
        store = tmp_path / f"store-{number}"
        _complete(shard, store)
        stores.append(store)
    measurement = pb.read_json(stores[0] / "measurement.json")
    measurement["geometry"]["diag_bridge_w"] += 0.1
    pb.atomic_json(stores[0] / "measurement.json", measurement)

    with pytest.raises(ValueError):
        merge_stores(canonical, stores, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_merge_rejects_store_metadata_change_during_validation(tmp_path, monkeypatch):
    canonical = _canonical(tmp_path, count=2)
    shards = split_bundle(canonical, tmp_path / "split", shard_size=1)
    stores = []
    for number, shard in enumerate(shards):
        store = tmp_path / f"store-{number}"
        _complete(shard, store)
        stores.append(store)
    original = pb.validate_store
    changed = False

    def validate_then_change(store_dir, *, require_complete=True):
        nonlocal changed
        result = original(store_dir, require_complete=require_complete)
        if Path(store_dir).resolve() == stores[0].resolve() and not changed:
            path = stores[0] / "results.json"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            changed = True
        return result

    monkeypatch.setattr(pb, "validate_store", validate_then_change)
    with pytest.raises(ValueError, match="metadata changed"):
        merge_stores(canonical, stores, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_merge_rejects_child_artifact_hash_failure_before_output(tmp_path):
    canonical = _canonical(tmp_path, count=2)
    shards = split_bundle(canonical, tmp_path / "split", shard_size=1)
    stores = []
    for number, shard in enumerate(shards):
        store = tmp_path / f"store-{number}"
        _complete(shard, store)
        stores.append(store)
    result = next(iter(pb.read_json(stores[0] / "results.json").values()))
    (stores[0] / result["sample_file"]).write_bytes(b"tampered")

    with pytest.raises(ValueError):
        merge_stores(canonical, stores, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_snapshot_successes_copies_only_current_valid_results(tmp_path):
    input_dir = _canonical(tmp_path, count=4)
    store = tmp_path / "store"
    _complete(input_dir, store)
    results = pb.read_json(store / "results.json")
    results["p001"] = {"error": "RPC failure"}
    del results["p002"]
    pb.atomic_json(store / "results.json", results)
    source_results_sha = pb.file_sha256(store / "results.json")

    snapshot = snapshot_successes([(input_dir, store)], tmp_path / "snapshot")

    assert [row["id"] for row in pb.read_json(snapshot / "manifest.json")] == ["p000", "p003"]
    assert list(pb.read_json(snapshot / "results.json")) == ["p000", "p003"]
    assert len(pb.validate_store(snapshot, require_complete=True)) == 2
    receipt = pb.read_json(snapshot / "source_bindings.json")
    assert receipt["partial_snapshot"] is True
    assert receipt["source_batches_may_be_incomplete"] is True
    assert receipt["counts"]["unique_selected_rows"] == 2
    assert receipt["sources"][0]["partial_source_results_sha256"] == source_results_sha
    assert pb.file_sha256(store / "results.json") == source_results_sha


def test_snapshot_successes_deduplicates_prefers_nonrepeat_and_excludes(tmp_path):
    canonical = _canonical(tmp_path, count=3)
    shards = split_bundle(canonical, tmp_path / "split", shard_size=1)
    first_rows = pb.read_json(shards[0] / "manifest.json")
    first_rows[0]["kind"] = "repeat"
    pb.atomic_json(shards[0] / "manifest.json", first_rows)
    second_rows = pb.read_json(shards[1] / "manifest.json")
    shutil.copy2(shards[0] / "p000.pt", shards[1] / "p001.pt")
    second_rows[0]["pattern_sha256"] = first_rows[0]["pattern_sha256"]
    pb.atomic_json(shards[1] / "manifest.json", second_rows)
    excluded = pb.read_json(shards[2] / "manifest.json")[0]["pattern_sha256"]
    pairs = []
    for number, shard in enumerate(shards):
        store = tmp_path / f"store-{number}"
        _complete(shard, store)
        pairs.append((shard, store))

    snapshot = snapshot_successes(pairs, tmp_path / "snapshot", [excluded])

    assert [row["id"] for row in pb.read_json(snapshot / "manifest.json")] == ["p001"]
    receipt = pb.read_json(snapshot / "source_bindings.json")
    assert receipt["counts"] == {
        "successful_source_rows": 3,
        "excluded_successful_rows": 1,
        "duplicate_successful_rows": 1,
        "unique_selected_rows": 1,
    }
    assert receipt["items"]["p001"]["source_store"] == str(pairs[1][1].resolve())


def test_snapshot_successes_rejects_input_store_manifest_drift(tmp_path):
    input_dir = _canonical(tmp_path, count=2)
    store = tmp_path / "store"
    _complete(input_dir, store)
    rows = pb.read_json(store / "manifest.json")
    rows[0]["prediction"] = [999.0]
    pb.atomic_json(store / "manifest.json", rows)

    with pytest.raises(ValueError, match="manifest differs"):
        snapshot_successes([(input_dir, store)], tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_snapshot_successes_rejects_tampered_artifact(tmp_path):
    input_dir = _canonical(tmp_path, count=2)
    store = tmp_path / "store"
    _complete(input_dir, store)
    entry = next(iter(pb.read_json(store / "results.json").values()))
    (store / entry["sample_file"]).write_bytes(b"tampered")

    with pytest.raises(ValueError):
        snapshot_successes([(input_dir, store)], tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_snapshot_successes_rejects_metadata_hash_race(tmp_path, monkeypatch):
    input_dir = _canonical(tmp_path, count=2)
    store = tmp_path / "store"
    _complete(input_dir, store)
    original = pb.validate_store
    changed = False

    def validate_then_change(store_dir, *, require_complete=True):
        nonlocal changed
        result = original(store_dir, require_complete=require_complete)
        if Path(store_dir).resolve() == store.resolve() and not changed:
            path = store / "results.json"
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            changed = True
        return result

    monkeypatch.setattr(pb, "validate_store", validate_then_change)
    with pytest.raises(ValueError, match="metadata changed"):
        snapshot_successes([(input_dir, store)], tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()
