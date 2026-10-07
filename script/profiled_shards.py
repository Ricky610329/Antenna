"""Split profiled inputs and merge completed child stores without touching HFSS.

The canonical input remains the source of IDs, manifest rows, predictions, and
measurement identity.  Child stores are only execution shards; their completed
artifacts are copied into a fresh offline view after strict replay validation.
"""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

from antenna.measurement import measurement_id, score_spec_id
from script import profiled_batch as pb


_IDENTITY_FILES = ("measurement.json", "score_spec.json", "config.yaml")
_INPUT_METADATA_FILES = (*_IDENTITY_FILES, "manifest.json")
_STORE_METADATA_FILES = ("measurement.json", "score_spec.json", "manifest.json", "results.json")


def _fresh_directory(target: Path) -> Path:
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))


def _finish_directory(temp: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    temp.replace(target)


def _cleanup_temp(temp: Path, target: Path) -> None:
    """Remove only the temporary sibling created for ``target``."""

    resolved = temp.resolve()
    expected_parent = target.resolve().parent
    expected_prefix = f".{target.name}.tmp-"
    if resolved.parent != expected_parent or not resolved.name.startswith(expected_prefix):
        raise RuntimeError(f"refusing unsafe temporary-directory cleanup: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _metadata_hashes(root: Path, names: Iterable[str]) -> dict[str, str]:
    return {name: pb.file_sha256(root / name) for name in names}


def _require_metadata_unchanged(root: Path, expected: dict[str, str], label: str) -> None:
    actual = _metadata_hashes(root, expected)
    if actual != expected:
        changed = sorted(name for name in expected if actual.get(name) != expected[name])
        raise ValueError(f"{label} metadata changed during operation: {changed}")


def _copy_file(source: Path, target: Path, *, expected_sha256: str | None = None) -> str:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    actual = pb.file_sha256(source)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"source artifact hash mismatch: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if pb.file_sha256(target) != actual:
            raise ValueError(f"artifact filename collision with different content: {target.name}")
    else:
        shutil.copy2(source, target)
    if pb.file_sha256(target) != actual:
        raise ValueError(f"copied artifact hash mismatch: {target}")
    return actual


def _relative_artifact(root: Path, value: str, *, nested: bool) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact path escapes store: {value}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"artifact path escapes store: {value}") from exc
    if not nested and resolved.parent != root.resolve():
        raise ValueError(f"sample artifact must be directly under store: {value}")
    return resolved


def split_bundle(input_dir, output_root, shard_size=16) -> list[Path]:
    """Split one canonical profiled input into fresh balanced round-robin shards.

    ``shard_size`` is the maximum shard size.  For the standard 48-item batch,
    the default produces three 16-item stores.  Manifest rows and pattern bytes
    are copied unchanged.
    """

    canonical = Path(input_dir).resolve()
    output = Path(output_root).resolve()
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size <= 0:
        raise ValueError("shard_size must be a positive integer")
    canonical_metadata = _metadata_hashes(canonical, _INPUT_METADATA_FILES)
    cfg = pb.load_profile_config(canonical / "config.yaml")
    if cfg is None:
        raise ValueError("canonical input must contain a profiled config.yaml")
    rows = pb.validate_input(canonical, cfg)
    shard_count = math.ceil(len(rows) / shard_size)
    assignments = [rows[index::shard_count] for index in range(shard_count)]
    if any(len(group) > shard_size for group in assignments):
        raise AssertionError("internal shard sizing failure")

    temp = _fresh_directory(output)
    try:
        receipt_shards = []
        for number, group in enumerate(assignments, start=1):
            name = f"shard-{number:03d}_input"
            child = temp / name
            child.mkdir()
            for marker in _IDENTITY_FILES:
                _copy_file(canonical / marker, child / marker)
            item_bindings = []
            for row in group:
                pattern_name = row.get("pattern_file", f"{row['id']}.pt")
                if pattern_name != f"{row['id']}.pt":
                    raise ValueError(f"profile pattern_file must equal id.pt: {row['id']}")
                source = canonical / pattern_name
                digest = _copy_file(source, child / pattern_name)
                item_bindings.append({"id": row["id"], "pattern_file": pattern_name,
                                      "file_sha256": digest,
                                      "pattern_sha256": row["pattern_sha256"]})
            pb.atomic_json(child / "manifest.json", group)
            pb.validate_input(child, cfg)
            receipt_shards.append({"name": name, "ids": [row["id"] for row in group],
                                   "items": item_bindings,
                                   "manifest_sha256": pb.file_sha256(child / "manifest.json")})

        receipt = {
            "schema_version": 1,
            "canonical_input": str(canonical),
            "canonical_manifest_sha256": pb.file_sha256(canonical / "manifest.json"),
            "canonical_metadata_sha256": canonical_metadata,
            "config_sha256": pb.file_sha256(canonical / "config.yaml"),
            "measurement_id": measurement_id(pb.read_json(canonical / "measurement.json")),
            "score_spec_id": score_spec_id(pb.read_json(canonical / "score_spec.json")),
            "round_robin": True,
            "shard_size": shard_size,
            "shards": receipt_shards,
        }
        pb.atomic_json(temp / "split_receipt.json", receipt)
        _require_metadata_unchanged(canonical, canonical_metadata, "canonical input")
        _finish_directory(temp, output)
    except Exception:
        _cleanup_temp(temp, output)
        raise
    return [output / entry["name"] for entry in receipt_shards]


def merge_stores(canonical_input, store_dirs: Iterable, output_dir) -> Path:
    """Validate completed shards and create a fresh immutable offline store view."""

    canonical = Path(canonical_input).resolve()
    output = Path(output_dir).resolve()
    stores = [Path(path).resolve() for path in store_dirs]
    if not stores:
        raise ValueError("at least one completed child store is required")
    if len(set(stores)) != len(stores):
        raise ValueError("child store list contains duplicates")

    canonical_metadata = _metadata_hashes(canonical, _INPUT_METADATA_FILES)
    cfg = pb.load_profile_config(canonical / "config.yaml")
    if cfg is None:
        raise ValueError("canonical input must contain a profiled config.yaml")
    canonical_rows = pb.validate_input(canonical, cfg)
    canonical_ids = [row["id"] for row in canonical_rows]
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("canonical manifest contains duplicate ids")
    canonical_by_id = {row["id"]: row for row in canonical_rows}
    canonical_measurement = pb.read_json(canonical / "measurement.json")
    canonical_score = pb.read_json(canonical / "score_spec.json")
    canonical_mid = measurement_id(canonical_measurement)
    canonical_sid = score_spec_id(canonical_score)

    merged_results = {}
    bindings = {}
    store_receipts = []
    seen = set()
    validated = []
    for store in stores:
        store_metadata = _metadata_hashes(store, _STORE_METADATA_FILES)
        results = pb.validate_store(store, require_complete=True)
        _require_metadata_unchanged(store, store_metadata, f"child store {store}")
        child_measurement = pb.read_json(store / "measurement.json")
        child_score = pb.read_json(store / "score_spec.json")
        if measurement_id(child_measurement) != canonical_mid:
            raise ValueError(f"child measurement identity differs from canonical: {store}")
        if score_spec_id(child_score) != canonical_sid:
            raise ValueError(f"child score identity differs from canonical: {store}")
        child_rows = pb.read_json(store / "manifest.json")
        child_ids = [row["id"] for row in child_rows]
        if len(set(child_ids)) != len(child_ids):
            raise ValueError(f"child manifest contains duplicate ids: {store}")
        overlap = seen.intersection(child_ids)
        if overlap:
            raise ValueError(f"child stores overlap ids: {sorted(overlap)[:3]}")
        for row in child_rows:
            expected = canonical_by_id.get(row["id"])
            if expected is None:
                raise ValueError(f"child contains id outside canonical manifest: {row['id']}")
            if row != expected:
                raise ValueError(f"child manifest row differs from canonical: {row['id']}")
        seen.update(child_ids)
        manifest_digest = store_metadata["manifest.json"]
        results_digest = store_metadata["results.json"]
        store_receipts.append({"store": str(store), "ids": child_ids,
                               "manifest_sha256": manifest_digest,
                               "results_sha256": results_digest,
                               "metadata_sha256": store_metadata})
        validated.append((store, child_rows, results, manifest_digest, results_digest,
                          store_metadata))

    missing = [name for name in canonical_ids if name not in seen]
    if missing or len(seen) != len(canonical_ids):
        raise ValueError(f"child stores are not an exact canonical union; missing={missing[:3]}")

    temp = _fresh_directory(output)
    try:
        for marker in _IDENTITY_FILES:
            _copy_file(canonical / marker, temp / marker)
        pb.atomic_json(temp / "manifest.json", canonical_rows)
        for store, child_rows, results, manifest_digest, results_digest, store_metadata in validated:
            for row in child_rows:
                name = row["id"]
                entry = results[name]
                sample_rel = Path(entry["sample_file"])
                sample_source = _relative_artifact(store, entry["sample_file"], nested=False)
                sample_digest = _copy_file(sample_source, temp / sample_rel,
                                           expected_sha256=entry["sample_sha256"])
                item = {
                    "source_store": str(store),
                    "source_manifest_sha256": manifest_digest,
                    "source_results_sha256": results_digest,
                    "source_sample_file": entry["sample_file"],
                    "source_sample_sha256": sample_digest,
                }
                if entry.get("rad_file"):
                    rad_rel = Path(entry["rad_file"])
                    rad_source = _relative_artifact(store, entry["rad_file"], nested=True)
                    rad_digest = _copy_file(rad_source, temp / rad_rel,
                                            expected_sha256=entry["rad_sha256"])
                    item.update({"source_rad_file": entry["rad_file"],
                                 "source_rad_sha256": rad_digest})
                bindings[name] = item
                merged_results[name] = entry

        ordered_results = {name: merged_results[name] for name in canonical_ids}
        pb.atomic_json(temp / "results.json", ordered_results)
        pb.atomic_json(temp / "source_bindings.json", {
            "schema_version": 1,
            "canonical_input": str(canonical),
            "canonical_manifest_sha256": pb.file_sha256(canonical / "manifest.json"),
            "canonical_metadata_sha256": canonical_metadata,
            "measurement_id": canonical_mid,
            "score_spec_id": canonical_sid,
            "stores": store_receipts,
            "items": {name: bindings[name] for name in canonical_ids},
        })
        pb.validate_store(temp, require_complete=True)
        _require_metadata_unchanged(canonical, canonical_metadata, "canonical input")
        for store, _rows, _results, _manifest, _result, store_metadata in validated:
            _require_metadata_unchanged(store, store_metadata, f"child store {store}")
        _finish_directory(temp, output)
    except Exception:
        _cleanup_temp(temp, output)
        raise
    return output


def snapshot_successes(input_store_pairs: Iterable, output_dir,
                       exclude_pattern_hashes=()) -> Path:
    """Create a fresh complete store containing only validated current successes.

    Source stores may still be incomplete.  Their manifests and results are
    hash-bound before validation and checked again immediately before the
    snapshot is published.  Physical patterns are deduplicated, preferring a
    non-repeat row and then the caller's stable pair/manifest order.
    """

    output = Path(output_dir).resolve()
    pairs = []
    for value in input_store_pairs:
        try:
            input_dir, store_dir = value
        except (TypeError, ValueError) as exc:
            raise ValueError("input_store_pairs entries must be (input_dir, store_dir)") from exc
        pairs.append((Path(input_dir).resolve(), Path(store_dir).resolve()))
    if not pairs:
        raise ValueError("at least one input/store pair is required")
    excluded = {str(value) for value in exclude_pattern_hashes}

    candidates = []
    sources = []
    common_mid = None
    common_sid = None
    for pair_index, (input_dir, store_dir) in enumerate(pairs):
        input_metadata = _metadata_hashes(input_dir, _INPUT_METADATA_FILES)
        store_metadata = _metadata_hashes(store_dir, _STORE_METADATA_FILES)
        cfg = pb.load_profile_config(input_dir / "config.yaml")
        if cfg is None:
            raise ValueError(f"input must contain a profiled config.yaml: {input_dir}")
        input_rows = pb.validate_input(input_dir, cfg)
        results = pb.validate_store(store_dir, require_complete=False)
        _require_metadata_unchanged(input_dir, input_metadata, f"input {input_dir}")
        _require_metadata_unchanged(store_dir, store_metadata, f"source store {store_dir}")
        store_rows = pb.read_json(store_dir / "manifest.json")
        if store_rows != input_rows:
            raise ValueError(f"source store manifest differs from its input: {store_dir}")
        input_mid = measurement_id(pb.read_json(input_dir / "measurement.json"))
        input_sid = score_spec_id(pb.read_json(input_dir / "score_spec.json"))
        store_mid = measurement_id(pb.read_json(store_dir / "measurement.json"))
        store_sid = score_spec_id(pb.read_json(store_dir / "score_spec.json"))
        if (store_mid, store_sid) != (input_mid, input_sid):
            raise ValueError(f"source store identity differs from its input: {store_dir}")
        if common_mid is None:
            common_mid, common_sid = input_mid, input_sid
        elif (input_mid, input_sid) != (common_mid, common_sid):
            raise ValueError("input/store pairs do not share measurement and score identities")

        successful = 0
        for row_index, row in enumerate(input_rows):
            entry = results.get(row["id"])
            if not isinstance(entry, dict) or entry.get("status") != "ok" or "error" in entry:
                continue
            successful += 1
            candidates.append({
                "pair_index": pair_index,
                "row_index": row_index,
                "input": input_dir,
                "store": store_dir,
                "row": row,
                "entry": entry,
                "repeat": row.get("kind") in ("repeat", "notarize"),
                "pattern_sha256": row["pattern_sha256"],
                "store_manifest_sha256": store_metadata["manifest.json"],
                "store_results_sha256": store_metadata["results.json"],
            })
        sources.append({
            "input": str(input_dir),
            "store": str(store_dir),
            "input_metadata_sha256": input_metadata,
            "store_metadata_sha256": store_metadata,
            "partial_source_results_sha256": store_metadata["results.json"],
            "manifest_count": len(input_rows),
            "successful_count": successful,
        })

    winners = {}
    eligible_count = 0
    for candidate in candidates:
        digest = candidate["pattern_sha256"]
        if digest in excluded:
            continue
        eligible_count += 1
        priority = (candidate["repeat"], candidate["pair_index"], candidate["row_index"])
        previous = winners.get(digest)
        if previous is None or priority < previous[0]:
            winners[digest] = (priority, candidate)
    selected = sorted((value[1] for value in winners.values()),
                      key=lambda item: (item["pair_index"], item["row_index"]))
    if not selected:
        raise ValueError("no successful unique observations remain for snapshot")
    selected_ids = [item["row"]["id"] for item in selected]
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected successes contain an id collision across source inputs")

    temp = _fresh_directory(output)
    try:
        base_input = pairs[0][0]
        for marker in _IDENTITY_FILES:
            _copy_file(base_input / marker, temp / marker)
        selected_rows = [item["row"] for item in selected]
        pb.atomic_json(temp / "manifest.json", selected_rows)
        snapshot_results = {}
        bindings = {}
        for item in selected:
            row = item["row"]
            entry = item["entry"]
            store = item["store"]
            sample_source = _relative_artifact(store, entry["sample_file"], nested=False)
            sample_digest = _copy_file(sample_source, temp / entry["sample_file"],
                                       expected_sha256=entry["sample_sha256"])
            binding = {
                "source_input": str(item["input"]),
                "source_store": str(store),
                "source_manifest_sha256": item["store_manifest_sha256"],
                "partial_source_results_sha256": item["store_results_sha256"],
                "source_sample_file": entry["sample_file"],
                "source_sample_sha256": sample_digest,
            }
            if entry.get("rad_file"):
                rad_source = _relative_artifact(store, entry["rad_file"], nested=True)
                rad_digest = _copy_file(rad_source, temp / entry["rad_file"],
                                        expected_sha256=entry["rad_sha256"])
                binding.update({"source_rad_file": entry["rad_file"],
                                "source_rad_sha256": rad_digest})
            bindings[row["id"]] = binding
            snapshot_results[row["id"]] = entry

        pb.atomic_json(temp / "results.json", snapshot_results)
        pb.atomic_json(temp / "source_bindings.json", {
            "schema_version": 1,
            "partial_snapshot": True,
            "source_batches_may_be_incomplete": True,
            "measurement_id": common_mid,
            "score_spec_id": common_sid,
            "counts": {
                "successful_source_rows": len(candidates),
                "excluded_successful_rows": len(candidates) - eligible_count,
                "duplicate_successful_rows": eligible_count - len(selected),
                "unique_selected_rows": len(selected),
            },
            "excluded_pattern_sha256": sorted(excluded),
            "sources": sources,
            "items": {name: bindings[name] for name in selected_ids},
        })
        pb.validate_store(temp, require_complete=True)
        for index, (input_dir, store_dir) in enumerate(pairs):
            _require_metadata_unchanged(
                input_dir, sources[index]["input_metadata_sha256"], f"input {input_dir}")
            _require_metadata_unchanged(
                store_dir, sources[index]["store_metadata_sha256"],
                f"source store {store_dir}")
        _finish_directory(temp, output)
    except Exception:
        _cleanup_temp(temp, output)
        raise
    return output
