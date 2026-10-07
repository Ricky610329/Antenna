"""Read-only evidence check for an original and repeat R81 filter observation.

The generic profile score regards a zero margin as passing.  This utility has
the deliberately stricter confirmation rule requested for R81: both measured
observations must have ``worst_margin > 0``.  It does not run HFSS, train a
model, dispatch work, or assess the separate engineering checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from antenna.measurement import frequency_grid, measurement_id, score_spec_id
from script import profiled_batch as pb


_STORE_METADATA = ("measurement.json", "score_spec.json", "manifest.json", "results.json")


def _metadata_hashes(root: Path) -> dict:
    names = list(_STORE_METADATA)
    if (root / "source_bindings.json").is_file():
        names.append("source_bindings.json")
    return {name: pb.file_sha256(root / name) for name in names}


def _source_identity(root: Path, row_id: str, entry: dict, mid: str, sid: str) -> dict:
    """Validate an optional snapshot binding and return observation identity."""
    path = root / "source_bindings.json"
    if not path.is_file():
        return {
            "binding_present": False,
            "source_store": str(root),
            "source_row_id": row_id,
            "identity": [str(root), row_id],
        }

    document = pb.read_json(path)
    items = document.get("items") if isinstance(document, dict) else None
    if (not isinstance(items, dict)
            or document.get("schema_version") != 1
            or document.get("partial_snapshot") is not True
            or document.get("measurement_id") != mid
            or document.get("score_spec_id") != sid
            or row_id not in items):
        raise ValueError(f"{root}: invalid source_bindings for selected row {row_id}")
    binding = items[row_id]
    if (not isinstance(binding, dict)
            or not isinstance(binding.get("source_store"), str)
            or not binding["source_store"]
            or binding.get("source_sample_file") != entry.get("sample_file")
            or binding.get("source_sample_sha256") != entry.get("sample_sha256")):
        raise ValueError(f"{root}: selected row {row_id} source binding/hash differs")

    source_store = Path(binding["source_store"])
    if not source_store.is_absolute():
        source_store = root / source_store
    source_store = source_store.resolve()
    return {
        "binding_present": True,
        "source_store": str(source_store),
        # snapshot_successes preserves the source row id.
        "source_row_id": row_id,
        "source_sample_file": binding.get("source_sample_file"),
        "source_sample_sha256": binding["source_sample_sha256"],
        "identity": [str(source_store), row_id],
    }


def _read_observation(store, row_id, cfg) -> dict:
    root = Path(store).resolve()
    if not isinstance(row_id, str) or not row_id or Path(row_id).name != row_id:
        raise ValueError("observation id must be a safe nonempty filename component")
    before = _metadata_hashes(root)
    results = pb.validate_store(root, require_complete=True)
    manifest = pb.read_json(root / "manifest.json")
    after = _metadata_hashes(root)
    if before != after:
        raise ValueError(f"{root}: store metadata changed during validation")

    expected_mid = measurement_id(cfg.measurement)
    expected_sid = score_spec_id(cfg.score_spec)
    store_mid = measurement_id(pb.read_json(root / "measurement.json"))
    store_sid = score_spec_id(pb.read_json(root / "score_spec.json"))
    if (store_mid, store_sid) != (expected_mid, expected_sid):
        raise ValueError(f"{root}: store does not match expected measurement and score profile")

    rows = {row["id"]: row for row in manifest}
    if row_id not in rows or row_id not in results:
        raise ValueError(f"{root}: selected observation id not found: {row_id}")
    row, entry = rows[row_id], results[row_id]
    sample = (root / entry["sample_file"]).resolve()
    try:
        sample.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{root}: selected sample path escapes store") from exc
    sample_hash = pb.file_sha256(sample)
    if sample_hash != entry.get("sample_sha256"):
        raise ValueError(f"{root}: selected sample hash differs after validation")
    pattern, response = torch.load(sample, weights_only=True, map_location="cpu")
    if pb.file_sha256(sample) != sample_hash:
        raise ValueError(f"{root}: selected sample changed while being read")
    values = np.asarray(response, dtype=float)
    if values.shape != (3, 49) or not np.isfinite(values).all():
        raise ValueError(f"{root}: selected observation must be a complete finite 3x49 response")
    if pb.pattern_sha256(pattern) != entry.get("pattern_sha256"):
        raise ValueError(f"{root}: selected pattern hash differs")

    band_names = [band["name"] for band in cfg.score_spec["bands"]]
    if (set(entry.get("margins", {})) != set(band_names)
            or set(entry.get("worst_frequencies", {})) != set(band_names)
            or entry.get("worst_margin") is None):
        raise ValueError(f"{root}: selected observation lacks all five score bands")
    source = _source_identity(root, row_id, entry, store_mid, store_sid)
    final_metadata = _metadata_hashes(root)
    if final_metadata != after:
        raise ValueError(f"{root}: store metadata changed while selected evidence was read")
    return {
        "root": root,
        "row": row,
        "entry": entry,
        "pattern_sha256": entry["pattern_sha256"],
        "source": source,
        "evidence": {
            "store": str(root),
            "id": row_id,
            "sample_file": entry["sample_file"],
            "sample_sha256": sample_hash,
            "metadata_sha256": final_metadata,
            "source_binding": source,
            "margins_db": {name: float(entry["margins"][name]) for name in band_names},
            "worst_frequencies_ghz": {
                name: float(entry["worst_frequencies"][name]) for name in band_names
            },
            "worst_band": entry.get("worst_band"),
            "worst_margin_db": float(entry["worst_margin"]),
            "generic_all_passed_ge_zero": bool(entry.get("all_passed")),
        },
    }


def analyze_filter_confirmation(profile_config, candidate_store, candidate_id,
                                repeat_store, repeat_id) -> dict:
    """Return read-only evidence for one original/repeat dual-profile pair."""
    config_path = Path(profile_config).resolve()
    config_hash = pb.file_sha256(config_path)
    cfg = pb.load_profile_config(config_path)
    if cfg is None:
        raise ValueError("expected profile config is not a named measurement config")
    freqs = frequency_grid(cfg.measurement)
    if (cfg.port != "dual" or cfg.measurement["labels"] != ["S11", "S21", "S22"]
            or not np.array_equal(freqs, np.arange(16.0, 40.1, 0.5))
            or len(cfg.score_spec["bands"]) != 5):
        raise ValueError("confirmation requires a dual R81 3x49 profile with five score bands")
    if pb.file_sha256(config_path) != config_hash:
        raise ValueError("expected profile config changed while being read")

    candidate_root, repeat_root = Path(candidate_store).resolve(), Path(repeat_store).resolve()
    if candidate_root == repeat_root and candidate_id == repeat_id:
        raise ValueError("candidate and repeat must be distinct observations")
    candidate = _read_observation(candidate_root, candidate_id, cfg)
    repeat = _read_observation(repeat_root, repeat_id, cfg)

    original_row, repeat_row = candidate["row"], repeat["row"]
    if (original_row.get("kind") == "repeat"
            or original_row.get("repeat") not in (None, False)):
        raise ValueError("candidate must be the original non-repeat observation")
    if (repeat_row.get("kind") != "repeat" or repeat_row.get("repeat") is not True
            or repeat_row.get("repeat_reason") != "repeatability"):
        raise ValueError("repeat row must have kind=repeat, repeat=true, repeat_reason=repeatability")
    if repeat_row.get("parent_batch_id") != candidate_id:
        raise ValueError("repeat parent_batch_id must equal the selected candidate id")
    if candidate["pattern_sha256"] != repeat["pattern_sha256"]:
        raise ValueError("candidate and repeat patterns differ")
    if candidate["source"]["identity"] == repeat["source"]["identity"]:
        raise ValueError("candidate and repeat are copies of the same source observation")

    candidate_wm = candidate["evidence"]["worst_margin_db"]
    repeat_wm = repeat["evidence"]["worst_margin_db"]
    failed = []
    if not candidate_wm > 0.0:
        failed.append("candidate_worst_margin_not_strictly_positive")
    if not repeat_wm > 0.0:
        failed.append("repeat_worst_margin_not_strictly_positive")
    confirmed = not failed
    reason = ("both_measured_worst_margins_strictly_positive" if confirmed
              else "+".join(failed))
    return {
        "schema_version": 1,
        "analysis_kind": "r81_original_repeat_performance_confirmation",
        "expected_profile": {
            "config_path": str(config_path),
            "config_sha256": config_hash,
            "measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec),
            "response_shape": [3, 49],
            "frequency_ghz": freqs.tolist(),
            "score_bands": [band["name"] for band in cfg.score_spec["bands"]],
        },
        "candidate": candidate["evidence"],
        "repeat": repeat["evidence"],
        "pair_min_worst_margin_db": min(candidate_wm, repeat_wm),
        "performance_target_confirmed": confirmed,
        "confirmation_reason": reason,
        "strict_confirmation_rule": "candidate.worst_margin_db > 0 and repeat.worst_margin_db > 0",
        "engineering_checks_not_assessed": True,
        "engineering_checks_remaining": [
            "R81 smoke validation",
            "Discrete sweep validation",
            "mesh convergence within 0.3 dB",
        ],
        "campaign_complete": False,
        "limitations": [
            "This is evidence for one measured original plus one measured repeat only.",
            "A zero worst margin remains a valid replayed result but does not meet the strict target.",
            "Equal response hashes are allowed for distinct observations from distinct trusted worker metadata.",
            "Source-observation identity relies on trusted worker/snapshot metadata.",
            "This evidence does not replace smoke, Discrete-sweep, or 0.3 dB mesh-convergence checks.",
        ],
    }


def write_fresh_json(path, result) -> Path:
    """Write a new receipt without ever replacing an existing path."""
    output = Path(path).resolve()
    for role in ("candidate", "repeat"):
        store = Path(result[role]["store"]).resolve()
        try:
            output.relative_to(store)
        except ValueError:
            continue
        raise ValueError(f"output must not be inside the {role} input store")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-config", required=True, type=Path)
    parser.add_argument("--candidate-store", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--repeat-store", required=True, type=Path)
    parser.add_argument("--repeat-id", required=True)
    parser.add_argument("--out", type=Path,
                        help="optional fresh JSON receipt; existing files are never overwritten")
    args = parser.parse_args()
    result = analyze_filter_confirmation(
        args.profile_config, args.candidate_store, args.candidate_id,
        args.repeat_store, args.repeat_id)
    if args.out is not None:
        write_fresh_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
