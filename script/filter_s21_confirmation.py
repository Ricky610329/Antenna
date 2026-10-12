"""Read-only original/repeat confirmation for the current three-band S21 profile.

This bounded utility replays already-complete native Torch observations.  It
does not run HFSS, train or score a model, dispatch work, or assess an
engineering/formal gate.  Confirmation requires a distinct, caller-pinned
source-bound repeat of the same pattern and a strictly positive three-band
worst margin in both observations.
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import numpy as np

from antenna.measurement import frequency_grid, measurement_id, score_spec_id
from script import profiled_batch as pb
from script.filter_confirmation import _read_observation


REPO = Path(__file__).resolve().parents[1]
CANONICAL_PROFILE = REPO / "configs" / "dual_r81_s21_discrete.yaml"
_EXPECTED_BANDS = (
    {"name": "s21_pass", "label": "S21", "start_ghz": 26.0,
     "stop_ghz": 30.0, "relation": "ge", "threshold": -3.0},
    {"name": "s21_low_stop", "label": "S21", "start_ghz": 16.0,
     "stop_ghz": 20.0, "relation": "le", "threshold": -20.0},
    {"name": "s21_high_stop", "label": "S21", "start_ghz": 36.0,
     "stop_ghz": 40.0, "relation": "le", "threshold": -20.0},
)


def _load_exact_profile(profile_config: Path):
    """Load only a byte-identical copy of the repository's current profile."""
    config_path = Path(profile_config).resolve()
    canonical_path = CANONICAL_PROFILE.resolve()
    canonical_hash = pb.file_sha256(canonical_path)
    config_hash = pb.file_sha256(config_path)
    if config_hash != canonical_hash:
        raise ValueError("profile config must be an exact copy of the canonical current profile")
    cfg = pb.load_profile_config(config_path)
    canonical = pb.load_profile_config(canonical_path)
    if cfg is None or canonical is None:
        raise ValueError("canonical current profile must be a named measurement config")
    if (cfg.measurement != canonical.measurement
            or cfg.score_spec != canonical.score_spec
            or cfg.name != canonical.name
            or cfg.port != canonical.port
            or cfg.scope != canonical.scope
            or cfg.runtime != canonical.runtime):
        raise ValueError("profile config differs from the canonical current profile")

    freqs = frequency_grid(cfg.measurement)
    bands = tuple(cfg.score_spec["bands"])
    if (cfg.port != "dual"
            or cfg.measurement["labels"] != ["S11", "S21", "S22"]
            or cfg.measurement["geometry"] != {
                "geom": "p01", "pixel_count": 25, "diag_bridge_w": 0.075}
            or cfg.measurement["sweep"].get("type") != "Discrete"
            or not np.array_equal(freqs, np.arange(16.0, 40.1, 0.5))
            or bands != _EXPECTED_BANDS):
        raise ValueError("canonical current profile is not the exact R81 three-S21-band profile")
    if pb.file_sha256(config_path) != config_hash or pb.file_sha256(canonical_path) != canonical_hash:
        raise ValueError("profile config changed while being read")
    return config_path, config_hash, canonical_path, canonical_hash, cfg, freqs


def _require_native_profile(store: Path, cfg) -> None:
    """Require native store documents, rather than an ID-only profile match."""
    root = Path(store).resolve()
    measurement = pb.read_json(root / "measurement.json")
    score_spec = pb.read_json(root / "score_spec.json")
    if measurement != cfg.measurement or score_spec != cfg.score_spec:
        raise ValueError(f"{root}: native profile documents differ from the canonical current profile")


def _require_sha256(value: str, role: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in value)):
        raise ValueError(f"{role} source_bindings SHA256 pin must be 64 hexadecimal characters")
    return value.lower()


def _role_fields(row: dict) -> dict:
    return {name: row.get(name) for name in (
        "id", "kind", "repeat", "repeat_reason", "parent_batch_id",
        "pattern_sha256", "measurement_id", "score_spec_id")}


def _read_bound_observation(store: Path, row_id: str, cfg,
                            expected_bindings_sha256: str, role: str) -> dict:
    """Replay a snapshot and its directly bound native source observation."""
    root = Path(store).resolve()
    bindings_path = root / "source_bindings.json"
    expected = _require_sha256(expected_bindings_sha256, role)
    if not bindings_path.is_file():
        raise ValueError(f"{role} snapshot requires source_bindings.json")
    if pb.file_sha256(bindings_path) != expected:
        raise ValueError(f"{role} source_bindings SHA256 differs from caller pin")

    _require_native_profile(root, cfg)
    snapshot = _read_observation(root, row_id, cfg)
    if not snapshot["source"].get("binding_present"):
        raise ValueError(f"{role} observation must retain a frozen source binding")
    if pb.file_sha256(bindings_path) != expected:
        raise ValueError(f"{role} source_bindings changed while snapshot was read")

    source_root = Path(snapshot["source"]["source_store"]).resolve()
    if source_root == root:
        raise ValueError(f"{role} source binding must point to an underlying native store")
    if (source_root / "source_bindings.json").exists():
        raise ValueError(f"{role} bound source must be a native store, not another snapshot")
    _require_native_profile(source_root, cfg)
    source = _read_observation(source_root, row_id, cfg)
    if source["source"].get("binding_present"):
        raise ValueError(f"{role} bound source unexpectedly has snapshot provenance")

    snapshot_entry, source_entry = snapshot["entry"], source["entry"]
    if (snapshot_entry.get("sample_file") != source_entry.get("sample_file")
            or snapshot_entry.get("sample_sha256") != source_entry.get("sample_sha256")
            or snapshot["pattern_sha256"] != source["pattern_sha256"]):
        raise ValueError(f"{role} snapshot differs from its bound native source observation")
    if _role_fields(snapshot["row"]) != _role_fields(source["row"]):
        raise ValueError(f"{role} snapshot/source role metadata differs")
    if pb.file_sha256(bindings_path) != expected:
        raise ValueError(f"{role} source_bindings changed while native source was read")

    snapshot["source_bindings_sha256"] = expected
    snapshot["bound_source"] = source
    snapshot["actual_source_identity"] = [str(source_root), row_id]
    return snapshot


def _band_facts(cfg, candidate: dict, repeat: dict) -> list[dict]:
    facts = []
    for band in cfg.score_spec["bands"]:
        name = band["name"]
        facts.append({
            "name": name,
            "label": band["label"],
            "start_ghz": float(band["start_ghz"]),
            "stop_ghz": float(band["stop_ghz"]),
            "relation": band["relation"],
            "threshold_db": float(band["threshold"]),
            "candidate_margin_db": candidate["evidence"]["margins_db"][name],
            "candidate_worst_frequency_ghz": candidate["evidence"][
                "worst_frequencies_ghz"][name],
            "repeat_margin_db": repeat["evidence"]["margins_db"][name],
            "repeat_worst_frequency_ghz": repeat["evidence"][
                "worst_frequencies_ghz"][name],
        })
    return facts


def analyze_filter_s21_confirmation(
        profile_config, candidate_store, candidate_id, repeat_store, repeat_id, *,
        candidate_source_bindings_sha256, repeat_source_bindings_sha256) -> dict:
    """Return additive read-only evidence for one original/repeat S21 pair."""
    (config_path, config_hash, canonical_path, canonical_hash,
     cfg, freqs) = _load_exact_profile(Path(profile_config))
    candidate_root = Path(candidate_store).resolve()
    repeat_root = Path(repeat_store).resolve()
    if candidate_root == repeat_root and candidate_id == repeat_id:
        raise ValueError("candidate and repeat must be distinct observations")

    candidate = _read_bound_observation(
        candidate_root, candidate_id, cfg,
        candidate_source_bindings_sha256, "candidate")
    repeat = _read_bound_observation(
        repeat_root, repeat_id, cfg,
        repeat_source_bindings_sha256, "repeat")

    if candidate["actual_source_identity"] == repeat["actual_source_identity"]:
        raise ValueError("candidate and repeat are copies of the same actual source observation")

    original_row, repeat_row = candidate["row"], repeat["row"]
    if (original_row.get("kind") == "repeat"
            or original_row.get("repeat") not in (None, False)):
        raise ValueError("candidate must be the original non-repeat observation")
    if (repeat_row.get("kind") != "repeat"
            or repeat_row.get("repeat") is not True
            or repeat_row.get("repeat_reason") != "repeatability"):
        raise ValueError(
            "repeat row must have kind=repeat, repeat=true, repeat_reason=repeatability")
    if repeat_row.get("parent_batch_id") != candidate_id:
        raise ValueError("repeat parent_batch_id must equal the selected candidate id")
    if candidate["pattern_sha256"] != repeat["pattern_sha256"]:
        raise ValueError("candidate and repeat patterns differ")

    candidate_wm = candidate["evidence"]["worst_margin_db"]
    repeat_wm = repeat["evidence"]["worst_margin_db"]
    failures = []
    if not candidate_wm > 0.0:
        failures.append("candidate_worst_margin_not_strictly_positive")
    if not repeat_wm > 0.0:
        failures.append("repeat_worst_margin_not_strictly_positive")
    confirmed = not failures
    reason = ("both_measured_s21_worst_margins_strictly_positive" if confirmed
              else "+".join(failures))

    pair_binding = {
        "candidate_id": candidate_id,
        "candidate_kind": original_row.get("kind"),
        "candidate_repeat": original_row.get("repeat"),
        "repeat_id": repeat_id,
        "repeat_kind": repeat_row.get("kind"),
        "repeat_repeat": repeat_row.get("repeat"),
        "repeat_reason": repeat_row.get("repeat_reason"),
        "repeat_parent_batch_id": repeat_row.get("parent_batch_id"),
        "pattern_sha256": candidate["pattern_sha256"],
        "same_pattern": True,
        "candidate_source_identity": candidate["actual_source_identity"],
        "repeat_source_identity": repeat["actual_source_identity"],
        "candidate_source_bindings_sha256": candidate["source_bindings_sha256"],
        "repeat_source_bindings_sha256": repeat["source_bindings_sha256"],
        "distinct_bound_source_observations": True,
    }
    return {
        "schema_version": 1,
        "analysis_kind": "r81_three_s21_band_original_repeat_confirmation",
        "identity": {
            "measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec),
            "candidate": candidate["actual_source_identity"],
            "repeat": repeat["actual_source_identity"],
            "pattern_sha256": candidate["pattern_sha256"],
        },
        "expected_profile": {
            "config_path": str(config_path),
            "config_sha256": config_hash,
            "canonical_config_path": str(canonical_path),
            "canonical_config_sha256": canonical_hash,
            "native_profile_documents_exact": True,
            "measurement_id": measurement_id(cfg.measurement),
            "score_spec_id": score_spec_id(cfg.score_spec),
            "response_labels": list(cfg.measurement["labels"]),
            "response_shape": [3, 49],
            "frequency_ghz": freqs.tolist(),
            "score_bands": [band["name"] for band in cfg.score_spec["bands"]],
        },
        "candidate": candidate["evidence"],
        "repeat": repeat["evidence"],
        "bound_native_sources": {
            "candidate": candidate["bound_source"]["evidence"],
            "repeat": repeat["bound_source"]["evidence"],
        },
        "pair_binding": pair_binding,
        "source_band_facts": _band_facts(cfg, candidate, repeat),
        "pair_min_worst_margin_db": min(candidate_wm, repeat_wm),
        "s21_positive_margin_repeat_confirmed": confirmed,
        "confirmation_reason": reason,
        "strict_confirmation_rule": (
            "candidate.three_band_worst_margin_db > 0 and "
            "repeat.three_band_worst_margin_db > 0"),
        "zero_margin_policy": (
            "WM = 0 satisfies the generic score threshold but is not confirmed here"),
        "formal_gate_satisfied": False,
        "engineering_gate_satisfied": False,
        "campaign_complete": False,
        "provenance_assurance": {
            "level": "caller_pinned_snapshot_binding_plus_native_source_replay",
            "candidate_source_bindings_sha256": candidate["source_bindings_sha256"],
            "repeat_source_bindings_sha256": repeat["source_bindings_sha256"],
            "independent_hfss_execution_cryptographically_proven": False,
        },
        "limitations": [
            "This confirms only the three specified S21 bands for one original and one distinct source-bound repeat.",
            "Distinct source identity relies on caller-pinned trusted worker/snapshot provenance; it is not cryptographic proof of independent HFSS execution.",
            "S11 and S22 are retained as raw measured channels but do not gate this S21-only result.",
            "Soft buffer diagnostics are not part of the three-band worst margin.",
            "The evidence uses a coarse 49-point 16--40 GHz grid at 0.5 GHz spacing.",
            "No behavior outside 16--40 GHz is measured or inferred.",
            "No statistical confidence limits follow from one original/repeat pair.",
            "This result does not establish a formal gate, engineering gate, or completed campaign.",
        ],
    }


def _evidence_roots(result: dict) -> list[Path]:
    roots = []
    for role in ("candidate", "repeat"):
        roots.append(Path(result[role]["store"]).resolve())
        roots.append(Path(result[role]["source_binding"]["source_store"]).resolve())
    return roots


def write_fresh_json(path, result) -> Path:
    """Atomically publish a complete receipt without replacing any path."""
    # Serialization happens before creating a directory, draft, or final path.
    payload = (json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    output = Path(path).resolve()
    for root in _evidence_roots(result):
        try:
            output.relative_to(root)
        except ValueError:
            continue
        raise ValueError(f"output must not be inside evidence root: {root}")
    if output.exists():
        raise FileExistsError(f"create-only receipt already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    draft = output.parent / f".filter-s21-confirmation-draft-{uuid.uuid4().hex}.json"
    try:
        with draft.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-volume hard link atomically exposes complete bytes and fails
        # if another writer created the final path first.
        os.link(draft, output)
    finally:
        if draft.exists():
            draft.unlink()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-config", required=True, type=Path)
    parser.add_argument("--candidate-store", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-source-bindings-sha256", required=True)
    parser.add_argument("--repeat-store", required=True, type=Path)
    parser.add_argument("--repeat-id", required=True)
    parser.add_argument("--repeat-source-bindings-sha256", required=True)
    parser.add_argument(
        "--out", type=Path,
        help="optional fresh JSON receipt outside both stores; existing files are never overwritten")
    args = parser.parse_args()
    result = analyze_filter_s21_confirmation(
        args.profile_config, args.candidate_store, args.candidate_id,
        args.repeat_store, args.repeat_id,
        candidate_source_bindings_sha256=args.candidate_source_bindings_sha256,
        repeat_source_bindings_sha256=args.repeat_source_bindings_sha256)
    if args.out is not None:
        write_fresh_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
