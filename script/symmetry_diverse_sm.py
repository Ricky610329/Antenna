# -*- coding: utf-8 -*-
"""Prospective diversity-aware symmetric single-port SM pool generation.

The module is deliberately read-only with respect to the training workdir.  It
binds one immutable cumulative data version, uses measured rows only as
mutation parents, and returns the ordinary :mod:`script.symmetry_sm_pool`
result contract with an additional JSON-safe ``diversity_audit`` mapping.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from script import exploration as ex
from script import profiled_batch as pb
from script import symmetry_sm_pool as sm


_POLICY_KEYS = {
    "schema_version", "name", "candidate_pool_size", "parent_top_fraction",
    "parent_count", "max_parents_per_canonical_group", "min_parent_groups",
    "min_parent_physical_hamming", "min_selected_physical_hamming",
    "max_selected_per_canonical_group", "arm_order", "arm_weights",
    "disagreement_mean_percentile", "source_pool_fractions", "ranking",
    "parent_ranking", "strict_no_relaxation", "measurement_profile",
    "measurement_changes", "training_protocol_changes", "dispatch", "seed", "limits",
}
_LITERALS = {
    "schema_version": 1,
    "name": "r80_diverse_sm_v1",
    "candidate_pool_size": 20_000,
    "parent_top_fraction": 0.25,
    "parent_count": 32,
    "max_parents_per_canonical_group": 2,
    "min_parent_groups": 8,
    "min_parent_physical_hamming": 64,
    "min_selected_physical_hamming": 64,
    "max_selected_per_canonical_group": 2,
    "arm_order": ["global_lcb", "parent_lcb", "high_disagreement"],
    "arm_weights": [1, 1, 1],
    "disagreement_mean_percentile": 60,
    "source_pool_fractions": {"fresh_random": 0.5, "parent_variant": 0.5},
    "ranking": ("descending score, then pattern_sha256; greedily take the highest "
                "feasible candidate under whole-cohort constraints"),
    "parent_ranking": ("actual WM top quartile; highest-WM first, then existing "
                       "maxmin_hamming_indices with canonical-group cap"),
    "strict_no_relaxation": True,
    "measurement_profile": "configs/single_r80_symmetry_factory.yaml",
    "measurement_changes": False,
    "training_protocol_changes": False,
    "dispatch": ("existing factory transaction, same priority1 and 16-pattern shards, "
                 "existing 96 outstanding and 5000 valid-plus-reserved limits"),
    "seed": 80100956,
    "limits": [
        "Predictions guide candidate selection; HFSS remains the truth.",
        ("This changes pool generation and arm allocation together; it is not a "
         "single-factor causal diversity comparison."),
        ("Top-quartile parent descriptors are development information, not independent "
         "model-validation evidence."),
        "LOW blind backlog remains enabled and unchanged.",
        "No bridge-width, frequency, score-spec, emforge or worker runtime change.",
        ("First new cohort must be evaluated by origin, saved prediction error, actual WM "
         "and geometry coverage before claiming improvement."),
    ],
}
_POSITIVE_INTS = {
    "schema_version", "candidate_pool_size", "parent_count",
    "max_parents_per_canonical_group", "min_parent_groups",
    "min_parent_physical_hamming", "min_selected_physical_hamming",
    "max_selected_per_canonical_group", "disagreement_mean_percentile", "seed",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("training manifest file path must be a non-empty string")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"training manifest path escapes workdir: {relative}") from exc
    if not path.is_file():
        raise ValueError(f"training manifest file is missing: {relative}")
    return path


def _hex64(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be lowercase 64-hex")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_policy(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("diversity policy must be a JSON object")
    unknown, missing = set(raw) - _POLICY_KEYS, _POLICY_KEYS - set(raw)
    if unknown or missing:
        raise ValueError(f"diversity policy keys differ; unknown={sorted(unknown)}, "
                         f"missing={sorted(missing)}")
    value = dict(raw)
    for name in _POSITIVE_INTS:
        _positive_int(value[name], name)
    for name in ("parent_top_fraction",):
        if (isinstance(value[name], bool) or not isinstance(value[name], (int, float)) or
                not math.isfinite(float(value[name])) or not 0.0 < float(value[name]) <= 1.0):
            raise ValueError(f"{name} must be finite and in (0, 1]")
    weights = value["arm_weights"]
    if (not isinstance(weights, list) or len(weights) != 3 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                not math.isfinite(float(x)) or float(x) <= 0 for x in weights)):
        raise ValueError("arm_weights must contain three finite positive numbers")
    fractions = value["source_pool_fractions"]
    if not isinstance(fractions, Mapping):
        raise ValueError("source_pool_fractions must be an object")
    for key, number in fractions.items():
        if (isinstance(number, bool) or not isinstance(number, (int, float)) or
                not math.isfinite(float(number))):
            raise ValueError(f"source_pool_fractions.{key} must be finite")
    if not isinstance(value["limits"], list) or any(
            not isinstance(item, str) or not item for item in value["limits"]):
        raise ValueError("limits must be a list of non-empty strings")
    for name, expected in _LITERALS.items():
        if value[name] != expected:
            raise ValueError(f"diversity policy field {name} differs from the fixed v1 contract")
    return value


def load_policy(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the fixed prospective v1 search policy."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return _validate_policy(_read_json(source))


def _wm(response: np.ndarray, frequencies: Sequence[float]) -> float:
    response = np.asarray(response, dtype=np.float64)
    freq = np.asarray(frequencies, dtype=np.float64)
    if response.shape != (2, len(freq)) or not np.isfinite(response).all():
        raise ValueError("measured response must be finite with shape (2, frequency_count)")
    band = (freq >= sm.FACTORY_BAND_GHZ[0]) & (freq <= sm.FACTORY_BAND_GHZ[1])
    if int(band.sum()) != 7 or not np.allclose(freq[band], np.arange(26.5, 29.51, 0.5)):
        raise ValueError("training protocol does not provide the exact 26.5--29.5 GHz WM band")
    s11_margin = -10.0 - float(response[0, band].max())
    gain_margin = float(response[1, band].min()) - 4.0
    value = min(s11_margin, gain_margin)
    if not math.isfinite(value):
        raise ValueError("measured WM is non-finite")
    return value


def _training_snapshot(training_workdir: Path, version: int,
                       profile_cfg: Any) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    root = Path(training_workdir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    version = _positive_int(version, "version")
    fixed = {
        "state": root / "state.json", "protocol": root / "protocol.json",
        "manifest": root / "data" / f"data-v{version:03d}" / "manifest.json",
        "receipt": root / "data" / f"data-v{version:03d}" / "receipt.json",
    }
    if any(not path.is_file() for path in fixed.values()):
        raise ValueError("training version is incomplete")
    before = {name: pb.file_sha256(path) for name, path in fixed.items()}
    state, protocol = _read_json(fixed["state"]), _read_json(fixed["protocol"])
    manifest, receipt = _read_json(fixed["manifest"]), _read_json(fixed["receipt"])
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("training manifest must be a non-empty list")
    if state.get("latest_data_version") != version or receipt.get("data_version") != version:
        raise ValueError("requested version is not the exact latest immutable training version")
    if ex.content_id(manifest) != receipt.get("manifest_id"):
        raise ValueError("training manifest_id is invalid")
    from antenna.measurement import measurement_id, score_spec_id
    protocol_id = protocol.get("protocol_id")
    if (_hex64(protocol_id, "protocol_id") !=
            ex.content_id({key: value for key, value in protocol.items()
                           if key != "protocol_id"})):
        raise ValueError("training protocol_id is invalid")
    if (protocol.get("measurement_id") != measurement_id(profile_cfg.measurement) or
            protocol.get("score_spec_id") != score_spec_id(profile_cfg.score_spec)):
        raise ValueError("training protocol measurement differs from pool profile")
    frequencies = protocol.get("frequencies_ghz")
    if not isinstance(frequencies, list):
        raise ValueError("training protocol frequencies are missing")

    rows: list[dict[str, Any]] = []
    tracked: dict[str, Path] = dict(fixed)
    declared_artifacts: dict[str, str] = {}
    seen_ids: set[str] = set()
    for ordinal, source in enumerate(manifest):
        if not isinstance(source, Mapping):
            raise ValueError(f"training manifest row {ordinal} is not an object")
        row = dict(source)
        required = {"id", "sample_file", "rad_file", "source_sample_sha256",
                    "source_rad_sha256", "pattern_sha256", "lineage_id",
                    "canonical_group_id", "measurement_id", "score_spec_id",
                    "geometry_profile"}
        if not required <= set(row):
            raise ValueError(f"training manifest row {ordinal} lacks required metadata")
        row_id = str(row["id"])
        if not row_id or row_id in seen_ids:
            raise ValueError("training manifest ids must be non-empty and unique")
        seen_ids.add(row_id)
        sample, rad = _safe_file(root, row["sample_file"]), _safe_file(root, row["rad_file"])
        sample_hash = _hex64(row["source_sample_sha256"], "source_sample_sha256")
        rad_hash = _hex64(row["source_rad_sha256"], "source_rad_sha256")
        if pb.file_sha256(sample) != sample_hash or pb.file_sha256(rad) != rad_hash:
            raise ValueError(f"training artifact hash mismatch for {row_id}")
        pattern_value, response_value = torch.load(sample, weights_only=True, map_location="cpu")
        pattern = sm._checked_pattern(pattern_value, profile_cfg)
        pattern_hash = _hex64(row["pattern_sha256"], "pattern_sha256")
        if ex.pattern_sha256(pattern) != pattern_hash:
            raise ValueError(f"training pattern hash mismatch for {row_id}")
        response = np.asarray(response_value, dtype=np.float64)
        # Loading the radiation payload catches malformed cached tensors while its
        # immutable byte identity remains the authoritative row binding.
        rad_value = torch.load(rad, weights_only=True, map_location="cpu")
        ex.radiation_vector(rad_value)
        if (row["measurement_id"] != protocol["measurement_id"] or
                row["score_spec_id"] != protocol["score_spec_id"] or
                row["geometry_profile"] != protocol["geometry_profile"]):
            raise ValueError(f"training row profile metadata mismatch for {row_id}")
        rows.append({**row, "pattern": pattern, "actual_wm": _wm(response, frequencies)})
        tracked[f"sample:{row_id}"] = sample
        tracked[f"rad:{row_id}"] = rad
        declared_artifacts[f"sample:{row_id}"] = sample_hash
        declared_artifacts[f"rad:{row_id}"] = rad_hash

    unique = {row["pattern_sha256"] for row in rows}
    if receipt.get("cumulative_observations") != len(rows) or \
            receipt.get("cumulative_valid_unique") != len(unique):
        raise ValueError("training receipt counts differ from actual manifest")
    after = {name: pb.file_sha256(path) for name, path in tracked.items()}
    expected = {**before, **declared_artifacts}
    if after != expected:
        raise ValueError("training cache changed while reading the parent snapshot")
    binding = {
        "data_version": version, "manifest_id": receipt["manifest_id"],
        "cumulative_valid_unique": receipt["cumulative_valid_unique"],
        "measurement_id": protocol["measurement_id"],
        "score_spec_id": protocol["score_spec_id"], "protocol_id": protocol_id,
        "state_sha256": before["state"], "protocol_sha256": before["protocol"],
        "manifest_sha256": before["manifest"], "receipt_sha256": before["receipt"],
        "artifact_set_sha256": ex.content_id(sorted(
            (name, digest) for name, digest in after.items()
            if name.startswith(("sample:", "rad:")))),
    }
    return rows, binding, after


def _select_parents(rows: Sequence[dict[str, Any]], policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row["actual_wm"]), row["pattern_sha256"], row["id"]))
    top_n = max(1, int(math.ceil(len(ranked) * float(policy["parent_top_fraction"]))))
    eligible = ranked[:top_n]
    chosen: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()

    def allowed(row: Mapping[str, Any]) -> bool:
        group = str(row["canonical_group_id"])
        return (group_counts[group] < int(policy["max_parents_per_canonical_group"]) and
                all(ex.hamming(row["pattern"], prior["pattern"]) >=
                    int(policy["min_parent_physical_hamming"]) for prior in chosen))

    if eligible:
        first = eligible[0]
        chosen.append(first); group_counts[str(first["canonical_group_id"])] += 1
    while len(chosen) < min(int(policy["parent_count"]), len(eligible)):
        chosen_ids = {str(row["id"]) for row in chosen}
        candidates = [i for i, row in enumerate(eligible)
                      if str(row["id"]) not in chosen_ids and allowed(row)]
        if not candidates:
            break
        index = ex.maxmin_hamming_indices(
            [row["pattern"] for row in eligible], 1,
            references=[row["pattern"] for row in chosen], eligible=candidates)[0]
        row = eligible[index]
        chosen.append(row); group_counts[str(row["canonical_group_id"])] += 1
    if len(group_counts) < int(policy["min_parent_groups"]):
        raise RuntimeError("strict parent diversity has too few canonical groups")
    return chosen


def _generate_pool(parents: Sequence[dict[str, Any]], policy: Mapping[str, Any],
                   profile_cfg: Any, blocked: set[str], *, seed: int | None = None
                   ) -> list[dict[str, Any]]:
    target = int(policy["candidate_pool_size"])
    per_origin = {"fresh_random": target // 2,
                  "parent_variant": target - target // 2}
    accepted: dict[str, list[dict[str, Any]]] = {key: [] for key in per_origin}
    seen = set(blocked)
    generation_seed = int(policy["seed"]) if seed is None else _positive_int(seed, "pool seed")
    round_index = 0
    while any(len(accepted[key]) < count for key, count in per_origin.items()):
        missing = max(count - len(accepted[key]) for key, count in per_origin.items())
        request = max(64, 2 * missing)
        generated = ex.generate_candidates(
            parents, request, seed=generation_seed + round_index * 1_000_003, single=True)
        round_index += 1
        for raw in generated:
            group = raw.get("candidate_group")
            if group not in per_origin or len(accepted[group]) >= per_origin[group]:
                continue
            pattern = sm._checked_pattern(raw["pattern"], profile_cfg)
            sha = ex.pattern_sha256(pattern)
            if sha in seen:
                continue
            seen.add(sha)
            row = dict(raw)
            row.update({"pattern": pattern, "pattern_sha256": sha,
                        "proposal": group, "generation": 0})
            if group == "parent_variant":
                parent = next((item for item in parents if str(item["id"]) == str(row["parent_id"])), None)
                if parent is None:
                    raise RuntimeError("candidate generator returned an unknown mutation parent")
                row.update({
                    "source_id": str(parent["id"]),
                    "ancestry_id": str(parent["lineage_id"]),
                    "lineage_id": str(parent["lineage_id"]),
                    "canonical_group_id": str(parent["canonical_group_id"]),
                    "mutation_parent_pattern_sha256": parent["pattern_sha256"],
                    "mutation_parent_actual_wm": float(parent["actual_wm"]),
                })
            else:
                row.update({"source_id": sha, "ancestry_id": sha,
                            "lineage_id": sha, "canonical_group_id": sha})
            accepted[group].append(row)
        if round_index > 100:
            raise RuntimeError("could not fill strict excluded candidate pool")
    rows = accepted["fresh_random"] + accepted["parent_variant"]
    rows.sort(key=lambda row: (row["candidate_group"], row["pattern_sha256"]))
    parent_ids = {str(row["parent_id"]) for row in rows if row["candidate_group"] == "parent_variant"}
    expected_ids = {str(row["id"]) for row in parents}
    if parent_ids != expected_ids:
        raise RuntimeError("candidate generation did not use every selected mutation parent")
    return rows


def _largest_remainder(total: int, order: Sequence[str], weights: Sequence[float]) -> dict[str, int]:
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("selected_count must be a positive integer")
    weight_sum = float(sum(weights))
    raw = [total * float(weight) / weight_sum for weight in weights]
    counts = [int(math.floor(value)) for value in raw]
    remainder = total - sum(counts)
    ranks = sorted(range(len(order)), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in ranks[:remainder]:
        counts[index] += 1
    return dict(zip(order, counts))


def _select(rows: Sequence[dict[str, Any]], selected_count: int,
            policy: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    order = list(policy["arm_order"])
    quotas = _largest_remainder(selected_count, order, policy["arm_weights"])
    p60 = float(np.percentile([row["navigation_mean_score"] for row in rows],
                              float(policy["disagreement_mean_percentile"])))
    pools = {
        "global_lcb": sorted((row for row in rows if row["candidate_group"] == "fresh_random"),
                             key=lambda row: (-row["lcb_score"], row["pattern_sha256"])),
        "parent_lcb": sorted((row for row in rows if row["candidate_group"] == "parent_variant"),
                             key=lambda row: (-row["lcb_score"], row["pattern_sha256"])),
        "high_disagreement": sorted((row for row in rows
                                     if row["navigation_mean_score"] >= p60),
                                    key=lambda row: (-row["response_disagreement"],
                                                     row["pattern_sha256"])),
    }
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    group_counts: Counter[str] = Counter()
    cursors = {arm: 0 for arm in order}
    remaining = dict(quotas)
    minimum = int(policy["min_selected_physical_hamming"])
    cap = int(policy["max_selected_per_canonical_group"])
    while any(remaining.values()):
        progress = False
        for arm in order:
            if remaining[arm] == 0:
                continue
            pool = pools[arm]
            while cursors[arm] < len(pool):
                row = pool[cursors[arm]]; cursors[arm] += 1
                sha, group = row["pattern_sha256"], str(row["canonical_group_id"])
                if sha in used or group_counts[group] >= cap:
                    continue
                if any(ex.hamming(row["pattern"], prior["pattern"]) < minimum for prior in selected):
                    continue
                row["selection_arm"] = arm
                if arm == "high_disagreement":
                    row["disagreement_eligibility_mean_p60"] = p60
                selected.append(row); used.add(sha); group_counts[group] += 1
                remaining[arm] -= 1; progress = True
                break
        if not progress:
            raise RuntimeError(f"strict no-relaxation selection could not fill quotas: {remaining}")
    return selected, {"arm_quotas": quotas, "disagreement_eligibility_mean_p60": p60}


def build_diverse_pool(*, pool_cfg: sm.PoolConfig | Mapping[str, Any],
                       policy: Mapping[str, Any], training_workdir: str | Path,
                       version: int, exclusions: Sequence[str],
                       predictor: sm.Predictor) -> dict[str, Any]:
    """Build the fixed 20k diversity pool and strict equal-weight cohort."""
    policy = _validate_policy(policy)
    cfg = pool_cfg if isinstance(pool_cfg, sm.PoolConfig) else sm.PoolConfig.from_mapping(pool_cfg)
    cfg.validate()
    if cfg.candidate_pool_size != policy["candidate_pool_size"]:
        raise ValueError("PoolConfig candidate_pool_size differs from diversity policy")
    if Path(cfg.profile_config).name != Path(policy["measurement_profile"]).name:
        raise ValueError("PoolConfig measurement profile differs from diversity policy")
    profile_cfg = sm._profile(cfg.profile_config)
    measured, training_binding, tracked_before = _training_snapshot(
        Path(training_workdir), version, profile_cfg)
    generation, valid_count, model_role, predictor_binding = sm._resolve_predictor_binding(
        predictor, cfg, profile_cfg)
    exact_predictor_binding = {
        key: training_binding[key] for key in (
            "measurement_id", "score_spec_id", "protocol_id", "data_version",
            "manifest_id", "cumulative_valid_unique")}
    if (model_role != "current_profile" or predictor_binding is None or
            predictor_binding != exact_predictor_binding):
        raise ValueError("predictor binding differs from the immutable measured parent version")
    queued = {_hex64(str(item), "exclusion hash") for item in exclusions}
    measured_hashes = {row["pattern_sha256"] for row in measured}
    parents = _select_parents(measured, policy)
    blocked = measured_hashes | queued
    rows = _generate_pool(parents, policy, profile_cfg, blocked, seed=cfg.seed)
    if len(rows) != cfg.candidate_pool_size:
        raise RuntimeError(f"candidate pool has {len(rows)}/{cfg.candidate_pool_size} rows")
    sm._score_rows(rows, predictor, cfg)
    selected, selection_audit = _select(rows, cfg.selected_count, policy)
    pool_hashes = [row["pattern_sha256"] for row in rows]
    if (len(pool_hashes) != len(set(pool_hashes)) or set(pool_hashes) & blocked or
            len(selected) != cfg.selected_count):
        raise AssertionError("diverse pool uniqueness/exclusion/count failure")
    root = Path(training_workdir).resolve()
    tracked_after = {name: pb.file_sha256(_safe_file(root, str(path.relative_to(root))))
                     for name, path in ((name, _safe_file(root, row[path_key]))
                                        for row in measured
                                        for name, path_key in ((f"sample:{row['id']}", "sample_file"),
                                                               (f"rad:{row['id']}", "rad_file")))}
    for name in ("state", "protocol", "manifest", "receipt"):
        tracked_after[name] = pb.file_sha256(
            root / ("state.json" if name == "state" else "protocol.json" if name == "protocol"
                    else f"data/data-v{version:03d}/{name}.json"))
    if tracked_after != tracked_before:
        raise ValueError("training cache changed during diverse pool construction")

    parent_records = [{
        "id": row["id"], "pattern_sha256": row["pattern_sha256"],
        "sample_sha256": row["source_sample_sha256"], "rad_sha256": row["source_rad_sha256"],
        "lineage_id": row["lineage_id"], "canonical_group_id": row["canonical_group_id"],
        "actual_wm": float(row["actual_wm"]),
    } for row in parents]
    frozen_ranks = [{
        "rank": index + 1, "pattern_sha256": row["pattern_sha256"],
        "pattern_bits_hex": np.packbits(
            np.asarray(row["pattern"], dtype=np.uint8).reshape(-1)).tobytes().hex(),
        "candidate_group": row["candidate_group"],
        "parent_id": row.get("parent_id"),
        "lineage_id": row["lineage_id"],
        "canonical_group_id": row["canonical_group_id"],
        "selection_arm": row.get("selection_arm"),
        "lcb_score": float(row["lcb_score"]),
        "response_disagreement": float(row["response_disagreement"]),
        "navigation_mean_score": float(row["navigation_mean_score"]),
    } for index, row in enumerate(sorted(rows, key=lambda item: (-item["lcb_score"],
                                                                 item["pattern_sha256"])))]
    audit = {
        "schema_version": 1, "policy_name": policy["name"],
        "policy_sha256": ex.content_id(policy), "strict_no_relaxation": True,
        "policy_seed": policy["seed"], "pool_seed": cfg.seed,
        "training_binding": training_binding,
        "measured_parent_row_count": len(measured),
        "measured_unique_pattern_count": len(measured_hashes),
        "measured_exclusion_sha256": ex.content_id(sorted(measured_hashes)),
        "queued_exclusion_count": len(queued),
        "queued_exclusion_sha256": ex.content_id(sorted(queued)),
        "combined_exclusion_sha256": ex.content_id(sorted(blocked)),
        "selected_parents": parent_records,
        "selected_parent_ids_sha256": ex.content_id([row["id"] for row in parent_records]),
        "selected_parent_patterns_sha256": ex.content_id(
            [row["pattern_sha256"] for row in parent_records]),
        "generation_input_sha256": ex.content_id({
            "policy": ex.content_id(policy), "training": training_binding,
            "parents": parent_records, "exclusions": sorted(blocked), "pool_seed": cfg.seed,
        }),
        "candidate_count": len(rows),
        "candidate_origin_counts": dict(sorted(Counter(
            row["candidate_group"] for row in rows).items())),
        "candidate_pool_sha256": ex.content_id([{
            "pattern_sha256": row["pattern_sha256"],
            "candidate_group": row["candidate_group"],
            "parent_id": row.get("parent_id"),
        } for row in rows]),
        "frozen_lcb_ranks": frozen_ranks,
        "frozen_lcb_ranks_sha256": ex.content_id(frozen_ranks),
        "selection": selection_audit,
        "selected_count": len(selected),
        "selected_arm_counts": dict(sorted(Counter(
            row["selection_arm"] for row in selected).items())),
        "selected_canonical_group_count": len({row["canonical_group_id"] for row in selected}),
        "selected_min_physical_hamming": min(
            ex.hamming(left["pattern"], right["pattern"])
            for i, left in enumerate(selected) for right in selected[i + 1:]) if len(selected) > 1 else None,
        "selected_cohort_sha256": ex.content_id([{
            "pattern_sha256": row["pattern_sha256"], "selection_arm": row["selection_arm"],
            "canonical_group_id": row["canonical_group_id"],
        } for row in selected]),
        "predictor_binding": predictor_binding,
        "model_ids": list(predictor.model_ids),
    }
    return {
        "config": cfg, "profile_cfg": profile_cfg, "rows": rows,
        "selected_rows": selected, "model_ids": list(predictor.model_ids),
        "predictor_model_role": model_role, "predictor_binding": predictor_binding,
        "surrogate_generation": generation, "valid_observations_at_fit": valid_count,
        "legacy_geometry_limitation": None, "diversity_audit": audit,
    }

