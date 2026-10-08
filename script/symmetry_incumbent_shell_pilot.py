"""Pure construction and readout core for the predeclared R80 shell pilot.

This module does not inspect or mutate the factory queue.  The factory owns
capacity, dispatch, recovery and one-shot consumption.  Here, pattern choice is
completed before the sole prospective predictor call.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from antenna.measurement import measurement_id, score_spec_id
from script import exploration as ex
from script import profiled_batch as pb
from script import symmetry_sm_pool as sm


ALGORITHM_ID = "r80_exact_lr_independent_d1_stratified_v1"
REQUEST_KIND = "r80_incumbent_shell_pilot_request"
SHELL_ROLE = "incumbent_shell"
CONTROL_ROLE = "blind_control"
SHELL_ARM = "incumbent_shell_d1"
CONTROL_ARM = "pilot_blind_control"
ROW_BANDS = ((0, 4), (5, 9), (10, 14), (15, 19), (20, 24))
COL_BANDS = ((0, 3), (4, 7), (8, 12))
SELECTION_SALT = "r80-shell-pilot-v1"
EPS2 = (0.30, 0.30)
EPS3 = (0.30, 0.30, 0.50)


class PilotError(ValueError):
    """The pilot request or an immutable pilot artifact is invalid."""


class PilotUnavailable(PilotError):
    """The fixed selection cannot be filled without changing the protocol."""

    VALID_CODES = {"empty_stratum", "no_eligible_blind"}

    def __init__(self, code: str, detail: str):
        if code not in self.VALID_CODES:
            raise ValueError(f"invalid PilotUnavailable code: {code}")
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class PilotSnapshotChanged(PilotError):
    """A source changed during construction; the caller may retry a fresh cycle."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _content_id(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"cannot read JSON: {path}") from exc


def _file_binding(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise PilotError(f"bound file does not exist: {path}")
    return {"path": str(path), "sha256": pb.file_sha256(path)}


def _tree_hashes(root: Path) -> dict[str, str]:
    root = root.resolve()
    if not root.is_dir():
        raise PilotError(f"bound directory does not exist: {root}")
    return {path.relative_to(root).as_posix(): pb.file_sha256(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def _directory_binding(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": _content_id(_tree_hashes(path))}


def _strict_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != required:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise PilotError(f"{label} keys differ: {actual}")


def _bound_file(value: Mapping[str, Any], label: str) -> dict[str, str]:
    _strict_keys(value, {"path", "sha256"}, label)
    path = Path(str(value["path"])).resolve()
    actual = _file_binding(path)
    if dict(value) != actual:
        raise PilotError(f"{label} binding differs")
    return actual


def _bound_directory(value: Mapping[str, Any], label: str) -> dict[str, str]:
    _strict_keys(value, {"path", "sha256"}, label)
    path = Path(str(value["path"])).resolve()
    actual = _directory_binding(path)
    if dict(value) != actual:
        raise PilotError(f"{label} binding differs")
    return actual


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _require_finite_json(value: Any, label: str) -> Any:
    clean = _jsonable(value)
    try:
        _canonical(clean)
    except (TypeError, ValueError) as exc:
        raise PilotError(f"{label} is not finite JSON") from exc
    return clean


@dataclass(frozen=True)
class PilotRequest:
    path: Path
    raw: dict[str, Any]
    request_sha256: str
    pilot_id: str
    binding: dict[str, Any]
    profile_path: Path
    profile: Any
    protocol_binding: dict[str, str]
    addendum_binding: dict[str, str]
    training_protocol_binding: dict[str, str]
    anchor: dict[str, Any]
    blind_pool_binding: dict[str, str]
    priority: int = 1
    count: int = 16


@dataclass(frozen=True)
class PilotSelection:
    rows: tuple[dict[str, Any], ...]
    hashes: tuple[str, ...]
    role_counts: dict[str, int]
    exclusions: tuple[str, ...]
    exclusions_sha256: str


@dataclass(frozen=True)
class AnnotatedSelection:
    rows: tuple[dict[str, Any], ...]
    hashes: tuple[str, ...]
    role_counts: dict[str, int]
    exclusions: tuple[str, ...]
    exclusions_sha256: str
    predictor_binding: dict[str, Any]
    model_ids: tuple[dict[str, Any], ...]
    surrogate_generation: str
    valid_observations_at_fit: int
    model_hash: str


def load_request(path: str | Path, profile_config: str | Path) -> PilotRequest:
    """Load and fully bind a strict private pilot request."""

    request_path = Path(path).resolve()
    try:
        raw_bytes = request_path.read_bytes()
    except OSError as exc:
        raise PilotError(f"cannot read pilot request: {request_path}") from exc
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PilotError("pilot request is not UTF-8 JSON") from exc
    required = {"schema_version", "kind", "algorithm_id", "protocol_binding",
                "addendum_binding", "profile_binding", "training_protocol_binding",
                "anchor", "blind_pool_binding", "campaign", "selection_salt",
                "one_shot"}
    _strict_keys(raw, required, "request")
    if (raw["schema_version"] != 1 or raw["kind"] != REQUEST_KIND or
            raw["algorithm_id"] != ALGORITHM_ID or
            raw["selection_salt"] != SELECTION_SALT or raw["one_shot"] is not True):
        raise PilotError("request identity or one-shot policy differs")

    request_sha = hashlib.sha256(raw_bytes).hexdigest()
    pilot_id = hashlib.sha256(b"r80-incumbent-shell-pilot-v1\0" + raw_bytes).hexdigest()
    protocol_binding = _bound_file(raw["protocol_binding"], "protocol")
    addendum_binding = _bound_file(raw["addendum_binding"], "addendum")
    training_binding = _bound_file(raw["training_protocol_binding"], "training protocol")

    profile_path = Path(profile_config).resolve()
    profile_binding = _file_binding(profile_path)
    if raw["profile_binding"] != profile_binding:
        raise PilotError("profile binding differs")
    profile = pb.load_profile_config(profile_path)
    if profile is None or profile.port != "single":
        raise PilotError("pilot requires a named single-port profile")

    protocol = _read_json(Path(protocol_binding["path"]))
    if (not isinstance(protocol, dict) or protocol.get("schema_version") != 1 or
            protocol.get("algorithm_id") != ALGORITHM_ID or
            protocol.get("scope") != profile.scope):
        raise PilotError("bound protocol identity differs")
    campaign_required = {"target_valid_unique", "max_guided_outstanding",
                         "shard_size", "priority", "planned_kind",
                         "measurement_id", "score_spec_id", "profile_sha256",
                         "training_protocol_sha256"}
    _strict_keys(raw["campaign"], campaign_required, "campaign")
    campaign = raw["campaign"]
    expected_campaign = protocol.get("unchanged_campaign", {})
    dispatch = protocol.get("dispatch", {})
    if not isinstance(expected_campaign, Mapping) or not isinstance(dispatch, Mapping):
        raise PilotError("protocol campaign or dispatch schema differs")
    fixed_campaign_identity = {
        "target_valid_unique": 5000,
        "profile_sha256": profile_binding["sha256"],
        "measurement_id": measurement_id(profile.measurement),
        "score_spec_id": score_spec_id(profile.score_spec),
        "training_protocol_sha256": training_binding["sha256"],
    }
    if any(expected_campaign.get(key) != value
           for key, value in fixed_campaign_identity.items()):
        raise PilotError("protocol unchanged campaign identity differs")
    if campaign != {
            "target_valid_unique": fixed_campaign_identity["target_valid_unique"],
            "max_guided_outstanding": 96,
            "shard_size": int(dispatch.get("rows_per_job", -1)),
            "priority": int(dispatch.get("priority", -1)),
            "planned_kind": dispatch.get("planned_kind"),
            "measurement_id": measurement_id(profile.measurement),
            "score_spec_id": score_spec_id(profile.score_spec),
            "profile_sha256": profile_binding["sha256"],
            "training_protocol_sha256": training_binding["sha256"],
    }:
        raise PilotError("request campaign differs from protocol/profile")
    if (campaign["target_valid_unique"] != 5000 or
            campaign["max_guided_outstanding"] != 96 or
            campaign["shard_size"] != 16 or campaign["priority"] != 1 or
            campaign["planned_kind"] != "p"):
        raise PilotError("fixed campaign gates differ")

    anchor_keys = {"id", "pattern_sha256", "ancestry_id", "source_id",
                   "manifest_path", "manifest_sha256", "tensor_path", "tensor_sha256"}
    _strict_keys(raw["anchor"], anchor_keys, "anchor")
    anchor = dict(raw["anchor"])
    manifest = Path(str(anchor["manifest_path"])).resolve()
    tensor = Path(str(anchor["tensor_path"])).resolve()
    if (_file_binding(manifest)["sha256"] != anchor["manifest_sha256"] or
            _file_binding(tensor)["sha256"] != anchor["tensor_sha256"]):
        raise PilotError("anchor file binding differs")
    protocol_anchor = protocol.get("fixed_anchor", {})
    if not isinstance(protocol_anchor, Mapping):
        raise PilotError("protocol anchor schema differs")
    for key in ("id", "pattern_sha256", "ancestry_id"):
        if anchor[key] != protocol_anchor.get(key):
            raise PilotError(f"anchor {key} differs from protocol")
    manifest_rows = _read_json(manifest)
    if (not isinstance(manifest_rows, list) or
            any(not isinstance(row, Mapping) for row in manifest_rows)):
        raise PilotError("anchor manifest schema differs")
    matching = [row for row in manifest_rows if row.get("id") == anchor["id"]]
    if len(matching) != 1:
        raise PilotError("anchor manifest must contain exactly one bound id")
    source = matching[0]
    if (source.get("pattern_sha256") != anchor["pattern_sha256"] or
            source.get("ancestry_id") != anchor["ancestry_id"] or
            source.get("source_id") != anchor["source_id"]):
        raise PilotError("anchor manifest row differs")
    pattern = torch.load(tensor, weights_only=True, map_location="cpu")
    if pb.validate_pattern(pattern, profile) != anchor["pattern_sha256"]:
        raise PilotError("anchor tensor differs")

    blind_binding = _bound_directory(raw["blind_pool_binding"], "blind pool")
    blind_root = Path(blind_binding["path"])
    blind_cfg = pb.load_profile_config(blind_root / "config.yaml")
    if (blind_cfg is None or blind_cfg.scope != profile.scope or
            measurement_id(blind_cfg.measurement) != measurement_id(profile.measurement) or
            score_spec_id(blind_cfg.score_spec) != score_spec_id(profile.score_spec)):
        raise PilotError("blind pool profile differs")
    pb.validate_input(blind_root, blind_cfg)

    binding = {
        "request_path": str(request_path), "request_sha256": request_sha,
        "pilot_id": pilot_id, "profile": profile_binding,
        "protocol": protocol_binding, "addendum": addendum_binding,
        "training_protocol": training_binding,
    }
    return PilotRequest(
        path=request_path, raw=dict(raw), request_sha256=request_sha, pilot_id=pilot_id,
        binding=binding, profile_path=profile_path, profile=profile,
        protocol_binding=protocol_binding, addendum_binding=addendum_binding,
        training_protocol_binding=training_binding, anchor=anchor,
        blind_pool_binding=blind_binding,
    )


def _recheck_request(request: PilotRequest) -> None:
    if pb.file_sha256(request.path) != request.request_sha256:
        raise PilotError("pilot request changed after binding")
    for key, binding in (("protocol", request.protocol_binding),
                         ("addendum", request.addendum_binding),
                         ("training protocol", request.training_protocol_binding)):
        if _file_binding(Path(binding["path"])) != binding:
            raise PilotError(f"bound {key} changed")
    if pb.file_sha256(request.profile_path) != request.raw["profile_binding"]["sha256"]:
        raise PilotError("bound profile changed")


def _source_state(request: PilotRequest) -> dict[str, Any]:
    try:
        return {
            "anchor_manifest_sha256": pb.file_sha256(Path(request.anchor["manifest_path"])),
            "anchor_tensor_sha256": pb.file_sha256(Path(request.anchor["tensor_path"])),
            "blind_pool": _directory_binding(Path(request.blind_pool_binding["path"])),
        }
    except (OSError, PilotError) as exc:
        raise PilotSnapshotChanged("bound anchor or blind source disappeared") from exc


def _expected_source_state(request: PilotRequest) -> dict[str, Any]:
    return {
        "anchor_manifest_sha256": request.anchor["manifest_sha256"],
        "anchor_tensor_sha256": request.anchor["tensor_sha256"],
        "blind_pool": request.blind_pool_binding,
    }


def select_rows(request: PilotRequest, exclusions: Sequence[str] | set[str]) -> PilotSelection:
    """Choose the complete 15+1 cohort without consulting a predictor."""

    _recheck_request(request)
    before = _source_state(request)
    if before != _expected_source_state(request):
        raise PilotError("bound anchor or blind source differs")
    excluded = set(exclusions)
    if any(not isinstance(value, str) or len(value) != 64 for value in excluded):
        raise PilotError("exclusions must be SHA-256 strings")
    frozen_exclusions = tuple(sorted(excluded))

    anchor = np.asarray(torch.load(Path(request.anchor["tensor_path"]), weights_only=True,
                                   map_location="cpu"), dtype=bool)
    if pb.validate_pattern(anchor, request.profile) != request.anchor["pattern_sha256"]:
        raise PilotError("anchor tensor changed")
    rows: list[dict[str, Any]] = []
    for row_band_index, (r0, r1) in enumerate(ROW_BANDS):
        for col_band_index, (c0, c1) in enumerate(COL_BANDS):
            candidates = []
            for row in range(r0, r1 + 1):
                for col in range(c0, c1 + 1):
                    if (row, col) == (24, 12):
                        continue
                    pattern = anchor.copy()
                    pattern[row, col] = ~pattern[row, col]
                    pattern = ex.enforce_single_geometry(pattern)
                    independent_hamming = int(np.count_nonzero(
                        pattern[:, :13] != anchor[:, :13]))
                    physical_hamming = int(np.count_nonzero(pattern != anchor))
                    if independent_hamming != 1 or physical_hamming != (1 if col == 12 else 2):
                        raise PilotError("exact d1 projection invariant failed")
                    digest = pb.validate_pattern(pattern, request.profile)
                    if digest in excluded or digest == request.anchor["pattern_sha256"]:
                        continue
                    key = hashlib.sha256(
                        (f"{SELECTION_SALT}\0{request.anchor['pattern_sha256']}\0"
                         f"{row:02d},{col:02d}\0{digest}").encode("utf-8")).hexdigest()
                    candidates.append((key, row, col, digest, pattern))
            if not candidates:
                raise PilotUnavailable(
                    "empty_stratum", f"row_band={row_band_index},col_band={col_band_index}")
            _key, row, col, digest, pattern = min(candidates, key=lambda item: item[:4])
            ordinal = len(rows) + 1
            rows.append({
                "id": f"r80p{request.pilot_id[:8]}s{ordinal:02d}_{digest[:8]}",
                "pattern": pattern.copy(), "pattern_sha256": digest,
                "kind": "pilot", "pilot_id": request.pilot_id,
                "pilot_role": SHELL_ROLE, "selection_arm": SHELL_ARM,
                "proposal": "incumbent_shell_d1",
                "lineage_id": request.anchor["ancestry_id"],
                "ancestry_id": request.anchor["ancestry_id"],
                "source_id": request.anchor["source_id"],
                "anchor_id": request.anchor["id"],
                "anchor_pattern_sha256": request.anchor["pattern_sha256"],
                "independent_coordinate": [row, col],
                "row_stratum": row_band_index, "column_stratum": col_band_index,
                "independent_hamming_distance": 1,
                "physical_hamming_distance": physical_hamming,
            })
            excluded.add(digest)

    blind_root = Path(request.blind_pool_binding["path"])
    blind_rows = _read_json(blind_root / "manifest.json")
    control = next((row for row in blind_rows
                    if row.get("pattern_sha256") not in excluded), None)
    if control is None:
        raise PilotUnavailable("no_eligible_blind", "bound blind pool has no eligible row")
    control_tensor = torch.load(blind_root / f"{control['id']}.pt", weights_only=True,
                                map_location="cpu")
    control_pattern = np.asarray(control_tensor, dtype=bool)
    control_hash = pb.validate_pattern(control_pattern, request.profile)
    if control_hash != control.get("pattern_sha256"):
        raise PilotError("blind control tensor differs from manifest")
    lineage = str(control.get("lineage_id") or control.get("ancestry_id") or
                  control.get("source_id") or control["id"])
    source_id = str(control.get("source_id") or control["id"])
    rows.append({
        "id": f"r80p{request.pilot_id[:8]}c00_{control_hash[:8]}",
        "pattern": control_pattern.copy(), "pattern_sha256": control_hash,
        "kind": "pilot", "pilot_id": request.pilot_id,
        "pilot_role": CONTROL_ROLE, "selection_arm": CONTROL_ARM,
        "proposal": str(control.get("candidate_group") or control.get("source") or
                        "blind_control"),
        "lineage_id": lineage,
        "ancestry_id": str(control.get("ancestry_id") or lineage),
        "source_id": source_id,
        "blind_source_id": control["id"],
        "blind_source_pattern_sha256": control_hash,
        "blind_source_metadata": _require_finite_json(control, "blind source metadata"),
    })

    after = _source_state(request)
    if after != before:
        raise PilotSnapshotChanged("anchor or blind pool changed during selection")
    hashes = tuple(row["pattern_sha256"] for row in rows)
    if len(rows) != 16 or len(set(hashes)) != 16:
        raise PilotError("pilot selection is not 16 unique rows")
    role_counts = {SHELL_ROLE: 15, CONTROL_ROLE: 1}
    return PilotSelection(tuple(rows), hashes, role_counts, frozen_exclusions,
                          _content_id(frozen_exclusions))


def _predictor_identity(request: PilotRequest, predictor: Any) -> tuple[dict[str, Any],
                                                                          tuple[dict[str, Any], ...],
                                                                          str, int, str]:
    if getattr(predictor, "model_role", None) != "current_profile":
        raise PilotError("pilot predictor must have current_profile role")
    binding = getattr(predictor, "binding", None)
    required = {"measurement_id", "score_spec_id", "protocol_id", "data_version",
                "manifest_id", "cumulative_valid_unique"}
    _strict_keys(binding, required, "predictor binding")
    binding = _require_finite_json(binding, "predictor binding")
    if (binding["measurement_id"] != measurement_id(request.profile.measurement) or
            binding["score_spec_id"] != score_spec_id(request.profile.score_spec)):
        raise PilotError("predictor profile identity differs")
    if (isinstance(binding["data_version"], bool) or
            not isinstance(binding["data_version"], int) or binding["data_version"] <= 0 or
            isinstance(binding["cumulative_valid_unique"], bool) or
            not isinstance(binding["cumulative_valid_unique"], int) or
            binding["cumulative_valid_unique"] < 0):
        raise PilotError("predictor version or fit count is invalid")
    raw_models = getattr(predictor, "model_ids", None)
    if not isinstance(raw_models, Sequence) or isinstance(raw_models, (str, bytes)) or not raw_models:
        raise PilotError("predictor model_ids must be a nonempty sequence")
    model_ids = tuple(_require_finite_json(dict(value), "predictor model id")
                      for value in raw_models)
    for model in model_ids:
        for key in ("protocol_id", "manifest_id"):
            if key in model and str(model[key]) != str(binding[key]):
                raise PilotError(f"predictor model {key} differs")
        if "data_version" in model and int(model["data_version"]) != binding["data_version"]:
            raise PilotError("predictor model data_version differs")
    generation = f"data-v{binding['data_version']:03d}"
    return binding, model_ids, generation, binding["cumulative_valid_unique"], _content_id(model_ids)


def annotate_selection(request: PilotRequest, selection: PilotSelection,
                       predictor: Any) -> AnnotatedSelection:
    """Call the current predictor exactly once on the already frozen cohort."""

    _recheck_request(request)
    if selection.role_counts != {SHELL_ROLE: 15, CONTROL_ROLE: 1} or len(selection.rows) != 16:
        raise PilotError("selection composition differs")
    frozen_hashes = tuple(row["pattern_sha256"] for row in selection.rows)
    if frozen_hashes != selection.hashes:
        raise PilotError("selection hashes differ from rows")
    binding, model_ids, generation, fit_count, model_hash = _predictor_identity(
        request, predictor)
    patterns = [np.asarray(row["pattern"], dtype=bool).copy() for row in selection.rows]
    prediction = predictor.predict(patterns, batch_size=256)
    scores = sm.score_predictions(prediction)
    response = np.asarray(prediction.response_mean, dtype=np.float64)
    radiation = np.asarray(prediction.radiation, dtype=np.float64)
    theta = np.asarray(prediction.radiation_theta, dtype=np.float64)
    disagreement = np.asarray(prediction.response_disagreement, dtype=np.float64)
    if (response.shape != (16, 2, 17) or radiation.ndim != 3 or
            radiation.shape[:2] != (16, 2) or radiation.shape[2] != len(theta) or
            disagreement.shape != (16,) or
            not all(np.isfinite(v).all() for v in (response, radiation, theta, disagreement))):
        raise PilotError("predictor returned invalid pilot arrays")
    members = None
    if prediction.response_members is not None:
        members = np.asarray(prediction.response_members, dtype=np.float64)
        if members.ndim != 4 or members.shape[1:] != (16, 2, 17) or not np.isfinite(members).all():
            raise PilotError("predictor member curves are invalid")
    annotated = []
    for index, source in enumerate(selection.rows):
        row = {key: _jsonable(value) for key, value in source.items() if key != "pattern"}
        row["pattern"] = patterns[index]
        row.update({
            "request_sha256": request.request_sha256,
            "algorithm_id": ALGORITHM_ID,
            "predicted_before_hfss": True,
            "prediction_is_navigation_only": True,
            "surrogate_generation": generation,
            "valid_observations_at_fit": fit_count,
            "predictor_binding": binding,
            "predictor_model_ids": list(model_ids),
            "model_hash": model_hash,
            "predicted_s11": response[index, 0].tolist(),
            "predicted_gain": response[index, 1].tolist(),
            "predicted_radiation_phi0": radiation[index, 0].tolist(),
            "predicted_radiation_phi90": radiation[index, 1].tolist(),
            "predicted_radiation_theta": theta.tolist(),
            "predicted_response_members": (members[:, index].tolist() if members is not None else None),
        })
        for name, values in scores.items():
            row[name] = float(values[index])
        _require_finite_json({k: v for k, v in row.items() if k != "pattern"}, "annotated row")
        annotated.append(row)
    if tuple(row["pattern_sha256"] for row in annotated) != frozen_hashes:
        raise PilotError("annotation changed selection")
    return AnnotatedSelection(tuple(annotated), frozen_hashes, dict(selection.role_counts),
                              selection.exclusions, selection.exclusions_sha256, binding, model_ids,
                              generation, fit_count, model_hash)


def _completion_payload(root: Path, request: PilotRequest) -> dict[str, Any]:
    rows = _read_json(root / "manifest.json")
    return {
        "schema_version": 1, "pilot_id": request.pilot_id,
        "request_sha256": request.request_sha256,
        "count": len(rows),
        "manifest_sha256": pb.file_sha256(root / "manifest.json"),
        "pilot_audit_sha256": pb.file_sha256(root / "pilot_audit.json"),
        "tensor_sha256": {f"{row['id']}.pt": pb.file_sha256(root / f"{row['id']}.pt")
                           for row in rows},
    }


def write_pilot_bundle(request: PilotRequest, selection: AnnotatedSelection,
                       out: str | Path) -> Path:
    """Atomically publish a complete worker input tree for the annotated cohort."""

    _recheck_request(request)
    destination = Path(out).resolve()
    if destination.exists():
        validate_bundle(destination, request)
        return destination
    if not isinstance(selection, AnnotatedSelection):
        raise PilotError("bundle writing requires an annotated selection")
    if len(selection.rows) != request.count or selection.hashes != tuple(
            row["pattern_sha256"] for row in selection.rows):
        raise PilotError("annotated selection composition differs")
    _predictor_identity_from_annotation(selection)
    source_before = _source_state(request)
    if source_before != _expected_source_state(request):
        raise PilotError("bound source differs before bundle publication")

    pending = destination.with_name(f".{destination.name}.pending-{request.pilot_id[:12]}")
    if pending.exists():
        raise PilotError(f"pending pilot publication already exists: {pending}")
    pending.mkdir(parents=True)
    try:
        shutil.copy2(request.profile_path, pending / "config.yaml")
        pb.atomic_json(pending / "measurement.json", request.profile.measurement)
        pb.atomic_json(pending / "score_spec.json", request.profile.score_spec)
        manifest = []
        for source in selection.rows:
            pattern = np.asarray(source["pattern"], dtype=bool)
            if pb.validate_pattern(pattern, request.profile) != source["pattern_sha256"]:
                raise PilotError("annotated pattern differs")
            torch.save(torch.as_tensor(pattern, dtype=torch.float32), pending / f"{source['id']}.pt")
            manifest.append({key: _jsonable(value) for key, value in source.items()
                             if key != "pattern"} | {
                                 "port": "single",
                                 "measurement_id": measurement_id(request.profile.measurement),
                                 "score_spec_id": score_spec_id(request.profile.score_spec),
                             })
        pb.atomic_json(pending / "manifest.json", manifest)
        audit = {
            "schema_version": 1, "kind": "r80_incumbent_shell_pilot_audit",
            "pilot_id": request.pilot_id, "request_binding": request.binding,
            "algorithm_id": ALGORITHM_ID, "one_shot": True,
            "selection_preceded_prediction": True,
            "prediction_used_for_selection": False,
            "replacement_after_freeze": False,
            "count": 16, "role_counts": selection.role_counts,
            "ordered_ids": [row["id"] for row in manifest],
            "ordered_hashes": list(selection.hashes),
            "exclusion_hashes": list(selection.exclusions),
            "exclusions_sha256": selection.exclusions_sha256,
            "predictor_binding": selection.predictor_binding,
            "model_ids": list(selection.model_ids),
            "model_hash": selection.model_hash,
            "surrogate_generation": selection.surrogate_generation,
            "valid_observations_at_fit": selection.valid_observations_at_fit,
            "profile_binding": request.raw["profile_binding"],
            "protocol_binding": request.protocol_binding,
            "addendum_binding": request.addendum_binding,
            "training_protocol_binding": request.training_protocol_binding,
        }
        pb.atomic_json(pending / "pilot_audit.json", audit)
        pb.atomic_json(pending / "pilot_bundle_complete.json",
                       _completion_payload(pending, request))
        validate_bundle(pending, request)
        if _source_state(request) != source_before:
            raise PilotSnapshotChanged("anchor or blind pool changed during publication")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pending, destination)
    except Exception:
        # Preserve the private pending tree for diagnosis; never overwrite it.
        raise
    return destination


def _predictor_identity_from_annotation(selection: AnnotatedSelection) -> None:
    if (not selection.predictor_binding or not selection.model_ids or
            selection.surrogate_generation !=
            f"data-v{selection.predictor_binding['data_version']:03d}" or
            selection.valid_observations_at_fit !=
            selection.predictor_binding["cumulative_valid_unique"] or
            selection.model_hash != _content_id(selection.model_ids)):
        raise PilotError("annotated predictor identity differs")


def validate_bundle(out: str | Path, request: PilotRequest) -> dict[str, Any]:
    """Validate a staged or queued complete pilot bundle."""

    _recheck_request(request)
    root = Path(out).resolve()
    required_files = {"config.yaml", "measurement.json", "score_spec.json", "manifest.json",
                      "pilot_audit.json", "pilot_bundle_complete.json"}
    if not root.is_dir() or not required_files <= {p.name for p in root.iterdir() if p.is_file()}:
        raise PilotError("pilot bundle is incomplete")
    if pb.file_sha256(root / "config.yaml") != request.raw["profile_binding"]["sha256"]:
        raise PilotError("bundle profile bytes differ")
    cfg = pb.load_profile_config(root / "config.yaml")
    if cfg is None or cfg.scope != request.profile.scope:
        raise PilotError("bundle profile scope differs")
    rows = pb.validate_input(root, cfg)
    if len(rows) != 16:
        raise PilotError("pilot bundle must contain exactly 16 rows")
    hashes = [row.get("pattern_sha256") for row in rows]
    ids = [row.get("id") for row in rows]
    if len(set(hashes)) != 16 or len(set(ids)) != 16:
        raise PilotError("pilot bundle IDs or hashes are not unique")
    roles = [row.get("pilot_role") for row in rows]
    role_counts = {SHELL_ROLE: roles.count(SHELL_ROLE), CONTROL_ROLE: roles.count(CONTROL_ROLE)}
    if role_counts != {SHELL_ROLE: 15, CONTROL_ROLE: 1}:
        raise PilotError("pilot role counts differ")
    saved_response, saved_radiation, saved_theta = [], [], []
    saved_disagreement, saved_members = [], []
    for row in rows:
        if (row.get("pilot_id") != request.pilot_id or
                row.get("request_sha256") != request.request_sha256 or
                row.get("algorithm_id") != ALGORITHM_ID or row.get("kind") != "pilot"):
            raise PilotError("pilot row identity differs")
        if row["pilot_role"] == SHELL_ROLE:
            if (row.get("selection_arm") != SHELL_ARM or
                    row.get("lineage_id") != request.anchor["ancestry_id"] or
                    row.get("ancestry_id") != request.anchor["ancestry_id"] or
                    row.get("anchor_pattern_sha256") != request.anchor["pattern_sha256"]):
                raise PilotError("shell row lineage or role differs")
        elif (row.get("selection_arm") != CONTROL_ARM or
              not row.get("blind_source_id") or not row.get("lineage_id")):
            raise PilotError("control source lineage differs")
        s11 = np.asarray(row.get("predicted_s11"), dtype=float)
        gain = np.asarray(row.get("predicted_gain"), dtype=float)
        phi0 = np.asarray(row.get("predicted_radiation_phi0"), dtype=float)
        phi90 = np.asarray(row.get("predicted_radiation_phi90"), dtype=float)
        theta = np.asarray(row.get("predicted_radiation_theta"), dtype=float)
        if (s11.shape != (17,) or gain.shape != (17,) or theta.ndim != 1 or
                phi0.shape != theta.shape or phi90.shape != theta.shape or
                not all(np.isfinite(value).all()
                        for value in (s11, gain, phi0, phi90, theta))):
            raise PilotError(f"{row['id']}: saved prediction curves differ")
        members = row.get("predicted_response_members")
        if members is not None:
            members = np.asarray(members, dtype=float)
            if (members.ndim != 3 or members.shape[1:] != (2, 17) or
                    not np.isfinite(members).all()):
                raise PilotError(f"{row['id']}: saved member curves differ")
        try:
            disagreement = float(row.get("response_disagreement"))
        except (TypeError, ValueError) as exc:
            raise PilotError(f"{row['id']}: saved disagreement differs") from exc
        if not math.isfinite(disagreement):
            raise PilotError(f"{row['id']}: saved disagreement differs")
        saved_response.append(np.stack((s11, gain)))
        saved_radiation.append(np.stack((phi0, phi90)))
        saved_theta.append(theta)
        saved_disagreement.append(disagreement)
        saved_members.append(members)
        if row.get("predicted_before_hfss") is not True or row.get(
                "prediction_is_navigation_only") is not True:
            raise PilotError("prediction provenance differs")

    audit = _read_json(root / "pilot_audit.json")
    required_audit = {"schema_version", "kind", "pilot_id", "request_binding",
                      "algorithm_id", "one_shot", "selection_preceded_prediction",
                      "prediction_used_for_selection", "replacement_after_freeze", "count",
                      "role_counts", "ordered_ids", "ordered_hashes", "exclusions_sha256",
                      "exclusion_hashes",
                      "predictor_binding", "model_ids", "model_hash", "surrogate_generation",
                      "valid_observations_at_fit", "profile_binding", "protocol_binding",
                      "addendum_binding", "training_protocol_binding"}
    _strict_keys(audit, required_audit, "pilot audit")
    if (audit["pilot_id"] != request.pilot_id or audit["request_binding"] != request.binding or
            audit["schema_version"] != 1 or
            audit["kind"] != "r80_incumbent_shell_pilot_audit" or
            audit["algorithm_id"] != ALGORITHM_ID or audit["one_shot"] is not True or
            audit["ordered_ids"] != ids or audit["ordered_hashes"] != hashes or
            audit["role_counts"] != role_counts or audit["count"] != 16 or
            audit["exclusions_sha256"] != _content_id(audit["exclusion_hashes"]) or
            audit["prediction_used_for_selection"] is not False or
            audit["selection_preceded_prediction"] is not True or
            audit["replacement_after_freeze"] is not False or
            audit["profile_binding"] != request.raw["profile_binding"] or
            audit["protocol_binding"] != request.protocol_binding or
            audit["addendum_binding"] != request.addendum_binding or
            audit["training_protocol_binding"] != request.training_protocol_binding):
        raise PilotError("pilot audit differs from manifest")
    replay = select_rows(request, audit["exclusion_hashes"])
    if (list(replay.exclusions) != audit["exclusion_hashes"] or
            replay.exclusions_sha256 != audit["exclusions_sha256"] or
            list(replay.hashes) != hashes or
            [row["id"] for row in replay.rows] != ids or
            replay.role_counts != role_counts):
        raise PilotError("pilot manifest differs from deterministic selection replay")
    structural_fields = (
        "pilot_role", "selection_arm", "proposal", "lineage_id", "ancestry_id",
        "source_id", "anchor_id", "anchor_pattern_sha256", "independent_coordinate",
        "row_stratum", "column_stratum", "independent_hamming_distance",
        "physical_hamming_distance", "blind_source_id", "blind_source_pattern_sha256",
        "blind_source_metadata",
    )
    for actual, expected in zip(rows, replay.rows):
        for key in structural_fields:
            if actual.get(key) != _jsonable(expected.get(key)):
                raise PilotError(f"{actual['id']}: deterministic selection field {key} differs")
    binding = audit["predictor_binding"]
    predictor_keys = {"measurement_id", "score_spec_id", "protocol_id", "data_version",
                      "manifest_id", "cumulative_valid_unique"}
    _strict_keys(binding, predictor_keys, "bundle predictor binding")
    if (binding["measurement_id"] != measurement_id(request.profile.measurement) or
            binding["score_spec_id"] != score_spec_id(request.profile.score_spec) or
            audit["model_hash"] != _content_id(audit["model_ids"]) or
            audit["surrogate_generation"] != f"data-v{binding['data_version']:03d}" or
            audit["valid_observations_at_fit"] != binding["cumulative_valid_unique"]):
        raise PilotError("pilot audit predictor identity differs")
    for row in rows:
        if (row.get("predictor_binding") != binding or
                row.get("predictor_model_ids") != audit["model_ids"] or
                row.get("model_hash") != audit["model_hash"] or
                row.get("surrogate_generation") != audit["surrogate_generation"] or
                row.get("valid_observations_at_fit") != audit["valid_observations_at_fit"]):
            raise PilotError("manifest predictor identity differs from audit")
    if any(not np.array_equal(saved_theta[0], value) for value in saved_theta[1:]):
        raise PilotError("saved radiation theta grids differ")
    member_batch = None
    if any(value is not None for value in saved_members):
        first_shape = next(value.shape for value in saved_members if value is not None)
        if any(value is None or value.shape != first_shape for value in saved_members):
            raise PilotError("saved member ensemble shapes differ")
        member_batch = np.stack(saved_members, axis=1)
    replayed = sm.score_predictions(sm.PredictionBatch(
        np.stack(saved_response), np.asarray(saved_disagreement),
        np.stack(saved_radiation), saved_theta[0], member_batch))
    for index, row in enumerate(rows):
        for name, values in replayed.items():
            try:
                actual = float(row.get(name))
            except (TypeError, ValueError) as exc:
                raise PilotError(f"{row['id']}: predictor scalar {name} differs") from exc
            if not math.isfinite(actual) or abs(actual - float(values[index])) > 1e-9:
                raise PilotError(f"{row['id']}: predictor scalar {name} differs")
    completion = _read_json(root / "pilot_bundle_complete.json")
    if completion != _completion_payload(root, request):
        raise PilotError("pilot completion binding differs")
    return {
        "pilot_id": request.pilot_id, "request_binding": dict(request.binding),
        "ordered_ids": ids, "ordered_hashes": hashes, "ordered_roles": roles,
        "role_counts": role_counts,
        "predictor_binding": audit["predictor_binding"], "model_ids": audit["model_ids"],
        "model_hash": audit["model_hash"],
        "surrogate_generation": audit["surrogate_generation"],
        "valid_observations_at_fit": audit["valid_observations_at_fit"],
        "manifest_sha256": completion["manifest_sha256"],
        "pilot_audit_sha256": completion["pilot_audit_sha256"],
    }


def _metadata_hashes(input_dir: Path, store: Path) -> dict[str, Any]:
    return {
        "input": {name: pb.file_sha256(input_dir / name) for name in
                  ("config.yaml", "measurement.json", "score_spec.json", "manifest.json")},
        "store": {name: pb.file_sha256(store / name) for name in
                  ("measurement.json", "score_spec.json", "manifest.json", "results.json")},
    }


def _safe_source_file(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise PilotError(f"missing {label} path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PilotError(f"{label} escapes store") from exc
    if not path.is_file():
        raise PilotError(f"missing {label}: {path}")
    return path


def _metric_from_observation(store: Path, row: Mapping[str, Any],
                             entry: Mapping[str, Any], cfg: Any
                             ) -> tuple[dict[str, Any], dict[str, str]]:
    sample = _safe_source_file(store, entry.get("sample_file"), "sample")
    rad_path = _safe_source_file(store, entry.get("rad_file"), "radiation")
    sample_sha, rad_sha = pb.file_sha256(sample), pb.file_sha256(rad_path)
    if sample_sha != entry.get("sample_sha256") or rad_sha != entry.get("rad_sha256"):
        raise PilotSnapshotChanged("observation raw hash changed after physical validation")
    pattern, response = torch.load(sample, weights_only=True, map_location="cpu")
    if pb.pattern_sha256(pattern) != row.get("pattern_sha256"):
        raise PilotError("observation pattern differs from manifest")
    y = np.asarray(response, dtype=np.float64)
    freqs = np.asarray(entry.get("freqs"), dtype=np.float64)
    if (entry.get("labels") != ["S11", "Gain"] or y.shape != (2, len(freqs)) or
            not np.isfinite(y).all() or not np.isfinite(freqs).all()):
        raise PilotError("observation response differs")
    band = (freqs >= 26.5) & (freqs <= 29.5)
    if not np.any(band):
        raise PilotError("observation has no factory band")
    s11 = float(-10.0 - np.max(y[0, band]))
    gain = float(np.min(y[1, band]) - 4.0)
    rad = torch.load(rad_path, weights_only=True, map_location="cpu")
    theta = np.asarray(rad["theta"], dtype=np.float64)
    phi0 = np.asarray(rad["phi0"], dtype=np.float64)
    phi90 = np.asarray(rad["phi90"], dtype=np.float64)
    if (theta.ndim != 1 or phi0.shape != theta.shape or phi90.shape != theta.shape or
            not all(np.isfinite(v).all() for v in (theta, phi0, phi90))):
        raise PilotError("observation radiation differs")
    window = np.abs(theta) <= 45.0
    if not np.any(window):
        raise PilotError("radiation has no +/-45 degree samples")
    replay = pb.observation(response, rad, row, pattern, cfg, entry.get("time_s"))
    if any(entry.get(key) != value for key, value in replay.items()):
        raise PilotError("chosen observation entry differs from bound raw files")
    bore = int(np.argmin(np.abs(theta)))
    radiation = float(min(np.min(phi0[window]) - (phi0[bore] - 3.0),
                          np.min(phi90[window]) - (phi90[bore] - 3.0)))
    if pb.file_sha256(sample) != sample_sha or pb.file_sha256(rad_path) != rad_sha:
        raise PilotSnapshotChanged("observation raw file changed during metric construction")
    metric = {
        "id": str(row["id"]), "pattern_sha256": str(row["pattern_sha256"]),
        "s11_margin_db": s11, "gain_margin_db": gain,
        "factory_margin_db": min(s11, gain), "radiation_margin_db": radiation,
        "radiation_margin_clipped_db": float(np.clip(radiation, -6.0, 3.0)),
    }
    _require_finite_json(metric, "reference metric")
    return metric, {"sample_sha256": sample_sha, "rad_sha256": rad_sha,
                    "sample_path": str(sample), "rad_path": str(rad_path)}


def _frontier(rows: Sequence[Mapping[str, Any]], dimensions: tuple[str, ...]) -> dict[str, Any]:
    front = []
    for row in rows:
        values = tuple(float(row[key]) for key in dimensions)
        if not any(other is not row and
                   all(float(other[key]) >= value for key, value in zip(dimensions, values)) and
                   any(float(other[key]) > value for key, value in zip(dimensions, values))
                   for other in rows):
            front.append(row)
    front = sorted(front, key=lambda row: (row["id"], row["pattern_sha256"]))
    points = [{"id": row["id"], "pattern_sha256": row["pattern_sha256"],
               **{key: float(row[key]) for key in dimensions}} for row in front]
    return {"ids": [row["id"] for row in front], "points": points,
            "sha256": _content_id(points)}


def freeze_reference(pairs: Sequence[tuple[str | Path, str | Path]],
                     pair_bindings: Sequence[Mapping[str, Any]], request: PilotRequest,
                     out: str | Path) -> Path:
    """Freeze the preparation-time scalar frontier with physical raw bindings."""

    _recheck_request(request)
    destination = Path(out).resolve()
    if destination.exists():
        reference = _read_json(destination)
        if reference.get("request_binding") != request.binding:
            raise PilotError("existing reference belongs to another request")
        recheck_reference(destination)
        return destination
    if len(pairs) != len(pair_bindings) or not pairs:
        raise PilotError("reference pairs and bindings must be equal nonempty sequences")

    states = []
    candidates = []
    for pair_index, ((input_value, store_value), supplied) in enumerate(zip(pairs, pair_bindings)):
        input_dir, store = Path(input_value).resolve(), Path(store_value).resolve()
        try:
            before = _metadata_hashes(input_dir, store)
        except OSError as exc:
            raise PilotSnapshotChanged(f"reference pair disappeared: {store}") from exc
        expected = {
            "input": str(input_dir), "store": str(store),
            "input_metadata_sha256": before["input"],
            "store_metadata_sha256": before["store"],
        }
        if dict(supplied) != expected:
            raise PilotSnapshotChanged(f"pair binding differs at index {pair_index}")
        cfg = pb.load_profile_config(input_dir / "config.yaml")
        if (cfg is None or cfg.scope != request.profile.scope or
                measurement_id(cfg.measurement) != measurement_id(request.profile.measurement) or
                score_spec_id(cfg.score_spec) != score_spec_id(request.profile.score_spec)):
            raise PilotError("reference pair profile differs")
        manifest = pb.validate_input(input_dir, cfg)
        store_manifest = _read_json(store / "manifest.json")
        if store_manifest != manifest:
            raise PilotError("reference store manifest differs from input")
        results = _read_json(store / "results.json")
        if not isinstance(results, dict):
            raise PilotError("reference results must be a mapping")
        manifest_ids = {row["id"] for row in manifest}
        if any(name not in manifest_ids or not isinstance(entry, Mapping)
               for name, entry in results.items()):
            raise PilotError("reference results contain an unknown or invalid entry")
        states.append({"pair_index": pair_index, "input": str(input_dir), "store": str(store),
                       "metadata_before": before,
                       "physical_validation_proof": dict(supplied)})
        for row_index, row in enumerate(manifest):
            entry = results.get(row["id"])
            if (not isinstance(entry, Mapping) or entry.get("status") != "ok" or
                    "error" in entry):
                continue
            if (entry.get("measurement_id") != measurement_id(cfg.measurement) or
                    entry.get("score_spec_id") != score_spec_id(cfg.score_spec) or
                    entry.get("pattern_sha256") != row["pattern_sha256"]):
                raise PilotError("successful reference entry identity differs")
            priority = (row.get("kind") in ("repeat", "notarize"), pair_index, row_index)
            candidates.append((row["pattern_sha256"], priority, pair_index, row_index,
                               dict(row), dict(entry)))

    choices: dict[str, tuple[Any, ...]] = {}
    for item in candidates:
        digest, priority = item[0], item[1]
        if digest not in choices or priority < choices[digest][1]:
            choices[digest] = item
    ordered = sorted(choices.values(), key=lambda item: item[1])
    if not ordered:
        raise PilotError("reference has no successful unique observations")

    observations = []
    for digest, priority, pair_index, row_index, row, entry in ordered:
        store = Path(states[pair_index]["store"])
        cfg = pb.load_profile_config(Path(states[pair_index]["input"]) / "config.yaml")
        metric, raw_binding = _metric_from_observation(store, row, entry, cfg)
        observations.append({
            **metric, "representative_priority": list(priority),
            "pair_index": pair_index, "row_index": row_index,
            "manifest_row": _require_finite_json(row, "reference manifest row"),
            "observation_entry": _require_finite_json(entry, "reference observation entry"),
            **raw_binding,
        })
    for state in states:
        after = _metadata_hashes(Path(state["input"]), Path(state["store"]))
        state["metadata_after"] = after
        if after != state["metadata_before"]:
            raise PilotSnapshotChanged(f"pair changed during reference construction: {state['store']}")
    for row in observations:
        if (pb.file_sha256(Path(row["sample_path"])) != row["sample_sha256"] or
                pb.file_sha256(Path(row["rad_path"])) != row["rad_sha256"]):
            raise PilotSnapshotChanged(f"chosen raw file changed during construction: {row['id']}")

    best = max(float(row["factory_margin_db"]) for row in observations)
    reference = {
        "schema_version": 1, "kind": "r80_incumbent_shell_pilot_reference",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "request_binding": request.binding, "pilot_id": request.pilot_id,
        "profile_binding": request.raw["profile_binding"],
        "measurement_id": measurement_id(request.profile.measurement),
        "score_spec_id": score_spec_id(request.profile.score_spec),
        "representative_policy": "normal_before_repeat_notarize_then_pair_row_order",
        "representative_order": [row["pattern_sha256"] for row in observations],
        "valid_unique": len(observations), "B_factory_margin_db": best,
        "relevance_floor_db": best - 3.0,
        "prior_nonnegative": any(float(row["factory_margin_db"]) >= 0 for row in observations),
        "epsilon_db": {"s11": 0.30, "gain": 0.30, "radiation_clipped": 0.50},
        "radiation_clip_db": [-6.0, 3.0],
        "frontier_2d": _frontier(observations, ("s11_margin_db", "gain_margin_db")),
        "frontier_3d": _frontier(
            observations, ("s11_margin_db", "gain_margin_db",
                           "radiation_margin_clipped_db")),
        "pairs": states, "observations": observations,
    }
    reference["reference_id"] = _content_id(reference)
    pb.atomic_json(destination, reference)
    recheck_reference(destination)
    return destination


def recheck_reference(reference_path: str | Path) -> None:
    """Protect frozen representatives while accepting appended later successes."""

    reference = _read_json(Path(reference_path).resolve())
    if reference.get("kind") != "r80_incumbent_shell_pilot_reference":
        raise PilotError("reference kind differs")
    expected_id = reference.get("reference_id")
    check = dict(reference)
    check.pop("reference_id", None)
    if expected_id != _content_id(check):
        raise PilotError("reference content ID differs")
    pairs = reference.get("pairs")
    observations = reference.get("observations")
    if not isinstance(pairs, list) or not isinstance(observations, list):
        raise PilotError("reference sources differ")
    for state in pairs:
        input_dir, store = Path(state["input"]), Path(state["store"])
        current = _metadata_hashes(input_dir, store)
        for side in ("input", "store"):
            for name, digest in state["metadata_before"][side].items():
                if side == "store" and name == "results.json":
                    continue
                if current[side].get(name) != digest:
                    raise PilotError(f"frozen {side} metadata changed: {name}")
    current_results = {index: _read_json(Path(state["store"]) / "results.json")
                       for index, state in enumerate(pairs)}
    for row in observations:
        results = current_results[row["pair_index"]]
        if results.get(row["id"]) != row["observation_entry"]:
            raise PilotError(f"frozen observation entry changed: {row['id']}")
        if (pb.file_sha256(Path(row["sample_path"])) != row["sample_sha256"] or
                pb.file_sha256(Path(row["rad_path"])) != row["rad_sha256"]):
            raise PilotError(f"frozen raw observation changed: {row['id']}")


def _covered(point: tuple[float, ...], prior: Sequence[tuple[float, ...]],
             epsilon: tuple[float, ...]) -> bool:
    return any(all(old[i] >= point[i] - epsilon[i] for i in range(len(epsilon)))
               for old in prior)


def _truth_metric(row: Mapping[str, Any]) -> dict[str, Any]:
    required = {"id", "pattern_sha256", "pilot_role", "s11_margin_db", "gain_margin_db",
                "factory_margin_db", "radiation_margin_db", "radiation_margin_clipped_db"}
    if not required <= set(row):
        raise PilotError("truth metric fields differ")
    out = {key: row[key] for key in required}
    for key in ("s11_margin_db", "gain_margin_db", "factory_margin_db",
                "radiation_margin_db", "radiation_margin_clipped_db"):
        out[key] = float(out[key])
        if not math.isfinite(out[key]):
            raise PilotError("truth metric is nonfinite")
    if abs(out["factory_margin_db"] - min(out["s11_margin_db"], out["gain_margin_db"])) > 1e-9:
        raise PilotError("truth factory margin differs")
    if abs(out["radiation_margin_clipped_db"] -
           float(np.clip(out["radiation_margin_db"], -6.0, 3.0))) > 1e-9:
        raise PilotError("truth clipped radiation differs")
    return out


def readout(truth: Mapping[str, Any], reference: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Apply the fixed reference-only material criteria to terminal pilot truth."""

    if isinstance(reference, (str, Path)):
        recheck_reference(reference)
        ref = _read_json(Path(reference).resolve())
    else:
        ref = dict(reference)
    if ref.get("kind") != "r80_incumbent_shell_pilot_reference":
        raise PilotError("readout reference kind differs")
    check = dict(ref)
    expected_reference_id = check.pop("reference_id", None)
    if expected_reference_id != _content_id(check):
        raise PilotError("readout reference content ID differs")
    bundle = truth.get("bundle_validation")
    bundle_required = {"pilot_id", "request_binding", "ordered_ids", "ordered_hashes",
                       "ordered_roles", "role_counts", "manifest_sha256",
                       "pilot_audit_sha256"}
    if not isinstance(bundle, Mapping) or not bundle_required <= set(bundle):
        raise PilotError("readout requires validated bundle identity")
    ordered_ids = bundle["ordered_ids"]
    ordered_hashes = bundle["ordered_hashes"]
    ordered_roles = bundle["ordered_roles"]
    if (bundle["pilot_id"] != ref.get("pilot_id") or
            not isinstance(bundle["request_binding"], Mapping) or
            dict(bundle["request_binding"]) != ref.get("request_binding") or
            bundle["role_counts"] != {SHELL_ROLE: 15, CONTROL_ROLE: 1} or
            not isinstance(ordered_ids, list) or not isinstance(ordered_hashes, list) or
            not isinstance(ordered_roles, list) or
            len(ordered_ids) != 16 or len(ordered_hashes) != 16 or
            len(ordered_roles) != 16 or
            len(set(ordered_ids)) != 16 or len(set(ordered_hashes)) != 16 or
            ordered_roles.count(SHELL_ROLE) != 15 or
            ordered_roles.count(CONTROL_ROLE) != 1 or
            any(not isinstance(value, str) or not value for value in ordered_ids) or
            any(not isinstance(value, str) or len(value) != 64 for value in ordered_hashes) or
            any(not isinstance(bundle[key], str) or len(bundle[key]) != 64
                for key in ("manifest_sha256", "pilot_audit_sha256"))):
        raise PilotError("validated bundle identity differs")
    expected_rows = dict(zip(ordered_ids, zip(ordered_hashes, ordered_roles)))
    rows_raw = truth.get("rows", [])
    if isinstance(rows_raw, list):
        for row in rows_raw:
            if (not isinstance(row, Mapping) or row.get("id") not in expected_rows or
                    (row.get("pattern_sha256"), row.get("pilot_role")) !=
                    expected_rows.get(row.get("id"))):
                raise PilotError("truth row differs from validated pilot bundle")
    terminal = truth.get("all_terminal") is True
    marker_errors = truth.get("marker_errors")
    result_errors = truth.get("result_errors")
    complete = (truth.get("pilot_id") == ref.get("pilot_id") and terminal and
                marker_errors == 0 and result_errors == 0 and
                isinstance(rows_raw, list) and len(rows_raw) == 16)
    if not complete:
        return {
            "schema_version": 1, "kind": "r80_incumbent_shell_pilot_readout",
            "status": "incomplete", "pilot_id": ref.get("pilot_id"),
            "all_terminal": terminal, "valid_unique": len(rows_raw) if isinstance(rows_raw, list) else 0,
            "marker_errors": marker_errors, "result_errors": result_errors,
            "automatic_repeat": False,
        }
    rows = [_truth_metric(row) for row in rows_raw]
    if (len({row["id"] for row in rows}) != 16 or
            len({row["pattern_sha256"] for row in rows}) != 16 or
            {row["id"]: (row["pattern_sha256"], row["pilot_role"])
             for row in rows} != expected_rows or
            sum(row["pilot_role"] == SHELL_ROLE for row in rows) != 15 or
            sum(row["pilot_role"] == CONTROL_ROLE for row in rows) != 1):
        raise PilotError("terminal truth composition differs")
    prior = ref["observations"]
    prior2 = [(float(row["s11_margin_db"]), float(row["gain_margin_db"])) for row in prior]
    prior3 = [(float(row["s11_margin_db"]), float(row["gain_margin_db"]),
               float(row["radiation_margin_clipped_db"])) for row in prior]
    B = float(ref["B_factory_margin_db"])
    assessed = []
    for row in rows:
        relevant = row["s11_margin_db"] >= B - 3.0 and row["gain_margin_db"] >= B - 3.0
        conditions = {
            "frozen_B_improved_by_at_least_0_30_db": row["factory_margin_db"] - B >= 0.30,
            "relevant_uncovered_2d": relevant and not _covered(
                (row["s11_margin_db"], row["gain_margin_db"]), prior2, EPS2),
            "relevant_uncovered_3d": relevant and not _covered(
                (row["s11_margin_db"], row["gain_margin_db"],
                 row["radiation_margin_clipped_db"]), prior3, EPS3),
            "first_nonnegative_relative_to_frozen_reference": (
                not bool(ref["prior_nonnegative"]) and row["factory_margin_db"] >= 0.0),
        }
        assessed.append({**row, "relevant": relevant,
                         "material_conditions": conditions,
                         "material": any(conditions.values())})
    shell = [row for row in assessed if row["pilot_role"] == SHELL_ROLE]
    control = [row for row in assessed if row["pilot_role"] == CONTROL_ROLE][0]
    positive = any(row["material"] for row in shell)
    return {
        "schema_version": 1, "kind": "r80_incumbent_shell_pilot_readout",
        "status": "shell_positive" if positive else "shell_red",
        "pilot_id": ref["pilot_id"], "reference_id": ref["reference_id"],
        "reference_only": True, "campaign_first_claim": False,
        "all_terminal": True, "valid_unique": 16, "marker_errors": 0,
        "result_errors": 0, "B_factory_margin_db": B,
        "relevance_floor_db": B - 3.0,
        "shell_relevant_count": sum(row["relevant"] for row in shell),
        "shell_material_count": sum(row["material"] for row in shell),
        "shell_zero_relevant_is_red": not positive and not any(row["relevant"] for row in shell),
        "shell_rows": shell, "blind_control": control,
        "blind_control_can_make_shell_positive": False,
        "automatic_repeat": False, "retain_all_valid_truth": True,
        "independent_positive_repeat_needed_for_spec_claim": True,
    }
