"""Additive S21-only surrogate training for the R81 filter exploration.

This module deliberately does not modify :mod:`script.exploration` or the
five-margin ``filter_masked_prior_v1`` protocol.  It consumes an explicit
profiled input/store pair, freezes the currently successful rows with
``profiled_shards.snapshot_successes``, and writes a separate two-member SM2
ensemble below an explicit output directory.

Historical 24--32 GHz responses supervise only the S21 pass-band output.  The
two stop-band outputs are masked and a prior-only checkpoint is never eligible
for worst-margin ranking.  Once finite 3x49 current truth is available, all
three outputs are trained and candidates are ranked by the minimum predicted
margin across the authoritative S21 score specification.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from antenna.measurement import (frequency_grid, measurement_id, score_response,
                                 score_spec_id, validate_measurement,
                                 validate_score_spec)
from script import exploration as ex
from script import filter_prior as fp
from script import filter_training as ft
from script import profiled_batch as pb
from script.profiled_shards import snapshot_successes


PROTOCOL = "filter_s21_masked_prior_v1"
OUTPUT_DIM = 3
ENSEMBLE_SEEDS = (0, 1)
MAX_RANK_CANDIDATES = 4096
SPLIT_FILE = "s21_family_split.json"
DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "dual_r81_s21_discrete.yaml"
DEFAULT_REFERENCE_CONFIG = Path(__file__).parents[1] / "configs" / "dual_r81_wide_filter.yaml"


@dataclass(frozen=True)
class S21Data:
    x: torch.Tensor
    y: torch.Tensor
    rows: tuple[dict[str, Any], ...]
    references: tuple[dict[str, float], ...]
    receipt: dict[str, Any]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _is_sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(char in "0123456789abcdefABCDEF" for char in value))


def load_protocol(config_path: Path = DEFAULT_CONFIG,
                  reference_config_path: Path = DEFAULT_REFERENCE_CONFIG) -> dict[str, Any]:
    """Load the authoritative three-band spec and old S11/S22 references."""
    config_path = Path(config_path).resolve()
    reference_config_path = Path(reference_config_path).resolve()
    cfg = pb.load_profile_config(config_path)
    reference = pb.load_profile_config(reference_config_path)
    if cfg is None or reference is None:
        raise ValueError("S21 and reference configs must be valid profiled configs")
    canonical = pb.load_profile_config(DEFAULT_CONFIG.resolve())
    if canonical is None:
        raise ValueError("repository canonical S21 config is unavailable")
    fields = ("name", "port", "scope", "measurement", "score_spec", "runtime")
    if any(getattr(cfg, field) != getattr(canonical, field) for field in fields):
        raise ValueError("S21 config differs from the repository canonical profile")
    measurement = validate_measurement(cfg.measurement)
    score = validate_score_spec(cfg.score_spec, measurement=measurement)
    names = [band["name"] for band in score["bands"]]
    if names != ["s21_pass", "s21_low_stop", "s21_high_stop"]:
        raise ValueError("authoritative S21 config must contain pass/low-stop/high-stop in order")
    if (measurement["port"] != "dual" or measurement["labels"] != ["S11", "S21", "S22"]
            or not np.array_equal(frequency_grid(measurement), fp.R81_FREQUENCIES_GHZ)
            or any(band["label"] != "S21" for band in score["bands"])):
        raise ValueError("authoritative S21 config must use dual 3x49 S-parameters")
    old_measurement = validate_measurement(reference.measurement)
    old_score = validate_score_spec(reference.score_spec, measurement=old_measurement)
    reference_bands = [band for band in old_score["bands"]
                       if band["name"] in ("s11_match", "s22_match")]
    if [band["name"] for band in reference_bands] != ["s11_match", "s22_match"]:
        raise ValueError("reference config must retain the old S11/S22 bands")
    return {
        "config": cfg,
        "measurement": measurement,
        "score_spec": score,
        "band_names": tuple(names),
        "reference_score_spec": {"name": "s11_s22_reference_only", "bands": reference_bands},
        "config_binding": {"path": str(config_path), "sha256": _sha(config_path)},
        "reference_config_binding": {"path": str(reference_config_path),
                                     "sha256": _sha(reference_config_path)},
    }


def score_s21(response: torch.Tensor | np.ndarray, protocol: Mapping[str, Any],
              *, labels: Sequence[str], freqs_ghz: Sequence[float]) -> np.ndarray:
    """Return the three authoritative S21 margins from complete wide truth."""
    values = np.asarray(response, dtype=np.float32)
    freqs = np.asarray(freqs_ghz, dtype=np.float64)
    if tuple(labels) != ("S11", "S21", "S22") or values.shape != (3, 49):
        raise ValueError("S21 current truth must be labeled finite 3x49 S11/S21/S22")
    if not np.isfinite(values).all() or not np.array_equal(freqs, fp.R81_FREQUENCIES_GHZ):
        raise ValueError("S21 current truth must cover exactly 16..40 GHz / 0.5")
    scored = score_response(values, list(labels), freqs, protocol["score_spec"])
    margins = scored["margins"]
    result = np.asarray([margins[name] for name in protocol["band_names"]], dtype=np.float32)
    if result.shape != (OUTPUT_DIM,) or not np.isfinite(result).all():
        raise ValueError("S21 scoring did not produce three finite margins")
    return result


def _reference_diagnostics(response: np.ndarray, protocol: Mapping[str, Any]) -> dict[str, float]:
    scored = score_response(response, ["S11", "S21", "S22"], fp.R81_FREQUENCIES_GHZ,
                            protocol["reference_score_spec"])
    margins = scored["margins"]
    s21 = response[1]
    freqs = fp.R81_FREQUENCIES_GHZ
    left = (freqs > 20) & (freqs < 26)
    right = (freqs > 30) & (freqs < 36)
    return {
        "s11_match_margin_db": float(margins["s11_match"]),
        "s22_match_margin_db": float(margins["s22_match"]),
        "low_buffer_max_s21_db": float(np.max(s21[left])),
        "high_buffer_max_s21_db": float(np.max(s21[right])),
    }


def _pattern_from_input(input_dir: Path, row: Mapping[str, Any]) -> torch.Tensor:
    relative = row.get("pattern_file")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError(f"input row {row.get('id')} has invalid pattern_file")
    path = (input_dir / relative).resolve()
    try:
        path.relative_to(input_dir.resolve())
    except ValueError as exc:
        raise ValueError("input pattern escapes its directory") from exc
    pattern = torch.as_tensor(torch.load(path, weights_only=True, map_location="cpu"),
                              dtype=torch.float32).reshape(25, 25)
    if pb.pattern_sha256(pattern) != row.get("pattern_sha256"):
        raise ValueError(f"input pattern hash changed: {row.get('id')}")
    return pattern


def load_split_anchor(input_dir: Path, protocol: Mapping[str, Any]) -> tuple[torch.Tensor, list[dict]]:
    """Authenticate the canonical first input used to freeze train/holdout families."""
    input_dir = Path(input_dir).resolve()
    cfg = pb.load_profile_config(input_dir / "config.yaml")
    if cfg is None:
        raise ValueError("split input has no profiled config")
    if (measurement_id(cfg.measurement) != measurement_id(protocol["measurement"]) or
            score_spec_id(cfg.score_spec) != score_spec_id(protocol["score_spec"])):
        raise ValueError("split input does not use the authoritative S21 profile")
    rows = pb.validate_input(input_dir, cfg)
    if not rows:
        raise ValueError("split input is empty")
    patterns = torch.stack([_pattern_from_input(input_dir, row) for row in rows]).reshape(len(rows), -1)
    return patterns, [dict(row) for row in rows]


def _snapshot_token(pairs: Sequence[tuple[Path, Path]]) -> str:
    bindings = []
    for input_dir, store_dir in pairs:
        bindings.append({
            "input": str(input_dir.resolve()),
            "store": str(store_dir.resolve()),
            "input_manifest_sha256": _sha(input_dir / "manifest.json"),
            "store_manifest_sha256": _sha(store_dir / "manifest.json"),
            "store_results_sha256": _sha(store_dir / "results.json"),
        })
    return _id(bindings)


def _validate_snapshot_binding(snapshot: Path) -> dict[str, Any]:
    """Validate the immutable snapshot and its provenance closure."""
    snapshot = Path(snapshot).resolve()
    results = pb.validate_store(snapshot, require_complete=True)
    rows = pb.read_json(snapshot / "manifest.json")
    binding = pb.read_json(snapshot / "source_bindings.json")
    if (binding.get("schema_version") != 1 or binding.get("partial_snapshot") is not True
            or binding.get("source_batches_may_be_incomplete") is not True):
        raise ValueError("S21 snapshot provenance schema/role changed")
    sources = binding.get("sources")
    items = binding.get("items")
    if not isinstance(sources, list) or not isinstance(items, dict):
        raise ValueError("S21 snapshot provenance is incomplete")
    expected_token = _id([{
        "input": source["input"], "store": source["store"],
        "input_manifest_sha256": source["input_metadata_sha256"]["manifest.json"],
        "store_manifest_sha256": source["store_metadata_sha256"]["manifest.json"],
        "store_results_sha256": source["store_metadata_sha256"]["results.json"],
    } for source in sources])
    if snapshot.name != expected_token:
        raise ValueError("S21 snapshot directory/token provenance changed")
    row_ids = [row["id"] for row in rows]
    if set(items) != set(row_ids) or set(results) != set(row_ids):
        raise ValueError("S21 snapshot provenance item set changed")
    for row in rows:
        item = items[row["id"]]
        result = results[row["id"]]
        if (item.get("source_sample_sha256") != result.get("sample_sha256") or
                item.get("source_sample_file") != result.get("sample_file") or
                not Path(item.get("source_input", "")).is_absolute() or
                not Path(item.get("source_store", "")).is_absolute()):
            raise ValueError(f"S21 snapshot provenance changed: {row['id']}")
    return binding


def freeze_successes(pairs: Sequence[tuple[Path, Path]], output_dir: Path) -> Path:
    """Freeze current successes; source stores are allowed to remain incomplete."""
    resolved = [(Path(a).resolve(), Path(b).resolve()) for a, b in pairs]
    if not resolved:
        raise ValueError("training requires at least one --input-store pair")
    token = _snapshot_token(resolved)
    snapshots = Path(output_dir).resolve() / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    target = snapshots / token
    if target.exists():
        _validate_snapshot_binding(target)
        return target
    attempt = snapshots / f".snapshot-{uuid.uuid4().hex}"
    frozen = snapshot_successes(resolved, attempt)
    actual = pb.read_json(frozen / "source_bindings.json")
    actual_token = _id([{
        "input": source["input"], "store": source["store"],
        "input_manifest_sha256": source["input_metadata_sha256"]["manifest.json"],
        "store_manifest_sha256": source["store_metadata_sha256"]["manifest.json"],
        "store_results_sha256": source["store_metadata_sha256"]["results.json"],
    } for source in actual["sources"]])
    if actual_token != token:
        raise ValueError("partial sources changed before their snapshot was published")
    frozen.rename(target)
    _validate_snapshot_binding(target)
    return target


def load_current_snapshot(snapshot: Path, protocol: Mapping[str, Any]) -> S21Data:
    snapshot = Path(snapshot).resolve()
    cfg = pb.load_profile_config(snapshot / "config.yaml")
    if cfg is None:
        raise ValueError("snapshot lacks its profiled config")
    if (measurement_id(cfg.measurement) != measurement_id(protocol["measurement"]) or
            score_spec_id(cfg.score_spec) != score_spec_id(protocol["score_spec"])):
        raise ValueError("snapshot profile differs from authoritative S21 config")
    results = pb.validate_store(snapshot, require_complete=True)
    rows = pb.read_json(snapshot / "manifest.json")
    bindings = _validate_snapshot_binding(snapshot)
    xs, ys, references, receipts = [], [], [], []
    for row in rows:
        entry = results[row["id"]]
        sample_path = (snapshot / entry["sample_file"]).resolve()
        raw = sample_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sample_sha256"]:
            raise ValueError(f"snapshot sample hash changed: {row['id']}")
        pattern, response = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
        pattern = torch.as_tensor(pattern, dtype=torch.float32).reshape(25, 25)
        if pb.pattern_sha256(pattern) != row["pattern_sha256"]:
            raise ValueError(f"snapshot pattern hash changed: {row['id']}")
        response = np.asarray(response, dtype=np.float32)
        target = score_s21(response, protocol, labels=entry.get("labels", ["S11", "S21", "S22"]),
                           freqs_ghz=entry.get("freqs", fp.R81_FREQUENCIES_GHZ))
        xs.append(pattern.reshape(-1)); ys.append(torch.from_numpy(target))
        references.append(_reference_diagnostics(response, protocol))
        receipts.append({"id": row["id"], "pattern_sha256": row["pattern_sha256"],
                         "sample_sha256": entry["sample_sha256"],
                         "lineage_id": row.get("lineage_id"),
                         "canonical_group_id": row.get("canonical_group_id")})
    receipt = {
        "snapshot": str(snapshot), "manifest_sha256": _sha(snapshot / "manifest.json"),
        "results_sha256": _sha(snapshot / "results.json"),
        "source_bindings_sha256": _sha(snapshot / "source_bindings.json"),
        "samples_id": _id(receipts), "sample_count": len(receipts),
        "source_partial_snapshot": bool(bindings.get("partial_snapshot")),
    }
    return S21Data(torch.stack(xs), torch.stack(ys), tuple(dict(row) for row in rows),
                   tuple(references), receipt)


def _combined_score(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": "s21_prior_projection_with_reference_bands", "bands": [
        *protocol["reference_score_spec"]["bands"], *protocol["score_spec"]["bands"]]}


def load_historical_prior(protocol: Mapping[str, Any], manifest: Path, root: Path,
                          expected_sha256: str, holdout_rows: Sequence[Mapping[str, Any]]):
    manifest, root = Path(manifest).resolve(), Path(root).resolve()
    if not _is_sha(expected_sha256) or _sha(manifest) != expected_sha256.lower():
        raise ValueError("historical manifest SHA-256 differs from expected value")
    prior = fp.prepare_filter_prior(protocol["measurement"], _combined_score(protocol),
                                    manifest, root, current_holdout=holdout_rows)
    names = list(prior.band_names)
    indices = [names.index(name) for name in protocol["band_names"]]
    y = torch.as_tensor(prior.margins[:, indices], dtype=torch.float32)
    mask = torch.as_tensor(prior.margin_masks[:, indices], dtype=torch.bool)
    expected_mask = torch.tensor([True, False, False], dtype=torch.bool)
    if not torch.equal(mask, expected_mask.expand_as(mask)):
        raise ValueError("historical prior must supervise only the S21 pass output")
    rows = [{"id": row["id"], "pattern_sha256": row["pattern_sha256"],
             "sample_sha256": row["source_sample_sha256"],
             "raw_response_sha256": row["raw_response_sha256"]} for row in prior.records]
    receipt = {
        "manifest": str(manifest), "manifest_sha256": expected_sha256.lower(),
        "sample_root": str(root), "kept": len(prior.records),
        "excluded": len(prior.exclusions), "validated_samples_id": _id(rows),
        "exclusions_id": _id(list(prior.exclusions)),
        "target_names": list(protocol["band_names"]),
        "target_mask": [True, False, False],
        "historical_stop_truth_available": False,
        "historical_full_three_band_worst_margin_available": False,
        "audit": prior.audit,
    }
    x = torch.as_tensor(prior.patterns, dtype=torch.float32).reshape(len(prior.patterns), -1)
    return x, y, mask, receipt


def _split(anchor_rows: Sequence[Mapping[str, Any]], current_rows: Sequence[Mapping[str, Any]],
           output_dir: Path, fraction: float, seed: int) -> tuple[list[int], list[int], dict]:
    path = Path(output_dir) / SPLIT_FILE
    existing = {}
    if path.is_file():
        previous = pb.read_json(path)
        if (previous.get("protocol") != PROTOCOL or previous.get("seed") != seed or
                previous.get("holdout_fraction") != fraction):
            raise ValueError("S21 split ledger recipe changed")
        existing = previous["assignments"]
    _at, _ah, assignments = ft.plan_split(anchor_rows, fraction, seed, existing)
    train, holdout, assignments = ft.plan_split(current_rows, fraction, seed, assignments)
    document = {"schema_version": 1, "protocol": PROTOCOL, "seed": seed,
                "holdout_fraction": fraction,
                "rule": "connected physical/family aliases; assignments never move",
                "anchor_count": len(anchor_rows), "assignments": dict(sorted(assignments.items()))}
    if path.is_file() and any(existing.get(key) not in (None, value)
                              for key, value in assignments.items()):
        raise ValueError("S21 split ledger would move an existing family")
    return train, holdout, document


def _write_split(output_dir: Path, document: Mapping[str, Any]) -> None:
    """Publish a validated split without ever moving an existing alias."""
    path = Path(output_dir) / SPLIT_FILE
    if path.is_file():
        previous = pb.read_json(path)
        old = previous.get("assignments", {})
        if any(old.get(key) not in (None, value)
               for key, value in document["assignments"].items()):
            raise ValueError("S21 split ledger would move an existing family")
    ex._atomic_json(path, dict(document))


def _checkpoint_ok(saved: Mapping[str, Any], signature: str, seed: int, *, rank_ready: bool,
                   trained_mask: Sequence[bool], require_complete: bool = True):
    required = {"schema_version": 1, "training_protocol": PROTOCOL,
                "output_dim": OUTPUT_DIM, "signature": signature,
                "member_seed": seed, "rank_ready": rank_ready,
                "trained_output_mask": list(trained_mask)}
    if any(saved.get(key) != value for key, value in required.items()):
        raise ValueError("S21 checkpoint recipe/member/ranking binding mismatch")
    if require_complete and (saved.get("complete") is not True or
                             saved.get("epoch") != saved.get("epochs")):
        raise ValueError("S21 checkpoint is incomplete or has an invalid epoch receipt")


def _save_checkpoint(path: Path, *, model, optimizer, signature: str, seed: int,
                     epoch: int, epochs: int, mean, std, rank_ready: bool,
                     trained_mask: Sequence[bool], parent_sha256: str | None = None):
    payload = {
        "schema_version": 1, "training_protocol": PROTOCOL, "output_dim": OUTPUT_DIM,
        "signature": signature, "member_seed": seed, "epoch": epoch,
        "epochs": epochs, "complete": epoch >= epochs, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(), "mean": mean.cpu(), "std": std.cpu(),
        "rank_ready": rank_ready, "trained_output_mask": list(trained_mask),
        "historical_full_three_band_worst_margin_available": False,
        "parent_prior_sha256": parent_sha256,
    }
    ex._atomic_torch(path, payload)


def _train_phase(model_dir: Path, *, x, y, mask, mean, std, hidden_dims,
                 learning_rate, batch_size, epochs, signature, rank_ready,
                 trained_mask, parents: Mapping[int, Path] | None = None,
                 interrupt_after_epochs: int | None = None) -> list[Path]:
    paths, completed = [], 0
    for member_seed in ENSEMBLE_SEEDS:
        torch.manual_seed(member_seed); np.random.seed(member_seed)
        model = ex.CurveMLP(625, hidden_dims, OUTPUT_DIM)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        parent_sha = None
        if parents is not None:
            parent = parents[member_seed]
            saved_parent = torch.load(parent, weights_only=False, map_location="cpu")
            if not saved_parent.get("complete") or saved_parent.get("rank_ready"):
                raise ValueError("current fine-tune requires a complete prior-only parent")
            model.load_state_dict(saved_parent["model_state"])
            optimizer.load_state_dict(saved_parent["optimizer_state"])
            parent_sha = _sha(parent)
        path = model_dir / f"member-{member_seed}.pt"
        start = 0
        if path.is_file():
            saved = torch.load(path, weights_only=False, map_location="cpu")
            _checkpoint_ok(saved, signature, member_seed, rank_ready=rank_ready,
                           trained_mask=trained_mask,
                           require_complete=False)
            if (not torch.equal(saved["mean"], mean) or not torch.equal(saved["std"], std)
                    or saved.get("parent_prior_sha256") != parent_sha):
                raise ValueError("S21 checkpoint normalization/parent binding mismatch")
            if saved.get("complete"):
                paths.append(path); continue
            model.load_state_dict(saved["model_state"]); optimizer.load_state_dict(saved["optimizer_state"])
            torch.set_rng_state(saved["torch_rng_state"]); np.random.set_state(saved["numpy_rng_state"])
            start = int(saved["epoch"])
        for epoch in range(start, epochs):
            model.train()
            order = ex._epoch_order(len(x), member_seed, 0, epoch)
            for offset in range(0, len(order), batch_size):
                idx = torch.as_tensor(order[offset:offset + batch_size], dtype=torch.long)
                prediction = model((x[idx] - mean) / std)
                loss = ft.masked_mse(prediction, y[idx], mask[idx])
                if not torch.isfinite(loss):
                    raise ValueError("non-finite S21 training loss")
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            _save_checkpoint(path, model=model, optimizer=optimizer, signature=signature,
                             seed=member_seed, epoch=epoch + 1, epochs=epochs,
                             mean=mean, std=std, rank_ready=rank_ready,
                             trained_mask=trained_mask, parent_sha256=parent_sha)
            completed += 1
            if interrupt_after_epochs is not None and completed >= interrupt_after_epochs:
                raise ex.TrainingInterrupted("intentional S21 epoch-boundary interruption")
        paths.append(path)
    return paths


def pretrain(*, split_input: Path, output_dir: Path, prior_manifest: Path,
             prior_root: Path, expected_prior_sha256: str, config: Path = DEFAULT_CONFIG,
             reference_config: Path = DEFAULT_REFERENCE_CONFIG, pretrain_epochs: int = 30,
             batch_size: int = 32, learning_rate: float = 1e-3,
             hidden_dims: Sequence[int] = (128, 128), holdout_fraction: float = .2,
             seed: int = 811007, current_rows: Sequence[Mapping[str, Any]] = (),
             interrupt_after_epochs: int | None = None) -> tuple[list[Path], dict]:
    """Train/reuse the pass-only historical ensemble; ranking remains disabled."""
    if (int(pretrain_epochs) < 1 or int(batch_size) < 1 or float(learning_rate) <= 0 or
            not 0 < float(holdout_fraction) < 1 or tuple(hidden_dims) == () or
            any(int(value) < 1 for value in hidden_dims)):
        raise ValueError("invalid S21 pretraining recipe")
    protocol = load_protocol(config, reference_config)
    output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    anchor_x, anchor_rows = load_split_anchor(split_input, protocol)
    current_rows = list(current_rows) or list(anchor_rows)
    _train, current_hold, split = _split(anchor_rows, current_rows, output_dir,
                                         holdout_fraction, seed)
    # Historical leakage is defined against every frozen holdout alias, including
    # the canonical first 60 even before their wide observations finish.
    _anchor_train, anchor_hold, _ = ft.plan_split(anchor_rows, holdout_fraction, seed,
                                                   split["assignments"])
    holdout_rows = [anchor_rows[index] for index in anchor_hold]
    seen_holdout = {row["id"] for row in holdout_rows}
    holdout_rows += [current_rows[index] for index in current_hold
                     if current_rows[index]["id"] not in seen_holdout]
    prior_x, prior_y, prior_mask, prior_receipt = load_historical_prior(
        protocol, prior_manifest, prior_root, expected_prior_sha256, holdout_rows)
    anchor_train, _, _ = ft.plan_split(anchor_rows, holdout_fraction, seed,
                                       split["assignments"])
    if not anchor_train:
        raise ValueError("S21 prior normalization needs at least one anchor-train row")
    mean, std = ex._fit_norm(anchor_x[anchor_train].float())
    plan = {
        "schema_version": 1, "training_protocol": PROTOCOL, "stage": "historical_pass_only",
        "rank_ready": False, "trained_output_mask": [True, False, False],
        "historical_full_three_band_worst_margin_available": False,
        "protocol": {"criteria": protocol["score_spec"],
                     "config": protocol["config_binding"],
                     "s11_s22_reference": protocol["reference_config_binding"]},
        "split": split, "prior": prior_receipt,
        "recipe": {"epochs": int(pretrain_epochs), "batch_size": int(batch_size),
                   "learning_rate": float(learning_rate), "hidden_dims": list(hidden_dims),
                   "ensemble_seeds": list(ENSEMBLE_SEEDS)},
    }
    signature = _id(plan)
    model_dir = output_dir / "prior" / signature[:16]
    if (model_dir / "training_plan.json").is_file() and pb.read_json(model_dir / "training_plan.json") != plan:
        raise ValueError("existing S21 prior plan changed")
    _write_split(output_dir, split)
    ex._atomic_json(model_dir / "training_plan.json", plan)
    paths = _train_phase(model_dir, x=prior_x, y=prior_y, mask=prior_mask,
                         mean=mean, std=std, hidden_dims=hidden_dims,
                         learning_rate=learning_rate, batch_size=batch_size,
                         epochs=pretrain_epochs, signature=signature, rank_ready=False,
                         trained_mask=[True, False, False],
                         interrupt_after_epochs=interrupt_after_epochs)
    summary = {**plan, "signature": signature,
               "model_files": [path.name for path in paths],
               "model_sha256": {path.name: _sha(path) for path in paths}}
    ex._atomic_json(model_dir / "training_summary.json", summary)
    _write_completion_receipt(model_dir, summary)
    return paths, summary


def train(*, input_store_pairs: Sequence[tuple[Path, Path]], split_input: Path,
          output_dir: Path, prior_manifest: Path, prior_root: Path,
          expected_prior_sha256: str, config: Path = DEFAULT_CONFIG,
          reference_config: Path = DEFAULT_REFERENCE_CONFIG, min_successes: int = 20,
          pretrain_epochs: int = 30, epochs: int = 100, batch_size: int = 32,
          learning_rate: float = 1e-3, hidden_dims: Sequence[int] = (128, 128),
          holdout_fraction: float = .2, seed: int = 811007,
          interrupt_after_epochs: int | None = None) -> tuple[list[Path], dict]:
    """Freeze partial successes and train a cumulative three-margin ensemble."""
    if int(min_successes) < 20:
        raise ValueError("S21 cumulative training requires min_successes >= 20")
    if int(epochs) < 1:
        raise ValueError("S21 cumulative training requires positive epochs")
    output_dir = Path(output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(config, reference_config)
    snapshot = freeze_successes(input_store_pairs, output_dir)
    current = load_current_snapshot(snapshot, protocol)
    if len(current.rows) < int(min_successes):
        raise ValueError(f"only {len(current.rows)} successes; need at least {int(min_successes)}")
    anchor_x, anchor_rows = load_split_anchor(split_input, protocol)
    train_idx, hold_idx, split = _split(anchor_rows, current.rows, output_dir,
                                        holdout_fraction, seed)
    if not train_idx or not hold_idx:
        raise ValueError("S21 cumulative training needs current train and holdout families")
    parents, prior_summary = pretrain(
        split_input=split_input, output_dir=output_dir, prior_manifest=prior_manifest,
        prior_root=prior_root, expected_prior_sha256=expected_prior_sha256,
        config=config, reference_config=reference_config, pretrain_epochs=pretrain_epochs,
        batch_size=batch_size, learning_rate=learning_rate, hidden_dims=hidden_dims,
        holdout_fraction=holdout_fraction, seed=seed, current_rows=current.rows)
    train_x = current.x[train_idx].float(); train_y = current.y[train_idx].float()
    current_mask = torch.ones_like(train_y, dtype=torch.bool)
    anchor_train, _anchor_hold, _ = ft.plan_split(
        anchor_rows, holdout_fraction, seed, split["assignments"])
    mean, std = ex._fit_norm(anchor_x[anchor_train].float())
    parent_map = {seed_value: path for seed_value, path in zip(ENSEMBLE_SEEDS, parents)}
    plan = {
        "schema_version": 1, "training_protocol": PROTOCOL, "stage": "cumulative_wide_truth",
        "rank_ready": True, "ranking": "descending minimum of the three S21 margins",
        "target_names": list(protocol["band_names"]), "trained_output_mask": [True, True, True],
        "historical_full_three_band_worst_margin_available": False,
        "protocol": {"criteria": protocol["score_spec"], "config": protocol["config_binding"],
                     "s11_s22_reference": protocol["reference_config_binding"]},
        "snapshot": current.receipt, "split": split,
        "train_ids": [current.rows[i]["id"] for i in train_idx],
        "holdout_ids": [current.rows[i]["id"] for i in hold_idx],
        "prior_signature": prior_summary["signature"],
        "prior_model_sha256": prior_summary["model_sha256"],
        "recipe": {"epochs": int(epochs), "batch_size": int(batch_size),
                   "learning_rate": float(learning_rate), "hidden_dims": list(hidden_dims),
                   "ensemble_seeds": list(ENSEMBLE_SEEDS)},
    }
    signature = _id(plan)
    model_dir = output_dir / "models" / f"rows-{len(current.rows):05d}-{signature[:12]}"
    ex._atomic_json(model_dir / "training_plan.json", plan)
    paths = _train_phase(model_dir, x=train_x, y=train_y, mask=current_mask,
                         mean=mean, std=std, hidden_dims=hidden_dims,
                         learning_rate=learning_rate, batch_size=batch_size,
                         epochs=epochs, signature=signature, rank_ready=True,
                         trained_mask=[True, True, True], parents=parent_map,
                         interrupt_after_epochs=interrupt_after_epochs)
    predictions = []
    for member_seed, path in zip(ENSEMBLE_SEEDS, paths):
        saved = torch.load(path, weights_only=False, map_location="cpu")
        _checkpoint_ok(saved, signature, member_seed, rank_ready=True,
                       trained_mask=[True, True, True])
        model = ex.CurveMLP(625, hidden_dims, OUTPUT_DIM)
        model.load_state_dict(saved["model_state"]); model.eval()
        with torch.no_grad():
            predictions.append(model((current.x[hold_idx] - mean) / std))
    prediction = torch.stack(predictions).mean(0)
    target = current.y[hold_idx]
    absolute = torch.abs(prediction - target)
    if not torch.isfinite(absolute).all():
        raise ValueError("non-finite S21 holdout evidence")
    summary = {**plan, "signature": signature,
               "model_files": [path.name for path in paths],
               "model_sha256": {path.name: _sha(path) for path in paths},
               "validation": {
                   "source": "current_wide_holdout_only", "used_for_model_selection": False,
                   "prediction": prediction.tolist(), "target": target.tolist(),
                   "mae_by_band": dict(zip(protocol["band_names"], absolute.mean(0).tolist())),
                   "worst_three_margin_mae_db": float(torch.abs(prediction.min(1).values -
                                                                target.min(1).values).mean()),
                   "s11_s22_reference_and_soft_buffers": [current.references[i] for i in hold_idx],
               }}
    ex._atomic_json(model_dir / "training_summary.json", summary)
    _write_completion_receipt(model_dir, summary)
    return paths, summary


def _completion_receipt(model_dir: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "training_protocol": PROTOCOL,
            "signature": summary["signature"],
            "training_plan_sha256": _sha(model_dir / "training_plan.json"),
            "training_summary_sha256": _sha(model_dir / "training_summary.json"),
            "model_sha256": dict(summary["model_sha256"])}


def _write_completion_receipt(model_dir: Path, summary: Mapping[str, Any]) -> Path:
    path = model_dir / "completion_receipt.json"
    ex._atomic_json(path, _completion_receipt(model_dir, summary))
    return path


def _validated_model_summary(model_dir: Path, *, rank_ready: bool,
                             expected_completion_sha256: str) -> dict[str, Any]:
    plan = pb.read_json(model_dir / "training_plan.json")
    summary = pb.read_json(model_dir / "training_summary.json")
    receipt_path = model_dir / "completion_receipt.json"
    if not _is_sha(expected_completion_sha256) or _sha(receipt_path) != expected_completion_sha256.lower():
        raise ValueError("S21 completion receipt differs from its external binding")
    receipt = pb.read_json(receipt_path)
    signature = _id(plan)
    if (summary.get("signature") != signature or
            any(summary.get(key) != value for key, value in plan.items()) or
            summary.get("rank_ready") is not rank_ready):
        raise ValueError("S21 training plan/summary/signature binding changed")
    expected_files = [f"member-{seed}.pt" for seed in ENSEMBLE_SEEDS]
    if summary.get("model_files") != expected_files or set(summary.get("model_sha256", {})) != set(expected_files):
        raise ValueError("S21 ensemble inventory changed")
    if receipt != _completion_receipt(model_dir, summary):
        raise ValueError("S21 completion receipt content changed")
    return summary


def predict_and_rank(model_dir: Path, candidates: Sequence[Mapping[str, Any]], *,
                     expected_completion_sha256: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return predictions, min-of-three order, and a bound candidate receipt."""
    model_dir = Path(model_dir)
    summary = _validated_model_summary(
        model_dir, rank_ready=True, expected_completion_sha256=expected_completion_sha256)
    if (summary.get("training_protocol") != PROTOCOL or not summary.get("rank_ready") or
            summary.get("trained_output_mask") != [True, True, True]):
        raise ValueError("S21 ranking requires three-output wide-truth training")
    hidden = summary["recipe"]["hidden_dims"]
    if not candidates:
        raise ValueError("S21 ranking requires a non-empty candidate set")
    if len(candidates) > MAX_RANK_CANDIDATES:
        raise ValueError(f"S21 candidate count exceeds {MAX_RANK_CANDIDATES}")
    ids, hashes, patterns = [], [], []
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str) or not raw["id"]:
            raise ValueError(f"S21 candidate {index} lacks an id")
        pattern = np.asarray(raw.get("pattern"))
        if (pattern.shape != (25, 25) or not np.isfinite(pattern).all() or
                not np.isin(pattern, [0, 1]).all() or
                not pattern[:5, 10:15].all() or not pattern[20:, 10:15].all()):
            raise ValueError(f"S21 candidate {raw['id']} is not a valid dual pattern")
        digest = pb.pattern_sha256(pattern)
        if raw.get("pattern_sha256") != digest:
            raise ValueError(f"S21 candidate {raw['id']} pattern hash changed")
        ids.append(raw["id"]); hashes.append(digest); patterns.append(pattern.astype(np.float32))
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise ValueError("S21 candidates must have unique ids and physical patterns")
    x = torch.as_tensor(np.asarray(patterns), dtype=torch.float32).reshape(len(patterns), -1)
    predictions = []
    for member_seed in ENSEMBLE_SEEDS:
        path = model_dir / f"member-{member_seed}.pt"
        if _sha(path) != summary["model_sha256"].get(path.name):
            raise ValueError("S21 ensemble file hash changed")
        saved = torch.load(path, weights_only=False, map_location="cpu")
        _checkpoint_ok(saved, summary["signature"], member_seed, rank_ready=True,
                       trained_mask=[True, True, True])
        model = ex.CurveMLP(625, hidden, OUTPUT_DIM)
        model.load_state_dict(saved["model_state"]); model.eval()
        with torch.no_grad():
            predictions.append(model((x - saved["mean"]) / saved["std"]).numpy())
    mean = np.stack(predictions).mean(0)
    if not np.isfinite(mean).all():
        raise ValueError("non-finite S21 ensemble prediction")
    order = np.argsort(-mean.min(axis=1), kind="stable")
    receipt = {"schema_version": 1, "training_protocol": PROTOCOL,
               "model_signature": summary["signature"],
               "completion_receipt_sha256": expected_completion_sha256.lower(),
               "model_sha256": summary["model_sha256"],
               "candidate_set_id": _id([{"id": name, "pattern_sha256": digest}
                                         for name, digest in zip(ids, hashes)]),
               "candidate_count": len(ids), "candidate_ids": ids,
               "ranking": "descending minimum of the three S21 margins",
               "ordered_ids": [ids[index] for index in order.tolist()],
               "predicted_worst_margin": [float(mean[index].min()) for index in order.tolist()]}
    return mean, order, receipt


def readiness(*, split_input: Path, prior_manifest: Path, prior_root: Path,
              expected_prior_sha256: str, config: Path = DEFAULT_CONFIG,
              reference_config: Path = DEFAULT_REFERENCE_CONFIG,
              holdout_fraction: float = .2, seed: int = 811007) -> dict[str, Any]:
    """Read-only physical preflight for the config, anchor, and masked prior."""
    protocol = load_protocol(config, reference_config)
    anchor_x, anchor_rows = load_split_anchor(split_input, protocol)
    train_idx, hold_idx, assignments = ft.plan_split(anchor_rows, holdout_fraction, seed)
    if not train_idx or not hold_idx:
        raise ValueError("S21 readiness needs anchor train and holdout families")
    prior_x, _prior_y, prior_mask, prior_receipt = load_historical_prior(
        protocol, prior_manifest, prior_root, expected_prior_sha256,
        [anchor_rows[index] for index in hold_idx])
    mean, std = ex._fit_norm(anchor_x[train_idx].float())
    return {
        "status": "ready", "training_protocol": PROTOCOL,
        "config": protocol["config_binding"],
        "reference_config": protocol["reference_config_binding"],
        "criteria": protocol["score_spec"], "anchor_count": len(anchor_rows),
        "anchor_train_count": len(train_idx), "anchor_holdout_count": len(hold_idx),
        "split_assignments_id": _id(dict(sorted(assignments.items()))),
        "prior": prior_receipt, "prior_shape": list(prior_x.shape),
        "prior_target_mask": prior_mask[0].tolist(),
        "normalization_finite": bool(torch.isfinite(mean).all() and torch.isfinite(std).all()),
        "rank_ready": False,
        "reason": "historical prior has no observed low/high stop-band truth",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S21-only historical + cumulative SM2 trainer")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("readiness", "pretrain", "train"):
        item = sub.add_parser(name)
        item.add_argument("--split-input", type=Path, required=True)
        if name != "readiness":
            item.add_argument("--output-dir", type=Path, required=True)
        item.add_argument("--prior-manifest", type=Path, required=True)
        item.add_argument("--prior-root", type=Path, required=True)
        item.add_argument("--expected-prior-sha256", required=True)
        item.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        item.add_argument("--reference-config", type=Path, default=DEFAULT_REFERENCE_CONFIG)
        if name != "readiness":
            item.add_argument("--pretrain-epochs", type=int, default=30)
            item.add_argument("--epochs", type=int, default=100)
            item.add_argument("--batch-train-size", type=int, default=32)
            item.add_argument("--learning-rate", type=float, default=1e-3)
            item.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
        item.add_argument("--holdout-fraction", type=float, default=.2)
        item.add_argument("--seed", type=int, default=811007)
        item.add_argument("--min-successes", type=int, default=20)
        if name == "train":
            item.add_argument("--input-store", action="append", nargs=2,
                              metavar=("INPUT_DIR", "STORE_DIR"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "readiness":
        result = readiness(split_input=args.split_input, prior_manifest=args.prior_manifest,
                           prior_root=args.prior_root,
                           expected_prior_sha256=args.expected_prior_sha256,
                           config=args.config, reference_config=args.reference_config,
                           holdout_fraction=args.holdout_fraction, seed=args.seed)
        print(json.dumps(result, sort_keys=True))
        return 0
    common = dict(split_input=args.split_input, output_dir=args.output_dir,
                  prior_manifest=args.prior_manifest, prior_root=args.prior_root,
                  expected_prior_sha256=args.expected_prior_sha256, config=args.config,
                  reference_config=args.reference_config, pretrain_epochs=args.pretrain_epochs,
                  batch_size=args.batch_train_size, learning_rate=args.learning_rate,
                  hidden_dims=args.hidden_dims, holdout_fraction=args.holdout_fraction,
                  seed=args.seed)
    if args.command == "pretrain":
        paths, summary = pretrain(**common)
    else:
        paths, summary = train(input_store_pairs=[(Path(a), Path(b))
                                                  for a, b in args.input_store],
                               min_successes=args.min_successes, epochs=args.epochs, **common)
    print(json.dumps({"status": "complete", "signature": summary["signature"],
                      "rank_ready": summary["rank_ready"],
                      "completion_receipt_sha256": _sha(paths[0].parent / "completion_receipt.json"),
                      "models": [str(path) for path in paths]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
