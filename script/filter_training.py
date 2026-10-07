"""Opt-in masked historical-prior training for the R81 exploration loop.

This module is imported lazily by :mod:`script.exploration`.  It does not own a
controller or data lifecycle: preparation, feedback import, model training and
candidate selection remain the existing exploration operations.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from antenna.measurement import measurement_id, score_spec_id


PROTOCOL = "filter_masked_prior_v1"
SPLIT_FILE = "filter_family_split.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prior_spec(cfg: Mapping[str, Any]) -> dict[str, Any]:
    exp = cfg["exploration"]
    if exp.get("training_protocol") != PROTOCOL:
        raise ValueError(f"filter training requires training_protocol={PROTOCOL}")
    if exp.get("mode") != "dual_margin5" or cfg["measurement"].get("port") != "dual":
        raise ValueError("filter_masked_prior_v1 is dual-only")
    value = exp.get("filter_prior")
    if not isinstance(value, Mapping):
        raise ValueError("filter_masked_prior_v1 requires exploration.filter_prior")
    required = {"manifest", "sample_root", "expected_manifest_sha256"}
    if set(value) != required:
        raise ValueError(f"filter_prior must contain exactly {sorted(required)}")
    expected = value["expected_manifest_sha256"]
    if (not isinstance(expected, str) or len(expected) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in expected)):
        raise ValueError("filter_prior.expected_manifest_sha256 must be SHA-256")
    paths = {name: Path(value[name]) for name in ("manifest", "sample_root")}
    if not all(path.is_absolute() for path in paths.values()):
        raise ValueError("filter_prior manifest and sample_root must be absolute paths")
    return {"manifest": paths["manifest"].resolve(),
            "sample_root": paths["sample_root"].resolve(),
            "expected_manifest_sha256": expected.lower()}


def _prior_receipt(prior) -> dict[str, Any]:
    samples = [{"id": row["id"], "sample_file": row["sample_file"],
                "sample_sha256": row["source_sample_sha256"],
                "raw_response_sha256": row["raw_response_sha256"]}
               for row in prior.records]
    samples += [{"id": row["id"], "sample_sha256": row["sample_sha256"],
                 "excluded": True, "reasons": row["reasons"]}
                for row in prior.exclusions]
    samples.sort(key=lambda row: row["id"])
    return {"audit": prior.audit, "all_validated_samples": samples,
            "records_kept": list(prior.records), "exclusions": list(prior.exclusions)}


def load_prior(cfg: Mapping[str, Any], *, current_holdout=()):
    """Validate the explicit prior, including the caller-pinned manifest bytes."""
    from script.filter_prior import prepare_filter_prior

    spec = _prior_spec(cfg)
    if not spec["manifest"].is_file():
        raise ValueError("filter prior manifest does not exist")
    actual = _sha(spec["manifest"])
    if actual != spec["expected_manifest_sha256"]:
        raise ValueError("filter prior manifest SHA-256 differs from expected value")
    prior = prepare_filter_prior(
        cfg["measurement"], cfg["score_spec"], spec["manifest"], spec["sample_root"],
        current_holdout=current_holdout)
    if prior.audit["manifest_sha256"] != actual:
        raise ValueError("filter prior manifest changed during validation")
    return prior, _prior_receipt(prior)


def preflight(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Full prior validation used by prepare_run before it creates any output."""
    _prior, receipt = load_prior(cfg)
    return receipt


def row_aliases(row: Mapping[str, Any]) -> set[str]:
    aliases = {f"pattern:{row['pattern_sha256']}"}
    lineage = row.get("lineage_id")
    canonical = row.get("canonical_group_id")
    if lineage is not None:
        aliases.add(f"group:{lineage}")
    if canonical is not None:
        aliases.add(f"group:{canonical}")
    return aliases


def _components(rows: Sequence[Mapping[str, Any]]) -> list[tuple[list[int], set[str]]]:
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    owner: dict[str, int] = {}
    aliases = [row_aliases(row) for row in rows]
    for index, values in enumerate(aliases):
        for value in values:
            if value in owner:
                union(index, owner[value])
            else:
                owner[value] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    return [(indices, set().union(*(aliases[i] for i in indices)))
            for _root, indices in sorted(groups.items())]


def plan_split(rows: Sequence[Mapping[str, Any]], fraction: float, seed: int,
               existing: Mapping[str, str] | None = None) -> tuple[list[int], list[int], dict]:
    """Assign connected physical/family aliases once; never move an old alias."""
    assignments = dict(existing or {})
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("filter protocol requires a nonzero holdout fraction below one")
    if any(value not in ("train", "holdout") for value in assignments.values()):
        raise ValueError("filter split ledger contains an invalid assignment")
    train, holdout = [], []
    for indices, aliases in _components(rows):
        inherited = {assignments[name] for name in aliases if name in assignments}
        if len(inherited) > 1:
            raise ValueError("retroactive filter split conflict joins train and holdout aliases")
        if inherited:
            split = next(iter(inherited))
        else:
            token = f"{int(seed)}:{min(aliases)}".encode("utf-8")
            value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
            split = "holdout" if value < float(fraction) else "train"
        for alias in aliases:
            previous = assignments.get(alias)
            if previous is not None and previous != split:
                raise ValueError("retroactive filter split assignment changed")
            assignments[alias] = split
        (holdout if split == "holdout" else train).extend(indices)
    return sorted(train), sorted(holdout), assignments


def reject_feedback_split_conflict(work_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject a new row that bridges already frozen train/holdout families."""
    path = Path(work_dir) / SPLIT_FILE
    if not path.is_file():
        return
    document = json.loads(path.read_text(encoding="utf-8"))
    assignments = document.get("assignments", {})
    plan_split(rows, document["holdout_fraction"], document["seed"], assignments)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """True MSE over covered elements only; masked targets have zero gradient."""
    weight = mask.to(device=prediction.device, dtype=prediction.dtype)
    if weight.shape != prediction.shape:
        weight = torch.broadcast_to(weight, prediction.shape)
    denominator = weight.sum()
    if not bool(denominator > 0):
        raise ValueError("masked MSE requires at least one supervised element")
    return (((prediction - target) ** 2) * weight).sum() / denominator


def _load_current(work_dir: Path, rows: Sequence[Mapping[str, Any]], cfg, state):
    """Score the same frozen bytes whose profile, hash and geometry are checked."""
    from script import exploration as ex
    from script import profiled_batch as pb

    receipt, xs, ys, seen = [], [], [], set()
    for row in rows:
        if any(row.get(key) != state[key] for key in ("measurement_id", "score_spec_id")):
            raise ValueError(f"current row profile changed: {row['id']}")
        if (row.get("response_labels") != cfg["measurement"]["labels"] or
                not np.array_equal(row.get("response_freqs_ghz"), state["frequencies_ghz"])):
            raise ValueError(f"current row response axes changed: {row['id']}")
        sample = ex._safe_input_path(work_dir, row["scoped_sample_file"])
        raw = sample.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["scoped_sample_sha256"]:
            raise ValueError(f"current scoped sample hash changed: {row['id']}")
        pattern, response = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
        if pb.pattern_sha256(pattern) != row["pattern_sha256"]:
            raise ValueError(f"current pattern hash changed: {row['id']}")
        p = np.asarray(pattern)
        if not p[:5, 10:15].all() or not p[20:, 10:15].all():
            raise ValueError("current filter sample lacks feed pads")
        if row["pattern_sha256"] in seen:
            raise ValueError("current filter training requires unique physical patterns")
        seen.add(row["pattern_sha256"])
        if tuple(response.shape) != (3, 49) or not torch.isfinite(response).all():
            raise ValueError("current filter sample must contain complete finite 3x49 truth")
        if ex.fingerprint(pattern, response) != row["sample_fingerprint"]:
            raise ValueError(f"current sample fingerprint changed: {row['id']}")
        ys.append(torch.as_tensor(ex._score_dual(response, cfg["measurement"], cfg["score_spec"],
                                                row["response_labels"], row["response_freqs_ghz"])))
        xs.append(torch.as_tensor(pattern, dtype=torch.float32).reshape(-1))
        receipt.append({key: row.get(key) for key in (
            "id", "batch", "sample_fingerprint", "scoped_sample_file",
            "scoped_sample_sha256", "source_sample_sha256", "source_manifest",
            "source_manifest_sha256", "source_measurement_sha256", "source_score_spec_sha256",
            "pattern_sha256", "lineage_id", "canonical_group_id", "measurement_id",
            "score_spec_id", "response_labels", "response_freqs_ghz")})
    if not xs:
        raise ValueError("filter protocol has no current-profile training samples")
    return ex.TrainingSet(torch.stack(xs), torch.stack(ys), list(rows)), receipt


def _norm_receipt(mean: torch.Tensor, std: torch.Tensor) -> dict[str, Any]:
    raw = torch.stack((mean, std)).detach().cpu().numpy().astype("<f4").tobytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(),
            "mean": mean.detach().cpu().tolist(), "std": std.detach().cpu().tolist(),
            "fit_source": "current_profile_train_only"}


def validate_feedback_binding(work_dir, batch, state):
    """Bind imported rows to the selected batch and its completed feedback audit."""
    from script import exploration as ex

    work_dir = Path(work_dir)
    info = state["batches"][str(batch)]
    input_path = ex._safe_input_path(work_dir, info["input_dir"]) / "manifest.json"
    audit_path = ex._safe_input_path(work_dir, info["feedback_audit"])
    if (_sha(input_path) != info.get("input_manifest_sha256") or
            _sha(audit_path) != info.get("feedback_audit_sha256")):
        raise ValueError("filter feedback input/audit hash binding changed")
    selected = ex._manifest_rows(input_path)
    all_rows = ex._manifest_rows(work_dir / "dataset_manifest.json")
    if any(type(row.get("batch")) is not int or
           not state["batches"].get(str(row["batch"]), {}).get("feedback_audit")
           for row in all_rows):
        raise ValueError("filter feedback dataset binding includes an unaudited batch")
    imported = [row for row in all_rows if row["batch"] == batch]
    audit = ex._read_json(audit_path)
    if (len(imported) != info["selected"] or len(selected) != info["selected"] or
            len({row["id"] for row in imported}) != len(imported) or
            {row["id"] for row in imported} != {row["id"] for row in selected} or
            audit.get("input_manifest_sha256") != info["input_manifest_sha256"] or
            audit.get("dataset_entries_sha256") != ex.content_id(sorted(imported, key=lambda row: row["id"]))):
        raise ValueError("filter feedback dataset binding changed")


def validate_model_evidence(work_dir, batch, cfg, state):
    """Authenticate a completed batch's recipe, each member and its split."""
    from script import exploration as ex

    model_dir = Path(work_dir) / "models" / f"batch-{batch:03d}"
    plan = ex._read_json(model_dir / "training_plan.json")
    summary = ex._read_json(model_dir / "training_summary.json")
    expected_files = [f"models/batch-{batch:03d}/member-{seed}.pt"
                      for seed in cfg["exploration"]["ensemble_seeds"]]
    if (plan.get("through_batch") != batch or plan.get("training_protocol") != PROTOCOL or
            plan.get("exploration") != cfg["exploration"] or
            summary.get("model_files") != expected_files or
            any(summary.get(key) != value for key, value in plan.items())):
        raise ValueError("filter model batch/recipe binding mismatch")
    for seed in cfg["exploration"]["ensemble_seeds"]:
        path = model_dir / f"member-{seed}.pt"
        saved = torch.load(path, weights_only=False, map_location="cpu")
        signature = ex.content_id({**plan, "member_seed": seed})
        ex._validate_checkpoint(saved, state, "dual_margin5", 5, signature=signature)
        if (saved.get("member_seed") != seed or saved.get("through_batch") != batch or
                saved.get("geometry_profile") != cfg["exploration"]["geometry_profile"] or
                saved.get("training_protocol") != PROTOCOL or saved.get("complete") is not True or
                summary.get("member_signatures", {}).get(str(seed)) != signature or
                summary.get("model_sha256", {}).get(path.name) != _sha(path)):
            raise ValueError("filter prediction model hash/signature or member binding mismatch")
    return plan, summary


def train(work_dir: Path, through_batch: int, *, interrupt_after_epochs=None) -> list[Path]:
    """Train the opt-in protocol inside the existing exploration lifecycle."""
    from script import exploration as ex

    work_dir = Path(work_dir)
    cfg = ex.validate_exploration_config(ex._load_yaml(work_dir / "config.yaml"))
    exp = cfg["exploration"]
    state = ex._read_json(work_dir / "state.json")
    if ex.content_id(cfg) != state["config_hash"]:
        raise ValueError("filter training prepared configuration changed")
    if measurement_id(cfg["measurement"]) != state["measurement_id"]:
        raise ValueError("filter training measurement profile changed")
    if score_spec_id(cfg["score_spec"]) != state["score_spec_id"]:
        raise ValueError("filter training score profile changed")
    if (measurement_id(ex._read_json(work_dir / "measurement.json")) != state["measurement_id"] or
            score_spec_id(ex._read_json(work_dir / "score_spec.json")) != state["score_spec_id"]):
        raise ValueError("filter training workdir profile markers changed")
    if through_batch not in range(1, exp["max_batches"] + 1):
        raise ValueError("filter training batch is outside the prepared budget")
    for batch in range(1, through_batch + 1):
        if not state["batches"].get(str(batch), {}).get("feedback_audit"):
            raise ValueError(f"batch {batch} feedback must be audited before training")
        validate_feedback_binding(work_dir, batch, state)

    authenticated_batches = set(range(1, through_batch + 1))
    rows = [row for row in ex._manifest_rows(work_dir / "dataset_manifest.json")
            if row["batch"] in authenticated_batches]
    current, current_receipt = _load_current(work_dir, rows, cfg, state)
    split_path = work_dir / SPLIT_FILE
    existing = {}
    if split_path.is_file():
        split_doc = ex._read_json(split_path)
        if (split_doc.get("protocol") != PROTOCOL or split_doc.get("seed") != exp["seed"]
                or split_doc.get("holdout_fraction") != exp["holdout_fraction"]):
            raise ValueError("filter split ledger protocol differs")
        existing = split_doc.get("assignments", {})
    for earlier_batch in range(1, through_batch):
        plan, _summary = validate_model_evidence(work_dir, earlier_batch, cfg, state)
        if any(existing.get(alias) != split for alias, split in plan["split"]["assignments"].items()):
            raise ValueError("filter split ledger changed a previously authenticated assignment")
    train_idx, hold_idx, assignments = plan_split(
        current.rows, exp["holdout_fraction"], exp["seed"], existing)
    if not train_idx or not hold_idx:
        raise ValueError("filter protocol needs both current train and holdout families")
    split_doc = {"schema_version": 1, "protocol": PROTOCOL, "seed": exp["seed"],
                 "holdout_fraction": exp["holdout_fraction"],
                 "rule": "sha256(seed:min-connected-alias) threshold; assignments never move",
                 "assignments": dict(sorted(assignments.items()))}
    if split_path.is_file() and any(existing.get(k) not in (None, v)
                                    for k, v in assignments.items()):
        raise ValueError("filter split ledger would move an existing assignment")

    hold_rows = [current.rows[i] for i in hold_idx]
    prior, prior_receipt = load_prior(cfg, current_holdout=hold_rows)
    train_x = current.x[train_idx].float()
    mean, std = ex._fit_norm(train_x)
    norm_receipt = _norm_receipt(mean, std)
    signature_base = {
        "training_protocol": PROTOCOL, "through_batch": through_batch,
        "current_samples": current_receipt,
        "prior": prior_receipt,
        "split": split_doc,
        "train_ids": [current.rows[i]["id"] for i in train_idx],
        "holdout_ids": [current.rows[i]["id"] for i in hold_idx],
        "input_norm": norm_receipt,
        "exploration": exp,
    }

    prior_x = torch.as_tensor(prior.patterns, dtype=torch.float32).reshape(len(prior.patterns), -1)
    prior_y = torch.as_tensor(prior.margins, dtype=torch.float32)
    prior_mask = torch.as_tensor(prior.margin_masks, dtype=torch.bool)
    current_y = current.y[train_idx].float()
    current_mask = torch.ones_like(current_y, dtype=torch.bool)
    phases = []
    if exp["pretrain_epochs"]:
        phases.append(("historical_masked_prior", prior_x, prior_y, prior_mask,
                       exp["pretrain_epochs"]))
    phases.append(("current_five_margin_finetune", train_x, current_y, current_mask,
                   exp["epochs"]))

    output_dim = 5
    model_dir = work_dir / "models" / f"batch-{through_batch:03d}"
    paths, completed_epochs = [], 0
    signatures = {}
    # Validate every existing member before altering the split ledger or any model.
    for member_seed in exp["ensemble_seeds"]:
        path = model_dir / f"member-{member_seed}.pt"
        if path.exists():
            saved = torch.load(path, weights_only=False, map_location="cpu")
            signature = ex.content_id({**signature_base, "member_seed": member_seed})
            ex._validate_checkpoint(saved, state, exp["mode"], output_dim, signature=signature)
    ex._atomic_json(split_path, split_doc)
    ex._atomic_json(model_dir / "training_plan.json", signature_base)
    for member_seed in exp["ensemble_seeds"]:
        torch.manual_seed(member_seed); np.random.seed(member_seed)
        model = ex.CurveMLP(625, exp["hidden_dims"], output_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=exp["learning_rate"])
        path = model_dir / f"member-{member_seed}.pt"
        signature = ex.content_id({**signature_base, "member_seed": member_seed})
        signatures[str(member_seed)] = signature
        phase_start, epoch_start = 0, 0
        if path.exists():
            saved = torch.load(path, weights_only=False, map_location="cpu")
            ex._validate_checkpoint(saved, state, exp["mode"], output_dim, signature=signature)
            if saved.get("training_protocol") != PROTOCOL:
                raise ValueError("checkpoint filter training protocol mismatch")
            saved_norms = saved.get("norms")
            if (not isinstance(saved_norms, list) or len(saved_norms) != 1
                    or not torch.equal(saved_norms[0][0], mean)
                    or not torch.equal(saved_norms[0][1], std)):
                raise ValueError("checkpoint current-train input normalization mismatch")
            if saved.get("complete"):
                paths.append(path); continue
            model.load_state_dict(saved["model_state"])
            optimizer.load_state_dict(saved["optimizer_state"])
            torch.set_rng_state(saved["torch_rng_state"])
            np.random.set_state(saved["numpy_rng_state"])
            phase_start, epoch_start = int(saved["phase"]), int(saved["epoch"])
        for phase_i, (_name, x, y, mask, epochs) in enumerate(phases):
            if phase_i < phase_start:
                continue
            first_epoch = epoch_start if phase_i == phase_start else 0
            for epoch in range(first_epoch, epochs):
                model.train()
                order = ex._epoch_order(len(x), member_seed, phase_i, epoch)
                for start in range(0, len(order), exp["batch_train_size"]):
                    idx = torch.as_tensor(order[start:start + exp["batch_train_size"]],
                                          dtype=torch.long)
                    prediction = model((x[idx] - mean) / std)
                    loss = masked_mse(prediction, y[idx], mask[idx])
                    if not torch.isfinite(loss):
                        raise ValueError("non-finite filter training loss")
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                next_phase, next_epoch = phase_i, epoch + 1
                if next_epoch >= epochs:
                    next_phase, next_epoch = phase_i + 1, 0
                complete = next_phase >= len(phases)
                ex._save_member_checkpoint(
                    path, model=model, optimizer=optimizer, signature=signature,
                    mode=exp["mode"], output_dim=output_dim,
                    measurement_id_value=state["measurement_id"],
                    score_spec_id_value=state["score_spec_id"],
                    geometry_profile=exp["geometry_profile"], member_seed=member_seed,
                    phase=next_phase, epoch=next_epoch, norms=[(mean, std)],
                    complete=complete, through_batch=through_batch,
                    training_protocol=PROTOCOL)
                completed_epochs += 1
                if (interrupt_after_epochs is not None
                        and completed_epochs >= interrupt_after_epochs):
                    raise ex.TrainingInterrupted("intentional epoch-boundary interruption")
        paths.append(path)

    predictions = []
    for path in paths:
        saved = torch.load(path, weights_only=False, map_location="cpu")
        model = ex.CurveMLP(625, exp["hidden_dims"], output_dim)
        model.load_state_dict(saved["model_state"]); model.eval()
        with torch.no_grad():
            predictions.append(model((current.x[hold_idx] - mean) / std))
    prediction = torch.stack(predictions).mean(dim=0)
    target = current.y[hold_idx]
    absolute = torch.abs(prediction - target)
    mae_by_band = absolute.mean(dim=0)
    wm_mae = torch.abs(prediction.min(dim=1).values - target.min(dim=1).values).mean()
    if not all(bool(torch.isfinite(value).all()) for value in
               [*predictions, prediction, target, absolute, mae_by_band, wm_mae]):
        raise ValueError("non-finite filter holdout evidence")
    names = [band["name"] for band in cfg["score_spec"]["bands"]]
    evidence = {**signature_base, "member_signatures": signatures,
                "model_files": [path.relative_to(work_dir).as_posix() for path in paths],
                "model_sha256": {path.name: _sha(path) for path in paths},
                "n_total": len(current.rows), "n_train": len(train_idx), "n_holdout": len(hold_idx),
                "validation": {"source": "current_profile_holdout_only", "used_for_model_selection": False,
                               "band_names": names, "prediction": prediction.tolist(), "target": target.tolist(),
                               "mae_by_band": dict(zip(names, mae_by_band.tolist())),
                               "worst_margin_mae_db": float(wm_mae)}}
    ex._atomic_json(model_dir / "training_summary.json", evidence)
    state = ex._read_json(work_dir / "state.json")
    state["batches"][str(through_batch)]["trained_model_dir"] = model_dir.relative_to(work_dir).as_posix()
    ex._atomic_json(work_dir / "state.json", state)
    return paths
