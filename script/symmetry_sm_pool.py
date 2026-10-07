# -*- coding: utf-8 -*-
"""Historical-SM navigation for a large, blind, symmetric single-port pool.

This is a controller-only utility.  It does not train a model, discover training
data, run HFSS, or write to the historical dataset directory.  The old models
are used only as a proposal prior: every emitted pattern is re-projected to the
current exact left/right geometry and validated against the named 0.1 mm bridge
measurement profile before it can enter a worker bundle.

The legacy checkpoints predate the 0.1 mm diagonal-bridge geometry.  Therefore
``factory_score`` and ``navigation_score`` are rankings, never simulated truth
or acceptance specifications.  Symmetry is an observed design restriction; it
does not receive an invented performance threshold here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch
import yaml
from torch import nn

from antenna.models.surrogates import HFSSNet, ResCNNNet
from script import exploration as ex
from script import profiled_batch as pb


LEGACY_FREQUENCIES_GHZ = np.arange(24.0, 32.0 + 0.25, 0.5, dtype=np.float32)
FACTORY_BAND_GHZ = (26.5, 29.5)
LEGACY_LIMITATION = (
    "Historical surrogate weights predate the 0.1 mm diagonal-bridge measurement "
    "geometry. Predictions and scores are navigation priors only; HFSS under the "
    "named measurement profile is the new truth."
)
HISTORICAL_TRAINING_SCOPE = (
    "This utility loads only the explicitly named checkpoint weights and does not "
    "discover or mix training rows. The checkpoint training history may contain "
    "non-symmetric designs; exact symmetry is enforced on every new candidate."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class PoolConfig:
    """Explicit inputs and bounded search budget for :func:`build_pool`."""

    profile_config: Path
    candidate_pool_size: int = 10_000
    selected_count: int = 384
    id_prefix: str = "r80b2"
    initial_pool_size: int = 2_500
    generations: int = 3
    parent_count: int = 256
    min_mutation_flips: int = 2
    max_mutation_flips: int = 48
    max_per_ancestry: int = 24
    diversity_weight: float = 0.35
    radiation_weight: float = 0.20
    uncertainty_weight: float = 0.10
    seed: int = 0
    prediction_batch_size: int = 256
    update_every_valid_points: int = 48
    valid_observations_at_fit: int | None = None
    surrogate_generation: str | None = None
    model_dir: Path | None = None
    response_models: tuple[str, ...] = (
        "sm_reanchor108.pth", "sm_ens108_1.pth", "sm_ens108_2.pth"
    )
    radiation_model: str = "rad_head108.pth"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PoolConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown pool config fields: {sorted(unknown)}")
        data = dict(raw)
        if "profile_config" not in data:
            raise ValueError("profile_config is required")
        data["profile_config"] = Path(data["profile_config"])
        if data.get("model_dir") is not None:
            data["model_dir"] = Path(data["model_dir"])
        if "response_models" in data:
            data["response_models"] = tuple(str(v) for v in data["response_models"])
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        for name in ("candidate_pool_size", "selected_count", "initial_pool_size",
                     "generations", "parent_count", "min_mutation_flips",
                     "max_mutation_flips", "max_per_ancestry", "prediction_batch_size"):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.update_every_valid_points, bool) or
                int(self.update_every_valid_points) <= 0):
            raise ValueError("update_every_valid_points must be a positive integer")
        if self.valid_observations_at_fit is not None and (
                isinstance(self.valid_observations_at_fit, bool) or
                int(self.valid_observations_at_fit) < 0):
            raise ValueError("valid_observations_at_fit must be a non-negative integer or null")
        if self.surrogate_generation is not None and not str(self.surrogate_generation).strip():
            raise ValueError("surrogate_generation must be non-empty or null")
        if self.selected_count > self.candidate_pool_size:
            raise ValueError("selected_count cannot exceed candidate_pool_size")
        if self.initial_pool_size > self.candidate_pool_size:
            raise ValueError("initial_pool_size cannot exceed candidate_pool_size")
        if self.min_mutation_flips > self.max_mutation_flips:
            raise ValueError("min_mutation_flips cannot exceed max_mutation_flips")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.id_prefix):
            raise ValueError("id_prefix must contain only letters, digits, '_' or '-'")
        for name in ("diversity_weight", "radiation_weight", "uncertainty_weight"):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if len(self.response_models) < 1:
            raise ValueError("at least one response model is required")


@dataclass(frozen=True)
class PredictionBatch:
    response_mean: np.ndarray       # (N, 2, 17): S11, Gain
    response_disagreement: np.ndarray  # (N,)
    radiation: np.ndarray           # (N, 2, T): phi0, phi90
    radiation_theta: np.ndarray     # (T,)
    response_members: np.ndarray | None = None  # (M, N, 2, 17), when available


class Predictor(Protocol):
    model_ids: Sequence[Mapping[str, str]]

    def predict(self, patterns: Sequence[np.ndarray], *, batch_size: int) -> PredictionBatch:
        ...


class LegacySurrogatePredictor:
    """Read-only loader for the MLP/ResCNN v108 ensemble and v108 rad head.

    Only ``model_state_dict`` is loaded from the large training checkpoints; the
    optimizer and record payloads are deliberately ignored.  This avoids both
    hidden training and writes to ``model_dir``.
    """

    def __init__(self, model_dir: Path, response_models: Sequence[str], radiation_model: str):
        self.model_role = "historical_cold_start_only"
        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"model_dir does not exist: {self.model_dir}")
        self.response_models: list[nn.Module] = []
        self.model_ids: list[dict[str, str]] = []
        for name in response_models:
            path = self.model_dir / name
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload.get("model_state_dict") if isinstance(payload, Mapping) else None
            if not isinstance(state, Mapping):
                raise ValueError(f"{name} has no model_state_dict")
            title = str(payload.get("title", ""))
            if "ResCNNNet" in title or any(str(k).startswith("stem.") for k in state):
                model: nn.Module = ResCNNNet(625, (2, 17))
                architecture = "ResCNNNet"
            elif "HFSSNet" in title or any(str(k).startswith("fc_patch.") for k in state):
                model = HFSSNet(625, (2, 17))
                architecture = "HFSSNet"
            else:
                raise ValueError(f"unsupported response architecture in {name}: {title}")
            model.load_state_dict(state, strict=True)
            model.eval()
            model.requires_grad_(False)
            self.response_models.append(model)
            self.model_ids.append({"file": name, "sha256": _file_sha256(path),
                                   "architecture": architecture,
                                   "source": "antenna.models.surrogates"})

        rad_path = self.model_dir / radiation_model
        rad = torch.load(rad_path, map_location="cpu", weights_only=False)
        if not isinstance(rad, Mapping) or not {"state", "K", "theta"} <= set(rad):
            raise ValueError(f"invalid radiation checkpoint: {radiation_model}")
        k = int(rad["K"])
        self.radnet = nn.Sequential(nn.Linear(625, 512), nn.ReLU(),
                                    nn.Linear(512, 256), nn.ReLU(),
                                    nn.Linear(256, 2 * k))
        self.radnet.load_state_dict(rad["state"], strict=True)
        self.radnet.eval()
        self.radnet.requires_grad_(False)
        self.radiation_theta = np.asarray(rad["theta"], dtype=np.float32).reshape(-1)
        phi = np.pi * (self.radiation_theta - self.radiation_theta.min()) / max(
            float(np.ptp(self.radiation_theta)), 1e-12)
        self.rad_basis = torch.as_tensor(
            np.cos(np.arange(k).reshape(-1, 1) * phi.reshape(1, -1)), dtype=torch.float32)
        self.rad_k = k
        self.model_ids.append({"file": radiation_model, "sha256": _file_sha256(rad_path),
                               "architecture": "legacy_rad_mlp_cosine_head",
                               "source": "script.sm_invert.Inverter"})

    def predict(self, patterns: Sequence[np.ndarray], *, batch_size: int = 256) -> PredictionBatch:
        # The public boundary always projects with the shared controller function.
        checked = [ex.enforce_single_geometry(p) for p in patterns]
        x = torch.as_tensor(np.stack(checked).reshape(-1, 625), dtype=torch.float32)
        response_chunks, member_chunks, disagreement_chunks, rad_chunks = [], [], [], []
        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                xb = x[start:start + batch_size]
                members = torch.stack([model(xb) for model in self.response_models], dim=0)
                member_chunks.append(members.cpu())
                response_chunks.append(members.mean(dim=0).cpu())
                if len(self.response_models) == 1:
                    disagreement_chunks.append(torch.zeros(len(xb)))
                else:
                    disagreement_chunks.append(members.std(dim=0, unbiased=False).mean((1, 2)).cpu())
                coeff = self.radnet(xb).reshape(len(xb), 2, self.rad_k)
                rad_chunks.append((coeff @ self.rad_basis).cpu())
        member_array = torch.cat(response_chunks, dim=0).numpy()
        member_predictions = torch.cat(member_chunks, dim=1).numpy()
        return PredictionBatch(
            response_mean=member_array,
            response_disagreement=torch.cat(disagreement_chunks).numpy(),
            radiation=torch.cat(rad_chunks).numpy(),
            radiation_theta=self.radiation_theta.copy(),
            response_members=member_predictions,
        )


def score_predictions(pred: PredictionBatch, *, radiation_weight: float = 0.20,
                      uncertainty_weight: float = 0.10) -> dict[str, np.ndarray]:
    """Compute transparent navigation metrics without creating an HFSS gate."""
    y = np.asarray(pred.response_mean, dtype=np.float64)
    rad = np.asarray(pred.radiation, dtype=np.float64)
    disagreement = np.asarray(pred.response_disagreement, dtype=np.float64).reshape(-1)
    theta = np.asarray(pred.radiation_theta, dtype=np.float64).reshape(-1)
    n = len(y)
    if y.shape != (n, 2, 17) or rad.ndim != 3 or rad.shape[:2] != (n, 2):
        raise ValueError("prediction shapes must be response (N,2,17), radiation (N,2,T)")
    if rad.shape[2] != len(theta) or disagreement.shape != (n,):
        raise ValueError("prediction batch dimensions disagree")
    if not (np.isfinite(y).all() and np.isfinite(rad).all() and
            np.isfinite(theta).all() and np.isfinite(disagreement).all()):
        raise ValueError("predictions must be finite")
    band = ((LEGACY_FREQUENCIES_GHZ >= FACTORY_BAND_GHZ[0]) &
            (LEGACY_FREQUENCIES_GHZ <= FACTORY_BAND_GHZ[1]))
    s11_margin = -10.0 - y[:, 0, band].max(axis=1)
    gain_margin = y[:, 1, band].min(axis=1) - 4.0
    factory_score = np.minimum(s11_margin, gain_margin)
    if pred.response_members is not None:
        members = np.asarray(pred.response_members, dtype=np.float64)
        if members.ndim != 4 or members.shape[1:] != (n, 2, 17):
            raise ValueError("response_members must have shape (M,N,2,17)")
        member_s11 = -10.0 - members[:, :, 0, :][:, :, band].max(axis=2)
        member_gain = members[:, :, 1, :][:, :, band].min(axis=2) - 4.0
        member_factory = np.minimum(member_s11, member_gain)
        s11_margin = member_s11.mean(axis=0)
        gain_margin = member_gain.mean(axis=0)
        factory_score = member_factory.mean(axis=0)
        disagreement = member_factory.std(axis=0)
    window = np.abs(theta) <= 45.0
    if not np.any(window):
        raise ValueError("radiation theta has no samples inside +/-45 degrees")
    bore = int(np.argmin(np.abs(theta)))
    rad_margin = np.min(rad[:, :, window].min(axis=2) - (rad[:, :, bore] - 3.0), axis=1)
    # Radiation is a legacy shaping prior.  Clipping prevents its uncalibrated
    # magnitude from overwhelming the new, explicit S11/Gain factory score.
    navigation_mean_score = (factory_score +
                             float(radiation_weight) * np.clip(rad_margin, -6.0, 3.0))
    lcb_score = navigation_mean_score - disagreement
    navigation_score = navigation_mean_score - float(uncertainty_weight) * disagreement
    return {"s11_margin": s11_margin, "gain_margin": gain_margin,
            "factory_score": factory_score, "radiation_margin": rad_margin,
            "response_disagreement": disagreement,
            "navigation_mean_score": navigation_mean_score,
            "lcb_score": lcb_score, "navigation_score": navigation_score}


def _profile(path: Path):
    cfg = pb.load_profile_config(path)
    if cfg is None or cfg.port != "single":
        raise ValueError("profile_config must be a named single-port measurement profile")
    geometry = cfg.measurement["geometry"]
    if geometry.get("pixel_count") != 25 or float(geometry.get("diag_bridge_w", -1)) != 0.1:
        raise ValueError("pool requires the current 25x25, diag_bridge_w=0.1 mm profile")
    return cfg


def _resolve_predictor_binding(predictor: Predictor, cfg: PoolConfig,
                               profile_cfg: Any) -> tuple[str, int, str, dict[str, Any] | None]:
    """Return authoritative generation/count from the physical predictor identity."""
    from antenna.measurement import measurement_id, score_spec_id

    role = getattr(predictor, "model_role", None)
    if role == "historical_cold_start_only":
        generation, count, binding = "historical-v108-cold-start", 0, None
    elif role == "current_profile":
        binding = getattr(predictor, "binding", None)
        required = {"measurement_id", "score_spec_id", "protocol_id", "data_version",
                    "manifest_id", "cumulative_valid_unique"}
        if not isinstance(binding, Mapping) or set(binding) != required:
            raise ValueError(f"current-profile predictor binding must contain exactly {sorted(required)}")
        binding = dict(binding)
        if binding["measurement_id"] != measurement_id(profile_cfg.measurement):
            raise ValueError("predictor measurement_id does not match the pool profile")
        if binding["score_spec_id"] != score_spec_id(profile_cfg.score_spec):
            raise ValueError("predictor score_spec_id does not match the pool profile")
        if (isinstance(binding["data_version"], bool) or
                not isinstance(binding["data_version"], int) or binding["data_version"] <= 0):
            raise ValueError("predictor data_version must be a positive integer")
        if (isinstance(binding["cumulative_valid_unique"], bool) or
                not isinstance(binding["cumulative_valid_unique"], int) or
                binding["cumulative_valid_unique"] < 0):
            raise ValueError("predictor cumulative_valid_unique must be a non-negative integer")
        if not str(binding["protocol_id"]).strip() or not str(binding["manifest_id"]).strip():
            raise ValueError("predictor protocol_id and manifest_id must be non-empty")
        generation = f"data-v{binding['data_version']:03d}"
        count = binding["cumulative_valid_unique"]
        for model_id in predictor.model_ids:
            for key in ("protocol_id", "manifest_id"):
                if key in model_id and str(model_id[key]) != str(binding[key]):
                    raise ValueError(f"predictor model_id {key} disagrees with binding")
            if "data_version" in model_id and int(model_id["data_version"]) != binding["data_version"]:
                raise ValueError("predictor model_id data_version disagrees with binding")
    else:
        raise ValueError("predictor must declare model_role as historical_cold_start_only or current_profile")
    if cfg.surrogate_generation is not None and cfg.surrogate_generation != generation:
        raise ValueError("configured surrogate_generation disagrees with predictor binding")
    if (cfg.valid_observations_at_fit is not None and
            cfg.valid_observations_at_fit != count):
        raise ValueError("configured valid_observations_at_fit disagrees with predictor binding")
    return generation, count, role, binding


def _checked_pattern(pattern: Any, profile_cfg: Any) -> np.ndarray:
    p = ex.enforce_single_geometry(pattern)
    pb.validate_pattern(p, profile_cfg)  # includes physical bridge-symmetry gate
    return p


def _normalise_seeds(seed_rows: Sequence[Mapping[str, Any]], profile_cfg: Any,
                     excluded: set[str]) -> list[dict[str, Any]]:
    if not seed_rows:
        raise ValueError("seed_rows must be an explicit non-empty sequence")
    out, seen = [], set(excluded)
    for ordinal, row in enumerate(seed_rows):
        if "pattern" not in row:
            raise ValueError(f"seed row {ordinal} has no pattern")
        p = _checked_pattern(row["pattern"], profile_cfg)
        sha = ex.pattern_sha256(p)
        if sha in seen:
            continue
        seen.add(sha)
        source_id = str(row.get("id", f"seed-{ordinal:05d}"))
        ancestry_id = str(row.get("canonical_group_id") or row.get("lineage_id") or
                          row.get("ancestry_id") or source_id)
        out.append({"pattern": p, "pattern_sha256": sha, "ancestry_id": ancestry_id,
                    "source_id": source_id, "generation": 0, "proposal": "explicit_seed"})
    if not out:
        raise ValueError("all explicit seeds were duplicates or excluded")
    return out


def _append_unique(rows: list[dict[str, Any]], seen: set[str], pattern: Any,
                   profile_cfg: Any, **metadata: Any) -> bool:
    p = _checked_pattern(pattern, profile_cfg)
    sha = ex.pattern_sha256(p)
    if sha in seen:
        return False
    seen.add(sha)
    rows.append({"pattern": p, "pattern_sha256": sha, **metadata})
    return True


def _score_rows(rows: list[dict[str, Any]], predictor: Predictor, cfg: PoolConfig) -> None:
    if not rows:
        return
    pred = predictor.predict([r["pattern"] for r in rows], batch_size=cfg.prediction_batch_size)
    metrics = score_predictions(pred, radiation_weight=cfg.radiation_weight,
                                uncertainty_weight=cfg.uncertainty_weight)
    for i, row in enumerate(rows):
        row.update({name: float(values[i]) for name, values in metrics.items()})
        row["predicted_s11"] = pred.response_mean[i, 0].astype(float).tolist()
        row["predicted_gain"] = pred.response_mean[i, 1].astype(float).tolist()
        row["predicted_radiation_phi0"] = pred.radiation[i, 0].astype(float).tolist()
        row["predicted_radiation_phi90"] = pred.radiation[i, 1].astype(float).tolist()
        row["predicted_radiation_theta"] = pred.radiation_theta.astype(float).tolist()


def _balanced_parents(rows: Sequence[dict[str, Any]], n: int,
                      max_per_ancestry: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda r: (-r["lcb_score"], r["pattern_sha256"]))
    chosen, counts = [], {}
    for row in ranked:
        ancestry = row["ancestry_id"]
        if counts.get(ancestry, 0) >= max_per_ancestry:
            continue
        chosen.append(row)
        counts[ancestry] = counts.get(ancestry, 0) + 1
        if len(chosen) == n:
            break
    return chosen or ranked[:1]


def _generate_initial(seeds: Sequence[dict[str, Any]], target: int, rng: np.random.Generator,
                      profile_cfg: Any, seen: set[str], cfg: PoolConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        _append_unique(rows, seen, seed["pattern"], profile_cfg,
                       ancestry_id=seed["ancestry_id"], source_id=seed["source_id"],
                       generation=0, proposal="explicit_seed")
        if len(rows) >= target:
            return rows
    attempts = 0
    while len(rows) < target and attempts < target * 100:
        attempts += 1
        if rng.random() < 0.30:
            ancestry = f"random-{attempts % max(16, cfg.parent_count // 4):03d}"
            p = ex.random_single_pattern(rng)
            proposal, source_id = "random_exploration", ancestry
        else:
            parent = seeds[int(rng.integers(len(seeds)))]
            p = ex.mutate_pattern(parent["pattern"], rng, single=True,
                                  min_flips=cfg.min_mutation_flips,
                                  max_flips=cfg.max_mutation_flips)
            ancestry, source_id = parent["ancestry_id"], parent["source_id"]
            proposal = "seed_mutation"
        _append_unique(rows, seen, p, profile_cfg, ancestry_id=ancestry,
                       source_id=source_id, generation=0, proposal=proposal)
    if len(rows) < target:
        raise RuntimeError(f"could create only {len(rows)}/{target} unique initial candidates")
    return rows


def _generate_evolution(parents: Sequence[dict[str, Any]], target: int, generation: int,
                        rng: np.random.Generator, profile_cfg: Any, seen: set[str],
                        cfg: PoolConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < target and attempts < target * 100:
        attempts += 1
        if rng.random() < 0.30:
            ancestry = f"random-g{generation}-{attempts % max(16, cfg.parent_count // 4):03d}"
            p = ex.random_single_pattern(rng)
            source_id, proposal = ancestry, "random_exploration"
        else:
            parent = parents[int(rng.integers(len(parents)))]
            p = ex.mutate_pattern(parent["pattern"], rng, single=True,
                                  min_flips=cfg.min_mutation_flips,
                                  max_flips=cfg.max_mutation_flips)
            ancestry, source_id = parent["ancestry_id"], parent["source_id"]
            proposal = "sm_guided_mutation"
        _append_unique(rows, seen, p, profile_cfg, ancestry_id=ancestry,
                       source_id=source_id, generation=generation, proposal=proposal)
    if len(rows) < target:
        raise RuntimeError(f"could create only {len(rows)}/{target} unique generation candidates")
    return rows


def _largest_remainder_counts(total: int) -> dict[str, int]:
    """Paper III-E quotas, with deterministic ties: 32 -> 13/10/9."""
    order = ("lcb_performance", "high_disagreement", "blind_random")
    weights = dict(zip(order, (0.40, 0.30, 0.30)))
    raw = {name: total * weights[name] for name in order}
    counts = {name: int(math.floor(raw[name])) for name in order}
    remainder = total - sum(counts.values())
    ranked = sorted(order, key=lambda name: (-(raw[name] - counts[name]), order.index(name)))
    for name in ranked[:remainder]:
        counts[name] += 1
    return counts


def _select_scored_arm(pool: Sequence[dict[str, Any]], count: int, *, score_key: str,
                       arm: str, cfg: PoolConfig) -> list[dict[str, Any]]:
    ranked = sorted(pool, key=lambda r: (-float(r[score_key]), r["pattern_sha256"]))
    shortlist = ranked[:max(count * 12, count)]
    values = np.asarray([r[score_key] for r in shortlist], dtype=float)
    scale = float(np.std(values)) or 1.0
    picked: list[dict[str, Any]] = []
    ancestry_counts: dict[str, int] = {}
    while shortlist and len(picked) < count:
        best_i, best_value = None, -float("inf")
        for i, row in enumerate(shortlist):
            if ancestry_counts.get(row["ancestry_id"], 0) >= cfg.max_per_ancestry:
                continue
            diversity = (min(ex.hamming(row["pattern"], prior["pattern"]) for prior in picked) / 625.0
                         if picked else 1.0)
            value = float(row[score_key]) / scale + cfg.diversity_weight * diversity
            if value > best_value:
                best_i, best_value = i, value
        if best_i is None:
            best_i = 0  # ancestry cap relaxes only if otherwise unable to fill the arm
        row = shortlist.pop(best_i)
        row["selection_arm"] = arm
        ancestry_counts[row["ancestry_id"]] = ancestry_counts.get(row["ancestry_id"], 0) + 1
        picked.append(row)
    if len(picked) != count:
        raise RuntimeError(f"{arm} selected only {len(picked)}/{count}")
    return picked


def _select(rows: Sequence[dict[str, Any]], cfg: PoolConfig,
            rng: np.random.Generator) -> list[dict[str, Any]]:
    del rng  # Blind allocation has its own fixed stream, independent of SM rankings.
    counts = _largest_remainder_counts(cfg.selected_count)

    # Reserve blind candidates first so neither LCB nor disagreement can consume
    # the symmetric-random-origin quota.  Ordering uses only hashes and cfg.seed;
    # changing every prediction leaves this arm byte-for-byte unchanged.
    blind_n = counts["blind_random"]
    random_origin = sorted((r for r in rows if r["proposal"] == "random_exploration"),
                           key=lambda r: r["pattern_sha256"])
    blind_rng = np.random.default_rng(cfg.seed + 0xB11D)
    order = blind_rng.permutation(len(random_origin)).tolist()
    blind: list[dict[str, Any]] = []
    ancestry_counts: dict[str, int] = {}
    if blind_n:
        for i in order:
            row = random_origin[int(i)]
            if ancestry_counts.get(row["ancestry_id"], 0) >= cfg.max_per_ancestry:
                continue
            row["selection_arm"] = "blind_random"
            row["blind_origin_fallback"] = False
            ancestry_counts[row["ancestry_id"]] = ancestry_counts.get(row["ancestry_id"], 0) + 1
            blind.append(row)
            if len(blind) == blind_n:
                break
    if len(blind) < blind_n:
        used_blind = {r["pattern_sha256"] for r in blind}
        fallback = sorted((r for r in rows if r["pattern_sha256"] not in used_blind),
                          key=lambda r: r["pattern_sha256"])
        fallback_order = blind_rng.permutation(len(fallback)).tolist()
        for i in fallback_order:
            row = fallback[int(i)]
            row["selection_arm"] = "blind_random"
            row["blind_origin_fallback"] = True
            blind.append(row)
            if len(blind) == blind_n:
                break
    if len(blind) != blind_n:
        raise RuntimeError(f"blind_random selected only {len(blind)}/{blind_n}")

    used = {r["pattern_sha256"] for r in blind}
    remaining = [r for r in rows if r["pattern_sha256"] not in used]
    lcb = _select_scored_arm(remaining, counts["lcb_performance"], score_key="lcb_score",
                             arm="lcb_performance", cfg=cfg)
    used.update(r["pattern_sha256"] for r in lcb)

    disagreement = []
    if counts["high_disagreement"]:
        p60 = float(np.percentile([r["navigation_mean_score"] for r in rows], 60.0))
        disagreement_pool = [r for r in rows if r["pattern_sha256"] not in used and
                             r["navigation_mean_score"] >= p60]
        disagreement = _select_scored_arm(disagreement_pool, counts["high_disagreement"],
                                          score_key="response_disagreement",
                                          arm="high_disagreement", cfg=cfg)
        for row in disagreement:
            row["disagreement_eligibility_mean_p60"] = p60
    selected = lcb + disagreement + blind
    if len(selected) != cfg.selected_count or len({r["pattern_sha256"] for r in selected}) != len(selected):
        raise RuntimeError("three-arm selection count or uniqueness failure")
    return selected


def build_pool(config: PoolConfig | Mapping[str, Any], seed_rows: Sequence[Mapping[str, Any]],
               exclude_hashes: Sequence[str], predictor: Predictor | None = None) -> dict[str, Any]:
    """Build and score a deterministic, explicit-input candidate pool.

    ``seed_rows`` and ``exclude_hashes`` are the complete input scope.  No browser
    cache, SampleStore, NAS path, or historical folder is discovered here.
    """
    cfg = config if isinstance(config, PoolConfig) else PoolConfig.from_mapping(config)
    cfg.validate()
    profile_cfg = _profile(cfg.profile_config)
    excluded = {str(v) for v in exclude_hashes}
    seeds = _normalise_seeds(seed_rows, profile_cfg, excluded)
    if predictor is None:
        if cfg.model_dir is None:
            raise ValueError("model_dir is required when predictor is not supplied")
        predictor = LegacySurrogatePredictor(cfg.model_dir, cfg.response_models, cfg.radiation_model)
    predictor_generation, valid_at_fit, model_role, binding = _resolve_predictor_binding(
        predictor, cfg, profile_cfg)

    rng = np.random.default_rng(cfg.seed)
    seen = set(excluded)
    initial_n = min(cfg.initial_pool_size, cfg.candidate_pool_size)
    rows = _generate_initial(seeds, initial_n, rng, profile_cfg, seen, cfg)
    _score_rows(rows, predictor, cfg)
    remaining = cfg.candidate_pool_size - len(rows)
    for search_generation in range(1, cfg.generations + 1):
        if remaining <= 0:
            break
        parent_rows = _balanced_parents(rows, min(cfg.parent_count, len(rows)), cfg.max_per_ancestry)
        n_new = int(math.ceil(remaining / (cfg.generations - search_generation + 1)))
        new_rows = _generate_evolution(parent_rows, n_new, search_generation, rng,
                                       profile_cfg, seen, cfg)
        _score_rows(new_rows, predictor, cfg)
        rows.extend(new_rows)
        remaining = cfg.candidate_pool_size - len(rows)
    if len(rows) != cfg.candidate_pool_size:
        raise RuntimeError(f"pool size {len(rows)} != requested {cfg.candidate_pool_size}")
    selected = _select(rows, cfg, rng)
    hashes = [r["pattern_sha256"] for r in rows]
    if len(hashes) != len(set(hashes)) or set(hashes) & excluded:
        raise AssertionError("internal deduplication/exclusion failure")
    return {"config": cfg, "profile_cfg": profile_cfg, "rows": rows,
            "selected_rows": selected, "model_ids": list(predictor.model_ids),
            "predictor_model_role": model_role,
            "predictor_binding": binding,
            "surrogate_generation": predictor_generation,
            "valid_observations_at_fit": valid_at_fit,
            "legacy_geometry_limitation": (LEGACY_LIMITATION if model_role ==
                                             "historical_cold_start_only" else None)}


def write_bundle(result: Mapping[str, Any], output_dir: Path) -> Path:
    """Write selected rows as a worker bundle and validate it with profiled_batch."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg: PoolConfig = result["config"]
    profile_cfg = result["profile_cfg"]
    shutil.copy2(cfg.profile_config, output_dir / "config.yaml")
    pb.atomic_json(output_dir / "measurement.json", profile_cfg.measurement)
    pb.atomic_json(output_dir / "score_spec.json", profile_cfg.score_spec)
    from antenna.measurement import measurement_id, score_spec_id
    mid = measurement_id(profile_cfg.measurement)
    sid = score_spec_id(profile_cfg.score_spec)
    combined_model_hash = hashlib.sha256(json.dumps(result["model_ids"], sort_keys=True).encode()).hexdigest()
    manifest = []
    for ordinal, source in enumerate(result["selected_rows"]):
        pattern = _checked_pattern(source["pattern"], profile_cfg)
        sha = ex.pattern_sha256(pattern)
        pid = f"{cfg.id_prefix}_{ordinal:05d}_{sha[:8]}"
        torch.save(torch.as_tensor(pattern, dtype=torch.float32), output_dir / f"{pid}.pt")
        clean = {k: _jsonable(v) for k, v in source.items() if k != "pattern"}
        clean.update({"id": pid, "port": "single", "measurement_id": mid,
                      "score_spec_id": sid, "pattern_sha256": sha,
                      "model_hash": combined_model_hash,
                      "prediction_is_navigation_only": True,
                      "predicted_before_hfss": True,
                      "surrogate_generation": result["surrogate_generation"],
                      "valid_observations_at_fit": result["valid_observations_at_fit"],
                      "update_every_valid_points": cfg.update_every_valid_points})
        if result["legacy_geometry_limitation"]:
            clean["legacy_geometry_limitation"] = result["legacy_geometry_limitation"]
        manifest.append(clean)
    pb.atomic_json(output_dir / "manifest.json", manifest)
    pb.validate_input(output_dir, profile_cfg)
    return output_dir


def write_audit(result: Mapping[str, Any], output_dir: Path, elapsed_s: float) -> Path:
    rows, selected = result["rows"], result["selected_rows"]
    cfg: PoolConfig = result["config"]
    audit = {
        "schema_version": 1,
        "controller_only": True,
        "hfss_run": False,
        "surrogate_training": False,
        "gpu_training": False,
        "historical_data_discovery": False,
        "historical_training_scope": (HISTORICAL_TRAINING_SCOPE if
                                      result["predictor_model_role"] ==
                                      "historical_cold_start_only" else None),
        "predictor_model_role": result["predictor_model_role"],
        "historical_models_cold_start_only": (
            result["predictor_model_role"] == "historical_cold_start_only"),
        "surrogate_generation": result["surrogate_generation"],
        "valid_observations_at_fit": result["valid_observations_at_fit"],
        "predictor_binding": result["predictor_binding"],
        "update_every_valid_points": cfg.update_every_valid_points,
        "next_update_at_valid_observations": (result["valid_observations_at_fit"] +
                                                cfg.update_every_valid_points),
        "pre_hfss_predictions_preserved_in_manifest": True,
        "candidate_count": len(rows),
        "selected_count": len(selected),
        "selected_arm_counts": {
            arm: sum(r["selection_arm"] == arm for r in selected)
            for arm in ("lcb_performance", "high_disagreement", "blind_random")
        },
        "unique_ancestries_selected": len({r["ancestry_id"] for r in selected}),
        "factory_band_ghz": list(FACTORY_BAND_GHZ),
        "factory_targets": {"S11_max_db": -10.0, "Gain_min_dbi": 4.0},
        "radiation_navigation": {"window_deg": [-45.0, 45.0], "floor_from_boresight_db": 3.0,
                                 "hard_acceptance_gate": False},
        "symmetry_performance_threshold": None,
        "measurement_profile": str(cfg.profile_config),
        "diag_bridge_w_mm": 0.1,
        "model_ids": result["model_ids"],
        "model_source_files": ["antenna/models/surrogates.py", "script/sm_invert.py"],
        "legacy_geometry_limitation": result["legacy_geometry_limitation"],
        "elapsed_s": float(elapsed_s),
        "scored_patterns_per_s": float(len(rows) / elapsed_s) if elapsed_s > 0 else None,
        "config": _jsonable(asdict(cfg)),
    }
    path = Path(output_dir) / "sm_pool_audit.json"
    pb.atomic_json(path, audit)
    return path


def load_seed_rows(path: Path) -> list[dict[str, Any]]:
    """Load only the explicitly named JSON rows and their explicitly named tensors."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("rows") if isinstance(raw, Mapping) else raw
    if not isinstance(rows, list):
        raise ValueError("seed JSON must be a list or {'rows': [...]} mapping")
    out = []
    for i, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise ValueError(f"seed row {i} is not a mapping")
        row = dict(source)
        if "pattern" not in row:
            rel = row.get("pattern_file") or (f"{row['id']}.pt" if "id" in row else None)
            if rel is None:
                raise ValueError(f"seed row {i} has no pattern or pattern_file")
            tensor_path = (path.parent / rel).resolve()
            row["pattern"] = torch.load(tensor_path, map_location="cpu", weights_only=True)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a symmetric historical-SM navigation pool")
    parser.add_argument("--config", type=Path, required=True, help="explicit pool YAML/JSON")
    parser.add_argument("--seed-rows", type=Path, required=True, help="explicit seed row JSON")
    parser.add_argument("--exclude-hashes", type=Path, required=True, help="explicit JSON hash list")
    parser.add_argument("--output", type=Path, required=True, help="new local worker input directory")
    args = parser.parse_args()
    raw_cfg = (yaml.safe_load(args.config.read_text(encoding="utf-8"))
               if args.config.suffix.lower() in {".yaml", ".yml"}
               else json.loads(args.config.read_text(encoding="utf-8")))
    cfg = PoolConfig.from_mapping(raw_cfg)
    seeds = load_seed_rows(args.seed_rows)
    raw_excluded = json.loads(args.exclude_hashes.read_text(encoding="utf-8"))
    excludes = raw_excluded.get("hashes", raw_excluded) if isinstance(raw_excluded, Mapping) else raw_excluded
    if not isinstance(excludes, list):
        raise ValueError("exclude-hashes JSON must be a list or {'hashes': [...]} mapping")
    started = time.perf_counter()
    result = build_pool(cfg, seeds, excludes)
    write_bundle(result, args.output)
    audit = write_audit(result, args.output, time.perf_counter() - started)
    print(f"pool={len(result['rows'])} selected={len(result['selected_rows'])} bundle={args.output}")
    print(f"audit={audit}")
    print(f"WARNING: {LEGACY_LIMITATION}")


if __name__ == "__main__":
    main()
