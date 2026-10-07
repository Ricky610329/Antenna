from __future__ import annotations

import json
import shutil
import sys
import tempfile
import types
import unittest
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from antenna.measurement import measurement_id, score_spec_id
from script.exploration import TrainingInterrupted, file_sha256, pattern_sha256
from script.profiled_batch import observation
from script.symmetry_training import (
    fixed_group_split,
    CurrentProfilePredictor,
    ingest_profile_stores,
    initialize_workdir,
    load_current_predictor,
    prepare_legacy_manifest,
    register_legacy_prior,
    train_version,
)


def _measurement():
    return {
        "name": "symmetry-test",
        "port": "single",
        "labels": ["S11_dB", "Gain_dBi"],
        "geometry": {"geom": "single_v1", "pixel_count": 25, "diag_bridge_w": 0.1},
        "sweep": {"start_ghz": 24.0, "stop_ghz": 32.0, "step_ghz": 0.5,
                  "type": "Interpolating"},
        "solver": {"setup_freq_ghz": 28.0, "open_region_freq_ghz": 32.0,
                   "max_delta_s": 0.02, "max_passes": 15, "min_passes": 2,
                   "min_converged": 2},
    }


def _score():
    return {"name": "test-score", "bands": [
        {"name": "return", "label": "S11_dB", "start_ghz": 27.5,
         "stop_ghz": 28.5, "relation": "le", "threshold": -10.0},
        {"name": "gain", "label": "Gain_dBi", "start_ghz": 27.5,
         "stop_ghz": 28.5, "relation": "ge", "threshold": 0.0},
    ]}


def _symmetric_pattern(index: int) -> torch.Tensor:
    p = torch.zeros((25, 25), dtype=torch.float32)
    p[24, 12] = 1
    row = 2 + index % 20
    col = 1 + index % 11
    p[row, col] = p[row, 24 - col] = 1
    return p


def _rad(offset: float = 0.0):
    theta = torch.arange(-180, 181, 2, dtype=torch.float32)
    return {"theta": theta, "phi0": torch.cos(torch.deg2rad(theta)) + offset,
            "phi90": torch.sin(torch.deg2rad(theta)) + offset}


def _response(index: int) -> torch.Tensor:
    f = torch.arange(17, dtype=torch.float32)
    return torch.stack((-15.0 + 0.05 * f + index, 2.0 + 0.1 * f - index * 0.1))


def _write_completed_store(root: Path, specs: list[tuple[str, str, int]], *,
                           ancestry_only: bool = False) -> Path:
    root.mkdir(parents=True)
    measurement, score = _measurement(), _score()
    cfg = SimpleNamespace(port="single", measurement=measurement, score_spec=score)
    rows, results = [], {}
    (root / "rad").mkdir()
    for name, lineage, index in specs:
        pattern, response, rad = _symmetric_pattern(index), _response(index), _rad(index * 0.01)
        row = {"id": name, "port": "single", "measurement_id": measurement_id(measurement),
               "score_spec_id": score_spec_id(score), "pattern_sha256": pattern_sha256(pattern)}
        row["ancestry_id" if ancestry_only else "lineage_id"] = lineage
        entry = observation(response, rad, row, pattern, cfg, 1.0 + index)
        sample_path, rad_path = root / entry["sample_file"], root / entry["rad_file"]
        torch.save((pattern, response), sample_path)
        torch.save(rad, rad_path)
        entry["sample_sha256"] = file_sha256(sample_path)
        entry["rad_sha256"] = file_sha256(rad_path)
        rows.append(row); results[name] = entry
    for name, value in (("measurement.json", measurement), ("score_spec.json", score),
                        ("manifest.json", rows), ("results.json", results)):
        (root / name).write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return root


def _lineages(seed: int, fraction: float, count: int = 2):
    found = {"train": [], "holdout": []}
    index = 0
    while min(map(len, found.values())) < count:
        value = f"lineage-{index}"
        found[fixed_group_split(value, fraction, seed)].append(value)
        index += 1
    return found


def _init(root: Path, *, current_epochs: int = 2, legacy_epochs: int = 1,
          ensemble_seeds=(3,), initial_lineage_aliases=None):
    return initialize_workdir(
        root, _measurement(), _score(), geometry_profile="single_v1",
        holdout_seed=71, holdout_fraction=0.5, ensemble_seeds=ensemble_seeds, hidden_dims=(8,),
        batch_size=2, current_epochs=current_epochs, legacy_epochs=legacy_epochs,
        update_min=2, update_max=8,
        initial_lineage_aliases=initial_lineage_aliases,
        alias_source_sha256="ab" * 32 if initial_lineage_aliases else None,
    )


class SymmetryTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_ingests_independent_completed_stores_with_fixed_group_holdout(self):
        work = _init(self.tmp_path / "work")
        groups = _lineages(71, 0.5)
        one = _write_completed_store(self.tmp_path / "one", [("a", groups["train"][0], 1),
                                                               ("b", groups["holdout"][0], 2)])
        two = _write_completed_store(self.tmp_path / "two", [("c", groups["train"][0], 3),
                                                               ("d", groups["holdout"][1], 4)])
        version = ingest_profile_stores(work, [one, two], min_new=4, max_new=4)
        rows = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 4)
        by_lineage = {}
        for row in rows:
            by_lineage.setdefault(row["lineage_id"], set()).add(row["split"])
            self.assertEqual(row["source_sample_sha256"], file_sha256(work / row["sample_file"]))
            self.assertEqual(row["geometry_profile"], "single_v1")
        self.assertTrue(all(len(splits) == 1 for splits in by_lineage.values()))
        self.assertEqual(ingest_profile_stores(work, [one, two], min_new=4, max_new=4), version)

        sample = next(iter(one.glob("*.pt")))
        sample.write_bytes(sample.read_bytes() + b"tamper")
        with self.assertRaises(ValueError):
            ingest_profile_stores(work, [one], min_new=1, max_new=9)

    def test_unique_trigger_and_ancestry_group_survive_leaf_entry_ids(self):
        work = _init(self.tmp_path / "work")
        groups = _lineages(71, 0.5)
        store = _write_completed_store(self.tmp_path / "ancestry", [
            ("leaf-a", groups["train"][0], 6),
            ("leaf-b", groups["train"][0], 7),
            ("repeat-c", "conflicting-alias", 6),
        ], ancestry_only=True)
        version = ingest_profile_stores(work, [store], min_new=2, max_new=2)
        rows = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
        receipt = json.loads((version / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["new_valid_unique"], 2)
        self.assertEqual(receipt["new_observations"], 3)
        by_pattern = {}
        for row in rows:
            by_pattern.setdefault(row["pattern_sha256"], set()).add(row["canonical_group_id"])
        self.assertTrue(all(len(groups_for_pattern) == 1 for groups_for_pattern in by_pattern.values()))
        self.assertEqual({row["split"] for row in rows}, {"profile_train"})

    def test_seed_variant_aliases_share_one_immutable_family_group(self):
        aliases = {"seed-variant-a": "family-top", "seed-variant-b": "family-top"}
        work = _init(self.tmp_path / "work", initial_lineage_aliases=aliases)
        store = _write_completed_store(self.tmp_path / "aliased", [
            ("leaf-a", "seed-variant-a", 10), ("leaf-b", "seed-variant-b", 11),
        ], ancestry_only=True)
        version = ingest_profile_stores(work, [store], min_new=2, max_new=2)
        rows = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
        protocol = json.loads((work / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual({row["canonical_group_id"] for row in rows}, {"family-top"})
        self.assertEqual(len({row["split"] for row in rows}), 1)
        self.assertEqual(protocol["lineage_aliases"]["mapping"], aliases)
        self.assertEqual(protocol["lineage_aliases"]["source_sha256"], "ab" * 32)

    def test_legacy_prior_records_mismatch_and_allows_nonmirror_pattern(self):
        work = _init(self.tmp_path / "work")
        source = self.tmp_path / "legacy-source"
        (source / "samples").mkdir(parents=True); (source / "rad").mkdir()
        groups = _lineages(71, 0.5)
        rows = []
        for index, lineage in enumerate((groups["train"][0], groups["holdout"][0])):
            pattern = torch.zeros((25, 25)); pattern[1 + index, 2] = 1
            sample = source / "samples" / f"{index}.pt"
            rad = source / "rad" / f"{index}.pt"
            torch.save((pattern, _response(index)), sample); torch.save(_rad(index), rad)
            rows.append({"id": f"old-{index}", "status": "ok", "lineage_id": lineage,
                         "sample_file": sample.relative_to(source).as_posix(),
                         "rad_file": rad.relative_to(source).as_posix()})
        manifest = source / "manifest.json"
        manifest.write_text(json.dumps(rows), encoding="utf-8")
        result = register_legacy_prior(work, manifest, source, {
            "source_name": "historical-v108",
            "source_geometry_profile": "legacy-bridge-unknown",
            "compatibility_note": "2x17 sweep compatible; bridge/profile mismatch retained",
        })
        registered = json.loads(result.read_text(encoding="utf-8"))
        provenance = json.loads((work / "legacy/provenance.json").read_text(encoding="utf-8"))
        self.assertEqual({row["split"] for row in registered}, {"legacy_train", "legacy_holdout"})
        self.assertEqual(provenance["usage"], "pretraining_prior_only")
        self.assertTrue(provenance["nonmirror_patterns_allowed"])
        self.assertTrue(provenance["excluded_from_current_validation"])

    def test_prepares_only_unambiguous_legacy_response_radiation_pairs(self):
        source = self.tmp_path / "historical"
        store, inputs = source / "round1", source / "round1_input"
        (store / "rad").mkdir(parents=True); inputs.mkdir(parents=True)
        pattern = torch.zeros((25, 25)); pattern[3, 4] = 1
        torch.save((pattern, _response(2)), store / "fingerprint.pt")
        torch.save(pattern, inputs / "candidate.pt")
        torch.save(_rad(2), store / "rad/candidate.pt")
        (store / "hfss_setup.json").write_text(json.dumps({"diag_bridge_w": 0.075}),
                                                encoding="utf-8")
        (inputs / "manifest.json").write_text(json.dumps([
            {"id": "candidate", "source_id": "parent-1", "kind": "legacy"}
        ]), encoding="utf-8")
        output = self.tmp_path / "manifest.json"
        prepare_legacy_manifest(source, output)
        rows = json.loads(output.read_text(encoding="utf-8"))
        audit = json.loads(output.with_suffix(".json.inventory.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["lineage_id"], "parent-1")
        self.assertEqual(rows[0]["geometry_profile"], "legacy_diag_bridge_w_0.075")
        self.assertEqual(rows[0]["source_hfss_setup"], {"diag_bridge_w": 0.075})
        self.assertEqual(audit["eligible_before_cap"], 1)
        self.assertEqual(audit["counts"]["accepted"], 1)

    def test_training_resume_is_bitwise_and_validation_is_profile_only(self):
        work = _init(self.tmp_path / "prepared", ensemble_seeds=(3, 5))
        groups = _lineages(71, 0.5)
        store = _write_completed_store(self.tmp_path / "profile", [
            ("a", groups["train"][0], 1), ("b", groups["train"][1], 2),
            ("c", groups["holdout"][0], 3), ("d", groups["holdout"][1], 4),
        ])
        ingest_profile_stores(work, [store], min_new=4, max_new=4)

        legacy = self.tmp_path / "legacy"
        (legacy / "samples").mkdir(parents=True); (legacy / "rad").mkdir()
        legacy_rows = []
        legacy_lineages = (groups["train"][0], groups["holdout"][0], groups["train"][1])
        for index, lineage in enumerate(legacy_lineages):
            sample, rad = legacy / "samples" / f"{index}.pt", legacy / "rad" / f"{index}.pt"
            pattern = _symmetric_pattern(3) if index == 2 else torch.zeros((25, 25))
            if index != 2:
                pattern[index + 1, 2] = 1
            torch.save((pattern, _response(index + 5)), sample)
            torch.save(_rad(index + 5), rad)
            legacy_rows.append({"id": f"legacy-{index}", "lineage_id": lineage,
                                "sample_file": sample.relative_to(legacy).as_posix(),
                                "rad_file": rad.relative_to(legacy).as_posix()})
        legacy_manifest = legacy / "manifest.json"
        legacy_manifest.write_text(json.dumps(legacy_rows), encoding="utf-8")
        register_legacy_prior(work, legacy_manifest, legacy, {
            "source_name": "fixture", "source_geometry_profile": "legacy-0.0",
            "compatibility_note": "response/radiation shape compatible only",
        })

        uninterrupted = self.tmp_path / "uninterrupted"
        resumed = self.tmp_path / "resumed"
        shutil.copytree(work, uninterrupted); shutil.copytree(work, resumed)
        full_paths = train_version(uninterrupted)
        with self.assertRaises(TrainingInterrupted):
            train_version(resumed, interrupt_after_epochs=1)
        resumed_paths = train_version(resumed)
        for full_path, resumed_path in zip(full_paths, resumed_paths):
            full = torch.load(full_path, weights_only=False, map_location="cpu")
            recovered = torch.load(resumed_path, weights_only=False, map_location="cpu")
            self.assertTrue(full["complete"] and recovered["complete"])
            self.assertEqual(full["signature"], recovered["signature"])
            for name in full["model_state"]:
                self.assertTrue(torch.equal(full["model_state"][name], recovered["model_state"][name]))
        summary = json.loads((resumed / "models/data-v001/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["counts"], {"profile_train": 2, "profile_holdout": 2,
                                              "legacy_train": 1, "legacy_holdout": 1})
        self.assertEqual(len(summary["legacy_train_exclusions_for_current_holdout"]), 1)
        self.assertTrue(set(summary["profile_train_ids"]).isdisjoint(summary["profile_holdout_ids"]))
        self.assertEqual(summary["validation_policy"],
                         "current_profile_holdout_only_no_early_stopping")
        self.assertEqual(summary["metrics"][0]["profile_holdout"]["n"], 2)
        predictor = load_current_predictor(resumed)
        pool_stub = types.ModuleType("script.symmetry_sm_pool")
        pool_stub.PredictionBatch = namedtuple(
            "PredictionBatch",
            "response_mean response_disagreement radiation radiation_theta response_members",
        )
        with mock.patch.dict(sys.modules, {"script.symmetry_sm_pool": pool_stub}):
            prediction = predictor.predict([_symmetric_pattern(8), _symmetric_pattern(9)], batch_size=1)
        self.assertEqual(prediction.response_mean.shape, (2, 2, 17))
        self.assertEqual(prediction.response_members.shape, (2, 2, 2, 17))
        self.assertEqual(prediction.radiation.shape, (2, 2, 91))
        self.assertEqual(prediction.radiation_theta.shape, (91,))
        self.assertEqual(len(predictor.model_ids), 2)
        self.assertEqual(predictor.model_ids[0]["data_version"], "1")
        self.assertEqual(predictor.model_role, "current_profile")
        self.assertEqual(predictor.binding["cumulative_valid_unique"], 4)
        self.assertIn("input_norm", recovered)
        self.assertIn("target_norm", recovered)
        self.assertNotIn("norms", recovered)
        with self.assertRaises(ValueError):
            CurrentProfilePredictor([resumed_paths[0], resumed_paths[0]])
        shutil.copyfile(resumed_paths[0], resumed_paths[1])
        with self.assertRaises(ValueError):
            load_current_predictor(resumed)


if __name__ == "__main__":
    unittest.main()
