from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from script.exploration import TrainingInterrupted, content_id
from script.symmetry_training import (
    fixed_group_split,
    ingest_profile_stores,
    initialize_workdir,
    train_version,
)
from tests.test_symmetry_training import (
    _init,
    _lineages,
    _measurement,
    _score,
    _write_completed_store,
)


REFERENCE_PROTOCOL_ID = "12" * 32
DEFAULT_PROTOCOL_SHA256 = "1335298b18f2e122644a37b39f4026c13455630a54a95969d428c2fd8bf5e211"
DEFAULT_PROTOCOL_ID = "80213e65ee1da7aaf8d82c0ad7aad3385350747033ccf901b77d8c60e61d13ca"


def _init_development(root: Path, development_group: str, *, aliases=None,
                      current_epochs: int = 2) -> Path:
    return initialize_workdir(
        root, _measurement(), _score(), geometry_profile="single_v1",
        holdout_seed=71, holdout_fraction=0.5, ensemble_seeds=(3,), hidden_dims=(8,),
        batch_size=2, current_epochs=current_epochs, legacy_epochs=0,
        update_min=2, update_max=8,
        initial_lineage_aliases=aliases,
        alias_source_sha256="ab" * 32 if aliases else None,
        development_train_groups=(development_group,),
        reference_protocol_id=REFERENCE_PROTOCOL_ID,
    )


def _rewrite_manifest_receipt(version: Path, rows: list[dict[str, Any]]) -> None:
    (version / "manifest.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt = json.loads((version / "receipt.json").read_text(encoding="utf-8"))
    receipt["manifest_id"] = content_id(rows)
    (version / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _unknown_group_with_role(role: str) -> str:
    for index in range(10_000):
        candidate = f"unknown-development-group-{index}"
        if fixed_group_split(candidate, 0.5, 71) == role:
            return candidate
    raise AssertionError("unable to construct deterministic role fixture")


class DevelopmentTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def assert_nested_equal(self, left: Any, right: Any) -> None:
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_default_protocol_is_byte_exact_and_has_no_development_fields(self):
        work = _init(self.tmp_path / "default")
        raw = (work / "protocol.json").read_bytes()
        protocol = json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), DEFAULT_PROTOCOL_SHA256)
        self.assertEqual(protocol["protocol_id"], DEFAULT_PROTOCOL_ID)
        self.assertNotIn("development_train_groups", protocol)
        self.assertNotIn("reference_protocol_id", protocol)
        self.assertNotIn("development_role_policy", protocol)
        groups = _lineages(71, 0.5)
        store = _write_completed_store(self.tmp_path / "default-store", [
            ("train", groups["train"][0], 20), ("hold", groups["holdout"][0], 21),
        ])
        ingest_profile_stores(work, [store], min_new=2, max_new=2)
        checkpoint = torch.load(train_version(work)[0], weights_only=False, map_location="cpu")
        summary = json.loads(
            (work / "models/data-v001/summary.json").read_text(encoding="utf-8"))
        self.assertNotIn("development_training", checkpoint)
        self.assertNotIn("development_training", summary)
        self.assertEqual(summary["validation_policy"],
                         "current_profile_holdout_only_no_early_stopping")

    def test_reference_id_is_required_and_protocol_is_immutable(self):
        args = dict(
            measurement=_measurement(), score_spec=_score(), geometry_profile="single_v1",
            holdout_seed=71, holdout_fraction=0.5, ensemble_seeds=(3,), hidden_dims=(8,),
            batch_size=2, current_epochs=2, legacy_epochs=0, update_min=2, update_max=8,
        )
        with self.assertRaises(ValueError):
            initialize_workdir(self.tmp_path / "missing", **args,
                               development_train_groups=("family",))
        with self.assertRaises(ValueError):
            initialize_workdir(self.tmp_path / "bad", **args,
                               development_train_groups=("family",), reference_protocol_id="bad")
        with self.assertRaises(ValueError):
            initialize_workdir(self.tmp_path / "stray", **args,
                               reference_protocol_id=REFERENCE_PROTOCOL_ID)

        existing = initialize_workdir(self.tmp_path / "immutable", **args)
        self.assertEqual(existing, self.tmp_path / "immutable")
        with self.assertRaises(ValueError):
            initialize_workdir(existing, **args, development_train_groups=("family",),
                               reference_protocol_id=REFERENCE_PROTOCOL_ID)

        opt_in = initialize_workdir(
            self.tmp_path / "opt-in", **args, development_train_groups=("family",),
            reference_protocol_id=REFERENCE_PROTOCOL_ID)
        self.assertEqual(initialize_workdir(
            opt_in, **args, development_train_groups=("family",),
            reference_protocol_id=REFERENCE_PROTOCOL_ID), opt_in)
        with self.assertRaises(ValueError):
            initialize_workdir(opt_in, **args, development_train_groups=("family",),
                               reference_protocol_id="34" * 32)

    def test_development_group_identifiers_are_strict_nonempty_strings(self):
        for bad_group in (None, 17, True, b"family", ""):
            with self.subTest(bad_group=bad_group), self.assertRaises((TypeError, ValueError)):
                initialize_workdir(
                    self.tmp_path / f"bad-{type(bad_group).__name__}-{repr(bad_group)}",
                    _measurement(), _score(), geometry_profile="single_v1",
                    development_train_groups=(bad_group,),
                    reference_protocol_id=REFERENCE_PROTOCOL_ID,
                )

    def test_alias_closure_is_whole_group_and_other_roles_stay_reference_roles(self):
        groups = _lineages(71, 0.5)
        family = groups["holdout"][0]
        aliases = {"variant-a": family, "variant-b": family}
        work = _init_development(self.tmp_path / "work", "variant-a", aliases=aliases)
        store = _write_completed_store(self.tmp_path / "store", [
            ("merge-first", "unbound-alias", 1), ("a", "variant-a", 1),
            ("b", "variant-b", 2),
            ("c", groups["train"][0], 3), ("d", groups["holdout"][1], 4),
        ])
        version = ingest_profile_stores(work, [store], min_new=4, max_new=4)
        rows = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
        protocol = json.loads((work / "protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol["development_train_groups"], [family])
        family_rows = [row for row in rows if row["canonical_group_id"] == family]
        self.assertEqual(len(family_rows), 3)
        self.assertIn("unbound-alias", {row["lineage_id"] for row in family_rows})
        self.assertEqual({row["split"] for row in family_rows}, {"profile_train"})
        self.assertEqual({row["reference_split"] for row in family_rows},
                         {"profile_holdout"})
        other_rows = [row for row in rows if row["canonical_group_id"] != family]
        self.assertTrue(all(row["split"] == row["reference_split"] for row in other_rows))

    def test_later_same_family_arrivals_keep_declared_training_role(self):
        groups = _lineages(71, 0.5)
        family = groups["holdout"][0]
        aliases = {"variant-a": family, "variant-b": family}
        work = _init_development(self.tmp_path / "work", family, aliases=aliases)
        first = _write_completed_store(self.tmp_path / "first", [
            ("a", "variant-a", 5), ("h", groups["holdout"][1], 6),
        ])
        ingest_profile_stores(work, [first], min_new=2, max_new=2)
        second = _write_completed_store(self.tmp_path / "second", [
            ("b", "variant-b", 7), ("t", groups["train"][0], 8),
        ])
        version = ingest_profile_stores(work, [second], min_new=2, max_new=2)
        rows = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
        family_rows = [row for row in rows if row["canonical_group_id"] == family]
        self.assertEqual(len(family_rows), 2)
        self.assertTrue(all(row["split"] == "profile_train" for row in family_rows))
        self.assertTrue(all(row["reference_split"] == "profile_holdout"
                            for row in family_rows))

    def test_training_rejects_declared_development_group_absent_from_manifest(self):
        groups = _lineages(71, 0.5)
        work = _init_development(self.tmp_path / "work", "declared-but-absent")
        store = _write_completed_store(self.tmp_path / "store", [
            ("train", groups["train"][0], 30),
            ("hold", groups["holdout"][0], 31),
        ])
        ingest_profile_stores(work, [store], min_new=2, max_new=2)
        with self.assertRaisesRegex(ValueError, "development groups.*absent"):
            train_version(work)

    def test_tampered_effective_or_reference_roles_are_rejected_before_training(self):
        groups = _lineages(71, 0.5)
        family = groups["holdout"][0]
        work = _init_development(self.tmp_path / "work", family)
        store = _write_completed_store(self.tmp_path / "store", [
            ("a", family, 10), ("b", family, 11),
            ("c", groups["train"][0], 12), ("d", groups["holdout"][1], 13),
        ])
        version = ingest_profile_stores(work, [store], min_new=4, max_new=4)
        rows = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
        target = next(row for row in rows if row["canonical_group_id"] == family)
        target["split"] = "profile_holdout"
        _rewrite_manifest_receipt(version, rows)
        with self.assertRaisesRegex(ValueError, "profile role proof mismatch"):
            train_version(work)

        target["split"] = "profile_train"
        target["reference_split"] = "profile_train"
        _rewrite_manifest_receipt(version, rows)
        with self.assertRaisesRegex(ValueError, "profile role proof mismatch"):
            train_version(work)

    def test_rehashed_canonical_identity_tamper_cannot_detach_lineage_closure(self):
        groups = _lineages(71, 0.5)
        family = groups["holdout"][0]
        aliases = {"variant-a": family}
        work = _init_development(self.tmp_path / "work", family, aliases=aliases)
        store = _write_completed_store(self.tmp_path / "store", [
            ("a", "variant-a", 22), ("b", family, 23),
            ("c", groups["train"][0], 24), ("d", groups["holdout"][1], 25),
        ])
        version = ingest_profile_stores(work, [store], min_new=4, max_new=4)
        rows = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
        target = next(row for row in rows if row["lineage_id"] == "variant-a")
        target["canonical_group_id"] = _unknown_group_with_role("holdout")
        target["split"] = "profile_holdout"
        target["reference_split"] = "profile_holdout"
        _rewrite_manifest_receipt(version, rows)
        with self.assertRaisesRegex(ValueError, "lineage|canonical|group"):
            train_version(work)

    def test_state_cannot_retarget_initial_protocol_alias_root(self):
        groups = _lineages(71, 0.5)
        family = groups["holdout"][0]
        aliases = {"variant-a": family}
        work = _init_development(self.tmp_path / "work", family, aliases=aliases)
        store = _write_completed_store(self.tmp_path / "store", [
            ("a", "variant-a", 26), ("b", family, 27),
            ("c", groups["train"][0], 28), ("d", groups["holdout"][1], 29),
        ])
        version = ingest_profile_stores(work, [store], min_new=4, max_new=4)
        state_path = work / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        replacement = _unknown_group_with_role("train")
        state["lineage_aliases"][family] = replacement
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
        rows = json.loads((version / "manifest.json").read_text(encoding="utf-8"))
        for row in rows:
            if row["canonical_group_id"] == family:
                row["split"] = "profile_train"
                row["reference_split"] = "profile_train"
        _rewrite_manifest_receipt(version, rows)
        with self.assertRaisesRegex(ValueError, "alias|canonical|protocol|role"):
            train_version(work)

    def test_true_tiny_sgd_resume_is_bitwise_with_role_binding(self):
        groups = _lineages(71, 0.5)
        family = groups["holdout"][0]
        aliases = {"variant-a": family, "variant-b": family}
        work = _init_development(self.tmp_path / "prepared", family, aliases=aliases)
        store = _write_completed_store(self.tmp_path / "store", [
            ("a", "variant-a", 14), ("b", "variant-b", 15),
            ("c", groups["train"][0], 16), ("d", groups["holdout"][1], 17),
        ])
        ingest_profile_stores(work, [store], min_new=4, max_new=4)
        direct, resumed = self.tmp_path / "direct", self.tmp_path / "resumed"
        shutil.copytree(work, direct); shutil.copytree(work, resumed)
        direct_path = train_version(direct)[0]
        with self.assertRaises(TrainingInterrupted):
            train_version(resumed, interrupt_after_epochs=1)
        resumed_path = train_version(resumed)[0]
        full = torch.load(direct_path, weights_only=False, map_location="cpu")
        recovered = torch.load(resumed_path, weights_only=False, map_location="cpu")
        self.assertTrue(full["complete"] and recovered["complete"])
        self.assert_nested_equal(full["model_state"], recovered["model_state"])
        self.assert_nested_equal(full["optimizer_state"], recovered["optimizer_state"])
        self.assertEqual(full["development_training"], recovered["development_training"])
        self.assertEqual(set(full["binding"]), {
            "measurement_id", "score_spec_id", "protocol_id", "data_version",
            "manifest_id", "cumulative_valid_unique",
        })
        expected_counts = {
            "development_rows": 2,
            "development_reference_holdout_rows": 2,
            "effective_profile_train": 3,
            "effective_profile_holdout": 1,
            "reference_profile_train": 1,
            "reference_profile_holdout": 3,
        }
        self.assertEqual(full["development_training"]["counts"], expected_counts)
        summary = json.loads(
            (resumed / "models/data-v001/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["development_training"]["counts"], expected_counts)
        self.assertEqual(summary["validation_policy"],
                         "remaining_reference_group_holdouts_descriptive_only_no_early_stopping")


if __name__ == "__main__":
    unittest.main()
