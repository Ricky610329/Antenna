from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from antenna.measurement import measurement_id, score_spec_id
from script import profiled_batch as pb
from script import symmetry_factory_cycle as cycle
from script import symmetry_factory_watch as watch
from script import symmetry_training as training


REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "configs" / "single_r80_symmetry_factory.yaml"
POLICY = REPO / "configs" / "r80_diverse_sm_search_v1.json"
RETRY_WORKERS = ["140.123.106.216", "140.123.106.218", "140.123.106.37"]


class CurrentPredictor:
    model_role = "current_profile"
    model_ids = [{"file": "fixture", "sha256": "0" * 64}]
    binding = {"fixture": "current"}


def _pattern(index: int) -> torch.Tensor:
    value = torch.zeros((25, 25), dtype=torch.float32)
    value[24, 12] = 1
    for bit in range(10):
        if index & (1 << bit):
            row, col = 1 + bit * 2, 1 + bit % 11
            value[row, col] = value[row, 24 - col] = 1
    return value


def _bundle(root: Path, count: int, *, prefix: str = "item") -> Path:
    root.mkdir(parents=True)
    shutil.copy2(PROFILE, root / "config.yaml")
    cfg = pb.load_profile_config(root / "config.yaml")
    pb.atomic_json(root / "measurement.json", cfg.measurement)
    pb.atomic_json(root / "score_spec.json", cfg.score_spec)
    mid, sid = measurement_id(cfg.measurement), score_spec_id(cfg.score_spec)
    rows = []
    for index in range(count):
        pattern = _pattern(index)
        name = f"{prefix}-{index:05d}"
        torch.save(pattern, root / f"{name}.pt")
        rows.append({
            "id": name, "pattern_file": f"{name}.pt", "port": "single",
            "measurement_id": mid, "score_spec_id": sid,
            "pattern_sha256": pb.pattern_sha256(pattern),
            "lineage_id": f"line-{index}", "prediction": [float(index)],
        })
    pb.atomic_json(root / "manifest.json", rows)
    return root


def _training(root: Path) -> Path:
    cfg = pb.load_profile_config(PROFILE)
    return training.initialize_workdir(
        root, cfg.measurement, cfg.score_spec, geometry_profile="single_v1",
        holdout_seed=7, holdout_fraction=0.2, ensemble_seeds=(1,), hidden_dims=(8,),
        legacy_epochs=0, current_epochs=1, update_min=48, update_max=96)


def _dataset(root: Path) -> Path:
    root.mkdir(); (root / "jobs_state").mkdir()
    pb.atomic_json(root / "jobs.json", [])
    return root


class DiverseFactoryIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = _dataset(self.root / "dataset")
        self.training = _training(self.root / "training")
        self.seeds = _bundle(self.root / "seeds", 4)
        self.policy = self.root / "search-policy.json"
        shutil.copy2(POLICY, self.policy)
        self.predictor = CurrentPredictor()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _prepare(self, work: Path, *, search_policy: Path | None, diverse: bool = True):
        calls = []

        def build_diverse(**kwargs):
            calls.append(kwargs)
            cfg = kwargs["pool_cfg"]
            return {"count": kwargs["pool_cfg"].selected_count,
                    "config": cfg,
                    "diversity_audit": {
                        "schema_version": 1,
                        "policy_sha256": cycle._content_id(kwargs["policy"]),
                        "pool_seed": cfg.seed,
                        "candidate_count": cfg.candidate_pool_size,
                        "selected_count": cfg.selected_count,
                        "selected_arm_counts": {
                            "global_lcb": 16, "parent_lcb": 16,
                            "high_disagreement": 16,
                        },
                    }}

        def build_legacy(config, _seeds, exclusions, predictor=None):
            calls.append({"pool_cfg": config, "exclusions": exclusions,
                          "predictor": predictor, "legacy": True})
            return {"count": config.selected_count, "config": config}

        def write_bundle(result, output):
            return _bundle(Path(output), result["count"], prefix=result["config"].id_prefix)

        def write_audit(_result, output, _elapsed):
            pb.atomic_json(Path(output) / "sm_pool_audit.json", {"fixture": True})

        with mock.patch.object(cycle, "_trained", return_value=(
                1, set(), {"manifest_id": "bound", "data_version": 1})), \
                mock.patch.object(cycle.training, "load_current_predictor",
                                  return_value=self.predictor), \
                mock.patch.object(cycle.diverse, "build_diverse_pool",
                                  side_effect=build_diverse), \
                mock.patch.object(cycle.sm, "build_pool", side_effect=build_legacy), \
                mock.patch.object(cycle.sm, "write_bundle", side_effect=write_bundle), \
                mock.patch.object(cycle.sm, "write_audit", side_effect=write_audit):
            with mock.patch.object(
                    cycle.pb, "validate_pattern",
                    side_effect=lambda pattern, _cfg: pb.pattern_sha256(pattern)):
                receipt_path = cycle.run_once(
                    local_workdir=work, dataset_root=self.dataset,
                    profile_config=PROFILE, training_workdir=self.training,
                    seed_inputs=[self.seeds], search_policy=search_policy,
                    expected_retry_workers=RETRY_WORKERS)
        self.assertEqual(bool(calls[0].get("legacy")), not diverse)
        return receipt_path, pb.read_json(receipt_path), calls

    def test_diverse_policy_uses_new_pool_and_binds_audits(self):
        receipt_path, receipt, calls = self._prepare(
            self.root / "cycle", search_policy=self.policy)
        call = calls[0]
        self.assertEqual(call["pool_cfg"].candidate_pool_size, 20_000)
        self.assertEqual(call["pool_cfg"].parent_count, 32)
        self.assertEqual(call["pool_cfg"].seed, 80_100_957)
        self.assertEqual(call["version"], 1)
        self.assertEqual(call["training_workdir"], self.training.resolve())
        self.assertEqual([job["count"] for job in receipt["planned_jobs"]], [16, 16, 16])
        self.assertEqual(receipt["search_policy"]["sha256"], pb.file_sha256(self.policy))
        audit = receipt["guided_pool_audit"]
        self.assertEqual(
            pb.read_json(Path(audit["bundle"]) / "sm_pool_audit.json")[
                "selected_arm_counts"],
            {"global_lcb": 16, "parent_lcb": 16, "high_disagreement": 16})
        self.assertEqual(audit["diversity_search_audit_sha256"], pb.file_sha256(
            Path(audit["bundle"]) / "diversity_search_audit.json"))
        marker = pb.read_json(Path(audit["bundle"]) / "factory_bundle_complete.json")
        self.assertEqual(marker["search_policy"], receipt["search_policy"])
        self.assertEqual(marker["diversity_search_audit_sha256"],
                         audit["diversity_search_audit_sha256"])
        self.assertTrue(receipt_path.is_file())

    def test_absent_policy_preserves_legacy_pool_and_receipt_shape(self):
        receipt_path, receipt, calls = self._prepare(
            self.root / "legacy-cycle", search_policy=None, diverse=False)
        self.assertEqual(calls[0]["pool_cfg"].candidate_pool_size, 10_000)
        self.assertNotIn("search_policy", receipt)
        self.assertNotIn("guided_pool_audit", receipt)
        marker = pb.read_json(receipt_path.parent / "guided_bundle/factory_bundle_complete.json")
        self.assertNotIn("search_policy", marker)
        self.assertNotIn("diversity_search_audit_sha256", marker)

    def test_active_prepared_cycle_rejects_policy_byte_change(self):
        work = self.root / "recovery-cycle"
        self._prepare(work, search_policy=self.policy)
        value = json.loads(self.policy.read_text(encoding="utf-8"))
        self.policy.write_text(json.dumps(value, indent=4), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different search policy binding"):
            cycle.run_once(
                local_workdir=work, dataset_root=self.dataset, profile_config=PROFILE,
                training_workdir=self.training, seed_inputs=[self.seeds],
                search_policy=self.policy, expected_retry_workers=RETRY_WORKERS)

    def test_dispatch_rejects_policy_byte_change_before_queue_add(self):
        receipt_path, _receipt, _calls = self._prepare(
            self.root / "dispatch-cycle", search_policy=self.policy)
        value = json.loads(self.policy.read_text(encoding="utf-8"))
        self.policy.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
        added = []
        with self.assertRaisesRegex(ValueError, "prepared binding"):
            cycle.commit_dispatch(
                receipt_path,
                queue_add=lambda *_args: added.append(True),
                duplicate_check=lambda *_args: 1)
        self.assertEqual(added, [])
        self.assertEqual(pb.read_json(self.dataset / "jobs.json"), [])

    def test_dispatch_rejects_diversity_audit_change_before_queue_add(self):
        receipt_path, receipt, _calls = self._prepare(
            self.root / "audit-dispatch-cycle", search_policy=self.policy)
        audit_path = (Path(receipt["guided_pool_audit"]["bundle"]) /
                      "diversity_search_audit.json")
        value = pb.read_json(audit_path); value["tampered"] = True
        pb.atomic_json(audit_path, value)
        added = []
        with self.assertRaisesRegex(ValueError, "pool audit differs"):
            cycle.commit_dispatch(
                receipt_path,
                queue_add=lambda *_args: added.append(True),
                duplicate_check=lambda *_args: 1)
        self.assertEqual(added, [])
        self.assertEqual(pb.read_json(self.dataset / "jobs.json"), [])

    def test_watch_passes_and_revalidates_bound_policy(self):
        settings = self.root / "watch-settings.json"
        payload = {
            "local_workdir": str(self.root / "watch"),
            "dataset_root": str(self.dataset), "profile_config": str(PROFILE),
            "training_workdir": str(self.training), "seed_inputs": [str(self.seeds)],
            "search_policy": str(self.policy), "interval_seconds": 1800,
            "expected_retry_workers": RETRY_WORKERS,
        }
        settings.write_text(json.dumps(payload), encoding="utf-8")
        dispatches = []

        def prepare(**kwargs):
            self.assertEqual(kwargs["search_policy"], str(self.policy.resolve()))
            _policy, binding = cycle._load_search_policy(self.policy, PROFILE)
            receipt = Path(kwargs["local_workdir"]) / "receipt.json"
            pb.atomic_json(receipt, {
                "status": "prepared", "valid_unique": 1, "pending_unique": 0,
                "target_valid_unique": 5000, "latest_data_version": 1,
                "expected_retry_workers": RETRY_WORKERS, "search_policy": binding,
            })
            return receipt

        def dispatch(path):
            value = pb.read_json(path); value["status"] = "dispatched"
            pb.atomic_json(path, value); dispatches.append(Path(path))

        result = watch.run(
            settings, prepare=prepare, dispatch=dispatch, max_cycles=1,
            campaign_stop=lambda _settings: None)
        self.assertEqual(result["status"], "bounded_run_complete")
        self.assertEqual(len(dispatches), 1)
        self.assertEqual(result["search_policy"]["sha256"], pb.file_sha256(self.policy))


if __name__ == "__main__":
    unittest.main()
