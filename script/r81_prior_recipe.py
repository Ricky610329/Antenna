"""Build the opt-in R81 masked-prior exploration config.

This is deliberately a config-only operation.  It neither extracts nor loads
the historical samples, and it never prepares or dispatches an exploration
run.  Full prior/sample validation remains the existing ``prepare_run`` gate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from script import exploration as ex
from script import filter_training


BASE_CONFIG = REPO / "configs" / "dual_r81_wide_filter.yaml"
EXPECTED_PRIOR_MANIFEST_SHA256 = (
    "8fcd05614e7851e9c0f7043c00540068df37380423ee4469db0afed90505c777"
)
PRODUCTION_OVERRIDES = {
    "training_protocol": "filter_masked_prior_v1",
    "pretrain_epochs": 30,
    "epochs": 100,
    "holdout_fraction": 0.2,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base_config() -> dict[str, Any]:
    """Load the tracked R81 base without modifying it."""
    return ex._load_yaml(BASE_CONFIG)


def build_recipe(
    prior_manifest: str | Path,
    prior_sample_root: str | Path,
    *,
    base_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the R81 config with only the masked-prior fields changed.

    Paths are required to be absolute, but this pure builder does not touch
    them.  The CLI performs the small existence/hash gate before it writes a
    config; the exploration lifecycle performs the complete prior preflight.
    """
    manifest = Path(prior_manifest)
    sample_root = Path(prior_sample_root)
    if not manifest.is_absolute() or not sample_root.is_absolute():
        raise ValueError("prior manifest and sample root must be absolute paths")

    base = copy.deepcopy(dict(base_config if base_config is not None else load_base_config()))
    before = copy.deepcopy(base)
    exploration = copy.deepcopy(dict(base.get("exploration", {})))
    exploration.update(PRODUCTION_OVERRIDES)
    exploration["filter_prior"] = {
        "manifest": str(manifest),
        "sample_root": str(sample_root),
        "expected_manifest_sha256": EXPECTED_PRIOR_MANIFEST_SHA256,
    }
    base["exploration"] = exploration

    checked = ex.validate_exploration_config(base)
    spec = filter_training._prior_spec(checked)
    if spec["expected_manifest_sha256"] != EXPECTED_PRIOR_MANIFEST_SHA256:
        raise ValueError("masked-prior manifest binding changed during validation")

    for key in ("measurement", "score_spec", "runtime"):
        if checked.get(key) != before.get(key):
            raise ValueError(f"recipe changed base scientific field: {key}")
    unchanged_exploration = set(before["exploration"]) - set(PRODUCTION_OVERRIDES)
    for key in unchanged_exploration:
        if checked["exploration"].get(key) != before["exploration"].get(key):
            raise ValueError(f"recipe changed base exploration field: {key}")
    return checked


def validate_artifact_location(config: Mapping[str, Any]) -> None:
    """Check only the caller's extracted manifest/root location.

    This intentionally does not load 21,034 samples or apply current holdout
    exclusions.  Those are execution-time gates in ``filter_training``.
    """
    spec = filter_training._prior_spec(config)
    manifest = spec["manifest"]
    sample_root = spec["sample_root"]
    if not manifest.is_file():
        raise ValueError("prior manifest does not exist")
    if not sample_root.is_dir():
        raise ValueError("prior sample root does not exist")
    if _sha256(manifest) != EXPECTED_PRIOR_MANIFEST_SHA256:
        raise ValueError("prior manifest SHA-256 differs from the frozen R81 prior")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a config-only R81 masked historical-prior recipe."
    )
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--prior-sample-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    config = build_recipe(args.prior_manifest, args.prior_sample_root)
    validate_artifact_location(config)
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"refusing to overwrite existing config: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(config, sort_keys=False)
    with output.open("x", encoding="utf-8", newline="") as stream:
        stream.write(payload)
    # Reparse the exact emitted bytes through the production schema API.
    ex.validate_exploration_config(ex._load_yaml(output))
    print(f"wrote config-only recipe: {output}")
    print("full historical-prior preflight and execution gates have not run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
