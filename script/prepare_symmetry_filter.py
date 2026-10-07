"""Prepare the bounded R80/R81 experiment locally; never start HFSS.

History is read only from the named input bundles below. Old response values are
used solely to choose dual seed strata, never relabelled as wide-band truth.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import io
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

from antenna.measurement import measurement_id, score_spec_id
from script import exploration as ex
from script import profiled_batch as pb

REPO = Path(__file__).resolve().parents[1]
SCOPE = "symmetry_filter_20261007"
DUAL_STORES = [f"dedust_r79b{b}{part}" for b in (1, 2, 3) for part in "abc"] + [
    f"dedust_smp073{part}" for part in "abc"]


def write_seeds(out, rows):
    out.mkdir(parents=True)
    manifest = []
    for i, row in enumerate(rows):
        item = {k: v for k, v in row.items() if k not in {"pattern", "old_margins"}}
        item["id"] = f"seed_{i:03d}"
        item["pattern_file"] = item["id"] + ".pt"
        item["pattern_sha256"] = ex.pattern_sha256(row["pattern"])
        torch.save(torch.as_tensor(row["pattern"], dtype=torch.float32), out / item["pattern_file"])
        manifest.append(item)
    pb.atomic_json(out / "manifest.json", manifest)


def single_history_seeds(root):
    single = []
    sdir = root / "dedust_r55sym_input"
    for row in pb.read_json(sdir / "manifest.json"):
        path = sdir / (row["id"] + ".pt")
        p = torch.load(path, weights_only=True, map_location="cpu").reshape(25, 25)
        single.append({"pattern": p.numpy(), "original_id": row["id"],
                       "lineage_id": row.get("parent", row["id"]),
                       "source": "R55 historical geometry only; new bridge profile will be measured",
                       "source_file": path.relative_to(root).as_posix(),
                       "source_sha256": pb.file_sha256(path)})
    if len(single) != 34:
        raise ValueError(f"expected 34 R55 single seeds, found {len(single)}")
    return single


def dual_history_seeds(root):
    dual, files = [], []
    for store in DUAL_STORES:
        inp = root / (store + "_input")
        results = pb.read_json(root / store / "results.json")
        files.extend([inp / "manifest.json", root / store / "results.json"])
        for row in pb.read_json(inp / "manifest.json"):
            result = results.get(row["id"], {})
            if result.get("geom") != "p01" or result.get("dbw") != .075 or "error" in result:
                continue
            if not all(np.isfinite(result.get(k, np.nan)) for k in ("m1", "m2", "m3", "m4")):
                continue
            path = inp / (row["id"] + ".pt")
            p = torch.load(path, weights_only=True, map_location="cpu").reshape(25, 25).numpy()
            if not (p[:5, 10:15].all() and p[20:, 10:15].all()):
                raise ValueError(f"historical dual lacks port pads: {path}")
            dual.append({"pattern": p, "original_id": row["id"],
                         # Family grouping is conservative when exact parent is missing.
                         "lineage_id": row.get("parent", row.get("family", row["id"])),
                         "source": "historical narrow-band geometry; not wide-band truth",
                         "source_file": path.relative_to(root).as_posix(),
                         "source_sha256": pb.file_sha256(path),
                         "old_margins": {k: result[k] for k in ("m1", "m2", "m3", "m4")}})
    unique = {}
    for row in dual:
        unique.setdefault(ex.pattern_sha256(row["pattern"]), row)
    pool = list(unique.values())
    if len(pool) < 40:
        raise ValueError("fewer than 40 eligible historical dual seeds")
    # Twenty old robust-margin leaders; no claim of performance under new bands.
    def old_wm(row):
        m = row["old_margins"]
        return min(m["m1"] + 2, m["m2"] + 2, m["m3"], m["m4"] + 5)
    ranked = sorted(pool, key=lambda r: (-old_wm(r), r["original_id"]))
    chosen = [{**r, "source_group": "history"} for r in ranked[:20]]
    used = {ex.pattern_sha256(r["pattern"]) for r in chosen}
    for metric in ("m3", "m4"):
        eligible = sorted([r for r in pool if ex.pattern_sha256(r["pattern"]) not in used],
                          key=lambda r: (-r["old_margins"][metric], r["original_id"]))
        for r in eligible[:10]:
            chosen.append({**r, "source_group": "specialist", "specialist_metric": metric})
            used.add(ex.pattern_sha256(r["pattern"]))
    rng = np.random.default_rng(811007)
    for i in range(20):
        p = rng.random((25, 25)) < rng.uniform(.3, .7)
        p[:5, 10:15] = True
        p[20:, 10:15] = True
        chosen.append({"pattern": p, "source_group": "random", "source": "fresh random geometry",
                       "lineage_id": f"r81_random_{i:02d}", "seed": 811007})
    evidence = {"source_root": str(root), "dual_patterns_read": len(dual),
                "dual_unique_patterns": len(pool),
                "source_metadata": [{"file": p.relative_to(root).as_posix(), "sha256": pb.file_sha256(p)}
                                    for p in files], "old_responses_imported_as_truth": False}
    return chosen, evidence


def history_seeds(root):
    """Compatibility helper for the explicit combined preparation phase."""
    single = single_history_seeds(root)
    dual, evidence = dual_history_seeds(root)
    evidence["single_patterns_read"] = len(single)
    return single, dual, evidence


def auxiliary_bundle(source, output, indices, *, config=None, repeat=False):
    """Copy chosen geometries into an immutable separate measurement bundle."""
    cfgdata = config or yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    (output / "config.yaml").write_text(yaml.safe_dump(cfgdata, sort_keys=False), encoding="utf-8")
    cfg = pb.load_profile_config(output / "config.yaml")
    rows = pb.read_json(source / "manifest.json")
    selected = []
    for ordinal, idx in enumerate(indices):
        old = rows[idx]
        row = copy.deepcopy(old)
        row["id"] = f"{output.name.removesuffix('_input')}_{ordinal:02d}_{old['pattern_sha256'][:8]}"
        row["parent_batch_id"] = old["id"]
        row["pattern_file"] = row["id"] + ".pt"
        row["measurement_id"] = measurement_id(cfg.measurement)
        row["score_spec_id"] = score_spec_id(cfg.score_spec)
        row["repeat"] = bool(repeat)
        # Engineering checks intentionally remeasure planned formal geometries.
        row["kind"] = "repeat"
        row["repeat_reason"] = "repeatability" if repeat else "wide_solver_engineering_check"
        row["selection_arm"] = "repeat" if repeat else "wide_measurement_check"
        shutil.copy2(source / old["pattern_file"], output / row["pattern_file"])
        selected.append(row)
    pb.atomic_json(output / "measurement.json", cfg.measurement)
    pb.atomic_json(output / "score_spec.json", cfg.score_spec)
    pb.atomic_json(output / "manifest.json", selected)
    pb.validate_input(output, cfg)
    return output


def prepare(history_root, out, *, phase="single"):
    if phase not in {"single", "combined"}:
        raise ValueError(f"unsupported preparation phase: {phase}")
    out = out.resolve()
    out.relative_to((REPO / "tmp").resolve())
    if out.exists():
        raise FileExistsError(f"use a fresh output, preserving earlier evidence: {out}")
    history_root = history_root.resolve()
    single = single_history_seeds(history_root)
    evidence = {"source_root": str(history_root), "single_patterns_read": len(single),
                "dual_history_read": False, "old_responses_imported_as_truth": False}
    dual = None
    if phase == "combined":
        dual, dual_evidence = dual_history_seeds(history_root)
        evidence.update(dual_evidence)
        evidence["dual_history_read"] = True
    out.mkdir(parents=True)
    dataset = out / "dataset"
    dataset.mkdir()
    write_seeds(out / "single_seeds", single)
    singles = ex.prepare_run(REPO / "configs/single_r80_symmetry_explore.yaml", dataset,
                            out / "single", [out / "single_seeds"])
    single_name = "dedust_r80b1_input"
    shutil.copytree(singles, dataset / single_name)
    pb.validate_input(singles, pb.load_profile_config(singles / "config.yaml"))
    # Two geometrically separated representatives, each measured two more times.
    sm = pb.read_json(singles / "manifest.json")
    pats = [torch.load(singles / r["pattern_file"], weights_only=True).numpy() for r in sm]
    reps = ex.maxmin_hamming_indices(pats, 2)
    for number in (1, 2):
        auxiliary_bundle(singles, dataset / f"dedust_r80repeat{number}_input", reps, repeat=True)
    ready = ["dedust_r80b1", "dedust_r80repeat1", "dedust_r80repeat2"]
    prepared_inputs = [name + "_input" for name in ready]
    not_queued = ["dedust_r81b1"]
    if phase == "combined":
        write_seeds(out / "dual_seeds", dual)
        duals = ex.prepare_run(REPO / "configs/dual_r81_wide_filter.yaml", dataset,
                              out / "dual", [out / "dual_seeds"])
        shutil.copytree(duals, dataset / "dedust_r81b1_input")
        pb.validate_input(duals, pb.load_profile_config(duals / "config.yaml"))
        # Six base measurements (two per seed stratum), and two checked using both
        # Discrete sweep and a tighter mesh. The full 60 remain unqueued until review.
        dm = pb.read_json(duals / "manifest.json")
        indices = []
        for arm in ("history", "specialist", "random"):
            indices.extend([i for i, r in enumerate(dm) if r["selection_arm"] == arm][:2])
        auxiliary_bundle(duals, dataset / "dedust_r81smoke_input", indices)
        dc = yaml.safe_load((duals / "config.yaml").read_text(encoding="utf-8"))
        dc["measurement"]["name"] += "_discrete"
        dc["measurement"]["sweep"]["type"] = "Discrete"
        auxiliary_bundle(duals, dataset / "dedust_r81discrete_input", indices[:2], config=dc)
        tight = copy.deepcopy(dc)
        tight["measurement"]["name"] += "_mesh005"
        tight["measurement"]["solver"].update(max_delta_s=.005, max_passes=20,
                                                min_passes=5, min_converged=3)
        tight["runtime"]["timeout"] = 3600
        auxiliary_bundle(duals, dataset / "dedust_r81mesh_input", indices[:2], config=tight)
        ready = ["dedust_r81smoke", "dedust_r81discrete", "dedust_r81mesh", *ready]
        prepared_inputs = [name + "_input" for name in ready]
    # jobs-add is deliberately a separate check-dup/dispatch step.
    evidence.update({"scope": SCOPE, "phase": phase, "prepared_jobs": ready,
                     "prepared_inputs": prepared_inputs, "prepared_measurements":
                     sum(len(pb.read_json(dataset / (name + "_input") / "manifest.json"))
                         for name in ready),
                     "not_queued": not_queued, "deferred": ["R81 wide-filter profile"],
                     "single_selected": len(sm), "dual_selected": len(dm) if phase == "combined" else 0,
                     "single_repeat_ids": [sm[i]["id"] for i in reps],
                     "hfss_started": False})
    pb.atomic_json(out / "preparation.json", evidence)
    print(json.dumps({"output": str(out), "phase": phase, "ready": ready,
                      "wide_full_batch": "deferred" if phase == "single" else "awaiting measurement check"}))


def measurement_check(dataset):
    configs, results = {}, {}
    for name in ("smoke", "discrete", "mesh"):
        store = dataset / ("dedust_r81" + name)
        cfg = pb.load_profile_config(dataset / (store.name + "_input") / "config.yaml")
        pb.verify_completed(store, cfg)
        rows = pb.read_json(store / "manifest.json")
        if len(rows) != (6 if name == "smoke" else 2):
            raise ValueError(f"unexpected {name} check sample count")
        res = pb.read_json(store / "results.json")
        if any(res.get(r["id"], {}).get("status") != "ok" for r in rows):
            raise ValueError(f"incomplete {name} results")
        results[name] = {r["parent_batch_id"]: (r, res[r["id"]], store) for r in rows}
        configs[name] = cfg
    comparisons = []
    for reference, alternate in (("smoke", "discrete"), ("discrete", "mesh")):
        for parent, (_row, result, store) in results[alternate].items():
            ref = results[reference][parent]
            x0, y0 = torch.load(ref[2] / ref[1]["sample_file"], weights_only=True)
            x1, y1 = torch.load(store / result["sample_file"], weights_only=True)
            if not torch.equal(x0, x1) or tuple(y0.shape) != (3, 49) or tuple(y1.shape) != (3, 49):
                raise ValueError("engineering comparisons require identical patterns and 49-point curves")
            point_difference = float(torch.max(torch.abs(y1 - y0)))
            margin_difference = max(abs(result["margins"][k] - ref[1]["margins"][k]) for k in result["margins"])
            comparisons.append({"parent": parent, "reference": reference, "alternate": alternate,
                                "max_curve_difference_db": point_difference,
                                "max_margin_difference_db": margin_difference})
    passed = all(max(c["max_curve_difference_db"], c["max_margin_difference_db"]) <= .3 for c in comparisons)
    return {"threshold_db": .3, "comparisons": comparisons, "engineering_check_passed": passed,
            "full_batch_release_allowed": passed, "claims_outside_16_40_ghz": False}


def package_input_names(receipt, jobs):
    """Return the explicitly prepared inputs and reject receipt/queue drift."""
    if {j["store"] for j in jobs} != set(receipt["prepared_jobs"]):
        raise ValueError("queue differs from prepared initial job list")
    job_inputs = [j["input"] for j in jobs]
    if len(job_inputs) != len(set(job_inputs)):
        raise ValueError("queue contains duplicate prepared inputs")
    declared = receipt.get("prepared_inputs")
    if declared is not None:
        if len(declared) != len(set(declared)) or set(declared) != set(job_inputs):
            raise ValueError("queue inputs differ from prepared input list")
    return sorted(declared if declared is not None else job_inputs)


def package(preparation, out):
    """Bundle committed code and verified prepared inputs; no results or secrets."""
    out = out.resolve()
    out.relative_to((REPO / "tmp").resolve())
    if out.exists():
        raise FileExistsError(out)
    dataset = preparation / "dataset"
    receipt = pb.read_json(preparation / "preparation.json")
    jobs = pb.read_json(dataset / "jobs.json")
    input_names = package_input_names(receipt, jobs)
    for job in jobs:
        if job.get("scope") != SCOPE:
            raise ValueError("queue contains another scope")
        cfg = pb.load_profile_config(dataset / job["config"])
        pb.validate_input(dataset / job["input"], cfg)
    prefixes = ["antenna", "script", "configs", "requirements.txt"]
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--name-only", "--", *prefixes], cwd=REPO, text=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", *prefixes], cwd=REPO, text=True)
    if dirty.strip() or untracked.strip():
        raise ValueError("commit reviewed worker source before packaging")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", "HEAD", *prefixes], cwd=REPO)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as source, zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as dest:
        for info in source.infolist():
            if not info.is_dir():
                dest.writestr("worker/" + info.filename, source.read(info))
        dest.writestr("worker/WORKER_REVISION.txt", rev + "\n")
        for name in input_names:
            folder = dataset / name
            for path in sorted(folder.iterdir()):
                if path.suffix in {".pt", ".json", ".yaml"}:
                    dest.write(path, "dataset/" + path.relative_to(dataset).as_posix())
        dest.write(dataset / "jobs.json", "dataset/jobs.json")
        dest.write(preparation / "preparation.json", "preparation.json")
        dest.write(REPO / "docs/log/symmetry-filter-20261007-runbook.md", "RUNBOOK.md")
    return {"package": str(out), "sha256": pb.file_sha256(out), "code_commit": rev,
            "jobs": len(jobs), "hfss_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--history-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phase", choices=("single", "combined"), default="single")
    p = sub.add_parser("check-wide")
    p.add_argument("--dataset-root", type=Path, required=True)
    p = sub.add_parser("pack")
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.history_root, args.out, phase=args.phase)
    elif args.command == "pack":
        print(json.dumps(package(args.preparation, args.out), indent=2))
    else:
        report = measurement_check(args.dataset_root)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["engineering_check_passed"] else 1)


if __name__ == "__main__":
    main()
