"""有明確量測身分的批次介面；HFSS 生命週期仍由 dedust.run 管理。"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from antenna.measurement import (frequency_grid, measurement_id, score_response,
                                score_spec_id, validate_measurement, validate_score_spec)


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pattern_sha256(pattern):
    arr = np.asarray(pattern)
    if arr.shape != (25, 25) or not np.isfinite(arr).all() or not np.isin(arr, [0, 1]).all():
        raise ValueError("pattern 必須為有限的二值 25×25")
    return hashlib.sha256(np.packbits(arr.astype(bool)).tobytes()).hexdigest()


def load_profile_config(path):
    """舊設定回 None，讓原 load_config 處理；新設定不假裝為 train.py 的 targets。"""
    try:
        with open(path, encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except FileNotFoundError:
        return None
    if not isinstance(data, dict) or "measurement" not in data:
        return None
    unknown = set(data) - {"name", "port", "scope", "measurement", "score_spec", "exploration", "runtime"}
    if unknown:
        raise ValueError(f"新批次 config 有未知欄位：{sorted(unknown)}")
    measurement = validate_measurement(data["measurement"])
    score = validate_score_spec(data["score_spec"], measurement=measurement)
    if data["port"] != measurement["port"]:
        raise ValueError("config port 與 measurement 不符")
    if measurement["geometry"]["pixel_count"] != 25:
        raise ValueError("本輪固定 25×25")
    if not measurement["geometry"]["diag_bridge_w"]:
        raise ValueError("具名 profile 的 diag_bridge_w 必須是正值；0/None 不可傳入本輪模擬器")
    if measurement["port"] == "single":
        if not np.array_equal(frequency_grid(measurement), np.arange(24, 32.1, .5)):
            raise ValueError("本輪 single 只支援 24–32 GHz / 0.5")
        if measurement["solver"]["setup_freq_ghz"] != 28 or measurement["solver"]["open_region_freq_ghz"] != 28:
            raise ValueError("single 求解頻率固定 28 GHz")
        if measurement["labels"] != ["S11", "Gain"] or score["bands"]:
            raise ValueError("對稱探索固定 S11/Gain，且不設性能評分門檻")
    elif measurement["labels"] != ["S11", "S21", "S22"]:
        raise ValueError("dual labels 必須是 S11/S21/S22")
    from script.batch_scope import validate_scope
    scope = validate_scope(data.get("scope"))
    if not scope:
        raise ValueError("新量測批次必須有 scope")
    runtime = data.get("runtime", {})
    if not isinstance(runtime, dict) or set(runtime) - {"timeout"}:
        raise ValueError("profile runtime 只接受 timeout")
    if "timeout" in runtime and (isinstance(runtime["timeout"], bool)
                                 or not isinstance(runtime["timeout"], int)
                                 or runtime["timeout"] <= 0):
        raise ValueError("profile runtime.timeout 必須是正整數秒")
    return SimpleNamespace(**{**data, "scope": scope, "measurement": measurement,
                              "score_spec": score, "radiation": {}, "targets": {},
                              "runtime": runtime, "exploration": data.get("exploration", {})})


def simulator_kwargs(cfg):
    m = cfg.measurement
    g, s = m["geometry"], m["solver"]
    if g["geom"] != ("p01" if cfg.port == "dual" else "single_v1"):
        raise ValueError("未支援的幾何版本；不能只改 metadata 冒充另一種幾何")
    kwargs = {k: s[k] for k in ("max_delta_s", "max_passes", "min_passes", "min_converged")}
    kwargs.update(pixel_count=g["pixel_count"], diag_bridge_w=g["diag_bridge_w"])
    if cfg.port == "dual":
        sw = m["sweep"]
        kwargs.update(sweep_start=sw["start_ghz"], sweep_end=sw["stop_ghz"], sweep_step=sw["step_ghz"],
                      setup_frequency=s["setup_freq_ghz"], open_region_frequency=s["open_region_freq_ghz"])
    return kwargs


def validate_pattern(pattern, cfg):
    digest = pattern_sha256(pattern)
    p = np.asarray(pattern).astype(bool)
    if cfg.port == "single":
        if not np.array_equal(p, p[:, ::-1]) or not p[24, 12]:
            raise ValueError("對稱探索要求左右完全鏡射且保留饋點")
        # 使用模擬器同一份橋接幾何，避免 bits 對稱但實際橋不對稱。
        from antenna.patch.patch_simulator.single_port import diag_bridge_sites
        sites, _ = diag_bridge_sites(p, cfg.measurement["geometry"]["diag_bridge_w"], .2)
        actual = {(round(x, 8), round(y, 8), round(w, 8)) for x, y, w in sites}
        mirrored = {(round(x, 8), round(5 - y, 8), round(w, 8)) for x, y, w in sites}
        if actual != mirrored:
            raise ValueError("實際橋接幾何破壞左右對稱")
    elif not p[:5, 10:15].all() or not p[20:, 10:15].all():
        raise ValueError("dual 必須保留上下饋墊")
    return digest


def validate_input(input_dir, cfg):
    root = Path(input_dir)
    if measurement_id(read_json(root / "measurement.json")) != measurement_id(cfg.measurement):
        raise ValueError("輸入與 config 的 measurement_id 不符")
    if score_spec_id(read_json(root / "score_spec.json")) != score_spec_id(cfg.score_spec):
        raise ValueError("輸入與 config 的 score_spec_id 不符")
    if (root / "hfss_setup.json").exists():
        raise ValueError("具名量測不接受第二份 hfss_setup 覆蓋；請建立新的 measurement")
    rows = read_json(root / "manifest.json")
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest 必須為非空列表")
    seen = set()
    for row in rows:
        name = row["id"]
        if not isinstance(name, str) or Path(name).name != name or any(x in name for x in ("/", "\\", ":")) or name in seen:
            raise ValueError("manifest id 重複或不是安全檔名")
        seen.add(name)
        if row.get("port") != cfg.port or row.get("measurement_id") != measurement_id(cfg.measurement):
            raise ValueError(f"{name}: port/measurement_id 不符")
        if row.get("score_spec_id") != score_spec_id(cfg.score_spec):
            raise ValueError(f"{name}: score_spec_id 不符")
        p = torch.load(root / f"{name}.pt", weights_only=True, map_location="cpu")
        if row.get("pattern_sha256") != validate_pattern(p, cfg):
            raise ValueError(f"{name}: pattern hash 不符")
    simulator_kwargs(cfg)
    return rows


def prepare_store(input_dir, store_dir, cfg):
    """先完整驗證身分、輸入與舊成果，通過後才建目錄與 snapshot。"""
    rows = validate_input(input_dir, cfg)
    root = Path(store_dir)
    marker = root / "measurement.json"
    if root.exists() and any(root.iterdir()):
        if not marker.exists() or measurement_id(read_json(marker)) != measurement_id(cfg.measurement):
            raise ValueError("拒絕覆寫不同量測／無身分的既有 store")
        if read_json(root / "manifest.json") != rows:
            raise ValueError("續跑 manifest 已改變")
        if score_spec_id(read_json(root / "score_spec.json")) != score_spec_id(cfg.score_spec):
            raise ValueError("評分已改變；請離線 rescore，不要重新跑 HFSS")
        verify_completed(root, cfg)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, cfg.measurement)
    atomic_json(root / "score_spec.json", cfg.score_spec)
    atomic_json(root / "manifest.json", rows)
    return rows


def observation(response, rad, row, pattern, cfg, elapsed):
    from antenna.utils.store import fingerprint
    freqs = frequency_grid(cfg.measurement)
    labels = cfg.measurement["labels"]
    y = np.asarray(response, dtype=float)
    if y.shape != (len(labels), len(freqs)) or not np.isfinite(y).all():
        raise ValueError("HFSS 響應 shape/finite 不符")
    pattern_tensor = torch.as_tensor(pattern)
    response_tensor = torch.as_tensor(response)
    entry = {"status": "ok", "measurement_id": measurement_id(cfg.measurement),
             "score_spec_id": score_spec_id(cfg.score_spec), "labels": labels, "freqs": freqs.tolist(),
             "pattern_sha256": validate_pattern(pattern, cfg), "lineage_id": row.get("lineage_id", row["id"]),
             "sample_file": fingerprint(pattern_tensor, response_tensor) + ".pt", "time_s": float(elapsed),
             "geom": cfg.measurement["geometry"]["geom"], "dbw": cfg.measurement["geometry"]["diag_bridge_w"]}
    if cfg.port == "single":
        if not isinstance(rad, dict) or any(rad.get(k) is None for k in ("theta", "phi0", "phi90")):
            raise ValueError("探索資料缺少完整的兩個場型切面")
        theta = np.asarray(rad["theta"], dtype=float)
        if theta.shape != (181,) or not np.array_equal(theta, np.arange(-180, 181, 2)):
            raise ValueError("場型角度網格必須為 -180..180，每2度")
        for key in ("phi0", "phi90"):
            cut = np.asarray(rad[key], dtype=float)
            if cut.shape != theta.shape or not np.isfinite(cut).all():
                raise ValueError(f"{key} 非完整有限場型")
        entry["gain28"] = float(y[1, 8])
        entry["s11_28"] = float(y[0, 8])
        entry["rad_file"] = f"rad/{row['id']}.pt"
        entry["performance_gate"] = False
    else:
        entry.update(score_response(y, labels, freqs, cfg.score_spec))
        from antenna.losses import dual_energy_max
        entry["energy_max"] = float(dual_energy_max(y))
    return entry


def verify_completed(store_dir, cfg, *, require_complete=False):
    """只讀重播已落地 observation；``require_complete`` 再要求 manifest 全數成功。"""
    root = Path(store_dir)
    rp = root / "results.json"
    results = read_json(rp) if rp.exists() else {}
    rows = {r["id"]: r for r in read_json(root / "manifest.json")}
    if not rows:
        raise ValueError("store manifest 不可為空")
    for name, entry in results.items():
        if name not in rows or not isinstance(entry, dict):
            raise ValueError("results.json 含非本批或非 dict 的紀錄")
        if "error" in entry:
            continue
        if entry.get("status") != "ok":
            raise ValueError("完成紀錄不是本批的有效 observation")
        if entry.get("measurement_id") != measurement_id(cfg.measurement):
            raise ValueError("完成紀錄 measurement_id 不符")
        path = root / entry["sample_file"]
        if path.parent.resolve() != root.resolve() or file_sha256(path) != entry.get("sample_sha256"):
            raise ValueError("完成樣本遺失／內容 hash 已改變")
        p, y = torch.load(path, weights_only=True, map_location="cpu")
        rad = None
        if cfg.port == "single":
            if not entry.get("rad_file"):
                raise ValueError("完成的 single observation 缺 rad_file")
            rad_path = (root / entry["rad_file"]).resolve()
            try:
                rad_path.relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError("場型路徑逃出 store") from exc
            if file_sha256(rad_path) != entry.get("rad_sha256"):
                raise ValueError("場型內容 hash 已改變")
            rad = torch.load(rad_path, weights_only=True, map_location="cpu")
        replay = observation(y, rad, rows[name], p, cfg, entry["time_s"])
        if replay["pattern_sha256"] != rows[name].get("pattern_sha256"):
            raise ValueError("完成樣本與 manifest 圖形不符")
        if any(entry.get(k) != v for k, v in replay.items()):
            raise ValueError("完成紀錄與原始樣本重播不符")
    if require_complete:
        completed = {name for name, entry in results.items()
                     if isinstance(entry, dict) and entry.get("status") == "ok" and "error" not in entry}
        if completed != set(rows):
            missing = sorted(set(rows) - completed)
            raise ValueError(f"profile 批次未完整成功：{missing[:20]}")
    return results


def check_duplicates(input_dir, dataset_root):
    root = Path(input_dir)
    mid = measurement_id(read_json(root / "measurement.json"))
    rows = read_json(root / "manifest.json")
    current, duplicates = {}, []
    for row in rows:
        p = torch.load(root / f"{row['id']}.pt", weights_only=True, map_location="cpu")
        key = pattern_sha256(p)
        if key != row.get("pattern_sha256"):
            raise ValueError("查重前發現 pattern hash 不符")
        if row.get("kind") in ("repeat", "notarize"):
            continue
        if key in current:
            duplicates.append((row["id"], current[key]))
        current[key] = row["id"]
    for other in Path(dataset_root).glob("*_input"):
        if other.resolve() == root.resolve() or not (other / "measurement.json").exists():
            continue
        if measurement_id(read_json(other / "measurement.json")) != mid:
            continue
        for row in read_json(other / "manifest.json"):
            if row.get("kind") in ("repeat", "notarize"):
                continue
            if row.get("pattern_sha256") in current:
                duplicates.append((current[row["pattern_sha256"]], f"{other.name}:{row['id']}"))
    if duplicates:
        raise ValueError(f"同量測條件下有重複：{duplicates[:10]}")
    return len(current)


def report(store_dir):
    root = Path(store_dir)
    rows = read_json(root / "manifest.json")
    result = read_json(root / "results.json") if (root / "results.json").exists() else {}
    print("| id | arm | status | observations |")
    print("|---|---|---|---|")
    for row in rows:
        r = result.get(row["id"], {})
        values = r.get("margins", {"gain28": r.get("gain28"), "s11_28": r.get("s11_28")})
        print(f"| {row['id']} | {row.get('selection_arm', row.get('arm', ''))} | {r.get('status', r.get('error', 'pending'))} | {values} |")


def validate_store(store_dir, *, require_complete=True):
    """不開 HFSS，僅由 store 自帶 snapshot 重播並驗證所有 hash。"""
    root = Path(store_dir)
    measurement = validate_measurement(read_json(root / "measurement.json"))
    score = validate_score_spec(read_json(root / "score_spec.json"), measurement=measurement)
    cfg = SimpleNamespace(port=measurement["port"], measurement=measurement, score_spec=score)
    return verify_completed(root, cfg, require_complete=require_complete)


def rescore_store(store_dir, score_spec_path, out_path):
    """以新門檻離線評分標準 tuple，另寫 summary，不修改原 results/score snapshot。"""
    root, out = Path(store_dir), Path(out_path)
    root_resolved, out_resolved = root.resolve(), out.resolve()
    if out.exists():
        raise ValueError("rescore --out 必須是不存在的新檔，不可覆寫任何檔案")
    try:
        out_resolved.relative_to(root_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("rescore --out 必須位於 store 外，避免覆寫任何原始 artifact")
    results = validate_store(root, require_complete=True)
    measurement = validate_measurement(read_json(root / "measurement.json"))
    new_score = validate_score_spec(read_json(score_spec_path), measurement=measurement)
    rows = {}
    for name, entry in sorted(results.items()):
        _pattern, response = torch.load(root / entry["sample_file"], weights_only=True, map_location="cpu")
        rows[name] = score_response(response, entry["labels"], entry["freqs"], new_score)
    summary = {
        "measurement_id": measurement_id(measurement),
        "source_score_spec_id": score_spec_id(read_json(root / "score_spec.json")),
        "score_spec_id": score_spec_id(new_score),
        "score_spec": new_score,
        "rows": rows,
    }
    atomic_json(out, summary)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="具名量測批次的離線驗證／重評分")
    sub = parser.add_subparsers(dest="cmd", required=True)
    validate_cmd = sub.add_parser("validate", help="重播 store 並驗 sample/rad hash，不開 HFSS")
    validate_cmd.add_argument("--store", required=True)
    rescore_cmd = sub.add_parser("rescore", help="用新 score spec 另寫 summary，不改原結果")
    rescore_cmd.add_argument("--store", required=True)
    rescore_cmd.add_argument("--score-spec", required=True, dest="score_spec")
    rescore_cmd.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "validate":
        results = validate_store(args.store, require_complete=True)
        print(f"驗證通過：{len(results)} 筆")
    else:
        summary = rescore_store(args.store, args.score_spec, args.out)
        print(f"重評分完成：{len(summary['rows'])} 筆 → {args.out}")


if __name__ == "__main__":
    main()
