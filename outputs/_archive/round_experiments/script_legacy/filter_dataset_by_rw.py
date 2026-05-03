"""Filter dataset jsonl entries to keep only specific ripple_weight."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, required=True)
    p.add_argument("--dst", type=str, required=True)
    p.add_argument("--rw", type=float, required=True)
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "patterns").mkdir(exist_ok=True)
    (dst / "responses").mkdir(exist_ok=True)

    with open(src / "entries.jsonl", "r", encoding="utf-8") as f, \
         open(dst / "entries.jsonl", "w", encoding="utf-8") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            filtered_pareto = [p for p in entry["pareto"] if p["ripple_weight"] == args.rw]
            if not filtered_pareto:
                continue
            for p in filtered_pareto:
                # Copy pattern + response files
                pat_src = src / p["pattern_file"]
                resp_src = src / p["response_file"]
                pat_dst = dst / p["pattern_file"]
                resp_dst = dst / p["response_file"]
                if not pat_dst.exists():
                    shutil.copy(pat_src, pat_dst)
                if not resp_dst.exists():
                    shutil.copy(resp_src, resp_dst)
            entry["pareto"] = filtered_pareto
            g.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Filtered {src} → {dst} (rw={args.rw})")


if __name__ == "__main__":
    main()
