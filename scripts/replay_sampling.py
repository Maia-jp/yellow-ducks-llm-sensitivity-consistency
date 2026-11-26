#!/usr/bin/env python3
"""Replay the notebook subsampling for DBPedia or WoS and print the sample IDs.

The script runs the notebook's own data cell (loader, balance_dataset, seed 42).
It stops before the notebook writes its data.json cache.
Use the output to check that two machines or two Python versions pick the same samples.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "original-paper"
NOTEBOOKS = {"dbpedia": "LLM DBPEDIA.ipynb", "wos": "LLM WoS.ipynb"}
CACHE_MARKER = "if not os.path.exists(processed_data_file):"


def data_cell(notebook: Path) -> str:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and source.lstrip().startswith("input_file ="):
            return source
    raise SystemExit(f"No data cell in {notebook}")


def point_path(source: str, variable: str, path: Path) -> str:
    pattern = rf"^{variable} = Path\(.*\)$"
    if not re.search(pattern, source, flags=re.M):
        raise SystemExit(f"The notebook has no line '{variable} = Path(...)'.")
    return re.sub(pattern, f"{variable} = Path({json.dumps(str(path))})", source, count=1, flags=re.M)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(NOTEBOOKS), required=True)
    parser.add_argument("--input-file", type=Path, required=True, help="DBPedia test.csv, or WoS X.txt")
    parser.add_argument("--target-file", type=Path, help="WoS YL1.txt")
    parser.add_argument("--output", type=Path, help="Write the JSON result to this file")
    parser.add_argument(
        "--subset-dir",
        type=Path,
        help="Write evaluation_subset_2000.jsonl and one_shot_examples.jsonl to this folder",
    )
    args = parser.parse_args()

    source = data_cell(CODE_DIR / NOTEBOOKS[args.dataset])
    source = point_path(source, "input_file", args.input_file.resolve())
    if args.dataset == "wos":
        if args.target_file is None:
            raise SystemExit("--target-file is required for wos.")
        source = point_path(source, "target_file", args.target_file.resolve())
    if CACHE_MARKER not in source:
        raise SystemExit("The notebook cache line changed. Review this script.")
    namespace = {"random": random, "csv": csv, "json": json, "re": re, "os": __import__("os"), "Path": Path}
    exec(compile(source.split(CACHE_MARKER)[0], NOTEBOOKS[args.dataset], "exec"), namespace)

    parse = namespace["parse_DBPEDIA"] if args.dataset == "dbpedia" else namespace["parse_WOS46985"]
    parse_args = [namespace["input_file"]] + ([namespace["target_file"]] if args.dataset == "wos" else [])
    samples, one_shot = namespace["balance_dataset"](
        namespace["class_labels"][:-1], parse(*parse_args), namespace["n_samples"]
    )
    ids = [s["id"] for s in samples]
    result = {
        "dataset": args.dataset,
        "python": sys.version.split()[0],
        "n_samples": len(ids),
        "n_unique": len(set(ids)),
        "class_counts": dict(sorted(collections.Counter(s["class"] for s in samples).items())),
        "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "one_shot_ids": [s["id"] for s in one_shot],
        "one_shot_overlap_with_eval": len(set(s["id"] for s in one_shot) & set(ids)),
        "ids": ids,
    }
    if args.subset_dir:
        args.subset_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in (("evaluation_subset_2000.jsonl", samples), ("one_shot_examples.jsonl", one_shot)):
            lines = (json.dumps(row, ensure_ascii=False) for row in rows)
            (args.subset_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    text = json.dumps(result)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(
        f"{result['dataset']}: python {result['python']}, {result['n_samples']} samples, "
        f"{result['n_unique']} unique, ids sha256 {result['ids_sha256'][:16]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
