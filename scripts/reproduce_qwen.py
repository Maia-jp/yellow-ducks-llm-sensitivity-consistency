#!/usr/bin/env python3
"""Run the paper's original evaluation code with Qwen3.5-4B."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "sensitivity-consistency-LLM"
NOTEBOOKS = {
    "cb": "LLM CB.ipynb",
    "dbpedia": "LLM DBPEDIA.ipynb",
    "rte": "LLM RTE.ipynb",
    "wos": "LLM WoS.ipynb",
}


def source_for(notebook: dict, starts_with: str) -> str:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith(starts_with):
            return source
    raise ValueError(f"Notebook has no code cell that starts with {starts_with!r}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(NOTEBOOKS), required=True)
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Override the notebook input file. This option applies to CB only.",
    )
    parser.add_argument("--smoke", action="store_true", help="Use two samples and three prompt variants. These are not paper results.")
    parser.add_argument("--max-samples", type=int, help="Run a smaller diagnostic sample. These are not full paper results.")
    args = parser.parse_args()

    if args.input_file is not None and args.dataset != "cb":
        raise SystemExit("--input-file is supported only for --dataset cb.")
    if args.dataset == "cb" and args.input_file is None:
        args.input_file = CODE_DIR / "data/CB/test_labeled.jsonl"
    if args.input_file is not None and not args.input_file.is_file():
        raise SystemExit(f"CB input file does not exist: {args.input_file}")

    notebook_path = CODE_DIR / NOTEBOOKS[args.dataset]
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {notebook_path}: {exc}") from exc

    sys.path.insert(0, str(CODE_DIR))
    sys.path.insert(0, str(ROOT / "scripts"))
    from qwen_model import register_qwen

    register_qwen()
    namespace: dict = {"__name__": "__qwen_reproduction__"}
    exec(compile(source_for(notebook, "import os"), str(notebook_path), "exec"), namespace)

    data_source = source_for(notebook, "input_file =")
    if args.input_file is not None:
        input_path = args.input_file.resolve()
        cache_path = input_path.with_name(input_path.name + ".parsed.json")
        original_input = "input_file = Path('./data/CB/train.jsonl')"
        original_cache = "processed_data_file = Path('./data/CB_RESULTS/data.json')"
        if original_input not in data_source or original_cache not in data_source:
            raise SystemExit(
                "The CB notebook input lines changed. Review the input override before running."
            )
        data_source = data_source.replace(
            original_input, f"input_file = Path({json.dumps(str(input_path))})", 1
        )
        data_source = data_source.replace(
            original_cache, f"processed_data_file = Path({json.dumps(str(cache_path))})", 1
        )
    exec(compile(data_source, str(notebook_path), "exec"), namespace)
    config_source = source_for(notebook, "question_to_rewrite =")
    exec(compile(config_source, str(notebook_path), "exec"), namespace)

    samples = namespace.get("samples", namespace.get("data_list"))
    if not isinstance(samples, list) or not samples:
        expected = namespace.get("input_file", "the notebook dataset file")
        raise SystemExit(f"No samples loaded. Check the original dataset files, including {expected}.")
    limit = 2 if args.smoke else args.max_samples
    if limit is not None:
        if limit < 1:
            raise SystemExit("--max-samples must be at least 1.")
        samples = samples[:limit]

    namespace["samples"] = samples
    namespace["Qs"] = [3] if args.smoke else namespace["Qs"]
    namespace["llms"] = ["qwen35"]
    namespace["results_folder"] = Path("results/qwen35") / args.dataset
    namespace["results_folder"].mkdir(parents=True, exist_ok=True)

    from evaluator import run_grid_search

    print(f"Dataset: {args.dataset}; samples: {len(samples)}; prompt variants: {namespace['Qs'][0]}")
    if args.smoke or args.max_samples:
        print("Diagnostic run only. Do not report these values as paper results.")
    run_grid_search(
        samples,
        namespace["llms"],
        namespace["prompt_types"],
        namespace["Qs"],
        namespace["temp_questions"],
        namespace["question_to_rewrite"],
        namespace["As"],
        namespace["temp_answers"],
        namespace["class_labels"],
        namespace["class_extractor_fun"],
        namespace["results_folder"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
