#!/usr/bin/env python3
"""Run the paper's original evaluation code for one dataset with a Qwen model.

The script runs the notebook cells that load the data and set the prompts,
then calls the unchanged run_grid_search() from evaluator.py. Only the model
list, the results folder and (with --smoke) the sample count change.

The Slurm jobs in jobs/qwen3.5-27b/ start it. It expects a running
OpenAI-compatible server (vLLM) and the datasets in WORK_DIR/data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "sensitivity-consistency-LLM"
LLM_NAME = "qwen35-27b"  # must not contain "_": evaluator.py splits result keys on it

# notebook, first line of the data cell, first line of the settings cell,
# folder that the data cell writes its cache file to
NOTEBOOKS = {
    "trec": ("LLM TREC.ipynb", "train_data =", "results_folder =", None),
    "cb": ("LLM CB.ipynb", "input_file =", "question_to_rewrite =", None),
    "rte": ("LLM RTE.ipynb", "input_file =", "question_to_rewrite =", "data/RTE_RESULTS"),
    "dbpedia": ("LLM DBPEDIA.ipynb", "input_file =", "question_to_rewrite =", "data/DBPEDIA_RESULTS"),
    "wos": ("LLM WoS.ipynb", "input_file =", "question_to_rewrite =", "data/WOS46985_RESULTS"),
}
CB_FILES = {"test_labeled": "data/CB/test_labeled.jsonl", "train": "data/CB/train.jsonl"}


def source_for(notebook: dict, starts_with: str) -> str:
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith(starts_with):
            return source
    raise SystemExit(f"The notebook has no code cell that starts with {starts_with!r}.")


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def check_server(api_base: str, model: str) -> None:
    """Send one classification request. Stop if the answer is empty or has thinking text."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a text classifier."},
            {"role": "user", "content": "Classify the question as Number, Location, Person, Description, Entity, or Abbreviation. Use one word.\nHere is the text: What is the capital of Denmark?"},
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    answer = post_json(api_base.rstrip("/") + "/chat/completions", payload)["choices"][0]["message"]["content"]
    print(f"Server check: answer {answer!r}", flush=True)
    if not isinstance(answer, str) or not answer.strip():
        raise SystemExit("The model server returned an empty answer.")
    if "<think>" in answer or "</think>" in answer:
        raise SystemExit("The answer has thinking text. enable_thinking=False did not work.")


def check_prompt_lengths(api_base: str, model: str, prompt_types: dict, samples: list, max_model_len: int) -> None:
    """Stop before the run if the longest prompt does not fit into max_model_len.

    The paper code retries a failed request forever, so a prompt that is too
    long would block the job until its time limit.
    """
    server_root = api_base.rstrip("/").removesuffix("/v1")
    longest = sorted(samples, key=lambda s: len(s["input"]), reverse=True)[:3]
    worst = 0
    for name, prompt in prompt_types.items():
        for sample in longest:
            messages = [
                {"role": "system" if role == "system" else "user", "content": text.replace("{summary}", sample["input"])}
                for role, text in prompt
            ]
            count = post_json(
                server_root + "/tokenize",
                {"model": model, "messages": messages, "add_generation_prompt": True,
                 "chat_template_kwargs": {"enable_thinking": False}},
            )["count"]
            worst = max(worst, count)
        print(f"Prompt length check: {name}: longest prompt {worst} tokens so far", flush=True)
    # Rephrased questions can be longer than the original. Keep room for them and the answer.
    if worst + 1024 > max_model_len:
        raise SystemExit(f"The longest prompt has {worst} tokens. Raise MAX_MODEL_LEN above {worst + 1024}.")


def summarize(results_folder: Path, prompt_types: dict, n_samples: int, class_labels: list) -> tuple[bool, float]:
    """Print the finished samples and the answer labels for each prompt type.

    Returns (all samples finished, share of answers without a class label).
    """
    all_done = True
    counts_total = [0.0] * len(class_labels)
    for prompt_type in prompt_types:
        result_file = results_folder / f"results_{prompt_type}.json"
        rows = json.loads(result_file.read_text()) if result_file.exists() else {}
        samples = [row for key, row in rows.items() if key.count("_") == 5]
        for row in samples:
            counts_total = [a + b for a, b in zip(counts_total, row["counts"])]
        print(f"{prompt_type}: {len(samples)} of {n_samples} samples finished", flush=True)
        all_done &= len(samples) >= n_samples
    answers = sum(counts_total)
    if answers:
        shares = ", ".join(f"{label} {count / answers:.0%}" for label, count in zip(class_labels, counts_total))
        print(f"Answer labels over {int(answers)} answers: {shares}", flush=True)
    na_share = counts_total[class_labels.index("N/A")] / answers if answers and "N/A" in class_labels else 0.0
    return all_done, na_share


def patch_ray(num_cpus: int, temp_dir: Path) -> None:
    """Start a private Ray cluster for this job with a fixed number of workers.

    evaluator.py calls ray.init() without arguments. Without this patch, Ray
    uses every CPU of the node and can join another user's Ray cluster.
    """
    import ray

    original_init = ray.init

    def init(*args, **kwargs):
        kwargs.setdefault("address", "local")
        kwargs.setdefault("num_cpus", num_cpus)
        kwargs.setdefault("include_dashboard", False)
        kwargs.setdefault("_temp_dir", str(temp_dir))
        return original_init(*args, **kwargs)

    ray.init = init


def check_ray_workers() -> None:
    """Stop if Ray workers cannot use the Qwen model.

    Workers register the model through scripts/sitecustomize.py. If that
    fails, every request fails and the paper code would retry forever.
    """
    import ray

    ray.init()
    try:
        @ray.remote
        def registered() -> bool:
            import util

            return LLM_NAME in util.llm_map

        if not ray.get(registered.remote()):
            raise SystemExit(f"Ray workers do not know the model {LLM_NAME!r}. Check PYTHONPATH and QWEN_REGISTER_PROVIDER.")
    finally:
        ray.shutdown()
    print("Ray check: workers know the model.", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(NOTEBOOKS), required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="Folder with data/. Results go to results/.")
    parser.add_argument("--cb-file", choices=sorted(CB_FILES), default="test_labeled", help="CB input file (CB only).")
    parser.add_argument("--api-base", default=os.getenv("QWEN_API_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-27B"))
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--ray-cpus", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="Two samples and three prompt variants. Not paper results.")
    parser.add_argument("--status-only", action="store_true",
                        help="Only report progress. Exit 0 if all samples are finished, else 3. Needs no server.")
    args = parser.parse_args()

    os.environ["QWEN_API_BASE"] = args.api_base
    os.environ["QWEN_MODEL"] = args.model
    work_dir = args.work_dir.resolve()
    os.chdir(work_dir)
    notebook_name, data_cell, settings_cell, cache_dir = NOTEBOOKS[args.dataset]
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(CODE_DIR))
    sys.path.insert(0, str(ROOT / "scripts"))
    from qwen_model import register_qwen

    register_qwen(LLM_NAME)
    ray_dir = Path("/tmp") / f"ray-{os.getenv('USER', 'user')}-{os.getenv('SLURM_JOB_ID', os.getpid())}"
    patch_ray(args.ray_cpus, ray_dir)

    notebook_path = CODE_DIR / notebook_name
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "__qwen_run__"}
    exec(compile(source_for(notebook, "import os"), str(notebook_path), "exec"), namespace)

    data_source = source_for(notebook, data_cell)
    run_name = args.dataset
    if args.dataset == "cb":
        run_name = f"cb-{args.cb_file}"
        input_path = work_dir / CB_FILES[args.cb_file]
        replacements = {
            "input_file = Path('./data/CB/train.jsonl')": f"input_file = Path({json.dumps(str(input_path))})",
            "processed_data_file = Path('./data/CB_RESULTS/data.json')":
                f"processed_data_file = Path({json.dumps(str(input_path) + '.parsed.json')})",
        }
        for old, new in replacements.items():
            if old not in data_source:
                raise SystemExit(f"The CB notebook changed. Line not found: {old}")
            data_source = data_source.replace(old, new, 1)
    exec(compile(data_source, str(notebook_path), "exec"), namespace)
    exec(compile(source_for(notebook, settings_cell), str(notebook_path), "exec"), namespace)

    samples = namespace.get("samples")
    if not isinstance(samples, list) or not samples:
        raise SystemExit("No samples loaded. Check the dataset files in data/.")
    if args.smoke:
        samples = samples[:2]
        namespace["Qs"] = [3]
        run_name += "-smoke"
    results_folder = work_dir / "results" / run_name
    results_folder.mkdir(parents=True, exist_ok=True)

    prompt_types = namespace["prompt_types"]
    print(
        f"Dataset {args.dataset}: {len(samples)} samples, prompt types {list(prompt_types)}, "
        f"Qs {namespace['Qs']}, As {namespace['As']}, temperatures {namespace['temp_questions']} / {namespace['temp_answers']}",
        flush=True,
    )
    if args.status_only:
        all_done, _ = summarize(results_folder, prompt_types, len(samples), namespace["class_labels"])
        return 0 if all_done else 3
    if args.smoke:
        print("Smoke run only. Do not report these values as results.", flush=True)

    check_server(args.api_base, args.model)
    check_prompt_lengths(args.api_base, args.model, prompt_types, samples, args.max_model_len)
    check_ray_workers()

    from evaluator import run_grid_search

    run_grid_search(
        samples,
        [LLM_NAME],
        prompt_types,
        namespace["Qs"],
        namespace["temp_questions"],
        namespace["question_to_rewrite"],
        namespace["As"],
        namespace["temp_answers"],
        namespace["class_labels"],
        namespace["class_extractor_fun"],
        results_folder,
    )

    all_done, na_share = summarize(results_folder, prompt_types, len(samples), namespace["class_labels"])
    shutil.rmtree(ray_dir, ignore_errors=True)
    if not all_done:
        raise SystemExit("Some samples are not finished. Resubmit the job to continue.")
    # In the smoke run, a high share of answers without a class label means that
    # the model output does not fit the paper's label extraction. Stop the chain.
    if args.smoke and na_share > 1 / 3:
        raise SystemExit(f"{na_share:.0%} of the answers have no class label (N/A). Check the answers before the full runs.")
    print(f"All prompt types finished. Results: {results_folder}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
