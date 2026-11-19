#!/usr/bin/env python3
"""Compute sensitivity, consistency, micro-F1 and N/A share for several models.

The formulas are the ones in mixtral8x7b-Instruct-v0.1_RESULTS/Results_and_paper_comparison.ipynb
(branch main), so DeepSeek and Mixtral are measured in the same way:

- Sensitivity (paper eq. 3-4): entropy of each sample's answer distribution,
  divided by log(number of classes, N/A included), averaged over samples.
- Consistency (paper eq. 5-7): 1 - total variation distance between the answer
  distributions of two samples of the same true class. Mean over all ordered
  pairs within each class, self-pairs included, pooled over classes.
- Micro-F1: over every single answer (all rephrasings), N/A counted as a wrong
  class. For single-label data this equals accuracy.

Usage (light work, numpy and pandas only):
  python3 compare_models.py --out OUTDIR \
      --model "Mixtral-8x7B=mixtral8x7b-Instruct-v0.1_RESULTS" \
      --model "DeepSeek-R1-7B=deepseek-r1-7b_RESULTS"

Each model folder holds one subfolder per dataset (CB_RESULTS, TREC_RESULTS,
RTE_RESULTS, DBPEDIA_RESULTS, WOS46985_RESULTS) with results_<strategy>.json.
Missing datasets or strategies are skipped. Only datasets present for all
models go into the comparison table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {
    "CB": "CB_RESULTS",
    "TREC": "TREC_RESULTS",
    "RTE": "RTE_RESULTS",
    "DBPedia": "DBPEDIA_RESULTS",
    "WoS": "WOS46985_RESULTS",
}
STRATEGIES = ["simple", "instruct", "fewshot"]


def load_samples(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [v for v in data.values() if isinstance(v, dict)]


def sensitivity(samples: list[dict]) -> float:
    values = []
    for sample in samples:
        p = np.asarray(sample["distribution"], dtype=float)
        nz = p[p > 0]
        values.append(float(-(nz * np.log(nz)).sum() / np.log(len(p))))
    return float(np.mean(values))


def consistency(samples: list[dict]) -> float:
    by_class: dict[str, list[np.ndarray]] = {}
    for sample in samples:
        by_class.setdefault(sample["target"], []).append(np.asarray(sample["distribution"], dtype=float))
    total, pairs = 0.0, 0
    for distributions in by_class.values():
        P = np.stack(distributions)
        tvd = 0.5 * np.abs(P[:, None, :] - P[None, :, :]).sum(axis=2)
        total += float((1.0 - tvd).sum())
        pairs += len(P) ** 2
    return total / pairs


def answers(samples: list[dict]) -> tuple[list[str], list[str]]:
    y_true, y_pred = [], []
    for sample in samples:
        for predictions in sample["info_answers"].values():
            for prediction in predictions:
                y_true.append(sample["target"])
                y_pred.append(prediction)
    return y_true, y_pred


def micro_f1(y_true: list[str], y_pred: list[str]) -> float:
    # Single-label task, N/A is a wrong class: micro-F1 = share of correct answers.
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def analyse(name: str, root: Path) -> list[dict]:
    rows = []
    for dataset, folder in DATASETS.items():
        for strategy in STRATEGIES:
            path = root / folder / f"results_{strategy}.json"
            if not path.exists():
                continue
            samples = load_samples(path)
            if not samples:
                continue
            y_true, y_pred = answers(samples)
            rows.append({
                "Model": name,
                "Dataset": dataset,
                "Strategy": strategy,
                "Samples": len(samples),
                "Answers": len(y_pred),
                "Sensitivity": sensitivity(samples),
                "Consistency": consistency(samples),
                "Micro-F1": micro_f1(y_true, y_pred),
                "N/A share": sum(p == "N/A" for p in y_pred) / len(y_pred),
            })
    return rows


def markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="NAME=FOLDER, repeat for each model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    rows = []
    for item in args.model:
        name, _, folder = item.partition("=")
        rows += analyse(name, Path(folder))
    if not rows:
        raise SystemExit("No result files found.")
    df = pd.DataFrame(rows)
    order = {d: i for i, d in enumerate(DATASETS)} | {s: i for i, s in enumerate(STRATEGIES)}
    df = df.sort_values(["Dataset", "Strategy", "Model"], key=lambda c: c.map(order) if c.name != "Model" else c)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "metrics_all.csv", index=False)

    models = list(dict.fromkeys(df["Model"]))
    common = [d for d in DATASETS if all(((df.Model == m) & (df.Dataset == d)).any() for m in models)]
    metrics = ["Sensitivity", "Consistency", "Micro-F1", "N/A share"]
    wide = df[df.Dataset.isin(common)].pivot_table(index=["Dataset", "Strategy"], columns="Model", values=metrics)
    wide = wide.reindex(pd.MultiIndex.from_product([common, STRATEGIES]), axis=0).dropna(how="all")
    flat = wide.copy()
    flat.columns = [f"{metric} {model}" for metric, model in flat.columns]
    flat = flat.reset_index().rename(columns={"level_0": "Dataset", "level_1": "Strategy"})
    flat.to_csv(args.out / "metrics_comparison.csv", index=False)

    report = ["# Model comparison", "", f"Models: {', '.join(models)}", "",
              "## All results", "", markdown(df.drop(columns=["Answers"])), ""]
    if common:
        report += ["## Datasets present for all models: " + ", ".join(common), "", markdown(flat), ""]
    (args.out / "metrics_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))

    if not args.no_plots and common:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sub = df[df.Dataset.isin(common)]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        labels = [f"{d}\n{s}" for d in common for s in STRATEGIES if ((sub.Dataset == d) & (sub.Strategy == s)).any()]
        x = np.arange(len(labels))
        width = 0.8 / len(models)
        for ax, metric in zip(axes, ["Sensitivity", "Consistency", "Micro-F1"]):
            for i, model in enumerate(models):
                vals = []
                for d in common:
                    for s in STRATEGIES:
                        if not ((sub.Dataset == d) & (sub.Strategy == s)).any():
                            continue
                        hit = sub[(sub.Model == model) & (sub.Dataset == d) & (sub.Strategy == s)][metric]
                        vals.append(float(hit.iloc[0]) if len(hit) else np.nan)
                ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width, label=model)
            ax.set_title(metric + (" (lower is better)" if metric == "Sensitivity" else " (higher is better)"))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(args.out / "metrics_comparison.png", dpi=150)
        print(f"Plot: {args.out / 'metrics_comparison.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
