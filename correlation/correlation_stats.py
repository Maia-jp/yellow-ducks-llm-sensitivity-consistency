#!/usr/bin/env python3
"""Statistical checks for Claim 1 (sensitivity-consistency correlation).

Complements "Sensiitivity-Consistency correlation analysis.ipynb" in this folder.
It builds the same 51 model-dataset-strategy configurations from ../models/*.zip
(same formulas as the notebook) and the 60 configurations of the paper's Table 1,
keeps the configurations with sensitivity >= 0.05, and reports:

  1. Pearson r, two-sided p-value and a 95% confidence interval for r
     (Fisher z-transform: z = atanh(r), SE = 1/sqrt(n-3)).
  2. A test of whether our r differs from the paper's r for the same models.
     Fisher z-test for two correlations: z = (atanh r1 - atanh r2) / sqrt(1/(n1-3) + 1/(n2-3)).
     This test assumes two independent samples. Ours and the paper's share models,
     datasets and strategies, so the p-value is only approximate.
  3. The statistical power of that test: the probability of a significant result
     (alpha = 0.05, two-sided) if the true correlations were -0.44 and -0.07.
     power = Phi(|delta| / SE - 1.96) + Phi(-|delta| / SE - 1.96), delta = atanh(r1) - atanh(r2).
  4. Per-model correlations, and correlations after removing the mean of each
     model or each dataset ("within-model" / "within-dataset"). Pooling groups
     with different mean levels can change a correlation, or even its sign
     (Simpson's paradox).

Run from this folder:  python3 correlation_stats.py
Needs numpy, pandas and scipy (the same packages as the notebook).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"
THRESHOLD = 0.05
ALPHA = 0.05

MODEL_FILES = {
    "Llama 3": "Llama-3-70B-Instruct_RESULTS.zip",
    "Mixtral": "mixtral8x7b-Instruct-v0.1_RESULTS.zip",
    "Qwen": "qwen3.5-27b_RESULTS.zip",
    "DeepSeek": "deepseek-r1-7b_RESULTS.zip",
}
STRATEGIES = {"simple": "Simple", "instruct": "Detail", "fewshot": "1-shot"}

# Table 1 of Errica et al. (2025), (sensitivity, consistency), checked against the PDF.
# Rows: TREC, CB, RTE, DBPedia, WoS. Columns: Simple, Detail, 1-shot.
PAPER = {
    "Llama 3": [[(.127, .693), (.141, .694), (.095, .686)], [(.018, .956), (.016, .963), (.014, .962)],
                [(.157, .701), (.213, .727), (.048, .755)], [(.018, .932), (.058, .901), (.030, .924)],
                [(.043, .479), (.034, .480), (.030, .468)]],
    "Mixtral": [[(.223, .657), (.182, .709), (.166, .640)], [(.203, .625), (.013, .695), (.017, .600)],
                [(.195, .669), (.176, .676), (.046, .754)], [(.053, .802), (.054, .808), (.069, .773)],
                [(.112, .483), (.124, .469), (.142, .472)]],
    "GPT-3.5": [[(.128, .573), (.148, .549), (.100, .649)], [(.274, .730), (.330, .732), (.288, .737)],
                [(.292, .719), (.346, .719), (.218, .703)], [(.026, .916), (.028, .895), (.026, .914)],
                [(.095, .482), (.093, .480), (.120, .486)]],
    "GPT-4o": [[(.042, .797), (.068, .708), (.037, .816)], [(.087, .814), (.081, .826), (.058, .880)],
               [(.233, .757), (.087, .823), (.074, .835)], [(.023, .939), (.022, .921), (.017, .922)],
               [(.048, .492), (.045, .493), (.046, .494)]],
}
DATASETS = ["TREC", "CB", "RTE", "DBPedia", "WoS"]


# ------------------------------------------------------------ our configurations
def identify_dataset(path: str) -> str | None:
    """Same rules as the notebook. CB uses the train file, like the paper's code."""
    path = path.lower()
    parts = path.split("/")
    if "cb-train" in parts:
        return "CB"
    for name, key in [("DBPedia", "dbpedia"), ("RTE", "rte"), ("TREC", "trec"), ("WoS", "wos")]:
        if key in parts:
            return name
    for name, key in [("DBPedia", "dbpedia_results"), ("TREC", "trec_results"), ("RTE", "rte_results"),
                      ("WoS", "wos_results"), ("WoS", "wos46985_results"), ("CB", "cb_results")]:
        if key in path:
            return name
    return None


def sensitivity(distributions: list[np.ndarray]) -> float:
    values = []
    for p in distributions:
        nz = p[p > 0]
        values.append(-(nz * np.log(nz)).sum() / np.log(len(p)))
    return float(np.mean(values))


def consistency(distributions: list[np.ndarray], targets: list[str]) -> float:
    total, pairs = 0.0, 0
    for target in set(targets):
        P = np.stack([d for d, t in zip(distributions, targets) if t == target])
        tvd = 0.5 * np.abs(P[:, None, :] - P[None, :, :]).sum(axis=2)
        total += float((1 - tvd).sum())
        pairs += len(P) ** 2
    return total / pairs


def our_configurations() -> pd.DataFrame:
    rows = []
    for model, filename in MODEL_FILES.items():
        with zipfile.ZipFile(MODELS_DIR / filename) as archive:
            for path in archive.namelist():
                dataset = identify_dataset(path)
                strategy = next((v for k, v in STRATEGIES.items() if path.endswith(f"results_{k}.json")), None)
                # Skip smoke tests and the extra CB run on test_labeled: CB uses the
                # train file, like the paper's code and the notebook.
                if dataset is None or strategy is None or "smoke" in path.lower() or "test_labeled" in path.lower():
                    continue
                data = json.loads(archive.read(path))
                samples = [s for s in data.values() if isinstance(s, dict) and "distribution" in s]
                dists = [np.asarray(s["distribution"], dtype=float) for s in samples]
                rows.append({"model": model, "dataset": dataset, "strategy": strategy,
                             "sensitivity": sensitivity(dists),
                             "consistency": consistency(dists, [s["target"] for s in samples])})
    df = pd.DataFrame(rows)
    assert not df.duplicated(["model", "dataset", "strategy"]).any(), "duplicate configuration"
    return df


def paper_configurations() -> pd.DataFrame:
    rows = []
    for model, per_dataset in PAPER.items():
        for dataset, per_strategy in zip(DATASETS, per_dataset):
            for strategy, (s, c) in zip(STRATEGIES.values(), per_strategy):
                rows.append({"model": model, "dataset": dataset, "strategy": strategy,
                             "sensitivity": s, "consistency": c})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- statistics
def pearson(df: pd.DataFrame) -> dict:
    n = len(df)
    if n < 4 or df["sensitivity"].nunique() < 2:
        return {"n": n, "r": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    r, p = stats.pearsonr(df["sensitivity"], df["consistency"])
    se = 1 / np.sqrt(n - 3)
    z = np.arctanh(r)
    return {"n": n, "r": r, "p": p, "ci_low": np.tanh(z - 1.96 * se), "ci_high": np.tanh(z + 1.96 * se)}


def fisher_difference(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """Two-sided p-value for H0: rho1 = rho2 (independent samples)."""
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(z), float(2 * stats.norm.sf(abs(z)))


def power_difference(rho1: float, rho2: float, n1: int, n2: int, alpha: float = ALPHA) -> float:
    """Power of the Fisher z-test to detect rho1 != rho2 with n1 and n2 configurations."""
    delta = abs(np.arctanh(rho1) - np.arctanh(rho2))
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    crit = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.cdf(delta / se - crit) + stats.norm.cdf(-delta / se - crit))


def demeaned(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in ["sensitivity", "consistency"]:
        out[col] = df[col] - df.groupby(by)[col].transform("mean")
    return out


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n## {title}\n")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    ours_all = our_configurations()
    paper_all = paper_configurations()
    ours = ours_all[ours_all["sensitivity"] >= THRESHOLD]
    paper = paper_all[paper_all["sensitivity"] >= THRESHOLD]
    print(f"Our configurations: {len(ours_all)}, with S >= {THRESHOLD}: {len(ours)}")
    print(f"Paper configurations: {len(paper_all)}, with S >= {THRESHOLD}: {len(paper)}")

    # 1. correlations with confidence intervals
    groups = [("Mixtral", ["Mixtral"]), ("Llama 3", ["Llama 3"]), ("Mixtral + Llama 3", ["Mixtral", "Llama 3"]),
              ("Qwen", ["Qwen"]), ("DeepSeek", ["DeepSeek"]), ("GPT-3.5", ["GPT-3.5"]), ("GPT-4o", ["GPT-4o"]),
              ("All four (ours: Mixtral, Llama 3, Qwen, DeepSeek)", list(MODEL_FILES), "ours"),
              ("All four (paper: Mixtral, Llama 3, GPT-3.5, GPT-4o)", list(PAPER), "paper")]
    rows = []
    for name, models, *only in groups:
        for source, data in [("ours", ours), ("paper", paper)]:
            if only and source != only[0]:
                continue
            sub = data[data["model"].isin(models)]
            if len(sub) == 0:
                continue
            rows.append({"group": name, "source": source, **pearson(sub)})
    table = pd.DataFrame(rows)
    show("1. Pearson r per group (S >= 0.05), 95% CI from the Fisher z-transform", table)

    # 2 and 3. difference ours vs paper, and power
    rows = []
    for name, models in [("Mixtral", ["Mixtral"]), ("Llama 3", ["Llama 3"]),
                         ("Mixtral + Llama 3", ["Mixtral", "Llama 3"])]:
        a = pearson(ours[ours["model"].isin(models)])
        b = pearson(paper[paper["model"].isin(models)])
        z, p = fisher_difference(a["r"], a["n"], b["r"], b["n"])
        rows.append({"group": name, "r_ours": a["r"], "n_ours": a["n"], "r_paper": b["r"], "n_paper": b["n"],
                     "z": z, "p_difference": p})
    show("2. Is our r different from the paper's r for the same models? (Fisher z-test)", pd.DataFrame(rows))

    a = pearson(ours[ours["model"].isin(["Mixtral", "Llama 3"])])
    b = pearson(paper[paper["model"].isin(["Mixtral", "Llama 3"])])
    rows = []
    for label, rho1, rho2 in [("our r vs the paper's reported -0.07", a["r"], -0.07),
                              ("-0.44 vs -0.07 (same as above, rounded)", -0.44, -0.07),
                              ("-0.44 vs 0 (no correlation)", -0.44, 0.0)]:
        rows.append({"comparison": label, "n1": a["n"], "n2": b["n"],
                     "power": power_difference(rho1, rho2, a["n"], b["n"])})
    rows.append({"comparison": "n per group needed for 80% power, -0.44 vs -0.07",
                 "n1": next(n for n in range(5, 1000) if power_difference(-0.44, -0.07, n, n) >= 0.8),
                 "n2": np.nan, "power": 0.8})
    show("3. Power of the Fisher z-test (alpha = 0.05, two-sided)", pd.DataFrame(rows))

    # 4. pooling: per-model and within-group correlations
    rows = []
    for source, data, models in [("paper", paper, list(PAPER)), ("paper", paper, ["Mixtral", "Llama 3"]),
                                 ("ours", ours, ["Mixtral", "Llama 3"]), ("ours", ours, list(MODEL_FILES))]:
        sub = data[data["model"].isin(models)]
        rows.append({"source": source, "models": " + ".join(models),
                     "pooled_r": pearson(sub)["r"],
                     "within_model_r": pearson(demeaned(sub, ["model"]))["r"],
                     "within_dataset_r": pearson(demeaned(sub, ["dataset"]))["r"],
                     "within_dataset_p": pearson(demeaned(sub, ["dataset"]))["p"]})
    show("4. Pooled vs within-group correlations (group means removed)", pd.DataFrame(rows))
    ds = ours[ours["model"] == "DeepSeek"]
    show("DeepSeek: its 6 configurations come from two datasets only",
         ds.groupby("dataset")[["sensitivity", "consistency"]].agg(["mean", "min", "max"]).round(4).reset_index())
    print(f"DeepSeek within-dataset r: {pearson(demeaned(ds, ['dataset']))['r']:.4f} (n = {len(ds)}, 3 strategies per dataset)")
    print("\nNote: removing group means costs degrees of freedom, so the p-values in part 4 are slightly optimistic.")

    means = paper_all.groupby("model")[["sensitivity", "consistency"]].mean()
    show("Paper Table 1: mean sensitivity and consistency per model (all 15 configurations)", means.reset_index())


if __name__ == "__main__":
    main()
