# Data for the sensitivity and consistency reproduction

This folder holds every file that the five notebooks read.
The notebooks read `./data/...` from the folder `original-paper`, so the paths work as they are.
We compared each file with its official source on 2026-10-27. The full record is in `docs/datasets/README.md`.

## Datasets

| Folder | Task | Paper setup | What the notebook evaluates | Explanation |
|---|---|---|---|---|
| `TREC/` | 6-class question type | 500 test samples | `TREC/test.txt`, 500 rows, no cut | [TREC/README.md](TREC/README.md) |
| `CB/` | 3-class entailment | 250 test samples | `CB/train.jsonl`, 250 rows, no cut (see the CB note) | [CB/README.md](CB/README.md) |
| `RTE/` | 2-class entailment | 2490 test samples | `RTE/train.jsonl`, 2490 rows, no cut | [RTE/README.md](RTE/README.md) |
| `DBPEDIA/` | 14-class ontology | 2000 balanced test samples | 2000 of 70,000 rows, cut with seed 42 | [DBPEDIA/README.md](DBPEDIA/README.md) |
| `WOS46985/` | 7-class field of study | 2000 balanced samples | 2000 of 46,985 rows, cut with seed 42 | [WOS46985/README.md](WOS46985/README.md) |

## Every cut in one place

| Dataset | Source size | Evaluated | How the cut works | Ready-made file |
|---|---|---|---|---|
| TREC | 500 test | 500 | none | not needed |
| CB | 250 train | 250 | none | `CB/test_labeled.jsonl` is the second option (paper text) |
| RTE | 2490 train | 2490 | none | not needed |
| DBPEDIA | 70,000 | 2000 | shuffle with seed 42, 142 for each class, 12 more at random | `DBPEDIA/evaluation_subset_2000.jsonl` |
| WoS | 46,985 | 2000 | shuffle with seed 42, 285 for each class, 5 more at random | `WOS46985/evaluation_subset_2000.jsonl` |

The 1-shot prompt uses examples from outside the evaluated set. DBPEDIA and WoS keep them in `one_shot_examples.jsonl`.
TREC, CB and RTE write them in the notebook prompt.

## Check, restore and rebuild

Run from the repository root. On ITU HPC, run these with `srun` or `sbatch`. Do not run them on the login node.

```bash
python3 scripts/fetch_datasets.py --verify-only   # compare every file with docs/datasets/data-manifest.json
python3 scripts/fetch_datasets.py                 # restore missing or damaged files, then verify
```

The script does these steps:

1. Download a file only when it is missing or its SHA-256 differs from the manifest.
2. Extract `WOS46985/X.txt` from `X.zip`. Git ignores `X.txt` (65 MB).
3. Build `CB/test_labeled.jsonl` from `CommitmentBank-items.csv`.
4. Write the DBPEDIA and WoS subset files with `scripts/replay_sampling.py`.

## Before you run a notebook

- Delete any `data/*_RESULTS/` folder from an older run. The notebooks cache the sampled data in `data/<DATASET>_RESULTS/data.json` and reuse it. An old cache hides a changed input file.
- Use Python 3.10. We got the same 2000 samples on Python 3.10, 3.11, 3.12, 3.13 and 3.14.
- Do not change the order of rows in a source file. The sample IDs are row numbers.

## Licenses and sources

| Dataset | License evidence we found | Citation |
|---|---|---|
| TREC | None found in the files or on the pages we read. Not verified. | Li and Roth 2002; Voorhees and Tice 2000 |
| CB | The GitHub repository has no license file. Not verified. | de Marneffe, Simons and Tonhauser 2019 |
| RTE | Not verified. SuperGLUE combines several RTE challenge sets. | Dagan et al. 2005 and later RTE challenges |
| DBPedia | CC BY-SA and GNU FDL. Stated in `DBPEDIA/readme.txt`. | Zhang, Zhao and LeCun 2015 |
| WoS-46985 | MIT-style permission notice. Stated in `WOS46985/ReadMe.txt`. | Kowsari et al. 2017 and Mendeley Data v6 |

The authors' repository (NEC Laboratories Europe GmbH) already contains the same data files.
The GitHub repository of this project is private. Check the licenses above before you make it public.
