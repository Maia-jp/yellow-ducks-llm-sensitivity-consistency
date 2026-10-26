# CB dataset findings timeline

This file records the evidence and work in the order we found it.
The first finding below was corrected after we found the original CB annotations.
The latest implementation uses the labeled CB test split.

## Finding

The paper says that its CommitmentBank (CB) evaluation used 250 test samples. The released CB notebook reads `data/CB/train.jsonl`. The paper and the released notebook therefore specify different splits.

The official SuperGLUE test set has 250 records, but its labels are hidden. The public train file has 250 labeled records. The train file supports the paper's Micro-F1 and class-based consistency calculations. The test file does not support these calculations without gold labels.

At the time of the first check, Qwen CB jobs used the released notebook path. They used the labeled train split. Report those runs as an evaluation of the released code, not as an exact reproduction of the paper's stated test split.

## Evidence

| Check | Evidence | Result |
|---|---|---|
| Paper split | [NAACL 2025 paper, Experiments](https://aclanthology.org/2025.naacl-long.73.pdf) | The paper describes CB as a three-class task with 250 test samples. |
| Notebook input | [Official CB notebook at the source commit](https://github.com/nec-research/sensitivity-consistency-LLM/blob/657355688d4b58a6e3a2ac20484518c5d402c8d8/LLM%20CB.ipynb) | The notebook reads `./data/CB/train.jsonl`. |
| Official data files | [CB data at the source commit](https://github.com/nec-research/sensitivity-consistency-LLM/tree/657355688d4b58a6e3a2ac20484518c5d402c8d8/data/CB) | The repository contains `train.jsonl`, `test.jsonl`, and `val.jsonl`. |
| Official test rules | [SuperGLUE FAQ](https://super.gluebenchmark.com/faq/) | The benchmark provides unlabeled test files. Submit predictions to get benchmark scores. |
| Dataset loader | [TensorFlow Datasets SuperGLUE loader](https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/text/super_glue.py) | The loader assigns label `-1` when a test row has no label. |
| Train file | SHA-256 `32017480d3a10508646e3efd64502c639f13d4d15077f278d2fd0992c33a3126` | 250 records. All records have labels: 115 entailment, 119 contradiction, and 16 neutral. |
| Test file | SHA-256 `860679c4ef6f56758f5530bfbcbf2fd7905667184144f3594ace9bb4c435618d` | 250 records. No record has a `label` field. |
| Validation file | SHA-256 `c6125587e9601560cb16ea223c33964c5a915568d2af68a81f651b6ef25395ec` | 56 records. All records have labels. |

The three file hashes above come from the source commit recorded in [the reproduction guide](reproduction-guide.md#source-and-attribution). The test file has 250 JSON records. It lacks a final newline, so `wc -l` reports 249 lines.

## HPC copy and job status

The earlier HPC check found 250 records in `sensitivity-consistency-LLM/data/CB/train.jsonl`. Its first record matched the official train file. The check did not compare full-file hashes.

The last successful SLURM check found jobs `169976` and `169977` pending. A later SSH check failed because the host name did not resolve. Current job status and the HPC file hash are not verified.

## Report guidance

Use one of these descriptions in the report:

- **Released-code run:** Use the labeled `train.jsonl` file. State that the notebook uses the train split, although the paper calls the 250 CB samples test samples.
- **Paper-split run:** Use the 250-record `test.jsonl` file. Obtain its gold labels before reporting Micro-F1 or class-based consistency.

## Recommended method

The initial recommendation was to use the labeled `train.jsonl` file to reproduce the released notebook and calculate all three metrics. The notebook selected this file. Do not use that recommendation for the paper-split evaluation. The later annotation check below provides labels for the test examples.

The official benchmark accepts predictions for the test split and calculates its benchmark scores. Its public rules do not provide the hidden labels. The benchmark also does not calculate this paper's sensitivity and class-based consistency metrics for our prompt variants.

The report must state both facts: the paper calls its CB samples test samples, and the released notebook uses the train split. Describe our run as a reproduction of the released code on the labeled train split. Do not describe it as a held-out test evaluation.

At that point, an exact run on the paper's stated test split appeared to need author-provided labels. The later annotation check below resolves this problem by reconstructing labels from the original public annotations.

## Web follow-up: recover the CB test labels

Date: 2026-10-26

This follow-up adds a way to recover the test labels from the original CB annotation data. It corrects the earlier conclusion that the public files cannot provide the labels.

The official SuperGLUE test file has 250 unlabeled records. The official benchmark asks users to submit predictions for test scoring. Its rules do not publish the hidden labels. [SuperGLUE FAQ](https://super.gluebenchmark.com/faq/) · [SuperGLUE CB split sizes](https://www.tensorflow.org/datasets/catalog/super_glue#super_gluecb)

The original CB repository publishes `CommitmentBank-items.csv`. This file has one row per CB item. It includes the context, target, hypothesis prompt, and annotator responses. [Original CB data](https://github.com/mcdm/CommitmentBank/blob/14ea91f56e3575b011d28fca5c37a1f5e92769b7/CommitmentBank-items.csv) · [Original CB data description](https://github.com/mcdm/CommitmentBank/blob/14ea91f56e3575b011d28fca5c37a1f5e92769b7/README.md)

The CB recast paper defines the label rule. At least 80% of responses from 1 to 3 mean entailment. At least 80% of zero responses mean neutral. At least 80% of responses from -3 to -1 mean contradiction. [Jiang and de Marneffe, 2019](https://aclanthology.org/D19-1630.pdf)

I matched all 250 SuperGLUE test records to one CB item. The match uses the premise (`Context` + `Target`) and the hypothesis (`Prompt`), with whitespace normalization. I applied the published 80% rule to each item's responses. The resulting counts are 113 entailment, 16 neutral, and 121 contradiction. These counts match the test split in Table 2 of the CB recast paper. The same rule reproduces all 250 train labels and all 56 validation labels.

The preparation script is [prepare_cb_test_labels.py](../scripts/prepare_cb_test_labels.py). It writes a new labeled test file. It does not change the original files. The script passed a run on all 250 official test records. It matched the published class counts.

To prepare the labeled test file, download the official `test.jsonl` file and `CommitmentBank-items.csv`. These commands use the source commits checked above:

```bash
mkdir -p data/CB
curl -fsSLo data/CB/test.jsonl \
  https://raw.githubusercontent.com/nec-research/sensitivity-consistency-LLM/657355688d4b58a6e3a2ac20484518c5d402c8d8/data/CB/test.jsonl
curl -fsSLo data/CommitmentBank-items.csv \
  https://raw.githubusercontent.com/mcdm/CommitmentBank/14ea91f56e3575b011d28fca5c37a1f5e92769b7/CommitmentBank-items.csv
python scripts/prepare_cb_test_labels.py \
  --test-jsonl data/CB/test.jsonl \
  --items-csv data/CommitmentBank-items.csv \
  --output-jsonl data/CB/test_labeled.jsonl
```

The source `CommitmentBank-items.csv` SHA-256 is `e06d30b2e197fe88b9917c049c534df74d35d37eddb2a55f6e7112c56e016401`. The prepared test file SHA-256 is `3c77ece9a973a65cc9889bc150291adc145bcc636a721abe48b5b72061ea73be`.

Use `test_labeled.jsonl` for the paper's stated test split. Keep the model and prompt settings fixed before you inspect test scores. Do not use the recovered test labels to select models or tune prompts.

The test labels are reconstructed from the original public annotations. We did not compare each reconstructed label with an author-provided SuperGLUE gold-label file. The published class counts and every available train and validation label match the reconstructed rule.

## Timeline of findings and work

| Date | Finding or work | Evidence or result |
|---|---|---|
| 2026-10-26 | Compared the paper's CB split with the released notebook. | The paper reports 250 test examples. The notebook reads the 250-row labeled train file. |
| 2026-10-26 | Checked the official SuperGLUE files and scoring rules. | The 250-row test file has no labels. The train and validation files have 250 and 56 labels. |
| 2026-10-26 | Checked the original CB annotation data and published label rule. | Each test example matched one original item. Applying the 80% rule produced the paper's test counts. The same rule reproduced all train and validation labels. |
| 2026-10-26 | Added `scripts/prepare_cb_test_labels.py`. | The script creates a new labeled test file and rejects missing, duplicate, or ambiguous matches. |
| 2026-10-26 | Updated the Qwen CB runner and run guide. | CB runs now use `test_labeled.jsonl` by default. Other dataset inputs stay unchanged. |
| 2026-10-26 | Re-ran the local data preparation checks. | 250 test rows matched. Counts were 113 entailment, 16 neutral, and 121 contradiction. No HPC inference ran for this change. |

The recovered labels support the paper's CB test evaluation. They are reconstructed labels, not an author-provided hidden-label file. Keep model and prompt choices fixed before you inspect test scores.
