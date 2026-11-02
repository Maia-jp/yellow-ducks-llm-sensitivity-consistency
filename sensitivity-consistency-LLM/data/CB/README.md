# CommitmentBank (CB)

## What the paper uses

A 3-class entailment task (entailment, contradiction, neutral) with 250 test samples. An extra class `N/A` holds invalid model answers.

## Files

| File | Rows | SHA-256 (start) | Role |
|---|---:|---|---|
| `train.jsonl` | 250 | `32017480d3a1` | **Read by the released notebook.** Has labels. |
| `val.jsonl` | 56 | `c6125587e960` | Source of the 3 fixed 1-shot examples. Has labels. |
| `test.jsonl` | 250 | `860679c4ef6f` | Official SuperGLUE test file. **No labels.** |
| `test_labeled.jsonl` | 250 | `3c77ece9a973` | Our rebuild of the test labels (see below). |
| `../CommitmentBank-items.csv` | 1200 items | `e06d30b2e197` | Original annotator responses. Input of the rebuild. |

Record format: `{"premise": ..., "hypothesis": ..., "idx": ..., "label": ...}`

## Source

- `train.jsonl`, `val.jsonl`, `test.jsonl`: SuperGLUE `CB.zip` (https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip). All three files are byte-identical to it.
- `CommitmentBank-items.csv`: https://github.com/mcdm/CommitmentBank at commit `14ea91f56e3575b011d28fca5c37a1f5e92769b7`.

## Class counts

| Class | `train.jsonl` | `val.jsonl` | `test_labeled.jsonl` |
|---|---:|---:|---:|
| entailment | 115 | 23 | 113 |
| contradiction | 119 | 28 | 121 |
| neutral | 16 | 5 | 16 |

## The CB split problem (read this first)

The paper says 250 **test** samples. The released notebook reads `train.jsonl`, which also has 250 rows.
The official `test.jsonl` has no labels, so the notebook code `json_obj['label']` would fail on it.
Therefore the authors' run most likely used `train.jsonl`. This is an inference from the code. We cannot prove it.

| Option | File | Fits | Use it for |
|---|---|---|---|
| A: released code | `train.jsonl` | The authors' code and probably the paper's numbers | Comparing our numbers with Table 1 |
| B: paper text | `test_labeled.jsonl` | The words "250 test samples" | A held-out evaluation |

Run both and label each result with its file name. Do not use option B labels to tune prompts or to choose models.
The unchanged notebook runs option A. To run option B, change `input_file` in the first data cell of `LLM CB.ipynb` to `./data/CB/test_labeled.jsonl`.

## How `test_labeled.jsonl` is built

1. Match each of the 250 test rows to one CB item. The key is the premise (`Context` + `Target`) and the hypothesis (`Prompt`), with whitespace normalized.
2. Apply the published rule of Jiang and de Marneffe (2019): at least 80% of the annotator responses must fall in one range. 1 to 3 is entailment. 0 is neutral. -3 to -1 is contradiction.
3. Write the result. The script is `scripts/prepare_cb_test_labels.py`.

Checks:

- All 250 test rows matched one item each. The counts (113, 16, 121) equal the test split in Table 2 of the CB recast paper. We took that table value from an earlier project note (removed from this branch) and did not read the paper again.
- The same rule reproduces all 250 train labels and all 56 validation labels.
- The labels are a reconstruction. No author-provided gold file for the SuperGLUE test split exists.

## 1-shot examples

The prompt in `LLM CB.ipynb` has 3 fixed examples (contradiction, entailment, neutral). All 3 premises are in `val.jsonl`. None is in `train.jsonl` or `test.jsonl`.

## The cut

None. The notebook uses all rows of the file it reads. The sample `id` is the `idx` field.

## Citation

Marie-Catherine de Marneffe, Mandy Simons and Judith Tonhauser. 2019. The CommitmentBank. Sinn und Bedeutung 23.
Nanjiang Jiang and Marie-Catherine de Marneffe. 2019. Evaluating BERT for natural language inference: A case study on the CommitmentBank. EMNLP 2019.
Alex Wang et al. 2019. SuperGLUE. NeurIPS 2019.
