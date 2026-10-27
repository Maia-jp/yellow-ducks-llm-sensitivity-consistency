# RTE (Recognizing Textual Entailment)

## What the paper uses

A binary entailment task with 2490 "test samples" (Dagan et al. 2005).

## Files

| File | Rows | SHA-256 (start) | Role |
|---|---:|---|---|
| `train.jsonl` | 2490 | `fdfb03443abe` | **Read by the notebook.** All 2490 rows are evaluated. |

Record format: `{"premise": ..., "hypothesis": ..., "label": "entailment" | "not_entailment", "idx": ...}`

## Source

SuperGLUE `RTE.zip` (https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip). `train.jsonl` is byte-identical to the file in the zip.

## Why 2490 "test" samples are really the train file

| SuperGLUE RTE split | Rows | Labels |
|---|---:|---|
| train | 2490 | yes |
| validation | 277 | yes |
| test | 3000 | hidden |

The paper count 2490 equals the train size. No 2490-row test set exists.
The notebook reads `./data/RTE/train.jsonl`, and its stored output from 2024-08 prints `2490`.
So `train.jsonl` is the closest possible data. The paper text calls it "test".
We did not copy the validation and test files. The notebook does not use them.

## Class counts

| Class | Rows |
|---|---:|
| entailment | 1249 |
| not_entailment | 1241 |

The notebook maps `not_entailment` to the class name `contradiction/neutral` in its prompts and answers.
Two rows share the same premise and hypothesis with another row. We kept them, as the notebook does.

## The cut

None. The notebook uses all 2490 rows. The sample `id` is the `idx` field.

## 1-shot examples

The prompt in `LLM RTE.ipynb` has 2 fixed examples:

- Example 1 (contradiction/neutral): the Dana Reeve premise. It is not in `train.jsonl`.
- Example 2 (entailment): the "kidnappers ... he's been to America" premise. **This is a CB item** (it is in `CB/val.jsonl`). It is not an RTE item. This is the authors' code. We did not change it.

We did not check the RTE validation file for example 1.

## Citation

Ido Dagan, Oren Glickman and Bernardo Magnini. 2005. The PASCAL Recognising Textual Entailment Challenge. MLCW 2005.
Alex Wang et al. 2019. SuperGLUE. NeurIPS 2019. (This version merges RTE challenge data. We did not recheck which challenges.)
