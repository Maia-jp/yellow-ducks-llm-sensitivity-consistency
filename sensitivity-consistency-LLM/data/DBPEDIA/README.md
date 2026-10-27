# DBPedia ontology classification

## What the paper uses

A 14-class ontology task (Zhang et al. 2015) with 2000 balanced test samples. The 1-shot prompt uses one example for each class from outside the evaluated samples.

## Files

| File | Rows | SHA-256 (start) | Role |
|---|---:|---|---|
| `test.csv` | 70,000 | `1f91caaf162f` | The full DBPedia test set. The notebook reads it and cuts 2000 rows. |
| `classes.txt` | 14 | `3390a11cf615` | Class names in the order of the class index (1 to 14). |
| `evaluation_subset_2000.jsonl` | 2000 | `473df6c29ee5` | **The 2000 evaluated samples**, in the notebook order. |
| `one_shot_examples.jsonl` | 14 | `ffddfe55773f` | The 1-shot examples, one for each class, in class order. |
| `readme.txt` | | `b5abe6e8539d` | The official readme. It states the license. |

Columns of `test.csv`: `Class` (1 to 14), `Title`, `Text`. The `Text` column is the model input.
Records of the two `.jsonl` files: `{"id": ..., "class": ..., "input": ...}`.

## Source

- Archive: `dbpedia_csv.tgz` from Zhang, Zhao and LeCun (fast.ai mirror https://s3.amazonaws.com/fast-ai-nlp/dbpedia_csv.tgz, SHA-256 `42db5221...`).
- Independent check: the Hugging Face dataset `fancyzhx/dbpedia_14` has the same 70,000 test rows (compared with whitespace normalized).

## Difference from the official file

`test.csv` has one added header line `Class,Title,Text` (+17 bytes). All 70,000 rows are equal. The notebook needs the header (`csv.DictReader`).
`classes.txt` is identical to the official file. It lists names without spaces (`EducationalInstitution`). The notebook uses its own names with spaces (`Educational Institution`).

## The cut: 70,000 rows to 2000

This is the code of the notebook (`balance_dataset`, `n_samples = 2000`, `random.seed(42)`):

1. Read all 70,000 rows. The sample `id` is the 1-based row number after the header.
2. `random.shuffle` the rows (seed 42).
3. Walk the shuffled rows. Keep a row while its class has fewer than `2000 // 14 = 142` rows. This gives 1988 rows.
4. Add 12 rows at random from the unused rows, until the set has 2000 rows.
5. The rows that stay unused are the pool for the 1-shot examples. The first unused row of each class (in shuffled order) is the 1-shot example.

Result:

- 2000 unique samples. Each class has 142 to 144 samples. Nine classes got 1 or 2 of the 12 extra rows.
- The 14 1-shot rows have the IDs 4513, 7360, 11650, 17014, 21220, 29918, 30205, 39512, 43021, 45004, 51301, 58998, 61869 and 67363. None is in the evaluated set.
- `evaluation_subset_2000.jsonl` equals the notebook's `data_list` (checked, same order). The order matters for `--max-samples` runs.
- The IDs have the SHA-256 `d80c8cf59cacb7d8...` (see `docs/datasets/sample-ids-dbpedia.json`).
- The same IDs came out on Python 3.10.18, 3.11.13, 3.12.9, 3.13.7 and 3.14.8.
- The sample at index 1 (ID 63788) is the "Darkman" film text. The authors' stored notebook output prints the same text at index 1.

Rebuild the two subset files:

```bash
python3 scripts/replay_sampling.py --dataset dbpedia \
  --input-file sensitivity-consistency-LLM/data/DBPEDIA/test.csv \
  --subset-dir sensitivity-consistency-LLM/data/DBPEDIA
```

## Citation

Xiang Zhang, Junbo Zhao and Yann LeCun. 2015. Character-level Convolutional Networks for Text Classification. NeurIPS 2015.
License: Creative Commons Attribution-ShareAlike and GNU Free Documentation License (from `readme.txt`).
