# Web of Science, WOS-46985

## What the paper uses

The 7-class Web of Science dataset (Kowsari et al.) with 2000 balanced samples. The 1-shot prompt uses one example for each class from outside the evaluated samples.

## Files

| File | Rows | SHA-256 (start) | Role |
|---|---:|---|---|
| `X.zip` | 1 file | `2e769e9d3adc` | Zip of `X.txt` made by the authors. Git tracks the zip. |
| `X.txt` | 46,985 | `82cc4794b577` | One abstract on each line. **Not in git (65 MB).** Run `python3 scripts/fetch_datasets.py` to extract it. |
| `YL1.txt` | 46,985 | `691872ed9eb0` | **Level-1 label (0 to 6). The notebook reads this file.** |
| `evaluation_subset_2000.jsonl` | 2000 | `e5c7c63dae07` | **The 2000 evaluated samples**, in the notebook order. |
| `one_shot_examples.jsonl` | 7 | `9550f2f5b6a4` | The 1-shot examples, one for each class, in class order. |
| `ReadMe.txt` | | `6da4ab1e624c` | The official readme with the permission notice. Keep it with the data. |

Line `i` of `YL1.txt` is the label of line `i` of `X.txt`. The sample `id` is the 1-based line number.
Records of the two `.jsonl` files: `{"id": ..., "class": ..., "input": ...}`.

## Source

- Mendeley Data, "Web of Science Dataset", version 6 (2018-03-15), DOI 10.17632/9rw3vkcfy4.6. File `WebOfScience.zip`, SHA-256 `b787d484bff88b0dcdb3fa291d06ec9d2f025dc2a67ce1045d0c688cd96ccf8a`. This equals the hash that Mendeley lists.
- `X.txt` and `YL1.txt` are byte-identical to the files in `WOS46985/` of that zip.
- We did not copy `Y.txt` (134 area labels) and `YL2.txt` (53 level-2 labels). The notebook does not read them.
- The zip also has `WOS5736/` and `WOS11967/`. We did not copy them. The paper uses WOS-46985.

## Classes

The notebook order and the official `Meta-data/Data.xlsx` agree:

| `YL1` value | Class | Rows in the full dataset |
|---:|---|---:|
| 0 | Computer Science | 6514 |
| 1 | Electrical Engineering | 5483 |
| 2 | Psychology | 7142 |
| 3 | Mechanical Engineering | 3297 |
| 4 | Civil Engineering | 4237 |
| 5 | Medical Science | 14,625 |
| 6 | Biochemistry | 5687 |

The full set is not balanced. The paper balances it by the cut below.

## The cut: 46,985 rows to 2000

This is the code of the notebook (`balance_dataset`, `n_samples = 2000`, `random.seed(42)`):

1. Read all rows with their labels.
2. `random.shuffle` the rows (seed 42).
3. Keep a row while its class has fewer than `2000 // 7 = 285` rows. This gives 1995 rows.
4. Add 5 rows at random from the unused rows, until the set has 2000 rows.
5. The first unused row of each class (in shuffled order) is the 1-shot example.

Result:

- 2000 unique samples. Class counts: Computer Science 286, Electrical Engineering 285, Psychology 286, Mechanical Engineering 285, Civil Engineering 285, Medical Science 287, Biochemistry 286.
- The 7 1-shot rows have the IDs 46694, 23822, 26312, 16814, 23528, 7204 and 46169 (class order). None is in the evaluated set.
- `evaluation_subset_2000.jsonl` equals the notebook's `data_list` (checked, same order).
- The IDs have the SHA-256 `0123a6e1855264f0...` (printed by `scripts/replay_sampling.py`, stored in `docs/datasets/data-manifest.json`).
- The same IDs came out on Python 3.10.18, 3.11.13, 3.12.9, 3.13.7 and 3.14.8.
- The authors' stored notebook output prints the same text at index 1 as our subset (a Biochemistry abstract about "nanoscale spacing between the plasma membrane and ... cortical actin").
- The abstracts have 190 to 4744 characters (median 1304).

Rebuild the two subset files (needs `X.txt`):

```bash
python3 scripts/replay_sampling.py --dataset wos \
  --input-file original-paper/data/WOS46985/X.txt \
  --target-file original-paper/data/WOS46985/YL1.txt \
  --subset-dir original-paper/data/WOS46985
```

## Citation and license

Kamran Kowsari, Donald E. Brown, Mojtaba Heidarysafa, Kiana Jafari Meimandi, Matthew S. Gerber and Laura E. Barnes. 2017. HDLTex: Hierarchical Deep Learning for Text Classification. ICMLA 2017.
Dataset: Kamran Kowsari et al. 2018. Web of Science Dataset v6. Mendeley Data. https://doi.org/10.17632/9rw3vkcfy4.6
License: the MIT-style permission notice at the top of `ReadMe.txt`. Copies must keep that notice.
