# Dataset fidelity

This folder records how close our datasets are to the datasets in the paper.
We checked each file against its official source on 2026-10-27.

## Where the data is

The datasets are committed in `original-paper/data/`. Each dataset folder has its own `README.md` with the source, the class counts, the cut and the 1-shot examples. Start with [the data index](../../original-paper/data/README.md).

| File in this folder | Purpose |
|---|---|
| `data-manifest.json` | SHA-256 of every data file, its source, and how we compared it with the official source |

## Check, restore and rebuild

Run from the repository root. On ITU HPC, run these with `srun` or `sbatch`. Do not run them on the login node.

```bash
python3 scripts/fetch_datasets.py --verify-only
python3 scripts/fetch_datasets.py
```

The second command downloads a file only when it is missing or damaged. It also extracts `WOS46985/X.txt` (git ignores it), builds `CB/test_labeled.jsonl`, and writes the DBPedia and WoS subset files.
Test on 2026-10-27: we deleted 8 files and damaged 1 in a copy. The script restored all 9 and all 21 files matched the manifest.

Check the subsampling on any machine:

```bash
python3 scripts/replay_sampling.py --dataset dbpedia --input-file original-paper/data/DBPEDIA/test.csv
python3 scripts/replay_sampling.py --dataset wos \
  --input-file original-paper/data/WOS46985/X.txt \
  --target-file original-paper/data/WOS46985/YL1.txt
```

The printed `ids sha256` must start with `d80c8cf59cacb7d8` (DBPedia) and `0123a6e1855264f0` (WoS).
We got the same IDs on Python 3.10.18, 3.11.13, 3.12.9, 3.13.7 and 3.14.8.

## Checks against sources (2026-10-27)

| Check | Result |
|---|---|
| Official source files, byte or line compare (TREC, SuperGLUE RTE and CB, DBPedia, WoS) | Equal. Differences are listed below. |
| Mendeley file hash for `WebOfScience.zip` | Equal to the hash that Mendeley lists |
| Independent mirror: Hugging Face `fancyzhx/dbpedia_14`, test split | 70,000 rows equal (whitespace normalized) |
| Independent mirror: Hugging Face `CogComp/trec`, test split | 500 rows, same order, same coarse labels |
| Independent mirror: Hugging Face `CogComp/trec`, train split | 5452 rows equal, except line 66 (see below) |
| The notebooks' own data cells on the committed files | CB 250, RTE 2490, DBPedia 2000, WoS 2000 samples. The DBPedia and WoS results equal the committed subset files in the same order. |
| Repaired TREC notebook loader | 500 test and 5452 train rows |
| Authors' stored notebook output | "Num test examples" is 250 (CB), 2490 (RTE), 2000 (DBPedia), 500 (TREC, paper text). Our counts equal them. |
| Paper text | TREC Entity samples 46 to 50 are the 5 questions quoted in the paper, in order |
| Restore test on a damaged copy | All files restored and verified |

## Result per dataset

| Dataset | Paper | Our data | Verified against | Status |
|---|---|---|---|---|
| TREC | 6 classes, 500 test | `TREC/test.txt`, 500 rows | cogcomp `TREC_10.label` | Same content. Sample order matches the paper (Entity 46 to 50). |
| CB | 3 classes, 250 test | `CB/train.jsonl` (released code) and `CB/test_labeled.jsonl` (paper text) | SuperGLUE `CB.zip`, original CB annotations | Both files are exact. See the CB note below. |
| RTE | 2 classes, 2490 "test" | `RTE/train.jsonl`, 2490 rows | SuperGLUE `RTE.zip` | Best possible. No 2490-row test set exists. |
| DBPedia | 14 classes, 2000 balanced | `DBPEDIA/test.csv`, 70000 rows, sampled by notebook | Zhang et al. `dbpedia_csv.tgz` | Same rows. Sampling is reproducible. |
| WoS | 7 classes, 2000 balanced | `WOS46985/*`, 46985 rows, sampled by notebook | Mendeley Data v6 | Byte-identical. Sampling is reproducible. |

## Differences we found

- **TREC `train.txt`:** line 66 has U+FFFD where the official latin-1 file has byte `0xF0`. The notebook reads UTF-8, so the authors re-encoded the file. The line is not evaluated. Both TREC files lack the final newline.
- **TREC notebook:** the upstream file is cut inside an image output. `LLM TREC.ipynb` in this repository is the repaired copy. It keeps the 7 complete cells (loader, prompts, 1-shot examples, grid search). The plot cells after the cut are lost.
- **DBPedia `test.csv`:** one header line `Class,Title,Text` added to the original file.
- **RTE:** the paper says "2490 test samples". SuperGLUE RTE has train 2490, validation 277, and test 3000 without labels. The notebook reads `train.jsonl` and its stored output prints `2490`. We use `train.jsonl`.
- **RTE 1-shot prompt:** example 2 ("kidnappers", "he's been to America") is a CB validation item. It is not an RTE item. This is the authors' code.
- **CB:** the authors' `parse_CB` reads `json_obj['label']` for every row. The official `test.jsonl` has no labels, so the authors' run could not have used it. The paper's CB numbers most likely come from `train.jsonl` (250 labeled rows). The paper text says "test". Run both files and label them clearly. The recovered test labels are a reconstruction, not an official file.
- **Paper Table 1:** the reference table copies the paper. See the next section.

## Paper Table 1 against the authors' stored outputs

The notebooks keep the authors' output from 2024-08. We compared "Avg Entropy over samples" with the sensitivity values in Table 1 for 48 cells (CB, RTE, DBPedia, WoS x Llama3, Mixtral, GPT-3.5, GPT-4o x 3 prompts). TREC has no stored output because the notebook is cut.

45 cells match within 0.001. Three differ:

| Dataset | Model | Prompt | Stored output | Paper Table 1 |
|---|---|---|---|---|
| CB | Mixtral | Detail | 0.1255 | 0.013 |
| CB | Mixtral | 1-shot | 0.1740 | 0.017 |
| WoS | Llama3 | 1-shot | 0.0353 | 0.030 |

The two CB cells differ by a factor of about 10. Appendix D of the paper gives a standard deviation of 0.188 for both cells. A value between 0 and 1 with mean 0.013 cannot have a standard deviation of 0.188. The stored outputs fit (standard deviation 0.1889 and 0.1885). Table 1 most likely has a decimal-point error in these two cells. The README reference table repeats the paper values. Use 0.125 and 0.174 as the expected CB Mixtral sensitivity for Detail and 1-shot, and cite the stored outputs as the reason.
The WoS Llama3 cell differs by 0.005. We have no explanation. The WoS notebook started Ray twice, so the stored output may come from a second run.

## What we did not verify

- Whether the authors' CB run used `train.jsonl`. We infer it from the code path.
- Model results. No model ran for this work.
- The plot cells of the TREC notebook.
- The data on HPC. Run `scripts/fetch_datasets.py --verify-only` there with `srun`.
- The licenses of the TREC, CB and RTE files. We found no license text. Check them before you make the repository public.
