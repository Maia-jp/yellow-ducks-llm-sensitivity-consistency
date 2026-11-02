# TREC question classification

## What the paper uses

A 6-class question task with 50 subclasses and 500 test samples. The 1-shot prompt uses one example for each class from outside the test set.

## Files

| File | Rows | SHA-256 (start) | Role |
|---|---|---|---|
| `test.txt` | 500 | `69b2d406aa15` | Evaluation. The notebook uses all 500 rows. |
| `train.txt` | 5452 | `202e3e1ad614` | The notebook parses it. The 1-shot examples come from it. |

Line format: `COARSE:subclass question text`. Example: `NUM:dist How far is it from Denver to Aspen ?`

## Source

- Page: https://cogcomp.seas.upenn.edu/Data/QA/QC/
- Official files: `train_5500.label` and `TREC_10.label` (SHA-256 `9e4c8bdc...` and `033f22c0...`).
- The 500 test questions are the TREC-10 questions. The train file is the 5500-question training set of Li and Roth (5452 rows, as the official file holds).

## Class counts

| Class | `train.txt` | `test.txt` |
|---|---:|---:|
| Number (`NUM`) | 896 | 113 |
| Location (`LOC`) | 835 | 81 |
| Person (`HUM`) | 1223 | 65 |
| Description (`DESC`) | 1162 | 138 |
| Entity (`ENTY`) | 1250 | 94 |
| Abbreviation (`ABBR`) | 86 | 9 |

The train file has 50 subclasses. The test file has 42.

## The cut

None. The notebook sets `samples = test_list`, which is all 500 test rows. The sample `id` is the 1-based line number.

## 1-shot examples

The prompt in the notebook has 6 fixed questions. All 6 are in `train.txt` and none is in `test.txt`:

| Class | Subclass in `train.txt` | Question |
|---|---|---|
| Description | `desc` | What 's the Olympic motto ? |
| Location | `country` | What country 's capital is Tirana ? |
| Number | `count` | How many people in the world speak French ? |
| Person | `ind` | Who is Snoopy 's arch-enemy ? |
| Abbreviation | `exp` | What does the abbreviation AIDS stand for ? |
| Entity | `color` | What 's the only color Johnny Cash wears on stage ? |

## Differences from the official files

- `train.txt` line 66 has the character U+FFFD. The official latin-1 file has byte `0xF0` there. The Hugging Face copy (`CogComp/trec`) reads "sister city". The notebook opens files as UTF-8, so the authors re-encoded the file. The line is not evaluated.
- Both files have no final newline. The official files have one. The notebook strips every line, so this has no effect.
- Content, order and labels of `test.txt` equal the official file and the Hugging Face copy.

## Evidence that the order equals the paper

The paper lists Entity samples 46 to 50: indigo, barometer, giraffe's tongue, twins, yak milk. In `test.txt` the Entity samples 46 to 50 are exactly these questions in this order.

## Notebook

The upstream `LLM TREC.ipynb` is cut inside an image output and is not valid JSON. `LLM TREC.ipynb` in the folder above is our repaired copy. It keeps the 7 complete cells and says so in its first cell. The plot cells after the cut are lost.

## Citation

Xin Li and Dan Roth. 2002. Learning Question Classifiers. COLING 2002.
Ellen M. Voorhees and Dawn M. Tice. 2000. Building a question answering test collection. SIGIR 2000.
