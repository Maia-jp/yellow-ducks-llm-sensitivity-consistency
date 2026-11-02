# Qwen3.5-27B jobs: one Slurm job for each dataset

These jobs run the paper's original code (`sensitivity-consistency-LLM/`) with `Qwen/Qwen3.5-27B` on ITU HPC (`hpc3`).
Each job is self-contained. It builds the Python environments, downloads the model and the dataset, starts vLLM and runs the paper code.
The login node only copies files and runs `sbatch` and `squeue`. Read [AGENTS.md](../../AGENTS.md) before you use the cluster.

## Files

| File | What it does |
|---|---|
| `trec.sbatch`, `cb.sbatch`, `rte.sbatch`, `dbpedia.sbatch`, `wos.sbatch` | The run for one dataset. 2 × A100 40 GB on `acltr` (node `cn13`). |
| `prepare.sbatch` | Optional. No GPU. Checks network and disk, builds the environments, downloads the model and the data. |
| `locks/*.txt` | Exact package versions. `vllm-0.32.0-cu130.txt` and `vllm-0.32.0-cu129.txt` are for vLLM. `eval-py310.txt` is for the paper code. |
| `../../scripts/hpc/qwen_common.sh` | The steps that all jobs share |
| `../../scripts/run_qwen.py` | Runs the notebook cells and `run_grid_search()` for one dataset |

## Settings

| Setting | Value | Why |
|---|---|---|
| Model | `Qwen/Qwen3.5-27B`, revision `fc05daec18b0a78c049392ed2e771dde82bdf654` | Fixed revision, so every run uses the same weights |
| vLLM | 0.32.0. The job reads the NVIDIA driver version and picks the CUDA 13.0 build (driver 580 or newer) or the CUDA 12.9 build (driver 525 or newer). | The default vLLM build needs CUDA 13 |
| Sampling | Temperature 0 and seed 42, from the notebooks. Thinking mode off. | The paper's settings |
| Prompts | 30 prompt rephrasings, 1 answer for each, 3 prompt types (simple, detail, 1-shot) | Unchanged notebook settings |
| CB input | `test_labeled.jsonl` (default) or `train.jsonl` with `CB_FILE=train` | See `docs/datasets/README.md` |
| Context length | 8192 tokens | The longest prompt has 2,937 tokens (WoS, 1-shot), measured with the Qwen tokenizer |

## Where the files go

Everything is in the home folder on the cluster, under `~/yellow-ducks/`:

| Folder | Content |
|---|---|
| `runs/qwen3.5-27b/<run>/results/<run>/results_<prompt type>.json` | The results. The paper code writes them after each finished sample. |
| `runs/qwen3.5-27b/<run>/data/` | The dataset, downloaded and checked against `docs/datasets/data-manifest.json` |
| `runs/qwen3.5-27b/<run>/logs/vllm-<job ID>.log` | The vLLM server log |
| `envs/`, `python/`, `tools/`, `hf/` | Environments, Python, uv and the model (about 56 GB) |

`<run>` is the dataset name, for example `rte`. For CB it is `cb-test_labeled` or `cb-train`.
The Slurm log files `qwen27b-<dataset>-<job ID>.out` and `.err` go to the folder from which you ran `sbatch`.

## How to run

1. On your own computer, copy the repository to the cluster. The jobs download the data, so the data folder is not copied:

   ```bash
   rsync -av --exclude .git --exclude 'sensitivity-consistency-LLM/data/' \
     ./ s25441@hpc3.itu.dk:~/yellow-ducks-llm-sensitivity-consistency/
   ```

2. Log in, go to the repository root, and check that you are on the login node, outside a job:

   ```bash
   cd ~/yellow-ducks-llm-sensitivity-consistency
   hostname; echo "${SLURM_JOB_ID:-none}"
   ```

3. Optional: run the preparation job, then wait until it ends and read `qwen27b-prepare-<job ID>.out`:

   ```bash
   sbatch jobs/qwen3.5-27b/prepare.sbatch
   ```

4. Recommended: run a short smoke test first (2 samples, 3 rephrasings). Its results go to a separate `-smoke` folder:

   ```bash
   sbatch --time=02:00:00 --export=ALL,SMOKE=1 jobs/qwen3.5-27b/cb.sbatch
   ```

5. Submit the five dataset jobs:

   ```bash
   for d in trec cb rte dbpedia wos; do sbatch jobs/qwen3.5-27b/$d.sbatch; done
   ```

6. Watch them with `squeue -u s25441`. When a job ends, read its `.out` and `.err` files.

`cn13` has 4 A100 40 GB GPUs, so 2 jobs can run at the same time. The other jobs wait in the queue.

### Other options

- Use 2 × L40S 48 GB on `cn19` in `scavenge` (24-hour limit, 2 jobs at a time): `sbatch --partition=scavenge --gres=gpu:l40s:2 --time=1-00:00:00 jobs/qwen3.5-27b/rte.sbatch`. If a job reaches the limit, submit it again. It continues from the results file.
- Use the A100 80 GB GPU on `cn7` (one job at a time): `sbatch --gres=gpu:a100_80gb:1 jobs/qwen3.5-27b/rte.sbatch`
- Run CB on the train file: `sbatch --export=ALL,CB_FILE=train jobs/qwen3.5-27b/cb.sbatch`

## If a job stops

Submit the same job again. The paper code skips the samples that are already in the results file.
The job stops by itself if vLLM stops, so it does not wait until the time limit.
If vLLM cannot start, the job tries again two times with smaller memory settings.

## What we checked before the first run (2026-10-30)

| Check | How | Result |
|---|---|---|
| Partitions, GPUs and nodes on `hpc3` | `sinfo` on the login node | `acltr` has `cn13` with 4 × `a100_40gb` and a 7-day limit |
| Both vLLM environments and the paper environment | `uv pip compile` for Linux x86_64 | All three resolve |
| Dataset download into an empty folder | `fetch_datasets.py --out` | All 21 files match the manifest |
| Full pipeline for TREC, CB (both files), RTE, DBPedia and WoS | `run_qwen.py --smoke` against a fake vLLM server on a Mac | All passed. Every request had temperature 0, seed 42 and thinking off. |
| Resume after a stop | Second run of the same smoke test | All finished samples were skipped |
| Locks between jobs | Shell test with a stub `squeue` | Stale locks are removed. A failed job removes its lock. |

Not checked before the first run: the NVIDIA driver version on `cn13`, network access from the compute nodes, the disk quota of the home folder, and vLLM with this model on A100 GPUs.
The preparation job and the smoke test check these.

## What happened in the runs (2026-10-30 to 2026-10-31)

All jobs ran on `cn19` with 2 × L40S 48 GB in `scavenge`, with the command from "Other options".

| Job | Slurm ID | Result |
|---|---|---|
| prepare | 171572 | Built the environments and downloaded the model and the data |
| CB smoke test | 171573 | Passed |
| TREC, CB, RTE, DBPedia, WoS | 171574, 171576, 171578, 171581, 171584 | All finished by 2026-10-31 10:44 |
| Continuation jobs | 171575, 171577, 171579 to 171580, 171582 to 171583, 171585 to 171587 | Ended with "Nothing to do" |

- The first vLLM start (max-num-seqs 64, CUDA graphs on) failed in every job.
- The second start (`--enforce-eager`, max-num-seqs 16) worked. All results come from this setting.
- RTE has about 4% "N/A" answers. The other datasets have about 0%.

## How to run another model

The model name is in 4 files. Copy the job folder and change these values:

| File | What to change |
|---|---|
| `scripts/hpc/qwen_common.sh` | `MODEL_ID`, `MODEL_REVISION` (a commit hash from Hugging Face), `MODEL_TAG`, `MODEL_MIN_FREE_GB`, `LOCK_FILES`, the `models--Qwen--Qwen3.5-27B` folder name in `ensure_model`, and the GPU memory check in `serve_model` (75000 MiB) |
| `scripts/run_qwen.py` | `LLM_NAME` (no `_` in the name) and the default of `--model` |
| `scripts/qwen_model.py` | `DEFAULT_NAME` (same value as `LLM_NAME`) and the default of `QWEN_MODEL` |
| `jobs/<new tag>/*.sbatch` | `--job-name`, `--output`, `--error`, `--gres` and `--partition` |

Then follow "How to run" with the new job folder.
If the new model needs a newer vLLM, make new lock files with `uv pip compile` on your own computer, and change `VLLM_VERSION`.
