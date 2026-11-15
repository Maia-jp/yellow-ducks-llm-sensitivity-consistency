# DeepSeek-R1 7B jobs: one Slurm job for each dataset

These jobs run the paper's original code (`sensitivity-consistency-LLM/`) with `deepseek-r1:7b` on ITU HPC (`hpc3`).
The model server is Ollama, from the ITU module `ollama/0.6.0-GCCcore-13.3.0-CUDA-12.6.0`.
The jobs are adapted from the Qwen3.5-27B jobs on the branch `qwen3.5-27b-jobs` (`jobs/qwen3.5-27b/`). The environment, lock and dataset steps are the same. vLLM is replaced by Ollama.
The login node only runs `git`, `sbatch` and `squeue`. Read [AGENTS.md](../../AGENTS.md) before you use the cluster.

**Status (2026-11-14): not yet run on the cluster.** See "What we checked" at the end. Run the smoke test first.

## Files

| File | What it does |
|---|---|
| `trec.sbatch`, `cb.sbatch`, `rte.sbatch`, `dbpedia.sbatch`, `wos.sbatch` | The run for one dataset. 1 GPU on an Ice Lake node in `scavenge` (`cn18` or `cn19`). |
| `locks/eval-py310.txt` | Exact package versions for the paper code. Same file as in `jobs/qwen3.5-27b/locks/`. |
| `slurm-logs/` | Slurm writes `deepseek7b-<dataset>-<job ID>.out` and `.err` here. Git ignores them. |
| `../../scripts/hpc/deepseek_common.sh` | The steps that all jobs share |
| `../../scripts/deepseek/run_deepseek.py` | Runs the notebook cells and `run_grid_search()` for one dataset |
| `../../scripts/deepseek/deepseek_model.py` | Adds the model to `util.llm_map` and removes the reasoning text from every answer |
| `../../scripts/deepseek/sitecustomize.py` | Makes the Ray workers know the model |

## Settings

| Setting | Value | Why |
|---|---|---|
| Model | `deepseek-r1:7b` from the Ollama library, ID `755ced02ce7b` (4.7 GB, Q4_K_M). The job stops if the tag has another ID. | Every run uses the same weights |
| Server | A private `ollama serve` for each job, on a free port (`OLLAMA_HOST`) | `cn19` runs an Ollama service of the node on the default port 11434. Without `OLLAMA_HOST` the commands talk to that service: another version, other settings, another model store. |
| Node | `--constraint=arch_icelake` | The Ollama module needs AVX-512. On `cn9` (AMD EPYC 7643) the model load stops with `SIGILL: illegal instruction`. |
| Sampling | Temperature 0 and seed 42, from the notebooks | The paper's settings |
| Prompts | 30 prompt rephrasings, 1 answer for each, 3 prompt types (simple, detail, 1-shot) | Unchanged notebook settings |
| Reasoning text | Removed. Only the text after `</think>` goes to the paper code. | See "Reasoning text" |
| Context length | 8192 tokens for one request (`CONTEXT_LENGTH`). The job makes the Ollama model `deepseek-r1-7b-ctx8192` with `num_ctx 8192`. | Ollama's default is 2048 tokens and it cuts longer prompts without an error. The longest prompt has about 2,940 tokens (WoS, 1-shot, Qwen tokenizer). |
| Answer limit | 2048 tokens for reasoning plus answer (`MAX_TOKENS`) | A request cannot run on without end |
| Parallel requests | 8 in Ollama (`NUM_PARALLEL`), 2 samples at a time in the paper code (`RAY_CPUS`) | One sample sends 30 requests at once |
| CB input | `test_labeled.jsonl` (default) or `train.jsonl` with `CB_FILE=train` | See `docs/datasets/README.md` |

Change a setting with `sbatch --export=ALL,NAME=value`.

## Reasoning text

DeepSeek-R1 is a reasoning model. With Ollama 0.6.0 every answer starts with reasoning text between `<think>` and `</think>`, and this cannot be turned off.
The paper code finds the class label with a substring search over the whole answer, and it takes the first label of a fixed list. The reasoning text names several labels, so it would decide the result.

`deepseek_model.py` removes the block:

- `<think>...</think>answer` gives `answer`.
- `<think>...` without the closing tag gives an empty answer. The model reached `MAX_TOKENS` inside the reasoning. The paper code counts it as `N/A`.
- The rephrased questions are cleaned in the same way.

This differs from the other models in the project, and the report must say so:

- Mixtral (paper) and Qwen3.5 (our runs, thinking mode off) answer directly. DeepSeek-R1 reasons before each answer.
- The run is slower, because each request writes some hundred tokens of reasoning.

## Where the files go

Everything is in the home folder on the cluster, under `~/yellow-ducks/`:

| Folder | Content |
|---|---|
| `runs/deepseek-r1-7b/<run>/results/<run>/results_<prompt type>.json` | The results. The paper code writes them after each finished sample. |
| `runs/deepseek-r1-7b/<run>/data/` | The dataset, downloaded and checked against `docs/datasets/data-manifest.json` |
| `runs/deepseek-r1-7b/<run>/logs/ollama-<job ID>.log` | The Ollama server log |
| `envs/`, `python/`, `tools/`, `cache/` | The Python environment, Python and uv |
| `~/.ollama/models/` | The model (4.7 GB). Not under `~/yellow-ducks/`. |

`<run>` is the dataset name, for example `rte`. For CB it is `cb-test_labeled` or `cb-train`. A smoke run adds `-smoke`.

## How to run

1. On your own computer, commit the branch and push it. On the login node, go to the repository root and get the branch:

   ```bash
   cd ~/yellow-ducks-llm-sensitivity-consistency
   hostname; echo "${SLURM_JOB_ID:-none}"
   git fetch && git checkout deepseek-r1-7b
   ```

2. Run the smoke test (2 samples, 3 rephrasings). The first job also builds the Python environment, so it takes longer:

   ```bash
   sbatch --time=02:00:00 --export=ALL,SMOKE=1 jobs/deepseek-r1-7b/trec.sbatch
   squeue -u "$USER"
   ```

3. When the job ends, read both log files:

   ```bash
   cat jobs/deepseek-r1-7b/slurm-logs/deepseek7b-trec-<job ID>.out
   cat jobs/deepseek-r1-7b/slurm-logs/deepseek7b-trec-<job ID>.err
   ```

   The `.out` file must end with `Job finished with exit code 0`. Read these lines:
   - `Server check`: the answer without reasoning must be one class name.
   - `Answer labels`: a high `N/A` share means that the answers do not fit the label search.
   - `Smoke run time`: a rough time for the full run of this dataset.

4. Decide which datasets to run (see "Size of the runs"), then submit them:

   ```bash
   for d in trec cb; do sbatch jobs/deepseek-r1-7b/$d.sbatch; done
   ```

5. `scavenge` has a 24-hour limit. If a job reaches it, submit the same job again. It continues from the results file.
   `cn18` is also in `acltr` (7-day limit): `sbatch --partition=acltr --time=3-00:00:00 jobs/deepseek-r1-7b/rte.sbatch`.

## Size of the runs

One sample needs 90 classification requests (30 rephrasings × 3 prompt types).

| Dataset | Samples | Requests |
|---|---|---|
| CB (`test_labeled`) | 250 | 22,500 |
| TREC | 500 | 45,000 |
| DBPedia | 2,000 | 180,000 |
| WoS | 2,000 | 180,000 |
| RTE | 2,490 | 224,100 |

We do not know the speed yet. The smoke test prints a rough time. Each job uses one GPU, so several datasets can run at the same time on different GPUs.

## If a job stops

Submit the same job again. The paper code skips the samples that are already in the results file.
The job stops by itself if Ollama stops, so it does not wait until the time limit.

| Text in the log | Cause |
|---|---|
| `The CPU of ... has no AVX-512` | The job is on a node that the Ollama module does not support. Keep `--constraint=arch_icelake`. |
| `Cannot load the module` | The module name changed. Check it with `module avail ollama` inside a job. |
| `has ID ..., expected 755ced02ce7b` | The tag `deepseek-r1:7b` points to other weights. Decide with the team, then use `ALLOW_OTHER_MODEL_ID=1` and record the ID. |
| `is not fully on the GPU` | Another process uses the GPU memory, or Ollama did not find the GPU. Read the Ollama log. |
| `Raise CONTEXT_LENGTH` | A prompt plus its answer does not fit into the context |

## What we checked

On 2026-11-14, by hand in interactive jobs:

| Check | Result |
|---|---|
| Ollama module on `cn9` (`arch_zen`, A30) | Model load fails with `SIGILL: illegal instruction`. The CPU has no AVX-512. |
| Ollama module on `cn19` (`arch_icelake`, L40S), private server with `OLLAMA_HOST` | `deepseek-r1:7b` answers. `ollama ps` shows `100% GPU`, 6.0 GB. The answer starts with a `<think>` block. |
| Default port on `cn19` | Taken by an Ollama service of the node (`/usr/local/bin/ollama serve`). |

On 2026-11-14, on a laptop, without the real model and without the real packages:

| Check | How | Result |
|---|---|---|
| Notebook cells of all five datasets | `run_deepseek.py --status-only` | All load: 500, 250, 2,490, 2,000 and 2,000 samples |
| Full job for TREC, smoke mode | `trec.sbatch` with stub commands (`ollama`, `module`, `squeue`, `nvidia-smi`) and a fake server that writes `<think>` text which names other labels | Passed. All 18 answers got the label after `</think>`. The stored rephrasings have no reasoning text. |
| Continuation | The same job again | "Nothing to do" |
| Node without AVX-512 | The same job on the laptop CPU | Stops with the AVX-512 text |
| Answer cut inside the reasoning | Fake server without `</think>` | The server check stops the run |
| Prompt longer than the context | Fake server, 6,000-token prompt | The prompt length check stops the run |

Not checked. The smoke test on the cluster checks these:

- `ThinkStrippingChatOpenAI` with the real `langchain-openai` 0.1.8. The laptop test used a stub with the same method names.
- Ray with real worker processes.
- `module load` inside a batch job.
- `ollama create` with `num_ctx`, and the `usage.prompt_tokens` field of the real Ollama API.
- The download of the datasets from a compute node, and the build of the Python environment in this home folder.
- The Ollama module on `arch_skylake` nodes. They have AVX-512, but we did not try them.
