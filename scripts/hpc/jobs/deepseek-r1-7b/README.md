# DeepSeek-R1 7B jobs: one Slurm job for each dataset

These jobs run the paper's original code (`original-paper/`) with `deepseek-r1:7b` on ITU HPC (`hpc3`).
The model server is Ollama. The jobs are adapted from the Qwen3.5-27B jobs on the branch `qwen3.5-27b-jobs` (`jobs/qwen3.5-27b/`). The environment, lock and dataset steps are the same. vLLM is replaced by Ollama.
The login node only runs `git`, `sbatch` and `squeue`. Read [the HPC guides](../../../../docs/hpc/) before you use the cluster.

**Status (2026-11-15): the full job has not yet finished on the cluster.** See "What we checked" at the end. Run the smoke test first.

## Two kinds of nodes

| | Default jobs (`*.sbatch`) | x86 jobs (`x86/*.sbatch`) |
|---|---|---|
| Node | `asus1`: ARM, NVIDIA GB10, 128 GB of memory shared by CPU and GPU | 1 GPU on a node with AVX-512 (`arch_icelake`: `cn18`, `cn19`) |
| Partition | `asus1` (3-day limit) | `scavenge` (1-day limit) or `acltr` |
| Ollama | The program of the node, `/usr/local/bin/ollama`, version 0.40.4 | ITU module `ollama/0.6.0-GCCcore-13.3.0-CUDA-12.6.0` |
| Reasoning text | In a separate field of the reply | Inside the answer text, between `<think>` and `</think>` |

**Use one kind of node for all results of the project.** The two Ollama versions are different programs, so their answers can differ.
A run folder remembers its Ollama version. A job with another version stops with an error and does not add to it.
We use `asus1`: on 2026-11-15 every GPU of the x86 nodes was in use and about 40 GPU jobs waited in the queue.

## Files

| File | What it does |
|---|---|
| `trec.sbatch`, `cb.sbatch`, `rte.sbatch`, `dbpedia.sbatch`, `wos.sbatch` | The run for one dataset on `asus1` |
| `x86/*.sbatch` | The same runs on an x86 node |
| `locks/eval-py310.txt` | Exact package versions for the paper code. Same file as in `jobs/qwen3.5-27b/locks/`. |
| `slurm-logs/` | Slurm writes `deepseek7b-<dataset>-<job ID>.out` and `.err` here. Git ignores them. |
| `../../../../scripts/hpc/deepseek_common.sh` | The steps that all jobs share |
| `../../../../scripts/deepseek/run_deepseek.py` | Runs the notebook cells and `run_grid_search()` for one dataset |
| `../../../../scripts/deepseek/deepseek_model.py` | Adds the model to `util.llm_map` and keeps the reasoning text away from the paper code |
| `../../../../scripts/deepseek/sitecustomize.py` | Makes the Ray workers know the model |

## Settings

| Setting | Value | Why |
|---|---|---|
| Model | `deepseek-r1:7b` from the Ollama library, ID `755ced02ce7b` (4.7 GB, Q4_K_M). The job stops if the tag has another ID. | Every run uses the same weights |
| Server | A private `ollama serve` for each job, on a free port (`OLLAMA_HOST`) | `asus1` and `cn19` run an Ollama service of the node on the default port 11434. A private server has our settings and our model store. |
| Memory | `--mem=32G` | Without `--mem` a job on `asus1` takes all memory of the node, and no other job can start there |
| Sampling | Temperature 0 and seed 42, from the notebooks | The paper's settings |
| Prompts | 30 prompt rephrasings, 1 answer for each, 3 prompt types (simple, detail, 1-shot) | Unchanged notebook settings |
| Reasoning | On (the model's default). The reasoning text does not go to the paper code. | See "Reasoning text" |
| Context length | 8192 tokens for one request (`CONTEXT_LENGTH`). The job makes the Ollama model `deepseek-r1-7b-ctx8192` with `num_ctx 8192`. | A fixed value on all nodes. Ollama cuts a prompt that is too long without an error. The longest prompt has about 2,940 tokens (WoS, 1-shot, Qwen tokenizer). |
| Answer limit | 2048 tokens for reasoning plus answer (`MAX_TOKENS`) | A request cannot run on without end |
| Parallel requests | 8 in Ollama (`NUM_PARALLEL`), 2 samples at a time in the paper code (`RAY_CPUS`) | One sample sends 30 requests at once |
| CB input | `test_labeled.jsonl` (default) or `train.jsonl` with `CB_FILE=train` | See `docs/datasets/README.md` |

Change a setting with `sbatch --export=ALL,NAME=value`.

## Reasoning text

DeepSeek-R1 is a reasoning model. It writes some hundred tokens of reasoning before each answer.
The paper code finds the class label with a substring search over the whole answer, and it takes the first label of a fixed list. The reasoning names several labels, so it must not reach that search.

- Ollama 0.40.4 (`asus1`) returns the reasoning in a separate field (`reasoning`). The answer text has only the answer.
- Ollama 0.6.0 (x86 module) puts the reasoning into the answer text. `deepseek_model.py` removes the `<think>...</think>` block.
- If the model reaches `MAX_TOKENS` inside the reasoning, the answer is empty. The paper code counts it as `N/A`.
- The rephrased questions are handled in the same way. White space around every answer is removed.

This differs from the other models in the project, and the report must say so:

- Mixtral (paper) and Qwen3.5 (our runs, thinking mode off) answer directly. DeepSeek-R1 reasons before each answer.
- The run is slower, because each request writes the reasoning too.

Open question for the team: newer Ollama versions may be able to turn the reasoning off for a request. We did not try it. It would make the run faster and closer to the Qwen runs, but it is not the model's default behaviour.

## Where the files go

Everything is in the home folder on the cluster, under `~/yellow-ducks/`:

| Folder | Content |
|---|---|
| `runs/deepseek-r1-7b/<run>/results/<run>/results_<prompt type>.json` | The results. The paper code writes them after each finished sample. |
| `runs/deepseek-r1-7b/<run>/ollama-version.txt` | The Ollama version of this run folder |
| `runs/deepseek-r1-7b/<run>/data/` | The dataset, downloaded and checked against `docs/datasets/data-manifest.json` |
| `runs/deepseek-r1-7b/<run>/logs/ollama-<job ID>.log` | The Ollama server log |
| `envs/eval-py310-<CPU type>/`, `python/`, `tools/`, `cache/` | The Python environment, Python and uv. ARM and x86 have separate folders. |
| `~/.ollama/models/` | The model (4.7 GB). Not under `~/yellow-ducks/`. |

`<run>` is the dataset name, for example `rte`. For CB it is `cb-test_labeled` or `cb-train`. A smoke run adds `-smoke`.

## How to run

1. On the login node, go to the repository root and get the branch:

   ```bash
   cd ~/yellow-ducks-llm-sensitivity-consistency
   hostname; echo "${SLURM_JOB_ID:-none}"
   git fetch && git checkout deepseek-r1-7b && git pull
   ```

2. Run the smoke test (2 samples, 3 rephrasings). The first job also builds the Python environment, so it takes longer:

   ```bash
   sbatch --time=02:00:00 --export=ALL,SMOKE=1 scripts/hpc/jobs/deepseek-r1-7b/trec.sbatch
   squeue -u "$USER"
   ```

3. When the job ends, read both log files:

   ```bash
   cat scripts/hpc/jobs/deepseek-r1-7b/slurm-logs/deepseek7b-trec-<job ID>.out
   cat scripts/hpc/jobs/deepseek-r1-7b/slurm-logs/deepseek7b-trec-<job ID>.err
   ```

   The `.out` file must end with `Job finished with exit code 0`. Read these lines:
   - `Server check`: the answer without reasoning must be one class name.
   - `Answer labels`: a high `N/A` share means that the answers do not fit the label search.
   - `Smoke run time`: a rough time for the full run of this dataset.

4. Decide which datasets to run (see "Size of the runs"), then submit them:

   ```bash
   for d in cb trec; do sbatch scripts/hpc/jobs/deepseek-r1-7b/$d.sbatch; done
   ```

5. The job files ask for 24 hours. If a job reaches the limit, submit the same job again. It continues from the results file.
   `asus1` allows 3 days: `sbatch --time=3-00:00:00 scripts/hpc/jobs/deepseek-r1-7b/rte.sbatch`. Other groups use the node too, so do not hold it longer than needed.

`asus1` has one GPU. Two jobs at the same time share it, so each one is slower.

## Size of the runs

One sample needs 90 classification requests (30 rephrasings × 3 prompt types).

| Dataset | Samples | Requests |
|---|---|---|
| CB (`test_labeled`) | 250 | 22,500 |
| TREC | 500 | 45,000 |
| DBPedia | 2,000 | 180,000 |
| WoS | 2,000 | 180,000 |
| RTE | 2,490 | 224,100 |

We do not know the speed yet. The smoke test prints a rough time.

## If a job stops

Submit the same job again. The paper code skips the samples that are already in the results file.
The job stops by itself if Ollama stops, so it does not wait until the time limit.

| Text in the log | Cause |
|---|---|
| `This run folder has results from Ollama ...` | The job runs on another kind of node than the earlier jobs of this run. Use the same kind. |
| `This node has no ollama program` | The default jobs need the Ollama program of the node. Run them on `asus1`. |
| `The CPU of ... has no AVX-512` | An x86 job is on a node that the Ollama module does not support. Keep `--constraint=arch_icelake`. |
| `Cannot load the module` | The module name changed. Check it with `module avail ollama` inside a job. |
| `has ID ..., expected 755ced02ce7b` | The tag `deepseek-r1:7b` points to other weights. Decide with the team, then use `ALLOW_OTHER_MODEL_ID=1` and record the ID. |
| `is not fully on the GPU` | Another process uses the GPU memory, or Ollama did not find the GPU. Read the Ollama log. |
| `shows a context of ... tokens, expected 8192` | Ollama did not take the context length from the served model |
| `Raise CONTEXT_LENGTH` | A prompt plus its answer does not fit into the context |

## What we checked

By hand in interactive jobs:

| Date | Check | Result |
|---|---|---|
| 2026-11-14 | Ollama module on `cn9` (`arch_zen`, A30) | Model load fails with `SIGILL: illegal instruction`. The CPU has no AVX-512. |
| 2026-11-14 | Ollama module on `cn19` (`arch_icelake`, L40S), private server | `deepseek-r1:7b` answers. `100% GPU`, 6.0 GB. The answer text starts with a `<think>` block. |
| 2026-11-15 | Ollama 0.40.4 of `asus1`, private server, the model store from the day before | `deepseek-r1:7b` (ID `755ced02ce7b`) answers `Location` to the test question. `100% GPU`. The reasoning is in the field `reasoning`, 258 completion tokens. The reply has `usage.prompt_tokens`. Without a context setting the context is 131,072 tokens (13 GB). |
| 2026-11-15 | Queue of the x86 GPU nodes | Every GPU in use. Two smoke jobs (`acltr`, `scavenge`) did not start. |

On a laptop, without the real model and without the real packages (stub commands, a fake server, and a stub of LangChain and Ray):

| Check | Result |
|---|---|
| Notebook cells of all five datasets (`run_deepseek.py --status-only`) | All load: 500, 250, 2,490, 2,000 and 2,000 samples |
| Full `trec.sbatch`, smoke mode, as on `asus1`: node program 0.40.4, reasoning in a separate field, no `squeue`, no Slurm GPU variable | Passed. All 18 answers got the right label. |
| Full `x86/trec.sbatch`, smoke mode: module, `<think>` text that names other labels | Passed. All 18 answers got the label after `</think>`. |
| The same job again | "Nothing to do" |
| An x86 job on a run folder with `asus1` results | Stops with the Ollama version error |
| Node without AVX-512, answer cut inside the reasoning, prompt longer than the context | Each one stops the run with a clear text |

Not checked. The smoke test on the cluster checks these:

- `ThinkStrippingChatOpenAI` with the real `langchain-openai` 0.1.8, and Ray with real worker processes.
- The build of the Python environment on ARM (`uv` and the packages of `locks/eval-py310.txt` for aarch64).
- `ollama create` with `num_ctx` on Ollama 0.40.4.
- The download of the datasets from `asus1`.
- The x86 jobs on the cluster: `module load` inside a batch job, and the module on `arch_skylake` nodes.
