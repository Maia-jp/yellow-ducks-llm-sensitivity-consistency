ITU HPC — Agent Setup and First Connection

## Goal
Connect from a local development machine to ITU's HPC cluster, verify SSH access, and submit a minimal SLURM job safely.

**Source:** *High Performance Computing at ITU* (Rob van der Goot, 2026), supplied by the user. See also https://hpc.itu.dk and https://slurm.schedmd.com/documentation.html.

## Rules for the coding agent
- Ask the user for their **ITU username**; do not guess it.
- Never request, store, print, or commit passwords, SSH private keys, or one-time codes. Authentication must occur interactively via the user's terminal or approved SSH agent.
- Do not run CPU/GPU-intensive workloads on the login node (`front`). Submit them through SLURM.
- **Hard rule:** Never run `pip`, `uv`, `conda`, `mamba` or any package manager on the login node. Never run Jupyter, VS Code Remote or other interactive tools there. ITU restricted the user's account on 2026-10-26 for this. Run installs only inside a SLURM job (`sbatch` or `srun`). See `AGENTS.md` in the project root.
- Before any remote command, run `echo "${SLURM_JOB_ID:-none}"`. If it prints `none`, you are on the login node.
- If a command fails with `fork: retry: Resource temporarily unavailable`, stop and tell the user. Do not retry.
- Do not cancel existing jobs, overwrite remote files, change SSH configuration, or install software without user approval.
- Run commands incrementally and show the user what will be executed before any remote modifications.
- Stop and ask for help if SSH authentication fails, if the cluster requires VPN/MFA, or if permissions are denied. The slides do not specify VPN/MFA prerequisites.

## Step 1 — Check local prerequisites
On macOS/Linux, check SSH and SCP:

```bash
ssh -V
command -v scp
```

If either is unavailable, stop and explain how to install an SSH client on the user's operating system.

## Step 2 — Obtain the ITU username
Ask the user to provide their ITU login username, without a domain suffix unless ITU explicitly instructs otherwise. Use it in place of `<ITU_USERNAME>` throughout this guide.

## Step 3 — Connect to the login node
From the local terminal:

```bash
ssh <ITU_USERNAME>@hpc.itu.dk
```

The first connection may prompt for SSH host-key verification. The agent should instruct the user to verify the fingerprint through an official ITU channel before accepting it. Enter any password/MFA response directly into the terminal, not into the chat.

If successful, run lightweight checks on the remote host:

```bash
whoami
hostname
pwd
```

**Expected:** `whoami` matches the ITU username. The login host may be named `front` (as shown in the slides); do not require an exact hostname.

## Step 4 — Inspect SLURM (on HPC)

```bash
command -v sbatch
command -v squeue
sinfo -o '%P %a %l %D'
squeue -u "$USER"
```

These are inspection commands only. The slides list the partitions `acltr` (GPU), `cores` (CPU), `cores_any` (additional CPU), and `scavenge` (all nodes). Partition configuration may change, so inspect the live cluster before submitting.

## Step 5 — Create a minimal job script (on HPC)
Create a dedicated directory and a small script:

```bash
mkdir -p ~/hpc-agent-test
cd ~/hpc-agent-test
cat > test.sh <<'SCRIPT'
#!/bin/bash
#SBATCH --job-name=agent-hello
#SBATCH --output=hello.%j.out
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --partition=scavenge

hostname
date
SCRIPT
```

The script uses the slide deck's example of a one-minute `scavenge` job. `%j` inserts the SLURM job ID into the output filename.

## Step 6 — Submit the job (on HPC)
Ask for approval before submitting, then run:

```bash
sbatch test.sh
```

Record the job ID printed in `Submitted batch job <JOB_ID>`.

## Step 7 — Monitor and read the output (on HPC)

```bash
squeue -u "$USER"
ls -l hello.*.out
cat hello.<JOB_ID>.out
```

Replace `<JOB_ID>` with the actual numeric ID. If the output file is not present yet, wait until the job finishes and check again. A successful run prints the compute-node hostname and date. Do not assume a missing `squeue` entry means success: inspect the output and, if available, `sacct -j <JOB_ID> --format=JobID,State,ExitCode`.

If the user explicitly requests cancellation:

```bash
scancel <JOB_ID>
```

## Step 8 — Transfer project files (from the local terminal)
Exit SSH or open a separate local terminal. For example:

```bash
scp ./train.py <ITU_USERNAME>@hpc.itu.dk:~/hpc-agent-test/
```

The supplied slides use `scp` to transfer a local file into HPC storage. Confirm the destination before uploading any sensitive data.

## Step 9 — Python job example (optional)
The slide deck gives this Python module and job configuration:

```bash
#!/bin/bash
#SBATCH --job-name=train.0
#SBATCH --output=train.%j.out
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --gres=gpu
#SBATCH --mem=30G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=scavenge

module load Python/3.12.3-GCCcore-13.3.0
python3 train.py --dataset_config configs/ewt.json
```

**Important:** The slides explicitly say `--gres` should appear before `--partition`. The Python module name is taken from the slides and must be checked against the modules actually available on the cluster. Only submit a GPU job if the user has a real workload and approves the requested resources.

For PyTorch, the slides suggest:

```bash
module spider torch
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
```

Modules must be loaded again in every new session and job script. Verify availability before loading.

## Step 10 — Report back to the user
Provide:
1. Whether SSH authentication succeeded.
2. The observed login hostname and username.
3. The available partitions.
4. The submitted test job ID, its final state, and output.
5. Any unresolved prerequisites or access errors.

Do not claim success unless the commands actually ran and the output was checked.

## Additional references
- ITU HPC documentation: https://hpc.itu.dk
- GPU job templates: https://hpc.itu.dk/scheduling/templates/gpu/
- Examples on the HPC filesystem: `/opt/itu/templates/`
- SLURM documentation: https://slurm.schedmd.com/documentation.html
