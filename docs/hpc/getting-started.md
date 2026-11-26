# ITU HPC: getting started and first connection

## Goal
Connect to ITU's HPC cluster, check access, and submit a small Slurm job safely.

## Access
```bash
ssh s25441@hpc3.itu.dk
```

This account needs no password. Do not guess other hosts or usernames.
The ITU username is `s25441`. Use it in place of `<ITU_USERNAME>` below.

Nothing in this guide was tested on `hpc3`. Check partition names, modules and paths on the cluster before you use them.

## Rules
- Never share, store, print, or commit passwords, SSH private keys, or one-time codes.
- Do not run CPU/GPU-intensive workloads on the login node. Submit them through Slurm.
- **Hard rule:** Never run `pip`, `uv`, `conda`, `mamba` or any package manager on the login node. Never run Jupyter, VS Code Remote or other interactive tools there. ITU forbids this. Run installs only inside a Slurm job (`sbatch` or `srun`).
- To check where you are, run `echo "${SLURM_JOB_ID:-none}"`. If it prints `none`, you are on the login node.
- If a command fails with `fork: retry: Resource temporarily unavailable`, stop. Do not retry.
- Do not cancel other people's jobs or overwrite shared files. Run few, light commands on the login node.

## Step 1: Check the local SSH client
```bash
ssh -V
command -v scp
```

## Step 2: Connect to the login node
```bash
ssh s25441@hpc3.itu.dk
```

If the first connection asks you to verify a host key, check it before you accept. Then run light checks:

```bash
whoami
hostname
echo "${SLURM_JOB_ID:-none}"
```

**Expected:** `whoami` prints `s25441`. `SLURM_JOB_ID` prints `none` on the login node.

## Step 3: Inspect Slurm (inspection only)
```bash
command -v sbatch
command -v squeue
sinfo -o '%P %a %l %D'
squeue -u "$USER"
```

Use a partition name from the `sinfo` output. The partitions can change, so read them from the live cluster.

## Step 4: Create a minimal job script
```bash
mkdir -p ~/hpc-test
cd ~/hpc-test
cat > test.sh <<'SCRIPT'
#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --output=hello.%j.out
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --partition=<PARTITION>

hostname
date
SCRIPT
```

Replace `<PARTITION>` with a partition from Step 3. `%j` inserts the job ID into the output file name.

## Step 5: Submit the job

```bash
sbatch test.sh
```

Record the job ID from `Submitted batch job <JOB_ID>`.

## Step 6: Monitor and read the output
```bash
squeue -u "$USER"
ls -l hello.*.out
cat hello.<JOB_ID>.out
```

Wait until the output file exists. A missing `squeue` entry does not prove success. Read the output. If `sacct` is available, also run `sacct -j <JOB_ID> --format=JobID,State,ExitCode`.

To cancel a job:

```bash
scancel <JOB_ID>
```

## Step 7: Transfer project files (from the local terminal)
```bash
scp ./file s25441@hpc3.itu.dk:~/hpc-test/
```

Confirm the destination before you upload anything sensitive.

## Step 8: Python and GPU jobs
Check which software the cluster offers before you write a job script. If the cluster uses environment modules, `module avail` lists them. Request a GPU only for a real workload. Put `--gres` before `--partition` if the cluster requires that order.
Install Python packages only inside a Slurm job. See [jobs-and-environments.md](jobs-and-environments.md) for the job file pattern and for how to build environments.

## References
- Slurm documentation: https://slurm.schedmd.com/documentation.html
