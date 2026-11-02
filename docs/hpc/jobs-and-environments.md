# ITU HPC: how jobs and environments work

Read this guide before you write, change or submit any Slurm job, and before you create a Python environment on the cluster.
The rules in [AGENTS.md](../../AGENTS.md) come first. If this guide and `AGENTS.md` differ, follow `AGENTS.md` and ask the user.

**Source:** the slides *Exercises for making environments* (`ENVS_HPC_SEP2026.pdf`) by Dr Inna Ermilova, Special Adviser and HPC administrator at ITU, September 2026. The user supplied them.
The slide examples use the instructor's account (`/home/iner`) and her node, GPU and module names. This guide replaces them with placeholders.
**Check every placeholder on the cluster before you use it.** Nothing here was tested on `hpc3`.

## The short version

1. The login node is only for editing files, copying files and running `sbatch`, `squeue` and `sinfo`.
2. Every install, download, script run and model job goes into a job file. Submit it with `sbatch`.
3. Build Python environments inside a job, never on the login node.
4. Put `export TMPDIR=$HOME/tmp` in every job file.
5. Read the output and error files of a job before you say that it worked.

## Why the login node is off limits

- The login node has no GPUs and no strong CPUs. A heavy process can overload it and send it to a draining state.
- ITU restricts accounts that run heavy processes on the login node. This already happened to this group on 2026-10-26 after a `pip` run.
- The slides name `pip install` as the example. `AGENTS.md` also forbids `conda`, `uv`, Jupyter, VS Code Remote and model servers on the login node.

## One-time set-up of your account

### 1. Check `.bashrc` for an active conda

After you log in, run:

```bash
cat ~/.bashrc
```

If the file has a block between `# >>> conda initialize >>>` and `# <<< conda initialize <<<`, comment out every line of it with `#`.
A conda that starts at every login conflicts with the environments below. Ask the user before you edit `.bashrc`.

### 2. Create a temporary folder

```bash
mkdir -p $HOME/tmp
export TMPDIR=$HOME/tmp
source ~/.bashrc
```

The slides say: add the line `export TMPDIR=$HOME/tmp` to **every** job script you run.

## The Slurm workflow

| Step | What to do |
|---|---|
| 1 | Create the `tmp` folder in your home folder (see above). |
| 2 | Create a folder for the project: `mkdir myproject`. |
| 3 | Put the files in your home folder: copy them with `scp`, or download them with `wget` or `curl`. Use a job for large downloads. |
| 4 | Write a job file in the same folder as the script that you want to run. |
| 5 | Submit it from that folder: `sbatch yourscript.job`. |
| 6 | Watch it: `squeue -u <ITU_USERNAME>`. |

States in `squeue`:

| State | Meaning |
|---|---|
| `PD` | Pending (waiting) |
| `CF` | Starting |
| `R` | Running |
| `CG` | Cancelling, for various reasons |
| not in the list | Done or failed. Read the `.out` and `.err` files. |

To see which nodes are idle, run `squeue` or `sinfo -N`. Another user can submit jobs while you look, so the picture can change.
Nodes in the `scavenge` partition usually run 24-hour jobs. Other nodes run 24 to 168 hours, depending on the account.

## A job file

This is the pattern from the slides, with placeholders:

```bash
#!/bin/bash
#SBATCH --job-name=<NAME>
#SBATCH --output=<NAME>.out
#SBATCH --error=<NAME>.err
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --nodelist=<NODE>          # optional. Pin a node only when you need it.
#SBATCH --gres=gpu:<GPU_TYPE>:<N>  # only for a GPU job. Read the names from the cluster.
#SBATCH --time=<HH:MM:SS>
#SBATCH --partition=<PARTITION>    # the slide examples use scavenge

set -e
set -x

module load GCCcore/13.3.0         # version from the slides. Check it with `module avail`.
export TMPDIR=$HOME/tmp

# your commands here
```

| Placeholder | Slide example | How to get the real value |
|---|---|---|
| `<PARTITION>` | `scavenge` | `sinfo -o '%P %a %l %D'` |
| `<NODE>` | `cn[3]`, `cn[7]`, `cn[19]` | `sinfo -N` |
| `<GPU_TYPE>:<N>` | `a100_80gb:1`, `l40s:4` | Ask the user or read the node description. |
| `<HH:MM:SS>` | `07:30:00` for environment jobs | Choose for the work. Do not request more than you need. |

`set -e` stops the job at the first error. `set -x` prints each command in the log.
Each slide example asks for a GPU node, also when it builds a CPU environment. You do not need a GPU to build an environment. Ask the user before you request GPU resources.

## Where to build an environment

The slides name four ways:

| Option | What it is | Notes from the slides |
|---|---|---|
| A. Conda module | Load the conda module and create the environment in your home folder | Shown in "Option A" below |
| B. Own Miniconda | Download Miniconda in a job and install it in your home folder | Shown in "Option B" below. On `dgx1` and `dgx2`, use the `aarch64` installer. |
| C. Pre-made environment in `/opt/itu` | Use an environment that ITU provides | Not shown in the slides. Look in `/opt/itu` and ask the user. |
| D. `/scratch` of a node | Build the environment on the node's local disk, and delete it later | Check the free space first. Other users may leave data there. Only works if the job runs on the same node. |

"Some of these solutions are very slow but might be the only working ones." Pick the simplest option that works.

### Machine types

All machines except `dgx1` and `dgx2` use the `x86_64` architecture. `dgx1` and `dgx2` use `aarch64`.
A conda install for one architecture does not work on the other. For a `dgx` machine, install conda and the environment there, separately.

### Check `/scratch` first (option D)

Submit a small job to the node that you plan to use (change the node name):

```bash
#!/bin/bash
#SBATCH --job-name=size_scr
#SBATCH --output=size_scr.out
#SBATCH --error=size_scr.err
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --nodelist=<NODE>
#SBATCH --time=00:10:00
#SBATCH --partition=<PARTITION>

set -e
set -x
hostname
df -h
```

Read `size_scr.out`. If a lot of space is used, ask the user before you continue.
For a fast start, one job can create the environment, install the packages and run the task on that node.

### Option A: conda module

Two jobs. The paths below are from the slides. Check the module version and the path with `module avail`.

**Job 1: create the environment.**

```bash
module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0

export CONDA_PKGS_DIRS=$HOME/miniconda3-25-5-1-1/cache
export TMPDIR=$HOME/tmp

# Accept the channel terms once, then create the environment
/opt/itu/easybuild/software/Miniconda3/25.5.1-1/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/opt/itu/easybuild/software/Miniconda3/25.5.1-1/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
/opt/itu/easybuild/software/Miniconda3/25.5.1-1/bin/conda create -y -p $HOME/miniconda3-25-5-1-1/envs/my_cpu_env python=3.12
```

**Job 2: install packages.**

```bash
module load Miniconda3/25.5.1-1
module load GCCcore/13.3.0

export CONDA_PKGS_DIRS=$HOME/miniconda3-25-5-1-1/cache
export TMPDIR=$HOME/tmp

# Load the conda shell function (required in batch jobs)
source /opt/itu/easybuild/software/Miniconda3/25.5.1-1/etc/profile.d/conda.sh
conda activate $HOME/miniconda3-25-5-1-1/envs/my_cpu_env

pip install numpy scipy torch      # replace with your packages
```

Put both blocks under the `#SBATCH` header of a job file (see "A job file").
The slide installs `torch` without GPU support in this example.

### Option B: your own Miniconda

Three jobs. Get the installer URL from https://repo.anaconda.com/miniconda/ (Miniconda3) or https://repo.anaconda.com/archive/ (Anaconda3). The URL must end in `.sh` and match the machine architecture, for example `...-Linux-x86_64.sh`.

**Job 1: download and install Miniconda.**

```bash
module load GCCcore/13.3.0

cd $HOME
mkdir MY_OWN_MINICONDA
mkdir CACHE_MY_OWN_MINICONDA
export CONDA_PKGS_DIRS=$HOME/CACHE_MY_OWN_MINICONDA
export TMPDIR=$HOME/tmp

cd MY_OWN_MINICONDA
wget https://repo.anaconda.com/miniconda/Miniconda3-py312_26.1.1-1-Linux-x86_64.sh     # version from the slides. Pick a current one.
bash Miniconda3-py312_26.1.1-1-Linux-x86_64.sh -b -p $HOME/MY_OWN_MINICONDA/Miniconda3-py312_26.1.1-1-Linux-x86_64
```

**Job 2: create the environment.**

```bash
module load GCCcore/13.3.0
cd $HOME
export CONDA_PKGS_DIRS=$HOME/CACHE_MY_OWN_MINICONDA
export TMPDIR=$HOME/tmp

CONDA=$HOME/MY_OWN_MINICONDA/Miniconda3-py312_26.1.1-1-Linux-x86_64/bin/conda
$CONDA tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$CONDA tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
$CONDA create -y -p $HOME/MY_OWN_MINICONDA/Miniconda3-py312_26.1.1-1-Linux-x86_64/envs/newpy_pip python=3.12
```

**Job 3: install packages.**

```bash
module load GCCcore/13.3.0
cd $HOME
export CONDA_PKGS_DIRS=$HOME/CACHE_MY_OWN_MINICONDA
export TMPDIR=$HOME/tmp

source $HOME/MY_OWN_MINICONDA/Miniconda3-py312_26.1.1-1-Linux-x86_64/etc/profile.d/conda.sh
conda activate $HOME/MY_OWN_MINICONDA/Miniconda3-py312_26.1.1-1-Linux-x86_64/envs/newpy_pip

pip install numpy scipy torch      # replace with your packages
```

### Run a script in the environment

Use the same header and the same `source` and `conda activate` lines, then run the script:

```bash
module load GCCcore/13.3.0
cd $HOME
export CONDA_PKGS_DIRS=<same cache folder as above>
export TMPDIR=$HOME/tmp

source <conda folder>/etc/profile.d/conda.sh
conda activate <environment folder>

python3 yourscript.py
```

The slide also shows a short check inside the job with `python3 - << 'EOF' ... EOF`. Use `python3 yourscript.py` for real work.

## Interactive session (the user approves it first)

The slides show how to start an interactive shell on a compute node, as an exercise:

```bash
srun --nodelist=<NODE> --ntasks-per-node=1 -p <PARTITION> --time=01:00:00 --pty bash -i
```

This needs a real terminal. An agent that runs commands without a terminal cannot use it. Prefer `sbatch` jobs. Never use it on the login node.

## Rules for agents (summary)

- Show the user each remote command before you run it. Use few, light commands.
- Never run a package manager on the login node. Put `pip`, `conda` and `uv` inside a job file.
- Put `export TMPDIR=$HOME/tmp` in every job file.
- Never guess node, GPU, partition or module names. Read them from the cluster, or ask the user.
- Request the smallest time and resources that the work needs. Do not request a GPU for an environment build.
- Do not cancel jobs, delete files or edit `.bashrc` without the user's approval.
- After you submit, record the job ID. When the job ends, read the `.out` and `.err` files before you report.
- Stop and tell the user if a command fails with `fork: retry: Resource temporarily unavailable`.

## Example for this project (not tested)

A job that checks the datasets without the login node. Run `sbatch` from the repository root after the user approves it:

```bash
#!/bin/bash
#SBATCH --job-name=check_data
#SBATCH --output=check_data.%j.out
#SBATCH --error=check_data.%j.err
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --partition=<PARTITION>

set -e
set -x
export TMPDIR=$HOME/tmp
hostname
python3 scripts/fetch_datasets.py
```

`fetch_datasets.py` uses only the Python standard library. It installs nothing. Check which `python3` the node offers: the paper code needs Python 3.10.
