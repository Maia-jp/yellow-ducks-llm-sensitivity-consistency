# Agent rules for this folder

These rules apply to every agent that works in this folder and its subfolders.
They are hard rules. Do not break them, even if the user's request seems to allow it.
If a task needs a rule to be broken, stop and ask the user first.

## ITU HPC cluster: hard rules

ITU HPC (`hpc.itu.dk`) restricted the user's account on 2026-10-26.
The cause: someone ran `pip` on the main login node. The Terms of Use forbid this.
After the restriction, the login shell could not start any command (`ulimit -Hu` = 2).
A person at ITU must remove the restriction. An agent cannot fix it.

1. **The login node is for light work only.** Allowed: `ls`, `cat`, `sbatch`, `squeue`, `sinfo`, `scancel` (with approval), file edits and `scp`.
2. **Never run a package manager on the login node.** This includes `pip`, `pip3`, `python -m pip`, `uv`, `conda`, `mamba` and `npm`.
   Run installs only inside a Slurm job: `sbatch`, or `srun` on a compute node.
3. **Never run Jupyter, VS Code Remote or other interactive tools on the login node.**
4. **Never run vLLM, model downloads, training or other heavy work on the login node.** Use Slurm.
5. **Never run scripts that install packages on the login node.** This includes every install or setup script.
6. **Before any remote command, check the host.** Run `hostname` and `echo "${SLURM_JOB_ID:-none}"`. If `SLURM_JOB_ID` is `none`, you are on the login node and rules 2 to 5 apply.
7. **Show the user each remote command before you run it.** Wait for approval for anything that changes remote state.
8. **If a remote command fails with `fork: retry: Resource temporarily unavailable`, stop.** Do not retry. Tell the user.
9. **Do not open extra SSH connections or run many remote commands in a loop.** Use one connection and few, light commands.
10. **Never ask for or store passwords, keys or one-time codes.** The user types them in their own terminal.

When you write instructions, scripts or README steps for HPC, apply the same rules.
Write "run this with `srun` or `sbatch`", never "run this on HPC".

## Source of the rules

- ITU HPC email to the user, 2026-10-26: "It is forbidden to use package managers like pip, conda etc. on the main login-node as well as to login with Vscode, Jupyter and other interactive tools."
- ITU HPC documentation: https://hpc.itu.dk
- Existing setup guide: `yellow-ducks-llm-sensitivity-consistency/docs/hpc/agent-setup.md`
