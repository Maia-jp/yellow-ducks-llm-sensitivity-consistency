# Shared HPC SSH connection

Connect to the ITU VPN. From the repository root, run:

```bash
./scripts/connect-hpc.sh
```

The script uses the ITU username `s25441`. To use another username, run:

```bash
./scripts/connect-hpc.sh ITU_USERNAME
```

Enter your password in the terminal. Keep the SSH connection open while the agent uses HPC.
The script reuses an existing shared connection when available.

The SSH socket is a local file that lets another SSH command use your authenticated connection.
The script uses `~/.ssh/itu-hpc-control` as the socket path.
The agent can use this command to check access:

```bash
ssh -S ~/.ssh/itu-hpc-control -o BatchMode=yes s25441@hpc.itu.dk hostname
```

## Rules for the shared connection

Use the shared connection to inspect files and to run `sbatch`, `squeue` and `sinfo`.
Do not run `pip`, `uv`, `conda`, Jupyter, VS Code Remote or vLLM on the login node. ITU forbids this.
Stop and tell the user if a command fails with `fork: retry: Resource temporarily unavailable`.
