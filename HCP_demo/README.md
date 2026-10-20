# Shared HPC SSH connection

Connect to the ITU VPN. From the workspace root, run:

```bash
./HCP_demo/connect.sh
```

The script uses the ITU username `s25441`. To use another username, run:

```bash
./HCP_demo/connect.sh ITU_USERNAME
```

Enter your password in the terminal. Keep the SSH connection open while the agent uses HPC.
The script reuses an existing shared connection when available.

The SSH socket is a local file that lets another SSH command use your authenticated connection.
The script uses `~/.ssh/itu-hpc-control` as the socket path.
The agent can use this command to check access:

```bash
ssh -S ~/.ssh/itu-hpc-control -o BatchMode=yes s25441@hpc.itu.dk hostname
```
