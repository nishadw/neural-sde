---
name: umiacs
description: Run, monitor and debug ripen jobs on the UMIACS Nexus class cluster (nexusclass00/01, SLURM partition "class"). Use for anything that trains, tests, extracts features, downloads data or otherwise computes on this cluster, and for checking job status or logs.
---

# Running ripen on UMIACS class accounts

Source: https://wiki.umiacs.umd.edu/umiacs/index.php/ClassAccounts

## Hard rules
- **Never compute on the submission node** (`nexusclass00` / `nexusclass01`). Training, tests, feature
  extraction, big downloads: all go through `sbatch` (or `srun --pty bash` for interactive work).
  Editing files, `git`, `squeue`, reading logs are fine on the login node.
- Class accounts may only use `--partition=class --account=class` and `--qos` in `default | medium | high`.
- Per-user caps across all running jobs: 32 CPUs, 4 GPUs, 256 GB RAM. Extra jobs just queue.
- Home (`/fs/classhomes/nwajge`) has a 30 GB quota. The repo, venv and all caches live in project storage:
  `/fs/class-projects/fall2026/cmsc498z/nwajge` (100 GB). Accounts are archived after the semester
  (fall: from January 1 of the next year), so push results to GitHub before then.

| QoS | MaxWall | Max per job |
|---|---|---|
| default | 3 days | 4 CPU, 1 GPU, 32 GB |
| medium | 2 days | 8 CPU, 2 GPU, 64 GB |
| high | 1 day | 16 CPU, 4 GPU, 128 GB |

Check live values with `show_qos`, `show_partition_qos`, `show_nodes -p class` (RTX A4000 / A5000 nodes).

## Layout on this cluster
- Repo: `/fs/class-projects/fall2026/cmsc498z/nwajge/ripen` (remote `git@github.com:nishadw/ripen.git`)
- `.ripen.env` (gitignored, sourced by `run_umiacs.sh`): `RIPEN_ENV` venv at `.../nwajge/envs/ripen`,
  `RIPEN_PYTHON` = uv-managed Python 3.11 (system python is 3.9, too old), pip/uv/torch caches under `.../nwajge/.cache`.
- `uv` is at `~/.local/bin/uv` with `UV_PYTHON_INSTALL_DIR=.../nwajge/.local/python`.
- Account/partition/QoS default to `class/class/default` when the hostname is `nexusclass*`; override with
  `RIPEN_QOS=medium` etc.

## Running
```bash
cd /fs/class-projects/fall2026/cmsc498z/nwajge/ripen
bash run_umiacs.sh pipeline          # setup -> data -> every experiment -> summary, chained with afterok
bash run_umiacs.sh setup             # one sbatch job: venv, install, CPU tests, smoke
bash run_umiacs.sh data              # one sbatch job: avocado download (resumable) + loader check, splits, features
bash run_umiacs.sh e1                # one experiment stage (e1..e6, synthetic, summary); needs setup+data done
RIPEN_AFTER=<jobid> bash run_umiacs.sh all   # make every submitted job wait on <jobid>
```
Every job logs to `results/logs/ripen-<name>-<jobid>.out`. One-off commands:
```bash
sbatch -p class -A class -q default --gres=gpu:1 -c 4 --mem=16G -t 2:00:00 -o results/logs/%x-%j.out \
  -J ripen-adhoc --wrap "source .ripen.env && source \$RIPEN_ENV/bin/activate && python scripts/train.py model=latent_sde data=avocado"
srun -p class -A class -q default --gres=gpu:1 -c 4 --mem=16G -t 1:00:00 --pty bash   # interactive
```

## Monitoring
- `squeue -u $USER` (state, reason); `squeue -u $USER -t PD -o '%i %j %r'` shows why jobs wait.
- `sacct -u $USER -S today -o JobID,JobName%28,State,Elapsed,ExitCode` for finished jobs.
- `tail -f results/logs/ripen-<name>-<jobid>.out`.
- `DependencyNeverSatisfied` means an upstream `afterok` job failed: read its log, fix, `scancel` the stuck
  dependents, and resubmit that stage (training writes to `results/runs/<run_name>`, so reruns overwrite).
- `scancel -u $USER -n ripen-e3-...` / `scancel <jobid>` to cancel.
