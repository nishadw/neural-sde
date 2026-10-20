#!/usr/bin/env bash
#SBATCH --job-name=ripen-synthetic
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# The fake-fruit test at full scale (CPU is enough).
#   sbatch --account=$RIPEN_ACCOUNT --partition=$RIPEN_PARTITION --qos=$RIPEN_QOS slurm/synthetic_recovery.sh
source "$(dirname "$0")/common.sh"; activate
python scripts/synthetic_recovery.py
