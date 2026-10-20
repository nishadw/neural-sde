#!/usr/bin/env bash
#SBATCH --job-name=ripen-grad
#SBATCH --output=results/logs/%x-%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-2
# Same latent SDE trained three ways: backprop through the solver, recursive checkpointing, stochastic adjoint.
#   sbatch --account=$RIPEN_ACCOUNT --partition=$RIPEN_PARTITION --qos=$RIPEN_QOS slurm/gradient_methods.sh avocado
source "$(dirname "$0")/common.sh"; activate
DATA="${1:-avocado}"
METHODS=(backprop checkpoint adjoint)
M="${METHODS[$SLURM_ARRAY_TASK_ID]}"
python scripts/train.py model=latent_sde data="$DATA" +experiment=gradient_methods model.method="$M"
