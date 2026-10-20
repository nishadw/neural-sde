#!/usr/bin/env bash
#SBATCH --job-name=ripen-logo
#SBATCH --output=results/logs/%x-%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8
# Generalisation to an unseen storage group: 3 held-out groups x 3 models (direct_reg, gru_d, latent_sde).
#   sbatch --account=$RIPEN_ACCOUNT --partition=$RIPEN_PARTITION --qos=$RIPEN_QOS slurm/logo.sh avocado
source "$(dirname "$0")/common.sh"; activate
DATA="${1:-avocado}"
MODELS=(direct_reg gru_d latent_sde)
GROUPS=(T10 T20 Tamb)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID % 3))]}"
G="${GROUPS[$((SLURM_ARRAY_TASK_ID / 3))]}"
python scripts/train.py model="$MODEL" data="$DATA" +experiment=logo splits.held_out="$G"
