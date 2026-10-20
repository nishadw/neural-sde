#!/usr/bin/env bash
#SBATCH --job-name=ripen-sampling
#SBATCH --output=results/logs/%x-%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
# Direction 3: train and test with photos every 2 or 3 days, for the three distributional sequence models.
#   sbatch --account=$RIPEN_ACCOUNT --partition=$RIPEN_PARTITION --qos=$RIPEN_QOS slurm/sampling_ablation.sh avocado
source "$(dirname "$0")/common.sh"; activate
DATA="${1:-avocado}"
MODELS=(gru_d latent_ode latent_sde)
EVERY=(2 3)
MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID % 3))]}"
K="${EVERY[$((SLURM_ARRAY_TASK_ID / 3))]}"
python scripts/train.py model="$MODEL" data="$DATA" +experiment=sampling_every"$K"
