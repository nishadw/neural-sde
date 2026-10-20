#!/usr/bin/env bash
#SBATCH --job-name=ripen-train
#SBATCH --output=results/logs/%x-%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
# Account / partition / qos are taken from slurm/common.sh at submit time:
#   sbatch --account=$RIPEN_ACCOUNT --partition=$RIPEN_PARTITION --qos=$RIPEN_QOS slurm/train_all_models.sh avocado
#
# One array task per model on the dataset given as $1 (default avocado). Extra hydra overrides go after it.
source "$(dirname "$0")/common.sh"; activate
DATA="${1:-avocado}"; shift || true
MODELS=(stage_linear direct_reg discrete_surv gru_gap gru_d ode_rnn latent_ode latent_sde)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "model=$MODEL data=$DATA extra=$*"
python scripts/train.py model="$MODEL" data="$DATA" "$@"
