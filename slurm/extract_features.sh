#!/usr/bin/env bash
#SBATCH --job-name=ripen-features
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
# Cache ResNet-18 features for every avocado photo (run once).
#   sbatch --account=$RIPEN_ACCOUNT --partition=$RIPEN_PARTITION --qos=$RIPEN_QOS slurm/extract_features.sh
source "$(dirname "$0")/common.sh"; activate
python scripts/extract_features.py --data avocado --root data/avocado --out data/features/avocado_resnet18.npz --device cuda
