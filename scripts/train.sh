#!/bin/bash
#================================================================
# SLURM Job Script: BERT Sentiment Analysis
# Northeastern University - Explorer Cluster
#================================================================
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --job-name=bert-sentiment
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "========================================"
echo "BERT Sentiment Analysis"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

# Load modules
module purge
module load anaconda3

# Activate environment
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate bert-sentiment

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

mkdir -p logs data/tokenized output

# Load .env if present (e.g. WANDB_API_KEY)
[ -f "$SLURM_SUBMIT_DIR/.env" ] && set -a && source "$SLURM_SUBMIT_DIR/.env" && set +a

export DATA_CACHE_DIR="$SLURM_SUBMIT_DIR/data/tokenized"
export OUTPUT_DIR="$SLURM_SUBMIT_DIR/output/run_${SLURM_JOB_ID}"

export WANDB_PROJECT="bert-sentiment-amazon"
export WANDB_RUN_NAME="bert-base-ep3-lr2e5-$(date +%Y%m%d_%H%M%S)"
export WANDB_INIT_TIMEOUT=300
export PYTHONUNBUFFERED=1

# Resume from checkpoint if one exists (set to "" to start fresh)
export RESUME_FROM_CHECKPOINT="$SLURM_SUBMIT_DIR/output/checkpoint-26000"

echo "Starting training..."
echo "========================================"

# Run training
python -u src/train.py

echo ""
echo "========================================"
echo "Training finished at $(date)"
echo "========================================"