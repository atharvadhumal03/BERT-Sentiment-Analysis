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
#SBATCH --time=12:00:00
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
source activate bert-sentiment

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi
echo ""

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

mkdir -p logs data/tokenized output

export DATA_CACHE_DIR="$SLURM_SUBMIT_DIR/data/tokenized"
export OUTPUT_DIR="$SLURM_SUBMIT_DIR/output"

echo "Starting training..."
echo "========================================"

# Run training
python src/train.py

echo ""
echo "========================================"
echo "Training finished at $(date)"
echo "========================================"