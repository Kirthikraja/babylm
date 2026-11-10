#!/bin/bash
#SBATCH --job-name=partition_corpus
#SBATCH --output=Logs/partition_corpus_%j.out
#SBATCH --error=Logs/partition_corpus_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu-mig-40g
#SBATCH --gres=gpu:1

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module purge
module load ALICE/default
module load CUDA/12.4.0
module load Miniconda3/24.7.1-0

source "$SLURM_SUBMIT_DIR/env.sh"
source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate "$ENV_NAME"

cd "$REPO_ROOT"
echo "Python: $(which python)"

# Build all 5 ablation conditions with DENSITY equalization to 50M tokens
python -u src/partition_corpus.py \
    --corpus data/raw/babylm_100M/corpus.train.txt \
    --out_dir data/ablation_100M \
    --condition all \
    --target_tokens 50000000 \
    --max_seq_len 1024

echo "Job finished: $(date)"
