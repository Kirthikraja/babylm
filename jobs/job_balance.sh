#!/bin/bash
# DENSITY-style diversity sampling (Sachdeva et al. ICLR 2026).
# Balances chunked sub-corpora to a common size using k-means + centroid
# selection.  Reads data/chunked_<scale>/ → writes data/balanced_<scale>/.
#
# Usage: sbatch jobs/job_balance.sh [10M|100M] [target_k]
#   e.g. sbatch jobs/job_balance.sh 10M 1005
#
#SBATCH --job-name=balance_corpus
#SBATCH --output=Logs/balance_%j.out
#SBATCH --error=Logs/balance_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1

SCALE="${1:-10M}"
TARGET_K="${2:-1005}"

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Corpus scale: $SCALE"
echo "Target k: $TARGET_K"
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

INPUT_DIR="$DATA_ROOT/chunked_${SCALE}"
OUTPUT_DIR="$DATA_ROOT/balanced_${SCALE}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: chunked data not found at $INPUT_DIR"
    echo "Run job_chunk.sh first."
    exit 1
fi

python -u src/balance_corpus.py \
    --corpus_scale "$SCALE" \
    --input_dir    "$INPUT_DIR" \
    --output_dir   "$OUTPUT_DIR" \
    --target_k     "$TARGET_K" \
    --drop switchboard \
    --model_name   gpt2 \
    --batch_size   8 \
    --seed         42

echo "========================================"
echo "Job finished: $(date)"
echo "Output written to: $OUTPUT_DIR"
echo "========================================"
