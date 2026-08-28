#!/bin/bash
#SBATCH --job-name=babylm_chunk
#SBATCH --output=Logs/chunk_%j.out
#SBATCH --error=Logs/chunk_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu-short

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0

source "$SLURM_SUBMIT_DIR/env.sh"
source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate "$ENV_NAME"

cd "$REPO_ROOT"
echo "Python: $(which python)"

# EOS-at-sentence-boundaries chunking (the WITH-CHUNKING condition)
python -u src/chunk_corpus.py --corpus_scale 100M

echo "Job finished: $(date)"
