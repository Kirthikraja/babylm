#!/bin/bash
# Usage: sbatch jobs/job_extract_think_vectors.sh flat
#        sbatch jobs/job_extract_think_vectors.sh flat --layer 8
#SBATCH --job-name=think_vec
#SBATCH --output=Logs/think_vectors_%j.out
#SBATCH --error=Logs/think_vectors_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1

CONDITION="${1:-flat}"
EXTRA_ARGS="${@:2}"   # e.g. --layer 8

echo "========================================"
echo "Job started:  $(date)"
echo "Node:         $HOSTNAME"
echo "Job ID:       $SLURM_JOB_ID"
echo "Condition:    $CONDITION"
echo "Extra args:   $EXTRA_ARGS"
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

python -u src/extract_think_vectors.py \
    --condition "$CONDITION" \
    $EXTRA_ARGS

echo "Job finished: $(date)"
