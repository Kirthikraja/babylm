#!/bin/bash
# Usage: sbatch jobs/job_fill_blimp.sh flat
#        sbatch jobs/job_fill_blimp.sh chunked
#        sbatch jobs/job_fill_blimp.sh balanced
#SBATCH --job-name=fill_blimp
#SBATCH --output=Logs/fill_blimp_%x_%j.out
#SBATCH --error=Logs/fill_blimp_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=7-00:00:00
#SBATCH --partition=gpu-mig-40g
#SBATCH --gres=gpu:1

CONDITION="${1:-flat}"

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Condition: $CONDITION"
echo "Mode: fill_blimp"
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

python -u src/eval_checkpoints.py \
    --condition "$CONDITION" \
    --fill_blimp

echo "Job finished: $(date)"
