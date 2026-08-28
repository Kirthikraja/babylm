#!/bin/bash
# Usage: sbatch jobs/job_eval_ewok.sh chunked
#        sbatch jobs/job_eval_ewok.sh flat
#SBATCH --job-name=eval_ewok
#SBATCH --output=Logs/eval_ewok_%j.out
#SBATCH --error=Logs/eval_ewok_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1

CONDITION="${1:-chunked}"

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Condition: $CONDITION"
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

MODEL_PATH="$MODELS_ROOT/$CONDITION/final"
echo "Model: $MODEL_PATH"

python -u src/eval_ewok.py \
    --model_path "$MODEL_PATH" \
    --condition "$CONDITION"

echo "Job finished: $(date)"
