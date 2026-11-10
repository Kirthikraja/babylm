#!/bin/bash
#SBATCH --job-name=eval_ablation
#SBATCH --output=Logs/eval_ablation_%j.out
#SBATCH --error=Logs/eval_ablation_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
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

# Run full evaluation battery (FBT + BLiMP + EWoK) on all 5 ablation conditions
python -u src/eval_checkpoints.py \
    --all_conditions

echo "Job finished: $(date)"
