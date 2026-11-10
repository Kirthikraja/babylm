#!/bin/bash
#SBATCH --job-name=think_vecs_ablation
#SBATCH --output=Logs/think_vecs_ablation_%j.out
#SBATCH --error=Logs/think_vecs_ablation_%j.err
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

# Extract think vectors (all 3 formulations) for all ablation conditions
python -u src/extract_think_vectors.py \
    --all_conditions

echo "Job finished: $(date)"
