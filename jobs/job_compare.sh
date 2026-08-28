#!/bin/bash
#SBATCH --job-name=compare_results
#SBATCH --output=Logs/compare_%j.out
#SBATCH --error=Logs/compare_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=cpu-short

echo "========================================"
echo "Job started: $(date)"
echo "========================================"

module purge
module load ALICE/default
module load Miniconda3/24.7.1-0

source "$SLURM_SUBMIT_DIR/env.sh"
source /easybuild/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate base
conda activate "$ENV_NAME"

cd "$REPO_ROOT"

python -u src/compare_results.py

echo "Job finished: $(date)"
