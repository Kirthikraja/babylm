#!/bin/bash
#SBATCH --job-name=plot_results
#SBATCH --output=Logs/plot_%j.out
#SBATCH --error=Logs/plot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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

python -u src/plot_results.py \
    --results_dir "$RESULTS_ROOT" \
    --out_dir "$RESULTS_ROOT/figures"

echo "Figures saved to $RESULTS_ROOT/figures"
echo "Job finished: $(date)"
