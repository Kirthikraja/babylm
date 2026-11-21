#!/bin/bash
#SBATCH --job-name=babylm_download
#SBATCH --output=Logs/download_%j.out
#SBATCH --error=Logs/download_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

export HF_HOME="/zfsstore/user/s4570170/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"

python -u src/download_data.py --scale 10M --cache_dir "$HF_HOME"

echo "Job finished: $(date)"
