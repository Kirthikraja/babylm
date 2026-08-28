#!/bin/bash
#SBATCH --job-name=train_chunked
#SBATCH --output=Logs/train_chunked_%j.out
#SBATCH --error=Logs/train_chunked_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-long      # check available GPU partitions: sinfo -s
#SBATCH --gres=gpu:1

echo "========================================"
echo "Job started: $(date)"
echo "Node: $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
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
echo "GPU check: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

python -u src/train_gpt2.py \
    --condition chunked \
    --corpus_scale 100M \
    --epochs 1 \
    --batch_size 32 \
    --grad_accum 4 \
    --lr 6e-4

echo "Job finished: $(date)"
