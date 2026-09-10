#!/bin/bash
# Runs FBT on ALL mention-order variants (960 items) for mention-order analysis.
# Usage: sbatch jobs/job_eval_fbt_all.sh chunked
#        sbatch jobs/job_eval_fbt_all.sh flat
#SBATCH --job-name=eval_fbt_all
#SBATCH --output=Logs/eval_fbt_all_%j.out
#SBATCH --error=Logs/eval_fbt_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

FBT_CSV="$REPO_ROOT/data/fbt/fb.csv"
if [ ! -f "$FBT_CSV" ]; then
    echo "ERROR: FBT stimuli not found at $FBT_CSV"
    exit 1
fi

MODEL_PATH="$MODELS_ROOT/$CONDITION/final"
echo "Model: $MODEL_PATH"

python -u src/eval_fbt.py \
    --model_path "$MODEL_PATH" \
    --condition "$CONDITION" \
    --output_dir "$RESULTS_ROOT/$CONDITION" \
    --all_items

# Rename output so it doesn't overwrite the filtered fbt.json
mv "$RESULTS_ROOT/$CONDITION/fbt.json" "$RESULTS_ROOT/$CONDITION/fbt_all.json"
echo "Saved to $RESULTS_ROOT/$CONDITION/fbt_all.json"

echo "Job finished: $(date)"
