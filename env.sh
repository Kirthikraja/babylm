#!/usr/bin/env bash
# env.sh — project-wide paths and conda env name.
# Source this at the top of every job script.

export REPO_ROOT="$HOME/babylm-gpt2-chunking"
export DATA_ROOT="$REPO_ROOT/data"
export RESULTS_ROOT="$REPO_ROOT/results"
export MODELS_ROOT="$REPO_ROOT/models"

export ENV_NAME="babylm"          # conda environment name
