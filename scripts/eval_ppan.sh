#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=small-pi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=./logs/run%j.out
#SBATCH --mem=0
#SBATCH --gres=gpu:2
#SBATCH --exclusive

# --------------------------------------------------------------------
# A script for evaluating PPAN. Supports multiple GPUs but only on a
# single node.
# --------------------------------------------------------------------

# Load .env file
set -a; source .env; set +a

module load cuda/12.1

# Add our ffmpeg binary to the path since it's not installed system-wide.
export PATH=$PPAN_FFMPEG_LOC:$PATH

# Load the virtual environment
source "$PPAN_VIRTUALENV"

# Install PPAN
pip install "$PPAN_REPO_ROOT"

python "$PPAN_REPO_ROOT"/ppan evaluate

deactivate
