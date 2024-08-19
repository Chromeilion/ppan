#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=cls_sgd
#SBATCH --nodelist=gpu003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=./logs/run%j.out
#SBATCH --mem=0
#SBATCH --gpus=2
#SBATCH --exclusive

# --------------------------------------------------------------------
# A script for installing and running PPAN in a Python virtualenv with
# Accelerate. A specific PPAN subcommand must be provided as an argument
# (pretrain, evaluate, finetune, etc.)
# The actual PPAN parameters can be controlled through environment variables.
# --------------------------------------------------------------------

if [ $# -ne 1 ]; then
  echo "Please specify the PPAN subcommand to run as an argument"
fi

# Load .env file
set -a; source .env; set +a

# Check that the Accelerate config file exists
ACCELERATE_CONFIG_LOC="${MLP_ACCELERATE_CONFIG:-./accelerate_config.yaml}"
if [ ! -f "$ACCELERATE_CONFIG_LOC" ]; then
    echo "Please create the Accelerate config file before running this
    script by running 'accelerate config --config_file $ACCELERATE_CONFIG_LOC'"
    exit 1
fi

# Add our ffmpeg binary to the path since it's not installed system-wide.
export PATH=$PPAN_FFMPEG_LOC:$PATH

module load cuda

# It's very important to recreate the virtualenv every time the job
# starts, as we want to guarantee that all our modules are correctly
# installed on the current node we're running on.
if [ -d "./venv" ]; then
  rm -r ./.venv
fi

#  Create a virtual env using the provided Python
"$PPAN_PYTHON_PREFIX"/bin/python3 -m venv .venv

source ./.venv/bin/activate

# Install wheel and upgrade pip
python -m pip install --upgrade pip
pip install wheel
pip install pybind11
# We have to manually install some packages first because madmoms
# deps are broken.
pip install Cython
pip install numpy
pip install git+https://github.com/CPJKU/madmom
pip install accelerate
# Install PPAN
pip install "$PPAN_REPO_ROOT"

# Run the script
accelerate launch --config_file "$ACCELERATE_CONFIG_LOC" "$PPAN_REPO_ROOT"/ppan "$1"

# Clean up the virtualenv after we're done
rm -r ./.venv
