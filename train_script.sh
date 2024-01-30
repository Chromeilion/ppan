#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=ppan_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=50gb
#SBATCH --time=02:00:00
#SBATCH --output=ppan_train%j.out
set -a; source .env; set +a
module load cuda

# It's very important to recreate the virtualenv every time the job
# starts, as we want to guarantee that all our modules are correctly
# installed on the current node we're running on.
if [ -d "./venv" ]; then
  rm -r ./venv
fi
python -m venv ./venv

# Check that the Accelerate config file exists
ACCELERATE_CONFIG_LOC="${MLP_ACCELERATE_CONFIG:-./accelerate_config.yaml}"
if [ ! -f "$ACCELERATE_CONFIG_LOC" ]; then
    echo "Please create the Accelerate config file before running this
    script by running 'accelerate config --config_file $ACCELERATE_CONFIG_LOC'"
    exit 1
fi

source ./venv/bin/activate
pip install cython
pip install -r requirements.txt
pip install .
accelerate launch --config_file "$ACCELERATE_CONFIG_LOC" ./ppan/__main__.py train

# Clean up the virtualenv after we're done
rm -r ./venv