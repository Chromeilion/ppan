#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=ppan_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=ppan_train%j.out
set -a; source .env; set +a

# Add our ffmpeg binary to the path since it's not installed.
export PATH=$PPAN_FFMPEG_LOC:$PATH

# Check that the Accelerate config file exists
ACCELERATE_CONFIG_LOC="${MLP_ACCELERATE_CONFIG:-./accelerate_config.yaml}"
if [ ! -f "$ACCELERATE_CONFIG_LOC" ]; then
    echo "Please create the Accelerate config file before running this
    script by running 'accelerate config --config_file $ACCELERATE_CONFIG_LOC'"
    exit 1
fi

module load cuda

# Let's compile our own Python to make sure we have everything we need.
PYTHON_VERSION=3.11.7
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
tar -xzf Python-$PYTHON_VERSION.tgz 
cd Python-$PYTHON_VERSION/ || exit
./configure --enable-optimizations CC="gcc -pthread" CXX="g++ -pthread"
make -j 24
cd ..

# It's very important to recreate the virtualenv every time the job
# starts, as we want to guarantee that all our modules are correctly
# installed on the current node we're running on.
if [ -d "./venv" ]; then
  rm -r ./venv
fi

python3 -m virtualenv --python="Python-$PYTHON_VERSION/python" .venv

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
pip install -r requirements.txt
pip install .
accelerate launch --config_file "$ACCELERATE_CONFIG_LOC" ppan train

# Clean up the virtualenv after we're done
rm -r ./.venv

