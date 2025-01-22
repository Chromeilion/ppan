#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --job-name=py_comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:10:00
#SBATCH --output=./logs/py_comp%j.out
#SBATCH --exclusive

# --------------------------------------------------------------------------
# A script for compiling a specific version of python to a prefix dir.
# If no prefix directory is specified (as the second argument), then the
# current dir is used.
# Exports PPAN_PYTHON_LOC which stores the location of the new binaries.
# --------------------------------------------------------------------------

set -a; source .env; set +a

# Sort out inputs
PYTHON_VERSION="$PPAN_PYTHON_VERSION"
PREFIX_DIR="$PPAN_PYTHON_PREFIX"
echo "Python ver: $PYTHON_VERSION, Prefix: $PREFIX_DIR"

# Actually compile
if ! [ -d "./Python-$PYTHON_VERSION/" ]; then
  # Dont download the file if it's already there
  wget -nc https://www.python.org/ftp/python/"$PYTHON_VERSION"/Python-"$PYTHON_VERSION".tgz
  tar -xzf Python-"$PYTHON_VERSION".tgz
fi

cd Python-"$PYTHON_VERSION"/ || exit
make clean
./configure --enable-optimizations CC="gcc -pthread" CXX="g++ -pthread" "--prefix=$PREFIX_DIR"
make -j "$(nproc --all)"
make install
cd ..

# Remove leftover files
rm -r ./Python-"$PYTHON_VERSION"/
rm ./Python-"$PYTHON_VERSION".tgz

# Let whoever called this script know where the new Python binaries are
export PPAN_PYTHON_LOC="$PREFIX_DIR/bin"
