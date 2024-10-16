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

module load cuda/12.1

# Add our ffmpeg binary to the path since it's not installed system-wide.
export PATH=$PPAN_FFMPEG_LOC:$PATH

# Load the virtual environment
source "$PPAN_VIRTUALENV"

# Install PPAN
pip install "$PPAN_REPO_ROOT"

# In case the SLURM cluster doesn't have a DNS, we find the IP address of the
# main node manually.
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
MASTER_HOSTNAME=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_ADDR_FULL=$(scontrol getaddrs $MASTER_HOSTNAME)
IFS=':' read -ra HOSTNAME_SPLIT <<< $MASTER_ADDR_FULL

MASTER_ADDR="$(echo -e "${HOSTNAME_SPLIT[1]}" | tr -d '[:space:]')"
MASTER_PORT=6000

echo "Master address: $MASTER_ADDR"

export LAUNCHER="accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "

export PROGRAM="$PPAN_REPO_ROOT/ppan $1"
export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

deactivate
