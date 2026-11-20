#!/bin/bash
#SBATCH --job-name=eval00001
#SBATCH --nodes=18
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

#SBATCH --account=e01-ICL-Magri
#SBATCH --partition=standard
#SBATCH --qos=standard


# load required modules
module swap PrgEnv-cray PrgEnv-gnu
module load cray-python

# load python environment
source /work/e01/e01/amole/smartsim_venv/bin/activate

# Set environment variables
# export OMP_NUM_THREADS=1
export SMARTSIM_LOG_LEVEL=DEBUG
export SMARTSIM_WLM_TRIALS=50  # default 10 stopped working  
# export SR_LOG_LEVEL=DEBUG
export SR_LOG_LEVEL=INFO
export SR_LOG_FILE=./log.sr
export SR_SOCKET_TIMEOUT=300000
export D4RL_DATASET_DIR=./.cache/torchrl/data/d4rl/datasets

# Launch script
# python launch_eval_run.py eval.training_name=training_sac_fluctuation_pen2 eval.model_id=1000448
python launch_eval_run.py eval.training_name=training_sac_fluctuation_pen000001 eval.model_id=1000448 
