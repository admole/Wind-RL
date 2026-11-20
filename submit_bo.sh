#!/bin/bash
#SBATCH --job-name=mobo
#SBATCH --nodes=34
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

# python launch_bo_run.py training_bo_del_test2 optim.beta=4.0 env.initial_reset_frames=50 env.n_parallel=4
# python launch_bo_run.py training_bo_del01 optim.beta=4.0 optim.mo_weight=0.1
python launch_mobo_run.py training_mobo
# python launch_mobo_run.py training_mobo_test env.initial_reset_frames=10 env.n_parallel=4 env.reset_frames=10 optim.episode_steps=2 
