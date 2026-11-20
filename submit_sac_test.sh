#!/bin/bash
#SBATCH --job-name=RL10-sac
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00

#SBATCH --account=e01-ICL-Magri
#SBATCH --partition=standard
#SBATCH --qos=short


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
# python launch_sac_run.py training_sasac_long_1_2 collector.total_frames=2000000 optim.entropy_lr=1e-05 checkpoint.load_from_checkpoint=True checkpoint.model_checkpoint_path=./training_sasac_long_1/sac/checkpoints checkpoint.model_checkpoint_id=1025024
# python launch_sac_run.py training_sasac_long_2_2 collector.total_frames=2000000 optim.entropy_lr=3e-06 checkpoint.load_from_checkpoint=True checkpoint.model_checkpoint_path=./training_sasac_long_2/sac/checkpoints checkpoint.model_checkpoint_id=1025024

# python launch_sac_run.py training_sasac_long_3_2 collector.total_frames=2000000 optim.entropy_lr=1e-05 optim.step_mult=2 checkpoint.load_from_checkpoint=True checkpoint.model_checkpoint_path=./training_sasac_long_3/sac/checkpoints checkpoint.model_checkpoint_id=1025024

# python launch_sac_run.py testing_sasac_10turb env.turbines=10 optim.entropy_lr=1e-05 env.initial_angles=[0,0,0,0,0,0,0,0,0,0] env.n_parallel=1
#
python launch_sac_run.py testing_sasac_fluc optim.entropy_lr=1e-05 env.n_parallel=2
