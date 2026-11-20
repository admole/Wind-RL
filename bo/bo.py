import time
import os
import sys
import hydra
import torch.cuda
import numpy as np
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from utils_bo import (
    make_parallel_env,
    BOTrainer
)
import logging
import shutil
import math


@hydra.main(version_base="1.2", config_path="./", config_name="config_bo")
def main(cfg: "DictConfig"):
    device = "cpu"  # Run on CPU only
    logging.info(f'Running on device: {device}.')

    logging_stream = sys.stdout if cfg.logger.logging_stream == 'stdout' else None
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        stream=logging_stream,
    )

    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # TO-DO: pass only the cfg into all functions...    
    params = {
        "n_turbines": cfg.env.turbines,
        "n_procs": cfg.env.n_processors_per_env,
        "n_envs": cfg.env.n_parallel,
        "probes_per_turbine": cfg.env.probes_per_turbine,
        "flow_field_directions": cfg.env.flow_field_directions,
        "turbine_diameter": cfg.env.turbine_diameter,
        "turbine_spacing": cfg.env.turbine_spacing,
        "max_yaw_speed": cfg.env.max_yaw_speed,
        "max_yaw_angle": cfg.env.max_yaw_angle,
        "dt": cfg.env.steps_per_frame * 0.2,
        "reset_frames": cfg.env.reset_frames,
        "run_steps": cfg.optim.episode_steps * cfg.env.steps_per_frame,
        "penalty_scale": cfg.env.penalty_scale,
        "penalty_exp": cfg.env.penalty_exp,
        "random_reset": cfg.env.random_reset,
        "initial_angles": cfg.env.initial_angles,
        "reward_average_steps": cfg.env.reward_average_steps,
        "velocity_penalty_scale": cfg.env.velocity_penalty_scale,
        "difference_penalty_scale": cfg.env.difference_penalty_scale,
        "fluctuation_penalty_scale": cfg.env.fluctuation_penalty_scale,
    }

    logging.info('Creating models')
    bo_env = make_parallel_env(cfg, params, cfg.env.n_parallel, device=device)

    # Initial reset to burn in simulation
    logging.info(f'Initial reset: collecting {(1 + cfg.env.initial_reset_frames // cfg.env.reset_frames)*cfg.env.reset_frames} frames ({1 + cfg.env.initial_reset_frames // cfg.env.reset_frames} resets with {cfg.env.reset_frames} frames each.).')
    # Each reset is cfg.env.reset_frames, we want a total of cfg.nv.initial_reset_frames many
    reset_td = bo_env.reset()
    for i in range(cfg.env.initial_reset_frames // cfg.env.reset_frames):
        reset_td = bo_env.reset(reset_td)
        logging.info(f"{100*i/(cfg.env.initial_reset_frames // cfg.env.reset_frames):.1f}% done with initial reset.")
    logging.info(f"100% done with initial reset.")

    # Define BO trainer
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
    trainer = BOTrainer(
        bo_env,
        n_initial_samples=cfg.optim.init_num_obs,
        episode_steps=cfg.optim.episode_steps,
        burnin_steps=cfg.env.reset_frames,
        beta=cfg.optim.beta,
        reset_iterations=cfg.optim.reset_iterations,
        save_directory_path=output_dir,
        mo_weight=cfg.optim.mo_weight,
    )
    if trainer.save_directory:
        logging.info(f"Logging files to {output_dir}.")

    # Initialise parameters
    logging.info("--------------------")
    logging.info(f"Initial random parameter search...")
    logging.info("--------------------")
    init_train_x, init_train_y = trainer.initialize_observations()

    logging.info("--------------------")
    logging.info(f"Optimising angles...")
    logging.info("--------------------")
    train_x, train_y = trainer.train(init_train_x, init_train_y, cfg.optim.train_steps)




if __name__ == "__main__":
    main()
