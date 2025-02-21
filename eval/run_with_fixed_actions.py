import os
import sys
sys.path.append(os.getcwd())
import time
import logging
import torch.optim
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import set_exploration_type, ExplorationType
import shutil
import hydra
import numpy as np
import shutil
import pickle
import math


def replay_rollout(env, angles):
    # actions is expected to be in the shape [#steps per env, #envs, #actions]
    alpha = []  # will be num_steps x num envs x something
    reward = []
    power = []
    action = []
    # initial reset
    td = env.reset()
    for k in range(angles.shape[0]):
        # first, add in the relevant action into the td
        td.set(("alpha"), angles[k])
        td.set(("action"), np.zeros_like(angles[k]))
        td = env.step(td)
        alpha.append(td.get(("next", "alpha")).detach().cpu().numpy())
        action.append(td.get("action").detach().cpu().numpy())
        reward.append(td.get(("next", "reward")).detach().cpu().numpy())
        power.append(td.get(("next", "power")).detach().cpu().numpy())
    alpha = np.stack(alpha, axis=0).transpose((1,0,2))
    action = np.stack(action, axis=0).transpose((1,0,2))
    reward = np.stack(reward, axis=0).transpose((1,0,2))
    power = np.stack(power, axis=0).transpose((1,0,2))
    return alpha, action, reward, power


@hydra.main(config_path="./", config_name="config_eval.yaml", version_base="1.2")
def main(cfg: "DictConfig"):
    device = "cpu"  # Run on CPU only
    logging.info(f'Running on device: {device}.')

    if 'sac' in cfg.eval.training_name.lower():
        from utils_sac import make_parallel_env_shifted, make_parallel_env
    else:
        raise Exception("Can not determine training algorithm")

    logging_stream = sys.stdout if cfg.logger.logging_stream == 'stdout' else None
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        stream=logging_stream,
    )   

    max_episode_length = cfg.eval.episode_length
    n_environments = cfg.eval.n_parallel
    num_rollouts = cfg.eval.num_rollouts

    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(hydra_dir, 'RESULTS')
    os.makedirs(results_dir, exist_ok=True)

    # TO-DO: why does params have n_envs as a parameter? This is passed explicitly wherever necessary

    params = {
        "n_turbines": cfg.env.turbines,
        "n_procs": cfg.env.n_processors_per_env,
        "n_envs": cfg.eval.n_parallel,
        "probes_per_turbine": cfg.env.probes_per_turbine,
        "flow_field_directions": cfg.env.flow_field_directions,
        "turbine_diameter": cfg.env.turbine_diameter,
        "turbine_spacing": cfg.env.turbine_spacing,
        "max_yaw_speed": cfg.env.max_yaw_speed,
        "max_yaw_angle": cfg.env.max_yaw_angle,
        "dt": cfg.env.steps_per_frame * 0.2,
        "reset_frames": cfg.env.reset_frames,
        "run_steps": max_episode_length * cfg.env.steps_per_frame,
        "penalty_scale": cfg.env.penalty_scale,
        "penalty_exp": cfg.env.penalty_exp,
        "random_reset": cfg.env.random_reset,
        "initial_angles": cfg.env.initial_angles,
        "reward_average_steps": cfg.env.reward_average_steps,
        "velocity_penalty_scale": cfg.env.velocity_penalty_scale,
        "difference_penalty_scale": cfg.env.difference_penalty_scale,
    }

    # Create environments
    # These are the ORIGINAL 16 envs (ie 1...16, the same ones used to compute the angles)
    # eval_env = make_parallel_env_shifted(
    eval_env = make_parallel_env(
        cfg,
        params,
        n_environments,
        device=device,
        eval_only=True,
    ) # this needs to be modified too, to make sure that we don't just load the same exact envs! Otherwise we are not doing anything

    # Load actions to replay
    with open('eval_logs.pkl', 'rb') as f:
        eval_logs = pickle.load(f)

    actions_replay = []
    for n_ep in range(1, num_rollouts+1):
        actions_per_episode = []
        for n_env in range(1, n_environments+1):
            actions = eval_logs[f'alphas_ENV_{n_env}_EPISODE_{n_ep}']  # n_steps x num_turbs
            actions_per_episode.append(actions)
        actions_replay.append(actions_per_episode)
    actions_replay = np.stack(actions_replay, axis=0)  # n_rollouts x n_envs x n_steps x num_turbs
    actions_replay = actions_replay.transpose((0, 2, 1, 3))  # n_rollouts x n_steps x n_envs x num_turbs

    # actions_replay = torch.zeros([cfg.eval.episode_length, cfg.eval.n_parallel, cfg.env.turbines])  # num steps per env x num envs x num actions

    # Checks to make sure shapes match
    assert len(actions_replay.shape) == 4
    assert actions_replay.shape[1] == cfg.eval.episode_length
    assert actions_replay.shape[0] == cfg.eval.num_rollouts  # should be 16
    assert actions_replay.shape[2] == cfg.eval.n_parallel  # should be 16
    assert actions_replay.shape[3] == cfg.env.turbines  # num of turbs

    # Initial reset to burn in simulation
    logging.info(f'Initial reset: collecting {(1 + cfg.eval.initial_reset_frames // cfg.env.reset_frames)*cfg.env.reset_frames} frames ({1 + cfg.eval.initial_reset_frames // cfg.env.reset_frames} resets with {cfg.env.reset_frames} frames each.).')
    # Each reset is cfg.env.reset_frames, we want a total of cfg.eval.initial_reset_frames many
    reset_td = eval_env.reset()
    for i in range(cfg.eval.initial_reset_frames // cfg.env.reset_frames):
        reset_td = eval_env.reset(reset_td)
        logging.info(f"{100*i/(cfg.eval.initial_reset_frames // cfg.env.reset_frames):.1f}% done with initial reset.")
    logging.info(f"100% done with initial reset.")

    num_inbetween_resets = math.ceil(float(cfg.eval.inbetween_reset_frames) / cfg.env.reset_frames)

    logging.info('Starting evaluation...')

    # Timing
    start_time = time.time()
    logs = {}

    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        for k in range(num_rollouts):
            logging.info(f"Commencing rollout #{k+1} across {n_environments} environments.")
            # IMPORTANT: double check whether the original rollout function actually executes a reset
            # either at the first call or at the end of an episode automatically???
            alpha, actions, rewards, power = replay_rollout(eval_env, actions_replay[k])

            for i in range(n_environments):
                logs[f"rewards_ENV_{i+1}_EPISODE_{k+1}"] = rewards[i]
                logs[f"alphas_ENV_{i+1}_EPISODE_{k+1}"] = alpha[i]
                logs[f"actions_ENV_{i+1}_EPISODE_{k+1}"] = actions[i]
                logs[f"power_ENV_{i+1}_EPISODE_{k+1}"] = power[i]
            
            # Reset environments before next episode
            for _ in range(num_inbetween_resets):
                eval_env.reset()

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Evaluation took {execution_time:.2f} seconds to finish")

    # Save logs to disk
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
    with open(output_dir + "fixed_actions_logs.pkl", "wb") as f:
        pickle.dump(logs, f)


if __name__ == "__main__":
    main()
