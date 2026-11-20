import os
import sys
sys.path.append(os.getcwd())
import time
import logging
import torch.optim
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs.utils import set_exploration_type, ExplorationType
import shutil
import hydra
import numpy as np
import shutil
import pickle
import math


def adjust_tensor_shapes(data, env):
    """
    Adjust the shapes of tensors for multiagent tensordict.
    """
    data.set(
        ("next", "agents", "done"),
        data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(data.get_item_shape(("next", env.reward_key))),
    )
    data.set(
        ("next", "agents", "terminated"),
        data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(data.get_item_shape(("next", env.reward_key))),
    )
    data.set(
        ("next", "done"),
        data.get(("next", "done"))
        .unsqueeze(-1)
        .expand(data.get_item_shape(("next", env.reward_key))),
    )
    data.set(
        ("next", "terminated"),
        data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(data.get_item_shape(("next", env.reward_key))),
    )
    return data


def make_constant_zero_policy(cfg):
    angles = torch.zeros([cfg.eval.n_parallel, cfg.env.turbines])
    actor = TensorDictModule(
        lambda x: angles,
        in_keys=["observation"], 
        out_keys=["action"]
    )
    return actor


@hydra.main(config_path="./", config_name="config_eval.yaml", version_base="1.2")
def main(cfg: "DictConfig"):
    device = "cpu"  # Run on CPU only
    logging.info(f'Running on device: {device}.')

    if 'ppo' in cfg.eval.training_name.lower():
        from utils_ppo import make_parallel_env, load_model, save_model, log_metrics
    elif 'sac' in cfg.eval.training_name.lower():
        from utils_sac import make_parallel_env, load_model, save_model, log_metrics
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
        "fluctuation_penalty_scale": cfg.env.fluctuation_penalty_scale,
    }

    # Create environments
    eval_env = make_parallel_env(
        cfg,
        params,
        n_environments,
        device=device,
        eval_only=True,
    )

    # Check if we want to use fixed angles throughout
    if cfg.eval.use_fixed_angles:
        actor = make_constant_zero_policy(cfg)
        actor.to(device)
        logging.info(f"Created actor which outputs constant angles {cfg.eval.fixed_angles}.")
    else:
        # Load the models to be evaluated
        # Copy the loaded models into the ppo/eval directory
        if not os.path.exists('models'):
            os.mkdir('models')
        for arch in ['actor', 'critic']:
            filename = f"{arch}_{cfg.eval.model_id}.pkl"
            if os.path.exists(filename):
                shutil.move(filename, "models")
        # Load actor and critic
        actor, critic = load_model(
            cfg=cfg,
            env_params=params,
            id=cfg.eval.model_id,
            path_to_model='models',
            )
        logging.info(f"Loaded actor_{cfg.eval.model_id}.pkl and critic_{cfg.eval.model_id}.pkl.")
        actor, critic = actor.to(device), critic.to(device)

    # Compute number of resets to be done in between eval runs
    num_inbetween_resets = math.ceil(float(cfg.eval.inbetween_reset_frames) / cfg.env.reset_frames)
    num_rollouts_per_env = cfg.eval.num_rollouts

    # Initial reset to burn in simulation
    logging.info(f'Initial reset: collecting {(1 + cfg.eval.initial_reset_frames // cfg.env.reset_frames)*cfg.env.reset_frames} frames ({1 + cfg.eval.initial_reset_frames // cfg.env.reset_frames} resets with {cfg.env.reset_frames} frames each.).')
    # Each reset is cfg.env.reset_frames, we want a total of cfg.eval.initial_reset_frames many
    reset_td = eval_env.reset()
    for i in range(cfg.eval.initial_reset_frames // cfg.env.reset_frames):
        reset_td = eval_env.reset(reset_td)
        logging.info(f"{100*i/(cfg.eval.initial_reset_frames // cfg.env.reset_frames):.1f}% done with initial reset.")
    logging.info(f"100% done with initial reset.")

    logging.info('Starting evaluation...')

    # Timing
    start_time = time.time()
    logs = {}

    with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
        for k in range(num_rollouts_per_env):
            logging.info(f"Commencing rollout #{k+1} across {n_environments} environments.")

            data = eval_env.rollout(
                max_steps=max_episode_length,
                policy=actor,
                auto_reset=True,
            )
            if cfg.multi_agent.use:
                data = adjust_tensor_shapes(data, eval_env)

            # Get rewards and episode lengths
            # No "agents" in single agent
            if cfg.multi_agent.use:
                episode_rewards = data["next", "agents", "episode_reward"][data["next", "done"]]
                reward_shape = data.get_item_shape(("next", eval_env.reward_key))
                episode_rewards = episode_rewards.view(reward_shape[-2], reward_shape[0]).mean(dim=0)
                episode_length = data["next", "step_count"][data["next", "done"].all(-2)]
                rewards = data["next", "agents", "reward"].squeeze().mean(dim=-1)
                alpha = data["agents", "alpha"].squeeze()
                actions = data["agents", "action"].squeeze()
            else:
                episode_rewards = data["next", "episode_reward"][data["next", "done"]]
                reward_shape = data.get_item_shape(("next", "reward"))
                episode_rewards = episode_rewards.view(reward_shape[-1], reward_shape[0]).mean(dim=0)
                episode_length = data["next", "step_count"][data["next", "done"]]
                rewards = data["next", "reward"].squeeze()
                observations = data["next", "observation"]
                alpha = data["alpha"].squeeze()
                actions = data["action"].squeeze()
                power = data.get(("next", "power")).squeeze()
                episode_power = data.get(("next", "episode_power"))[data["next", "done"]]
                episode_power = episode_power.view(reward_shape[-1], reward_shape[0]).mean(dim=0)

            if not len(episode_length) > 0:
                raise RuntimeWarning("The eval tensordict does not contain a finished episode.")
            
            for i in range(n_environments):
                logs[f"episode_reward_ENV_{i+1}_EPISODE_{k+1}"] = episode_rewards[i].item()
                logs[f"episode_length_ENV_{i+1}_EPISODE_{k+1}"] = episode_length[i].item()
                logs[f"rewards_ENV_{i+1}_EPISODE_{k+1}"] = rewards[i].detach().cpu().numpy()
                logs[f"alphas_ENV_{i+1}_EPISODE_{k+1}"] = alpha[i].detach().cpu().numpy()
                logs[f"actions_ENV_{i+1}_EPISODE_{k+1}"] = actions[i].detach().cpu().numpy()
                logs[f"observations_ENV_{i+1}_EPISODE_{k+1}"] = observations[i].detach().cpu().numpy()
                logs[f"episode_power_ENV_{i+1}_EPISODE_{k+1}"] = episode_power[i].item()
                logs[f"power_ENV_{i+1}_EPISODE_{k+1}"] = power[i].detach().cpu().numpy()
            
            # Reset environments before next episode
            for _ in range(num_inbetween_resets):
                eval_env.reset()

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Evaluation took {execution_time:.2f} seconds to finish")

    # Save logs to disk
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
    with open(output_dir + "eval_logs.pkl", "wb") as f:
        pickle.dump(logs, f)


if __name__ == "__main__":
    main()
