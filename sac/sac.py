import time
import os
import sys
import hydra
import torch.cuda
import numpy as np
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from utils_sac import (
    make_collector,
    make_loss_module,
    make_replay_buffer,
    make_saving_replay_buffer,
    make_sa_sac_agent,
    make_ma_sac_agents,
    make_sac_optimizer,
    make_parallel_env,
    log_metrics,
    should_log_now,
    save_model,
    load_model,
    update_data_shapes,
)
import logging
import shutil


@hydra.main(version_base="1.2", config_path="./", config_name="config_sac")
def main(cfg: "DictConfig"):
    device = "cpu"  # Run on CPU only
    logging.info(f'Running on device: {device}.')

    logging_stream = sys.stdout if cfg.logger.logging_stream == 'stdout' else None
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s', 
        stream=logging_stream,
    )

    total_frames = cfg.collector.total_frames
    frames_per_batch = cfg.collector.frames_per_batch

    hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(hydra_dir, 'RESULTS')
    os.makedirs(results_dir, exist_ok=True)

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
        "run_steps": cfg.collector.max_episode_length * cfg.env.steps_per_frame,
        "penalty_scale": cfg.env.penalty_scale,
        "penalty_exp": cfg.env.penalty_exp,
        "random_reset": cfg.env.random_reset,
        "initial_angles": cfg.env.initial_angles,
        "reward_average_steps": cfg.env.reward_average_steps,
        "velocity_penalty_scale": cfg.env.velocity_penalty_scale,
        "difference_penalty_scale": cfg.env.difference_penalty_scale,
        "fluctuation_penalty_scale": cfg.env.fluctuation_penalty_scale,
    }

    # Create agent
    logging.info('Creating models')
    if not cfg.checkpoint.load_from_checkpoint:
        # Create a new model
        if cfg.multi_agent.use:
            actor, critic = make_ma_sac_agents(cfg, params)
        else:
            actor, critic = make_sa_sac_agent(cfg, params)
        logging.info(f"Created new models in {'multi' if cfg.multi_agent.use else 'single'} agent mode.")
    else:
        # Load from specified checkpoint
        # Copy the loaded models into the /checkpoints directory
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        for arch in ['actor', 'critic']:
            filename = f"{arch}_{cfg.checkpoint.model_checkpoint_id}.pkl"
            if os.path.exists(filename):
                shutil.move(filename, "checkpoints")
        # Load actor and critic
        actor, critic = load_model(
            cfg=cfg,
            env_params=params,
            id=cfg.checkpoint.model_checkpoint_id,
            path_to_model='checkpoints',
            )
        logging.info(f"Loaded models. Starting training from frame {cfg.checkpoint.model_checkpoint_id}.")
    actor, critic = actor.to(device), critic.to(device)

    # How many frames have already been collected (if loading from checkpoint)?
    collected_frames = 0 if not cfg.checkpoint.load_from_checkpoint else cfg.checkpoint.model_checkpoint_id

    # Create SAC loss
    logging.info('Creating loss module')
    loss_module, target_net_updater = make_loss_module(cfg, params, actor, critic)

    # Create off-policy collector
    logging.info('Creating collector')
    train_env = make_parallel_env(cfg, params, cfg.env.n_parallel, device=device)
    collector = make_collector(cfg, train_env, actor, device)

    # Create replay buffer
    logging.info('Creating replay buffer')
    replay_buffer = make_replay_buffer(cfg)

    # Create replay buffer for saving
    if cfg.checkpoint.save_replay_buffer:
        saving_buffer = make_saving_replay_buffer(cfg)

    # Create optimizers
    (
        optimizer_actor,
        optimizer_critic,
        optimizer_alpha,
    ) = make_sac_optimizer(cfg, loss_module)

    # Main loop
    start_time = time.time()
    # pbar = tqdm.tqdm(total=cfg.collector.total_frames // cfg.env.frame_skip)
    num_console_updates = 1000

    init_random_frames = cfg.collector.init_random_frames
    num_updates = frames_per_batch * cfg.optim.step_mult
    prb = cfg.replay_buffer.prb

    # Initial reset to burn in simulation
    logging.info(f'Initial reset: collecting {(1 + cfg.env.initial_reset_frames // cfg.env.reset_frames)*cfg.env.reset_frames} frames.')
    # Each reset is cfg.env.reset_frames, we want a total of cfg.env.initial_reset_frames many
    reset_td = collector.reset()
    for i in range(cfg.env.initial_reset_frames // cfg.env.reset_frames):
        reset_td = collector.reset(reset_td)
        logging.info(f"{100*i/(cfg.env.initial_reset_frames // cfg.env.reset_frames)}% done with initial reset.")
    logging.info(f"100% done with initial reset.")

    logging.info('Starting training...')
    sampling_start = time.time()
    logs = {}
    train_start_time = sampling_start
    for i, tensordict in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start

        if cfg.multi_agent.use:
            tensordict = update_data_shapes(train_env, tensordict)

        # Update weights of the inference policy
        collector.update_policy_weights_()

        # If desired, save data to extra buffer for saving purposes
        if cfg.checkpoint.save_replay_buffer:
            saving_buffer.extend(tensordict.transpose(0,1))

        # Flatten tensordict
        tensordict = tensordict.reshape(-1)
        current_frames = tensordict.numel()
        
        # Add to replay buffer
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames

        # Console update
        # pbar.update(tensordict.numel())
        if should_log_now(cfg, collected_frames, num_console_updates):
            console_output = f'Frame {collected_frames}/{cfg.collector.total_frames}'
            time_passed = time.time() - train_start_time
            console_output += f' | {time_passed/60:.0f} min' if time_passed/60 > 1 else f' | <1 min'
            logging.info(console_output)

        # Optimization steps
        training_start = time.time()
        if collected_frames >= init_random_frames:
            losses = TensorDict({}, batch_size=[num_updates])
            for j in range(num_updates):
                # Sample from replay buffer
                sampled_tensordict = replay_buffer.sample().clone()
                sampled_tensordict = sampled_tensordict.to(device)

                # Compute loss
                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]

                # Update actor
                optimizer_actor.zero_grad()
                actor_loss.backward(retain_graph=False)
                optimizer_actor.step()

                # Update critic
                optimizer_critic.zero_grad()
                q_loss.backward()
                optimizer_critic.step()

                # Update alpha
                optimizer_alpha.zero_grad()
                alpha_loss.backward()
                optimizer_alpha.step()

                losses[j] = loss_td.select(
                    "loss_actor", "loss_qvalue", "loss_alpha"
                ).detach()

                # Update qnet_target params
                target_net_updater.step()

                # Update priority
                if prb:
                    replay_buffer.update_tensordict_priority(sampled_tensordict)

        training_time = time.time() - training_start

        # Logging
        if cfg.multi_agent.use:
            episode_rewards = tensordict[("next", "agents", "episode_reward")][tensordict["next", "done"]]
            episode_length = tensordict["next", "step_count"][tensordict["next", "done"].all(-2)]
        else:
            episode_rewards = tensordict[("next", "episode_reward")][tensordict["next", "done"]]
            episode_length = tensordict["next", "step_count"][tensordict["next", "done"]]

        if len(episode_length) > 0:
            log_info.update(
                {
                    "train/episode_reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item() / len(episode_length),
                }
            )
        else:  # if no end of an episode is contained in the batch, fill the logs with NaN
            log_info.update(
                {
                    "train/episode_reward": np.nan,
                    "train/episode_length": np.nan,
                }
            )
        if collected_frames >= init_random_frames:
            log_info["train/q_loss"] = losses.get("loss_qvalue").mean().item()
            log_info["train/actor_loss"] = losses.get("loss_actor").mean().item()
            log_info["train/alpha_loss"] = losses.get("loss_alpha").mean().item()
            log_info["train/alpha"] = loss_td["alpha"].item()
            log_info["train/entropy"] = loss_td["entropy"].item()
            log_info["train/sampling_time"] = sampling_time
            log_info["train/training_time"] = training_time

        if i % cfg.checkpoint.checkpoint_interval == 0 or i >= total_frames // frames_per_batch:
            # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            output_dir = os.getcwd() + '/checkpoints/'
            if cfg.checkpoint.save_replay_buffer:
                torch.save(saving_buffer._storage._storage, output_dir + 'replay_buffer_checkpoint.pt')
                logging.info(f"Checkpointed replay buffer. (Saved at {output_dir + 'replay_buffer_checkpoint'}). #Timesteps/environment stored: {len(saving_buffer)}")
            save_model(cfg, actor, critic, output_dir, collected_frames)
            logging.info(f"Checkpointed model. (Saved at {output_dir}actor_{collected_frames}.pkl and {output_dir}critic_{collected_frames}.pkl")
            
        log_metrics(logs, log_info)
        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
