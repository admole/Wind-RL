import os
import sys
sys.path.append(os.getcwd())
import time
import copy
import logging
import torch.optim
import tqdm
import math
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import generate_exp_name
from utils_ppo import make_parallel_env, make_ma_ppo_models, make_sa_ppo_models, load_model, save_model, log_metrics, update_data_shapes
import shutil
import hydra
import numpy as np
import shutil


@hydra.main(config_path="./", config_name="config_ppo", version_base="1.2")
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
    max_episode_length = cfg.collector.max_episode_length
    mini_batch_size = cfg.loss.mini_batch_size
    n_environments = cfg.env.n_parallel

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

    # Create models
    logging.info('Creating models')
    if not cfg.checkpoint.load_from_checkpoint:
        # Create a new model
        if cfg.multi_agent.use:
            actor, critic = make_ma_ppo_models(cfg, params)
        else:
            actor, critic = make_sa_ppo_models(cfg, params)
    else:
        # Load from specified checkpoint
        # Copy the loaded models into the ppo/checkpoints directory
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

    # Create environments
    logging.info(f'Creating {n_environments} parallel environments')
    train_env = make_parallel_env(cfg, params, n_environments, device=device)

    # How many frames have already been collected (if loading from checkpoint)?
    collected_frames = 0 if not cfg.checkpoint.load_from_checkpoint else cfg.checkpoint.model_checkpoint_id

    # Create collector
    logging.info(f'Creating data collector')
    collector = SyncDataCollector(
        train_env,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames-collected_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=max_episode_length
    )

    # Create data buffer
    logging.info(f'Creating data buffer')
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch),
        sampler=sampler,
        batch_size=mini_batch_size,
    )

    # Create replay buffer to remember entire history
    if cfg.checkpoint.save_replay_buffer:
        full_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(math.ceil(total_frames / n_environments)),
        )

    # Create loss and adv modules
    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=False,
    )
    if cfg.multi_agent.use:
        loss_module.set_keys(  # We have to tell the loss where to find the keys
            reward=train_env.reward_key,
            action=train_env.action_key,
            sample_log_prob=("agents", "sample_log_prob"),
            value=("agents", "state_value"),
            # These last 2 keys will be expanded to match the reward shape
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )

    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )
    if cfg.multi_agent.use:
        adv_module.set_keys(
            value=("agents", "state_value"),
            reward=train_env.reward_key,
        )

    # Create optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.optim.lr, eps=1e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.optim.lr, eps=1e-5)

    # Main loop
    start_time = time.time()
    #pbar = tqdm.tqdm(total=total_frames)
    num_mini_batches = frames_per_batch // mini_batch_size
    num_network_updates = (collected_frames // frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches
    total_network_updates = (total_frames // frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches

    # Initial reset
    logging.info(f'Initial reset: collecting {(cfg.env.initial_reset_frames // cfg.env.reset_frames)*cfg.env.reset_frames} frames.')
    # Each reset is cfg.env.reset_frames, we want a total of cfg.env.initial_reset_frames many
    reset_td = collector.reset()
    for i in range(cfg.env.initial_reset_frames // cfg.env.reset_frames):
        reset_td = collector.reset(reset_td)
        logging.info(f"{100*i/(cfg.env.initial_reset_frames // cfg.env.reset_frames)}% done with initial reset.")
    logging.info(f"100% done with initial reset.")

    sampling_start = time.time()

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = cfg.optim.lr
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    losses = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    # Set up empty dict for logging
    logs = {}

    logging.info(f'Starting main loop')
    for i, data in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        #pbar.update(data.numel())
        logging.info(f"Training step {collected_frames}/{total_frames}.")

        if cfg.multi_agent.use:
            data = update_data_shapes(train_env, data)

        # Get training rewards and episode lengths
        if cfg.multi_agent.use:
            episode_rewards = data[("next", "agents", "episode_reward")][data["next", "done"]]
            episode_length = data["next", "step_count"][data["next", "done"].all(-2)]
        else:
            episode_rewards = data[("next", "episode_reward")][data["next", "done"]]
            episode_length = data["next", "step_count"][data["next", "done"]]

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

        # Save to full buffer
        if cfg.checkpoint.save_replay_buffer:
            full_buffer.extend(data.transpose(0,1))

        training_start = time.time()
        for j in range(cfg_loss_ppo_epochs):

            # Compute GAE
            with torch.no_grad():
                data = adv_module(data)
            data_reshape = data.reshape(-1)

            # Update the data buffers
            data_buffer.extend(data_reshape)

            for k, batch in enumerate(data_buffer):

                # Get a data batch
                batch = batch.to(device)

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                if cfg_optim_anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in actor_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                    for group in critic_optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                if cfg_loss_anneal_clip_eps:
                    loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
                num_network_updates += 1

                # Exponential decay for the entropy loss
                beta = 1.0
                if cfg.optim.anneal_entropy:
                    decay_rate = -np.log(cfg.optim.anneal_entropy_scale)/cfg.optim.anneal_entropy_step
                    beta = np.exp(- decay_rate * collected_frames)

                # Forward pass PPO loss
                loss = loss_module(batch)
                loss.set("loss_entropy", beta * loss.get("loss_entropy"))
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
                critic_loss = loss.get("loss_critic")
                actor_loss = loss.get("loss_objective") + loss.get("loss_entropy")

                # Backward pass
                actor_loss.backward()
                critic_loss.backward()

                # Update the networks
                actor_optim.step()
                critic_optim.step()
                actor_optim.zero_grad()
                critic_optim.zero_grad()

        # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/lr": alpha * cfg_optim_lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/clip_epsilon": alpha * cfg_loss_clip_epsilon
                if cfg_loss_anneal_clip_eps
                else cfg_loss_clip_epsilon,
                "step": collected_frames,
            }
        )

        if i % cfg.checkpoint.checkpoint_interval == 0 or i >= total_frames // frames_per_batch:
            # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + '/'
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            output_dir = os.getcwd() + '/checkpoints/'
            if cfg.checkpoint.save_replay_buffer:
                torch.save(full_buffer._storage._storage, output_dir + 'replay_buffer_checkpoint.pt')
                logging.info(f"Checkpointed replay buffer. (Saved at {output_dir + 'replay_buffer_checkpoint'}). #Timesteps/environment: {len(full_buffer)}")
            save_model(cfg, actor, critic, output_dir, collected_frames)
            logging.info(f"Checkpointed model. (Saved at {output_dir}actor_{collected_frames}.pkl and {output_dir}critic_{collected_frames}.pkl")
            
        log_metrics(logs, log_info)
        collector.update_policy_weights_()
        sampling_start = time.time()

    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
