from collections import defaultdict
from typing import Optional

import numpy as np
import math
import torch
import tqdm
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from torch import nn
import time

from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.envs.utils import check_env_specs, step_mdp
from smartredis import Client
import logging


def ring_insert_shift(buffer: torch.Tensor, new_value: float) -> None:
    """
    Inserts `new_value` into `buffer` (size N) by shifting all elements left.
    The oldest value (buffer[0]) is dropped; new_value goes to buffer[-1].
    """
    # Shift all elements to the left
    buffer[:-1] = buffer[1:].clone()
    # Insert the new value at the end
    buffer[-1] = new_value
    return buffer


class TurbEnv(EnvBase):
    metadata = {}
    batch_locked = False

    def __init__(self,
                 params=None,
                 multi_agent=True,
                 seed=None,
                 save=False,
                 instance=None,
                 device="cpu",
                 ):

        super().__init__(device=device, batch_size=[])

        self.save = save
        self.probes_per_turbine = params["probes_per_turbine"]
        lookup = {"ux": 0, "uy": 1, "uz": 2}
        if not all(p in lookup.keys() for p in params["flow_field_directions"]):
            raise ValueError(f"The parameter 'flow_field_directions' must be a list containing only the elements 'ux', 'uy', 'uz'. Got {params['flow_field_directions']}")
        if not len(set(params["flow_field_directions"])) == len(params["flow_field_directions"]):
            raise ValueError(f"The parameter 'flow_field_directions' must not contain duplicates. Got {params['flow_field_directions']}.")
        self.obs_idxs = list(map(lookup.get, params["flow_field_directions"]))  # nice
        self.obs_per_probe = len(self.obs_idxs)
        self.obs_per_turbine = self.probes_per_turbine * self.obs_per_probe
        self.n_turbs = params["n_turbines"]
        self.total_probes = self.probes_per_turbine * self.n_turbs
        self.total_obs = self.obs_per_turbine * self.n_turbs
        self.max_speed = params["max_yaw_speed"]  # maximum angular velocity of wind turbine
        self.max_angle = params["max_yaw_angle"]
        self.dt = params["dt"]
        self.reset_frames = params["reset_frames"]
        self.instance = 0 if instance is None else instance+1
        self.penalty_scale = params["penalty_scale"]
        self.penalty_exponent = params["penalty_exp"]
        self.random_reset = bool(params["random_reset"])
        self.initial_angles = params["initial_angles"]
        if not len(self.initial_angles) == self.n_turbs:
            raise ValueError(f'Number of initial angles provided {len(self.initial_angles)}'
                             f'does not match number of number of turbines {self.n_turbs}')
        self.multi_agent = multi_agent  # If True, using multi agent, else use single agent
        self.reward_average_steps = params["reward_average_steps"]
        self.velocity_penalty_scale = params["velocity_penalty_scale"]
        self.difference_penalty_scale = params["difference_penalty_scale"]
        self.fluctuation_penalty_scale = params["fluctuation_penalty_scale"]

        # Create client
        self.client = Client(address=None, cluster=False)
        self.client.put_tensor(f"{self.instance}_yaws_done", np.array([0]))
        self.client.put_tensor(f"{self.instance}_sim_done", np.array([0]))

        # Set up spec for multi vs single agent
        if self.multi_agent:
            self._make_spec_ma()
        else:
            self._make_spec_sa()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)

        # Set up keys for single vs multi agent
        if self.multi_agent:
            self.env_action_key = ("agents", "action")
            self.env_alpha_key = ("agents", "alpha")
        else:
            self.env_action_key = "action"
            self.env_alpha_key = "alpha"

    @staticmethod
    def _normalise_probe_data(arr):
        # Each probe contains (U, V, W) = (u_x, u_y, u_z)
        # The range is typically 0 < U < 12, -4 < V,W < 4.
        loc = np.array([6., 0., 0.])
        scale = np.array([6., 4., 4.])
        return (arr-loc)/scale
    
    @staticmethod
    def _position_encoding(n_turbs):
        idxs = torch.arange(n_turbs).view([n_turbs, 1])
        idxs = (torch.pi/2) * idxs / (n_turbs-1)
        return torch.sin(idxs)

    def _communicate(self, new_alpha):
        ######### Communication with SmartRedis server ###########
        # Send yaws to X3D
        self.client.put_tensor(f"{self.instance}_yaws", new_alpha.detach().cpu().numpy().squeeze().astype(np.float64))
        # Set i_yaws_done flag to True (one)
        self.client.put_tensor(f"{self.instance}_yaws_done", np.ones(1)) # setting one as True

        # print(f"Set key {self.instance}_yaws_done to True.")
        # Poll whether X3D simulation is done
        while not self.client.poll_key(f'{self.instance}_sim_done', 100, 10) or not bool(self.client.get_tensor(f"{self.instance}_sim_done")[0]):
            continue

        # Set i_sim_done flag to false (zero)
        # print(f"Setting key {self.instance}_sim_done to False inside of communicate now.")
        self.client.put_tensor(f"{self.instance}_sim_done", np.array([0]))

        turbine_powers = self.client.get_tensor(f"{self.instance}_turbine_powers")  # [n_turbs]
        turbine_obs = np.zeros((self.total_probes, self.obs_per_probe))
        for i in range(self.total_probes):
            probe = self.client.get_tensor(f"{self.instance}_probe_{i+1}")
            turbine_obs[i] = self._normalise_probe_data(probe)[self.obs_idxs]  # [n_turbs*probes_per_turbine, obs_per_probe]
        turbine_obs = turbine_obs.reshape(self.n_turbs, self.probes_per_turbine, self.obs_per_probe)
        turbine_obs = turbine_obs.reshape(self.n_turbs, self.probes_per_turbine * self.obs_per_probe)  # [n_turbs, probes_per_turbine*obs_per_probe]

        # Process the outputs from the solver
        turbine_powers /= 1e06
        # farm_power = turbine_powers.mean(axis=-1)  # Compute mean over turbines
        # farm_power = np.broadcast_to(farm_power, (self.n_turbs,))  # repeat farm power for all turbines
        # farm_power = turbine_powers.mean(axis=-1)  # Compute mean over iterations many time steps
        # turbine_obs = turbine_obs.mean(axis=-1)  # Compute mean over iterations many time steps

        # Convert to Torch tensors
        power = torch.tensor(turbine_powers, dtype=torch.float32).to(self.device)
        observation = torch.tensor(turbine_obs, dtype=torch.float32).to(self.device)

        return power, observation

    def _step(self, tensordict):
        # All tensors are expected to be of shape [*batch_size, num_turbs, X]
        # where X = num_obs_per_turbine for observation
        #       X = num_actions_per_turbine for action
        #       X = 1 for reward

        # Retrieve action and previous alpha from tensordict
        action = tensordict.get(self.env_action_key)
        alpha = tensordict.get(self.env_alpha_key)
        old_power = tensordict.get(("next", "power"))
        u = action
        # u = u.clamp(-1., 1.)  # this should happen automatically
        u = u * self.max_speed  # since the actor outputs values between -1 and 1, correct scale here
        
        # Compute new alpha
        new_alpha = alpha + u * self.dt
        new_alpha = new_alpha.clamp(-self.max_angle, self.max_angle)

        # Retrieve power and observations
        power, observation = self._communicate(new_alpha)

        # Compute various penalty terms
        angle_penalty = torch.mean(self.penalty_scale * (new_alpha.squeeze()/self.max_angle)**(self.penalty_exponent))
        velocity_penalty = self.velocity_penalty_scale * torch.mean((u / self.max_speed)**2)
        difference_penalty = self.difference_penalty_scale * torch.mean((power.flatten().unsqueeze(0) - power.flatten().unsqueeze(1))**2)
        fluctuation_penalty = self.fluctuation_penalty_scale * torch.mean((power-old_power)**2)

        # power contains the individual turbine powers
        # take the mean over those to get the farm power (and keep scaling)
        power = torch.mean(power, axis=-1)

        # Compute instanteneous reward
        reward = power - angle_penalty - velocity_penalty - difference_penalty - fluctuation_penalty

        # reward now has the instanteneous reward
        # Next, we average the reward over a certain number of steps in the past
        reward_buffer = tensordict.get("reward_buffer")
        reward_buffer = ring_insert_shift(reward_buffer, reward.item())
        # compute mean but be careful in the beginning, when the buffer is filled with -1.0 as default value
        time_averaged_reward = torch.mean(reward_buffer[reward_buffer != -1.0])
        # expand the reward to be of shape (n_turbs,)
        time_averaged_reward = torch.broadcast_to(time_averaged_reward, (self.n_turbs,))
        power = torch.broadcast_to(power, (self.n_turbs,))
        # forget the instanteneous reward and use the time average
        reward = time_averaged_reward

        if len(reward.shape) < len(tensordict.shape) + 2:
            reward = reward.unsqueeze(dim=-1)
            power = power.unsqueeze(dim=-1)
        done = torch.zeros((*tensordict.shape, 1), dtype=torch.bool)

        pos_enc = self._position_encoding(self.n_turbs)

        if self.multi_agent:
            out = self._make_tensordict_ma(done, new_alpha, observation, reward, power, pos_enc)
        else:
            out = self._make_tensordict_sa(done, new_alpha, observation, reward, power, pos_enc, reward_buffer)

        # print(f"Hello again. I am instance {self.instance} This is the time at END of step: {time.time():.4f}.")

        return out

    def _reset(self, tensordict):
        logging.info(f"Resetting now")

        if self.random_reset:
            # Choose a set of random angles
            reset_angles = 0.75 * self.max_angle * (2 * torch.rand([self.n_turbs]) - 1)
        else:
            # Set angles to all 0
            reset_angles = torch.tensor(self.initial_angles)
        steps_to_change_angle = math.ceil(self.max_angle / (self.max_speed * self.dt))
        if tensordict is not None:
            alpha = tensordict.get(self.env_alpha_key).squeeze()
        else:
            alpha = torch.tensor(self.initial_angles)

        if not steps_to_change_angle <= self.reset_frames:
            raise ValueError(f"Must have at least {steps_to_change_angle} many reset frames. Only have {self.reset_frames}.")

        for i in range(steps_to_change_angle):
            interpolated_angle = (steps_to_change_angle-1-i)/(steps_to_change_angle-1)*alpha + i/(steps_to_change_angle-1)*reset_angles
            _, _ = self._communicate(new_alpha=interpolated_angle)
            
        for _ in range(self.reset_frames - steps_to_change_angle):
            _, _ = self._communicate(new_alpha=reset_angles)

        # for non batch-locked envs, the input tensordict shape dictates the number
        # of simulators run simultaneously. In other contexts, the initial
        # random state's shape will depend upon the environment batch-size instead.
        alpha = torch.zeros((*self.batch_size, self.n_turbs, 1), device=self.device)
        alpha[:, 0] = torch.tensor(self.initial_angles)
        observation = torch.zeros((*self.batch_size, self.n_turbs, self.obs_per_turbine), device=self.device)
        pos_enc = self._position_encoding(self.n_turbs)
        reward_buffer = -1.0 * torch.ones((*self.batch_size, self.reward_average_steps))

        if self.multi_agent:
            out = self._make_tensordict_ma(None, alpha, observation, None, None, pos_enc, reward_buffer)
        else:
            out = self._make_tensordict_sa(None, alpha, observation, None, None, pos_enc, reward_buffer)

        return out

    def _make_agent_spec(self):
        # Under the hood, this will populate self.output_spec["observation"]
        observation_spec = CompositeSpec(
                alpha=BoundedTensorSpec(
                    low=-self.max_angle,
                    high=self.max_angle,
                    shape=(*self.batch_size, 1),
                    dtype=torch.float32,
                    device=self.device
                ),
                pos_enc=UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size, 1),
                    dtype=torch.float32,
                    device=self.device
                ),
                observation=UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size, self.obs_per_turbine),
                    dtype=torch.float32,
                    device=self.device
                ),
                power=UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size, 1),
                    dtype=torch.float32,
                    device=self.device
                ),
                # we need to add the "params" to the observation specs, as we want
                # to pass it at each step during a rollout
                # params=self.make_composite_from_td(td_params["params"]),
                shape=(),
                device=self.device
                )
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        action_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(*self.batch_size, 1),
            dtype=torch.float32,
            device=self.device
        )
        reward_spec = UnboundedContinuousTensorSpec(
            shape=(*self.batch_size, 1),
            dtype=torch.float32,
            device=self.device
        )
        power_spec = UnboundedContinuousTensorSpec(
            shape=(*self.batch_size, 1),
            dtype=torch.float32,
            device=self.device
        )
        return action_spec, reward_spec, power_spec, observation_spec

    def _make_spec_ma(self):
        action_specs = []
        observation_specs = []
        reward_specs = []
        power_specs = []
        for _ in range(self.n_turbs):
            agent_i_action_spec, agent_i_reward_spec, agent_i_power_spec, agent_i_observation_spec = self._make_agent_spec()
            action_specs.append(agent_i_action_spec)
            reward_specs.append(agent_i_reward_spec)
            power_specs.append(agent_i_power_spec)
            observation_specs.append(agent_i_observation_spec)
        self.action_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {"action": torch.stack(action_specs, dim=0)}, shape=(self.n_turbs,)
                )
            }
        )
        self.reward_spec = CompositeSpec(
            {
                "agents": CompositeSpec(
                    {
                        "reward": torch.stack(reward_specs, dim=0),
                        "power": torch.stack(power_specs, dim=0)  # this is currently broken... needs to be done differently
                    }, 
                    shape=(self.n_turbs,),
                )
            }
        )
        self.observation_spec = CompositeSpec(
            {
                "agents": torch.stack(observation_specs, dim=0),
            },
            #shape=(self.n_turbs,)
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, EnvBase expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()

    def _make_spec_sa(self):
        observation_spec = CompositeSpec(
                alpha=BoundedTensorSpec(
                    low=-self.max_angle,
                    high=self.max_angle,
                    shape=(*self.batch_size, self.n_turbs),
                    dtype=torch.float32,
                    device=self.device
                ),
                observation=UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size, self.obs_per_turbine*self.n_turbs),
                    dtype=torch.float32,
                    device=self.device
                ),
                reward_buffer=UnboundedContinuousTensorSpec(
                    shape=(*self.batch_size, self.reward_average_steps),
                    dtype=torch.float32,
                    device=self.device
                ),
                shape=(),
                device=self.device
                )
        # action-spec will be automatically wrapped in input_spec when
        # `self.action_spec = spec` will be called supported
        action_spec = BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(*self.batch_size, self.n_turbs),
            dtype=torch.float32,
            device=self.device
        )
        reward_spec = CompositeSpec(
            reward=UnboundedContinuousTensorSpec(
                shape=(*self.batch_size, 1),
                dtype=torch.float32,
                device=self.device
            ),
            power=UnboundedContinuousTensorSpec(
                shape=(*self.batch_size, 1),
                dtype=torch.float32,
                device=self.device
            ),
            shape=(),
            device=self.device
        )

        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.observation_spec = observation_spec
        self.state_spec = self.observation_spec.clone()

    def _set_seed(self, seed: Optional[int]):
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        self.rng = rng

    def _make_tensordict_ma(self, done, new_alpha, observation, reward, power, pos_enc):
        source = {}
        if done is not None:
            source.update({"done": done})
        agent_tds = []
        for i in range(self.n_turbs):
            agent_out = {
                "alpha": new_alpha[..., i, :],
                "observation": observation[..., i, :],
                "pos_enc": pos_enc[..., i, :],
            }
            if reward is not None:
                agent_out.update({"reward": reward[..., i, :]})
            if power is not None:
                agent_out.update({"power": power[..., i, :]})
            agent_out = TensorDict(agent_out, ())
            agent_tds.append(agent_out)

        # agent_tds = torch.stack(agent_tds, dim=1)
        agent_tds = torch.stack(agent_tds)
        agent_tds = agent_tds.to_tensordict()
        source.update({"agents": agent_tds})

        out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )

        return out
    
    def _make_tensordict_sa(self, done, new_alpha, observation, reward, power, pos_enc, reward_buffer):
        source = {}
        batch_size = new_alpha.shape[:-2]
        if done is not None:
            source.update({"done": done})
        if reward is not None:
            source.update({"reward": reward[..., 0, :]})
        if power is not None:
            source.update({"power": power[..., 0, :]})
        source.update({
            "alpha": new_alpha.view(*batch_size, -1),
            "observation": observation.view(*batch_size, -1),
            "reward_buffer": reward_buffer.view(*batch_size, -1),
        })
        out = TensorDict(
            source=source,
            batch_size=self.batch_size,
            device=self.device,
        )

        return out

