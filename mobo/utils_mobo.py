import torch
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from pyDOE2 import lhs
from tqdm import tqdm
import pickle
import warnings
import logging

from torchrl.envs import (
    CatFrames,
    RewardSum,
    StepCounter,
    InitTracker,
    FiniteTensorDictCheck,
    TransformedEnv,
    ParallelEnv,
    Compose,
    ObservationNorm,
)


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def transforms(cfg, eval_only=False):
    observation_key = "observation"
    alpha_key = "alpha"
    alpha_norm_key = "alpha_normalised"
    transform_list = [
        InitTracker(),
        RewardSum(in_keys=["reward", "power"], reset_keys=["_reset", "_reset"]),  # episodic reward and power
        FiniteTensorDictCheck(),
        ObservationNorm(
            loc=0.,
            scale=(4/cfg.env.max_yaw_angle),
            in_keys=[alpha_key],
            out_keys=[alpha_norm_key]
        )
    ]
    if eval_only:
        transform_list.append(StepCounter(cfg.eval.episode_length))
    transforms = Compose(*transform_list)
    return transforms


def make_env(cfg, params, instance=None, save=False, device="cpu", add_transforms=True, eval_only=False):
    from WF_enviroment import TurbEnv
    env = TurbEnv(params, multi_agent=False, save=save, instance=instance, device=device)
    if add_transforms:
        env = TransformedEnv(env, transforms(cfg, eval_only))
    if eval_only:
        env.eval()
    return env


def make_parallel_env(cfg, params, num_envs, device="cpu", eval_only=False):
    function_list = [lambda i=i: make_env(cfg, params, instance=i, device=device, eval_only=eval_only) for i in
                     range(num_envs)]
    env = ParallelEnv(num_envs, function_list)
    return env

def rainflow_count(data):
    """
    Perform rainflow counting on a time series.
    Calculates an array of cycle amplitudes.
    Uses this to determine the damage equivalent load
    """
    # Extract turning points (peaks and valleys)
    # print(data.shape)
    cycles = []
    # for series in data:
    series = data
    tp = [series[0]]
    for i in range(1, len(series)-1):
        if (series[i] - series[i-1]) * (series[i+1] - series[i]) < 0:
            tp.append(series[i])
    tp.append(series[-1])
    tp = np.array(tp)

    stack = []

    for x in tp:
        stack.append(x)
        while len(stack) >= 3:
            s0, s1, s2 = stack[-3:]
            range1 = abs(s1 - s0)
            range2 = abs(s2 - s1)
            if range1 <= range2:
                cycles.append(range1)
                stack.pop(-2)  # remove middle element
            else:
                break

    # Remaining half cycles
    for i in range(len(stack) - 1):
        cycles.append(abs(stack[i+1] - stack[i]) / 2)

    m = 10

    rain = np.array(cycles)
    damage = np.sum((rain) ** m / len(rain)) ** (1/m)

    return damage


class BOTrainer:
    def __init__(self, env, n_initial_samples, episode_steps, burnin_steps, beta, reset_iterations,
                 bound=40., save_directory_path=None, mo_weight=0.0):
        self.N_DIMS = env.action_spec.shape[-1]
        self.env = env
        # TO-DO: figure out the lower and upper bounds for all the angles!
        # Consider that the flaps cannot touch and cannot overextend
        self.lower_bound = torch.zeros(self.N_DIMS)
        self.upper_bound = torch.ones(self.N_DIMS) * bound
        self.bounds_diff = self.upper_bound - self.lower_bound
        self.episode_time = episode_steps
        self.burnin_steps = burnin_steps
        self.total_steps = self.episode_time + self.burnin_steps
        self.beta = beta
        self.n_initial_samples = n_initial_samples
        self.save_directory = save_directory_path
        self.reset_iterations = reset_iterations
        self.mo_weight = mo_weight
        if save_directory_path is None:
            warnings.warn(f"save_directory_path parameter has not been specified. Results will not be saved.")

        logging.info(
            f'Using lower bounds {list(self.lower_bound.numpy())} and upper bounds {list(self.upper_bound.numpy())}.')

    def initialize_observations(self, save=True, reset_env=True):
        # Initialize observations
        lhs_samples = lhs(self.N_DIMS, samples=self.n_initial_samples)
        train_x = torch.tensor(lhs_samples, dtype=torch.float64).mul(self.bounds_diff).add(self.lower_bound)
        train_y = []
        for i in range(train_x.shape[0]):
            if (i % self.reset_iterations == 0) and reset_env:
                logging.info(f"Resetting environment...")
                self.env.reset()
            angles = train_x[i]
            new_y, info = self.experiment(angles)
            train_y.append(new_y)
            if save and self.save_directory is not None:
                self.save(self.save_directory, train_x, train_y, info, user_warnings=False)

            # Console logs
            console_output = f"Iteration {i}: "
            console_output += "Angles "
            console_output += ' / '.join([f"{v:.3f}" for v in angles.numpy()])
            console_output += ' | '
            # console_output += f"Value {float(train_y[-1]):.5f}"
            logging.info(console_output)

        train_y = torch.tensor(np.asarray(train_y)).unsqueeze(-1)

        return train_x, train_y

    def train_one_step(self, train_x, train_y):
        scaled_y = (train_y - train_y.mean()) / (train_y.std())
        scaled_x = (train_x - self.lower_bound) / self.bounds_diff

        # The model seems to converge better when we don't scale train_y...
        models = []
        for i in range(train_y.shape[-1]):
            train_y_i = train_y[..., i]
            models.append(SingleTaskGP(scaled_x, train_y_i))  # scaled_y
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        qnehvi_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            # ref_point=train_y.min(dim=0).values - 0.1,
            ref_point = torch.tensor([0.0, 0.0], dtype=torch.double),
            X_baseline=scaled_x,
            prune_baseline=True,
            sampler=qnehvi_sampler,
        )


        # Define and optimise acquisition function
        # UCB = UpperConfidenceBound(model, beta=self.beta)
        unit_bounds = torch.stack([torch.zeros(self.N_DIMS), torch.ones(self.N_DIMS)])
        new_x_scaled, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=unit_bounds,
            q=1,
            num_restarts=6,
            raw_samples=self.n_initial_samples,
            sequential=True,
        )
        # Run experiment
        new_x = self.bounds_diff * new_x_scaled + self.lower_bound
        new_y, info = self.experiment(new_x.flatten())

        new_y = new_y.view([1, 2, 1])
        # Add to training data
        new_train_x = torch.cat([train_x, new_x])
        new_train_y = torch.cat([train_y, new_y])

        return new_train_x, new_train_y, info

    def train(self, init_train_x, init_train_y, optimisation_iterations, save=True):
        train_x = init_train_x
        train_y = init_train_y

        # Bayesian Optimization loop
        for i in range(optimisation_iterations):
            if i % self.reset_iterations == 0:
                logging.info("Resetting environment...")
                self.env.reset()

            self.beta -= self.beta / optimisation_iterations
            train_x, train_y, info = self.train_one_step(train_x, train_y)
            if save and self.save_directory is not None:
                self.save(self.save_directory, train_x, train_y, info, user_warnings=False)

            # Console logs
            console_output = f"Iteration {i}: "
            console_output += "Angles "
            console_output += ' / '.join([f"{v:.3f}" for v in train_x[-1].numpy()])
            console_output += ' | '
            # console_output += f"Value {float(train_y[-1]):.5f}"
            logging.info(console_output)

        return train_x, train_y

    def experiment(self, angles):
        # Take repeated env steps and save all data received
        rewards = []
        observations = []
        num_envs = self.env.batch_size[0]

        for _ in range(self.total_steps):

            all_angles = angles.expand(num_envs, -1)
            actions = torch.zeros_like(all_angles)
            action_td = TensorDict({"action": actions, "alpha": all_angles}, batch_size=[num_envs])

            next_td = self.env.step(action_td)
            reward = next_td["next", "power"].mean()
            observation = next_td["next", "observation"].mean(0)
            rewards.append(reward)
            observations.append(observation)

        # Discard the initial burn in steps
        cropped_rewards = rewards[self.burnin_steps:]

        # Extract mean (and an estimate of measurement noise?)
        mean_reward = torch.tensor(np.mean(np.asarray(cropped_rewards)))
        damage = torch.tensor(rainflow_count(np.asarray(cropped_rewards)))
        # Think this damage should be computed per turbine then averaged
        # rather than calculated from the wind farm power series (mean 12 up)

        # mean_reward -= self.mo_weight * damage
        combined_reward = torch.stack((mean_reward, -damage), dim=-1)

        # Store any desired extra information in an info dictionary
        info = {'rewards': rewards, 'observations': observations}

        return combined_reward, info

    @staticmethod
    def save(path, train_x, train_y, info, user_warnings=True):
        import os
        if not (os.path.exists(path) and os.path.isdir(path)):
            os.mkdir(path)
            if user_warnings:
                warnings.warn(f"Directory {path} did not exist. Created this directory.")

        # Save angles and episodic rewards
        filepath = os.path.join(path, "BO_output.pkl")
        if user_warnings:
            if os.path.exists(filepath) and os.path.isfile(filepath):
                warnings.warn(f"{filepath} exists and will be overwritten.")
        output_dict = {'train_x': train_x, 'train_y': train_y, 'info': info}

        with open(filepath, 'wb') as f:
            pickle.dump(output_dict, f)

        # Save pressure observations
        filepath = os.path.join(path, "pressure_logs.dat")
        if user_warnings:
            if os.path.exists(filepath) and os.path.isfile(filepath):
                warnings.warn(f"{filepath} exists and will be overwritten.")
        with open(filepath, 'ab') as f:
            np.savetxt(f, info['observations'])

        # Save rewards
        filepath = os.path.join(path, "reward_logs.dat")
        if user_warnings:
            if os.path.exists(filepath) and os.path.isfile(filepath):
                warnings.warn(f"{filepath} exists and will be overwritten.")
        with open(filepath, 'ab') as f:
            np.savetxt(f, info['rewards'])

        return None
