import warnings
warnings.filterwarnings("ignore")
from torch import multiprocessing
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal
from tqdm import tqdm
import pandas as pd


# Device configuration
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# Hyperparameters
num_cells = 256
lr = 3e-4
max_grad_norm = 1.0
frames_per_batch = 1000     
total_frames = 1_000_000

total_runs = 3

gamma = 0.99  # discount factor

# Environment list
envs = ['HumanoidStandup-v4', 'Hopper-v4', 'HalfCheetah-v4', 'InvertedDoublePendulum-v4', 'InvertedPendulum-v4',
        'Reacher-v4', 'Swimmer-v4', 'Walker2d-v4']

#envs = ['Swimmer-v4']

def compute_returns(rewards, dones, gamma):
    """
    Compute discounted returns for a trajectory.
    
    Args:
        rewards (Tensor): 1D tensor of rewards.
        dones (Tensor): 1D tensor of booleans indicating episode termination.
        gamma (float): Discount factor.
    
    Returns:
        Tensor: A tensor of discounted returns.
    """
    returns = torch.zeros_like(rewards)
    R = 0.0
    # Loop backwards over time steps
    for t in reversed(range(len(rewards))):
        if dones[t]:
            R = 0.0
        R = rewards[t] + gamma * R
        returns[t] = R
    return returns

for env_name in envs:
    for run_id in range(1, total_runs + 1):
        print(f'########## Starting run {run_id} for {env_name} ##########\n')

        base_env = GymEnv(env_name, device=device)

        # Name of the run
        run_name = f'VPG_{env_name}_{run_id}_{total_frames}'

        env = TransformedEnv(
            base_env,
            Compose(
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ),
        )
        env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
        check_env_specs(env)

        
        # Build the policy network (actor)
        
        actor_net = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
            NormalParamExtractor(),
        )

        policy_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )

        policy_module = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": env.action_spec.space.low,
                "high": env.action_spec.space.high,
            },
            return_log_prob=True,
        )

        # Initialize the lazy layers using a dummy input.
        dummy_input = env.reset()["observation"].unsqueeze(0)
        with torch.no_grad():
            policy_module(dummy_input)

        print("Actor network initialized.")

        
        # Data Collector
        
        collector = SyncDataCollector(
            env,
            policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
        )

        
        # Training Setup
        
        optim = torch.optim.Adam(policy_module.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, total_frames // frames_per_batch, 0.0
        )

        logs = defaultdict(list)
        pbar = tqdm(total=total_frames)

        logs["step_count"] = []
        logs["Max Step Count (Test)"] = []
        logs["eval reward"] = []
        logs["lr"] = []
        logs["reward"] = []
        logs["Return (Test)"] = []

        
        # Training Loop
        
        for i, tensordict_data in enumerate(collector):
            tensordict_data = tensordict_data.to(device)

            # Retrieve rewards from the correct key.
            rewards = tensordict_data.get(("next", "reward"))
            if rewards is None:
                raise ValueError("Rewards not found in tensordict under key ('next', 'reward').")

            # Retrieve dones from the tensordict if available.
            if ("next", "done") in tensordict_data.keys(include_nested=True):
                dones = tensordict_data.get(("next", "done"))
            else:
                dones = tensordict_data.get("step_count") == 1

            # Compute discounted returns for the batch.
            returns = compute_returns(rewards, dones, gamma)
            #optional normalizing
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            # Retrieve log_probs.
            log_probs = tensordict_data.get("log_prob")
            # If log_prob is missing, try to get nested key ("action", "log_prob")
            if log_probs is None:
                if ("action", "log_prob") in tensordict_data.keys(include_nested=True):
                    log_probs = tensordict_data.get(("action", "log_prob"))

            # If still not found, compute the log_probs manually.
            if log_probs is None:
                # Assume that the taken action is stored under key "action"
                actions = tensordict_data.get("action")
                # Get the distribution from the policy_module
                dist = policy_module.get_dist(tensordict_data)
                log_probs = dist.log_prob(actions)

            # Compute the vanilla policy gradient (REINFORCE) loss.
            loss = - (log_probs * returns).mean()

            # Backpropagation and parameter update.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

            # Log training rewards and other metrics.
            logs["reward"].append(rewards.mean().item())
            logs["step_count"].append(tensordict_data.get("step_count").max().item())
            logs["lr"].append(optim.param_groups[0]["lr"])

            # Evaluation every 10 iterations.
            if i % 10 == 0:
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    eval_rollout = env.rollout(1000, policy_module)
                    logs["eval reward"].append(
                        eval_rollout.get(("next", "reward")).mean().item()
                    )
                    logs["Return (Test)"].append(
                        eval_rollout.get(("next", "reward")).sum().item()
                    )
                    logs["Max Step Count (Test)"].append(
                        eval_rollout.get("step_count").max().item()
                    )
                eval_str = (
                    f"Eval cumulative reward: {logs['Return (Test)'][-1]:.4f} "
                    f"(init: {logs['Return (Test)'][0]:.4f}), "
                    f"Eval step-count: {logs['Max Step Count (Test)'][-1]}"
                )
                del eval_rollout

            cum_reward_str = f"Avg reward: {logs['reward'][-1]:.4f} (init: {logs['reward'][0]:.4f})"
            stepcount_str = f"Step count (max): {logs['step_count'][-1]}"
            lr_str = f"LR: {logs['lr'][-1]:.4f}"

            pbar.update(frames_per_batch)
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
            scheduler.step()
        pbar.close()

        
        # Plotting
        
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(logs["reward"])
        plt.title("Training Rewards (Average)")

        plt.subplot(2, 2, 2)
        plt.plot(logs["step_count"])
        plt.title("Max Step Count (Training)")

        plt.subplot(2, 2, 3)
        plt.plot(logs["Return (Test)"])
        plt.title("Return (Test)")

        plt.subplot(2, 2, 4)
        plt.plot(logs["Max Step Count (Test)"])
        plt.title("Max Step Count (Test)")

        plt.savefig(f'../runs/VPG/plots_{run_name}.jpg', dpi=150)
        # plt.show()

        
        # Saving the Model and Logs
        
        save_path = f"../runs/VPG/{run_name}.pth"
        checkpoint = {
            "policy_state_dict": policy_module.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "hyperparameters": {
                "num_cells": num_cells,
                "lr": lr,
                "gamma": gamma,
            }
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

        # Make sure all logs have the same length.
        max_length = max(len(logs[key]) for key in logs)
        for key in logs:
            while len(logs[key]) < max_length:
                logs[key].append(None)
        df_logs = pd.DataFrame.from_dict(logs)
        df_logs.to_csv(f"../runs/VPG/logs_{run_name}.csv", index=False)
        print("Logs saved")
