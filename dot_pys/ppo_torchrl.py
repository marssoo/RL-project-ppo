# imports
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
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

# setting parameters

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
num_cells = 256  # number of cells in each layer i.e. output dim.
lr = 3e-4
max_grad_norm = 1.0

frames_per_batch = 1000
# For a complete training, bring the number of frames up to 1M


#total_frames = 250_000
#total_frames = 10_000
total_frames = 1_000_000
total_runs = 3

# PPO parameters
sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

# env
#env_name = "Hopper-v4"

envs = ['HalfCheetah-v4', 'Hopper-v4', 'InvertedDoublePendulum-v4', 'InvertedPendulum-v4',
        'Reacher-v4', 'Swimmer-v4', 'Walker2d-v4']

for env_name in envs:
    for run_id in range(1, total_runs+1):
        print(f'########## Starting run {run_id} for {env_name} ##########\n')
        base_env = GymEnv(env_name, device=device)

        #run_name
        run_id = 1
        run_name = f'PPO_{env_name}_{run_id}_{total_frames}'

        #transforms and normalization

        env = TransformedEnv(
            base_env,
            Compose(
                # normalize observations
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ),
        )

        env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

        #print("normalization constant shape:", env.transform[0].loc.shape)

        #print("observation_spec:", env.observation_spec)
        #print("reward_spec:", env.reward_spec)
        #print("input_spec:", env.input_spec)
        #print("action_spec (as defined by input_spec):", env.action_spec)
        #
        check_env_specs(env)
        #
        #rollout = env.rollout(3)
        #print("rollout of three steps:", rollout)
        #print("Shape of the rollout TensorDict:", rollout.batch_size)

        # policy

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
            # we'll need the log-prob for the numerator of the importance weights
        )

        # value network

        value_net = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(1, device=device),
        )

        value_module = ValueOperator(
            module=value_net,
            in_keys=["observation"],
        )

        policy_module(env.reset())
        value_module(env.reset())

        # data collector

        collector = SyncDataCollector(
            env,
            policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
        )

        # replay buffer

        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=frames_per_batch),
            sampler=SamplerWithoutReplacement(),
        )

        # loss function

        advantage_module = GAE(
            gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
        )

        loss_module = ClipPPOLoss(
            actor_network=policy_module,
            critic_network=value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coef=entropy_eps,
            # these keys match by default but we set this for completeness
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
        )

        optim = torch.optim.Adam(loss_module.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, total_frames // frames_per_batch, 0.0
        )

        # training loop 

        logs = defaultdict(list)
        pbar = tqdm(total=total_frames)
        eval_str = ""

        # We iterate over the collector until it reaches the total number of frames it was
        # designed to collect:
        for i, tensordict_data in enumerate(collector):
            # we now have a batch of data to work with. Let's learn something from it.
            for _ in range(num_epochs):
                # We'll need an "advantage" signal to make PPO work.
                # We re-compute it at each epoch as its value depends on the value
                # network which is updated in the inner loop.
                advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                replay_buffer.extend(data_view.cpu())
                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = replay_buffer.sample(sub_batch_size)
                    loss_vals = loss_module(subdata.to(device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()
                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optim.step()
                    optim.zero_grad()

            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            pbar.update(tensordict_data.numel())
            cum_reward_str = (
                f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
            )
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            stepcount_str = f"step count (max): {logs['step_count'][-1]}"
            logs["lr"].append(optim.param_groups[0]["lr"])
            lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
            if i % 10 == 0:
                # We evaluate the policy once every 10 batches of data.
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    # execute a rollout with the trained policy
                    eval_rollout = env.rollout(1000, policy_module)
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["Return (test)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    logs["Max step count (test)"].append(eval_rollout["step_count"].max().item())
                    eval_str = (
                        f"eval cumulative reward: {logs['Return (test)'][-1]: 4.4f} "
                        f"(init: {logs['Return (test)'][0]: 4.4f}), "
                        f"eval step-count: {logs['Max step count (test)'][-1]}"
                    )
                    del eval_rollout
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))


            scheduler.step()

        # results
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(logs["reward"])
        plt.title("training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(logs["Return (test)"])
        plt.title("Return (test)")
        plt.subplot(2, 2, 4)
        plt.plot(logs["Max step count (test)"])
        plt.title("Max step count (test)")
        plt.savefig(f'../runs/PPO/plots_{run_name}.jpg', dpi=150)
        #plt.show()

        # save logs and model

        # Define save path
        save_path = f"../runs/PPO/{run_name}.pth"

        # Create a dictionary
        checkpoint = {
            "policy_state_dict": policy_module.state_dict(),
            "value_state_dict": value_module.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "hyperparameters": {
                "num_cells": num_cells,
                "lr": lr,
                "gamma": gamma,
                "lmbda": lmbda,
                "entropy_eps": entropy_eps,
            }
        }

        # Save checkpoint
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

        import pandas as pd

        # Find the maximum length of any logged metric
        max_length = max(len(logs[key]) for key in logs)

        # Pad all lists to match max_length
        for key in logs:
            while len(logs[key]) < max_length:
                logs[key].append(None)  # Fill with None (or np.nan)

        # Convert to DataFrame
        df_logs = pd.DataFrame.from_dict(logs)

        # Save to CSV
        df_logs.to_csv(f"../runs/PPO/logs_{run_name}.csv", index=False)

        print("Logs saved")