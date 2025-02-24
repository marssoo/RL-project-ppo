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
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives.value import GAE
from tqdm import tqdm
import pandas as pd

# Device selection
is_fork = torch.multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# Hyperparameters
num_cells = 256
value_lr = 3e-4        # learning rate for value function updates
frames_per_batch = 1000

total_frames = 1_000_000
#total_frames = 40_000
total_runs = 3

gamma = 0.99
lmbda = 0.95

# TRPO-specific hyperparameters
max_kl = 1e-2                # trust region constraint
cg_iters = 10                # number of conjugate gradient iterations
cg_damping = 1e-2            # damping factor for Fisher-vector product
line_search_backtracks = 10  # number of backtracking steps in line search
line_search_accept_ratio = 0.1

# List of environments
envs = ['HumanoidStandup-v4', 'Hopper-v4', 'HalfCheetah-v4', 'InvertedDoublePendulum-v4', 'InvertedPendulum-v4',
        'Reacher-v4', 'Swimmer-v4', 'Walker2d-v4']

# ---- Helper Functions ----
def flat_params(model):
    return torch.cat([p.view(-1) for p in model.parameters()])

def set_flat_params(model, flat_vec):
    prev_ind = 0
    for p in model.parameters():
        flat_size = p.numel()
        p.data.copy_(flat_vec[prev_ind:prev_ind+flat_size].view_as(p))
        prev_ind += flat_size

def flat_grad(loss, model, retain_graph=False, create_graph=False):
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True
    )
    grad_list = []
    for grad, p in zip(grads, params):
        if grad is None:
            grad_list.append(torch.zeros_like(p))
        else:
            grad_list.append(grad.contiguous().view(-1))
    return torch.cat(grad_list)

def conjugate_gradient(f_Ax, b, nsteps, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    r_dot_r = torch.dot(r, r)
    for i in range(nsteps):
        Ap = f_Ax(p)
        alpha = r_dot_r / (torch.dot(p, Ap) + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_r_dot_r = torch.dot(r, r)
        if new_r_dot_r < residual_tol:
            break
        beta = new_r_dot_r / (r_dot_r + 1e-8)
        p = r + beta * p
        r_dot_r = new_r_dot_r
    return x

# ---- Main TRPO Loop ----
for env_name in envs:
    for run_id in range(1, total_runs+1):
        print(f'########## Starting run {run_id} for {env_name} ##########\n')

        # Initialize environment
        base_env = GymEnv(env_name, device=device)
        run_name = f'TRPO_{env_name}_{run_id}_{total_frames}'
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

        # Define policy (actor) network
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
        policy_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
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

        # Define value (critic) network
        value_net = nn.Sequential(
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(num_cells, device=device),
            nn.Tanh(),
            nn.LazyLinear(1, device=device),
        )
        value_module = ValueOperator(module=value_net, in_keys=["observation"])

        # Initialize LazyLinear layers with a dummy input
        dummy_input = env.reset()["observation"].unsqueeze(0)
        with torch.no_grad():
            policy_module(dummy_input)
            value_module(dummy_input)
        print("Actor and Critic networks initialized.")

        # Data collector & advantage estimator
        collector = SyncDataCollector(
            env,
            policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
        )
        advantage_module = GAE(
            gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
        )
        # Optimizer for the value network only
        optim_value = torch.optim.Adam(value_module.parameters(), lr=value_lr)

        # Incorporate a scheduler for the value network optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_value, total_frames // frames_per_batch, 0.0
        )

        # Logging
        logs = defaultdict(list)
        pbar = tqdm(total=total_frames)
        logs["step_count"] = []
        logs["eval reward"] = []
        logs["Return (Test)"] = []
        logs["Max Step Count (Test)"] = []
        logs["reward"] = []

        # Training loop
        for i, tensordict_data in enumerate(collector):
            # Compute advantages (and targets if provided by GAE)
            advantage_module(tensordict_data)

            # Extract data from collected batch
            observations = tensordict_data["observation"]
            actions = tensordict_data["action"]
            advantages = tensordict_data["advantage"]
            # Use computed returns if available, else approximate as advantage + value estimate
            if "return" in tensordict_data.keys():
                returns = tensordict_data["return"]
            else:
                with torch.no_grad():
                    returns = advantages + value_module(observations).squeeze(-1)

            with torch.no_grad():
                output = policy_module(observations)
                # Unpack the tuple: adjust indices if your tuple is structured differently.
                old_loc, old_scale, _, old_log_probs = output
                old_log_probs = old_log_probs.detach()
                old_loc = old_loc.detach()
                old_scale = old_scale.detach()

            # Closure to compute surrogate loss and KL divergence
            def get_loss_kl():
                new_output = policy_module(observations)
                new_loc, new_scale, new_action, new_log_probs = new_output

                # Compute the ratio between new and old log probabilities.
                ratio = torch.exp(new_log_probs - old_log_probs)
                surrogate_loss = -torch.mean(ratio * advantages)

                # Compute Gaussian KL divergence using the unsquashed parameters.
                kl = torch.mean(
                    torch.log(new_scale / old_scale) +
                    (old_scale**2 + (old_loc - new_loc)**2) / (2 * new_scale**2) - 0.5
                )
                return surrogate_loss, kl

            surrogate_loss, _ = get_loss_kl()
            # Compute gradient of the surrogate loss w.r.t. policy parameters.
            g = flat_grad(surrogate_loss, policy_module.module, retain_graph=True)

            # Define Fisher-vector product function
            def Fvp(v):
                _, kl = get_loss_kl()
                # Compute the gradient of KL with create_graph=True so that we can compute its derivative.
                kl_grad = flat_grad(kl, policy_module.module, retain_graph=True, create_graph=True)
                # Compute the directional derivative
                kl_v = (kl_grad * v).sum()
                # Compute the Hessian-vector product by differentiating kl_v.
                kl_hessian = flat_grad(kl_v, policy_module.module, retain_graph=True, create_graph=False)
                return kl_hessian + cg_damping * v

            # Compute step direction using conjugate gradients
            step_direction = conjugate_gradient(Fvp, -g, cg_iters)
            # Compute step size scaling
            shs = 0.5 * (step_direction * Fvp(step_direction)).sum(0, keepdim=True)
            if shs.item() <= 0:
                print("Non-positive curvature detected; skipping update.")
                continue
            step_size = torch.sqrt(2 * max_kl / (shs + 1e-8))
            fullstep = step_direction * step_size

            # Expected improvement
            expected_improve = (-g * fullstep).sum(0, keepdim=True)

            # Save old parameters
            old_params = flat_params(policy_module)
            # Line search to enforce KL constraint and improvement
            success = False
            for stepfrac in [0.5**i for i in range(line_search_backtracks)]:
                new_params = old_params + stepfrac * fullstep
                set_flat_params(policy_module, new_params)
                new_loss, new_kl = get_loss_kl()
                actual_improve = surrogate_loss - new_loss
                if actual_improve.item() > 0 and new_kl.item() <= max_kl:
                    success = True
                    break
            if not success:
                set_flat_params(policy_module, old_params)
            # --------- End TRPO Policy Update ----------

            # --------- Value Function Update ----------
            # Perform several gradient descent steps to update the value network
            # (Here we use 80 iterations per batch; adjust as needed.)
            for _ in range(80):
                optim_value.zero_grad()
                v_pred = value_module(observations).squeeze(-1)
                loss_value = ((v_pred - returns)**2).mean()
                loss_value.backward()
                optim_value.step()
            # --------- End Value Function Update ----------
            
            # Step the scheduler (adjusts learning rate of the value network optimizer)
            scheduler.step()

            # Log training information
            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            logs["step_count"].append(tensordict_data["step_count"].max().item())

            # Evaluation every 10 iterations
            if i % 10 == 0:
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    eval_rollout = env.rollout(1000, policy_module)
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["Return (Test)"].append(eval_rollout["next", "reward"].sum().item())
                    logs["Max Step Count (Test)"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"Eval Return: {logs['Return (Test)'][-1]:.4f} "
                    f"Max Step (Test): {logs['Max Step Count (Test)'][-1]}"
                )
            else:
                eval_str = ""

            # Update progress bar
            cum_reward_str = f"Avg reward: {logs['reward'][-1]:.4f}"
            stepcount_str = f"Step count (max): {logs['step_count'][-1]}"
            pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str]))
            pbar.update(frames_per_batch)
        pbar.close()

        # ---- Plotting Results ----
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
        plt.savefig(f'../runs/TRPO/plots_{run_name}.jpg', dpi=150)
        # plt.show()

        # ---- Saving Checkpoints and Logs ----
        save_path = f"../runs/TRPO/{run_name}.pth"
        checkpoint = {
            "policy_state_dict": policy_module.state_dict(),
            "value_state_dict": value_module.state_dict(),
            "optim_value_state_dict": optim_value.state_dict(),
            "hyperparameters": {
                "num_cells": num_cells,
                "value_lr": value_lr,
                "gamma": gamma,
                "lmbda": lmbda,
                "max_kl": max_kl,
                "cg_iters": cg_iters,
                "cg_damping": cg_damping,
            }
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

        # Ensure all log lists have the same length before saving to CSV
        max_length = max(len(logs[key]) for key in logs)
        for key in logs:
            while len(logs[key]) < max_length:
                logs[key].append(None)
        df_logs = pd.DataFrame.from_dict(logs)
        df_logs.to_csv(f"../runs/TRPO/logs_{run_name}.csv", index=False)
        print("Logs saved")
