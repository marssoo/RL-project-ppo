# imports
import warnings
warnings.filterwarnings("ignore")
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torch import nn, optim, multiprocessing
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives.value import GAE
from tqdm import tqdm
import pandas as pd

# device selection
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# TRPO hyperparameters
num_cells = 256
value_lr = 3e-4  # learning rate for value function
max_grad_norm = 1.0

frames_per_batch = 1000
#total_frames = 1_000_000
total_frames = 30_000
total_runs = 3

gamma = 0.99
lmbda = 0.95

# TRPO-specific parameters
max_kl = 0.01       # maximum KL divergence per update
cg_iters = 10       # number of conjugate gradient iterations
damping = 0.1       # damping coefficient for Hessian-vector product
ls_max_steps = 10   # maximum number of line search steps
ls_accept_ratio = 0.1

#envs = ['HumanoidStandup-v4', 'HalfCheetah-v4', 'Hopper-v4', 'InvertedDoublePendulum-v4', 'InvertedPendulum-v4',
#        'Reacher-v4', 'Swimmer-v4', 'Walker2d-v4']

envs = ['Swimmer-v4']

#########################################
# Helper functions for TRPO update
#########################################

def conjugate_gradient(Avp, b, nsteps, residual_tol=1e-10):
    """
    Solves Ax = b using the conjugate gradient method.
    Avp: function that returns A*v given v.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):
        Avp_p = Avp(p)
        alpha = rdotr / (torch.dot(p, Avp_p) + 1e-8)
        x += alpha * p
        r -= alpha * Avp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x

def flat_grad(loss, parameters, retain_graph=False):
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph)
    return parameters_to_vector(grads).detach()

def compute_kl(tensordict_data, new_policy, old_policy):
    # Select only the "observation" key from the tensordict
    obs_td = tensordict_data.select("observation")
    new_out = new_policy(obs_td)
    with torch.no_grad():
        old_out = old_policy(obs_td)
    new_log_prob = new_out["sample_log_prob"]
    with torch.no_grad():
        old_log_prob = old_out["sample_log_prob"]
    return (old_log_prob - new_log_prob).mean()


def line_search(policy_module, loss_fn, old_params, full_step, expected_improve_rate):
    """
    Backtracking line search.
    """
    accept_ratio = ls_accept_ratio
    stepfrac = 1.0
    for i in range(ls_max_steps):
        new_params = old_params + stepfrac * full_step
        vector_to_parameters(new_params, policy_module.parameters())
        loss_new = loss_fn()
        loss_improve = loss_fn_old - loss_new
        expected_improve = expected_improve_rate * stepfrac
        if loss_improve.item() > accept_ratio * expected_improve:
            return True, new_params
        stepfrac *= 0.5
    return False, old_params

#########################################
# Main loop over environments and runs
#########################################
for env_name in envs:
    for run_id in range(1, total_runs+1):
        print(f'########## Starting run {run_id} for {env_name} ##########\n')
        
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

        # Define actor (policy) network
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

        # Define critic (value) network
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

        # Data collector
        collector = SyncDataCollector(
            env,
            policy_module,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            split_trajs=False,
            device=device,
        )

        # Advantage estimator (GAE)
        advantage_module = GAE(gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True)

        # Optimizer for the value network
        optim_value = optim.Adam(value_module.parameters(), lr=value_lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim_value, total_frames // frames_per_batch, 0.0
        )

        logs = defaultdict(list)
        logs["step_count"] = []
        logs["reward"] = []
        logs["lr_value"] = []
        logs["eval reward"] = []
        logs["Return (Test)"] = []
        logs["Max Step Count (Test)"] = []
        pbar = tqdm(total=total_frames)

        for i, tensordict_data in enumerate(collector):
            # Compute advantages (this adds "advantage" to tensordict_data)
            advantage_module(tensordict_data)

            # ---------------------------
            # TRPO policy update
            # ---------------------------
            # Create a snapshot of the old policy for KL computation and ratio calculation.
            old_policy = copy.deepcopy(policy_module).to(device)
            # Compute old log probabilities on the batch (detach so that they are not part of the graph)
            with torch.no_grad():
                old_log_prob = old_policy(tensordict_data)["sample_log_prob"]

            # Define the surrogate loss function (to be minimized)
            def surrogate_loss_fn():
                new_log_prob = policy_module(tensordict_data)["sample_log_prob"]
                ratio = torch.exp(new_log_prob - old_log_prob)
                return -(ratio * tensordict_data["advantage"]).mean()

            loss = surrogate_loss_fn()
            # Save current loss for line search
            loss_fn_old = loss.detach()

            # Get flat gradients of the surrogate loss wrt policy parameters
            policy_params = list(policy_module.parameters())
            flat_grad_surrogate = flat_grad(loss, policy_params, retain_graph=True)

            # Define function to compute Hessian-vector product (Fvp) for the KL divergence.
            def Fvp(v):
                kl = compute_kl(tensordict_data, policy_module, old_policy)
                kl_grads = torch.autograd.grad(kl, policy_params, create_graph=True)
                flat_grad_kl = parameters_to_vector(kl_grads)
                grad_v = torch.dot(flat_grad_kl, v)
                hvp = torch.autograd.grad(grad_v, policy_params, retain_graph=True)
                flat_hvp = parameters_to_vector(hvp)
                return flat_hvp + damping * v

            # Solve for the step direction using conjugate gradient
            step_direction = conjugate_gradient(Fvp, flat_grad_surrogate, cg_iters)
            # Compute the step size that satisfies the KL constraint
            step_dir_dot = torch.dot(step_direction, Fvp(step_direction))
            max_step_size = torch.sqrt(2 * max_kl / (step_dir_dot + 1e-8))
            full_step = step_direction * max_step_size

            # Save current flat parameters
            old_flat_params = parameters_to_vector(policy_params).detach()
            expected_improve = torch.dot(flat_grad_surrogate, full_step)

            # Line search to find the new parameters
            success, new_flat_params = line_search(policy_module, surrogate_loss_fn, old_flat_params, full_step, expected_improve)
            if success:
                vector_to_parameters(new_flat_params, policy_module.parameters())
            else:
                vector_to_parameters(old_flat_params, policy_module.parameters())
                print("Line search failed; no parameter update performed.")

            # ---------------------------
            # Value network update (using simple gradient descent)
            # ---------------------------
            with torch.no_grad():
                current_value = value_module(tensordict_data)["value"]
                returns = tensordict_data["advantage"] + current_value
            value_pred = value_module(tensordict_data)["value"]
            value_loss = (value_pred - returns).pow(2).mean()

            optim_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_module.parameters(), max_grad_norm)
            optim_value.step()

            # ---------------------------
            # Logging and evaluation
            # ---------------------------
            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            logs["step_count"].append(tensordict_data["step_count"].max().item())
            logs["lr_value"].append(optim_value.param_groups[0]["lr"])

            if i % 10 == 0:
                with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                    eval_rollout = env.rollout(1000, policy_module)
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["Return (Test)"].append(eval_rollout["next", "reward"].sum().item())
                    logs["Max Step Count (Test)"].append(eval_rollout["step_count"].max().item())
                eval_str = (f"Eval return: {logs['Return (Test)'][-1]:.4f} "
                            f"(init: {logs['Return (Test)'][0]:.4f}), "
                            f"Max step (Test): {logs['Max Step Count (Test)'][-1]}")
                del eval_rollout
            pbar.update(frames_per_batch)
            pbar.set_description(f"Train reward: {logs['reward'][-1]:.4f}, " +
                                  f"Step count: {logs['step_count'][-1]}, " +
                                  f"Value loss: {value_loss.item():.4f}")
            scheduler.step()
        pbar.close()

        # Plotting results
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

        # Saving checkpoint
        save_path = f"../runs/TRPO/{run_name}.pth"
        checkpoint = {
            "policy_state_dict": policy_module.state_dict(),
            "value_state_dict": value_module.state_dict(),
            "optimizer_value_state_dict": optim_value.state_dict(),
            "hyperparameters": {
                "num_cells": num_cells,
                "value_lr": value_lr,
                "gamma": gamma,
                "lmbda": lmbda,
                "max_kl": max_kl,
                "cg_iters": cg_iters,
                "damping": damping,
            }
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

        # Save logs as CSV
        max_length = max(len(logs[key]) for key in logs)
        for key in logs:
            while len(logs[key]) < max_length:
                logs[key].append(None)
        df_logs = pd.DataFrame.from_dict(logs)
        df_logs.to_csv(f"../runs/TRPO/logs_{run_name}.csv", index=False)
        print("Logs saved")