import gym
from envs import point_gather
import numpy as np

import torch

# env = gym.make("MyEnv-v0")

# obs = env.reset()
# print(obs.shape)
# exit()

from argparse import ArgumentParser
from envs.point_gather import PointGatherEnv
import gym
import torch
from yaml import load, FullLoader

from cpo import CPO
from memory import Memory
from models import build_diag_gauss_policy, build_mlp
from simulators import SinglePathSimulator
from torch_utils.torch_utils import get_device

config_filename = "config.yaml"

parser = ArgumentParser(
    prog="train.py",
    description="Train a policy on the specified environment"
    " using Constrained Policy Optimization (Achaim 2017).",
)
parser.add_argument(
    "--continue",
    dest="continue_from_file",
    action="store_true",
    help="Set this flag to continue training from a previously "
    "saved session. Session will be overwritten if this flag is "
    "not set and a saved file associated with model-name already "
    "exists.",
)
parser.add_argument(
    "--model-name",
    type=str,
    dest="model_name",
    default="point_gather",
    help="The entry in config.yaml from which settings" "should be loaded.",
)
parser.add_argument(
    "--simulator",
    dest="simulator_type",
    type=str,
    default="single-path",
    choices=["single-path", "vine"],
    help="The type of simulator" " to use when collecting training experiences.",
)

args = parser.parse_args()
continue_from_file = args.continue_from_file
model_name = args.model_name
config = load(open(config_filename, "r"), FullLoader)[model_name]

state_dim = config["state_dim"]
action_dim = config["action_dim"]

n_episodes = config["n_episodes"]
env_name = config["env_name"]
n_episodes = config["n_episodes"]
# n_trajectories = config["n_trajectories"]
n_trajectories = 2
trajectory_len = 1_000
policy_dims = config["policy_hidden_dims"]
vf_dims = config["vf_hidden_dims"]
cf_dims = config["cf_hidden_dims"]
max_constraint_val = config["max_constraint_val"]
bias_red_cost = config["bias_red_cost"]

policy = build_diag_gauss_policy(state_dim, policy_dims, action_dim)
value_fun = build_mlp(state_dim, vf_dims, 1)
cost_fun = build_mlp(state_dim, cf_dims, 1)

device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")

policy.to(device)
value_fun.to(device)
cost_fun.to(device)


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv(
    [make_env("MyEnv-v0", 0 + i, i, False, "test") for i in range(2)]
)


# ALGO Logic: Storage setup


num_steps = 1_000
num_envs = 2
obs = torch.zeros((num_steps, 2) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((num_steps, 2) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((num_steps, 2)).to(device)
rewards = torch.zeros((num_steps, 2)).to(device)
costs = torch.zeros((num_steps, 2)).to(device)
dones = torch.zeros((num_steps, 2)).to(device)
values = torch.zeros((num_steps, 2)).to(device)

global_step = 0
next_obs = torch.Tensor(envs.reset()).to(device)
next_done = torch.zeros(2).to(device)

num_updates = 100
for update in range(1, num_updates + 1):
    # # Annealing the rate if instructed to do so.
    # if args.anneal_lr:
    #     frac = 1.0 - (update - 1.0) / num_updates
    #     lrnow = frac * args.learning_rate
    #     optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, num_steps):
        global_step += 1 * num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action_dists = policy(next_obs)
            action = action_dists.sample()
            logprob = action_dists.log_prob(action)
            value = value_fun(next_obs).flatten()
            values[step] = value

        import pdb

        pdb.set_trace()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, infos = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        costs[step] = torch.tensor(
            np.array([info["constraint_cost"] for info in infos])
        ).to(device)
        next_obs, next_done = (
            torch.Tensor(next_obs).to(device),
            torch.Tensor(done).to(device),
        )

    #     for item in info:
    #         if "episode" in item.keys():
    #             print(
    #                 f"global_step={global_step}, episodic_return={item['episode']['r']}"
    #             )
    #             writer.add_scalar(
    #                 "charts/episodic_return", item["episode"]["r"], global_step
    #             )
    #             writer.add_scalar(
    #                 "charts/episodic_length", item["episode"]["l"], global_step
    #             )
    #             break

    # # bootstrap value if not done
    # with torch.no_grad():
    #     next_value = agent.get_value(next_obs).reshape(1, -1)
    #     if args.gae:
    #         advantages = torch.zeros_like(rewards).to(device)
    #         lastgaelam = 0
    #         for t in reversed(range(args.num_steps)):
    #             if t == args.num_steps - 1:
    #                 nextnonterminal = 1.0 - next_done
    #                 nextvalues = next_value
    #             else:
    #                 nextnonterminal = 1.0 - dones[t + 1]
    #                 nextvalues = values[t + 1]
    #             delta = (
    #                 rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
    #             )
    #             advantages[t] = lastgaelam = (
    #                 delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    #             )
    #         returns = advantages + values
    #     else:
    #         returns = torch.zeros_like(rewards).to(device)
    #         for t in reversed(range(args.num_steps)):
    #             if t == args.num_steps - 1:
    #                 nextnonterminal = 1.0 - next_done
    #                 next_return = next_value
    #             else:
    #                 nextnonterminal = 1.0 - dones[t + 1]
    #                 next_return = returns[t + 1]
    #             returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
    #         advantages = returns - values

