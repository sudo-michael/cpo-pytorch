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
n_trajectories = config["n_trajectories"]
trajectory_len = config["max_timesteps"]
policy_dims = config["policy_hidden_dims"]
vf_dims = config["vf_hidden_dims"]
cf_dims = config["cf_hidden_dims"]
max_constraint_val = config["max_constraint_val"]
bias_red_cost = config["bias_red_cost"]
device = get_device()

policy = build_diag_gauss_policy(state_dim, policy_dims, action_dim)
value_fun = build_mlp(state_dim, vf_dims, 1)
cost_fun = build_mlp(state_dim, cf_dims, 1)

policy.to(device)
value_fun.to(device)
cost_fun.to(device)

# simulator = SinglePathSimulator(env_name, policy, 50, trajectory_len)


# import pdb;
# pdb.set_trace()

# make_env(env_name)

# memory = simulator.run_sim()
# memory.sample()

# import pdb


import time
from collections import deque
import numpy as np


class RecordEpisodeStatisticsC(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        # self.num_envs = 8
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
        self.episode_costs = None
        # self.violation_returns = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.cost_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_costs = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += rewards
        self.episode_lengths += 1
        costs = np.zeros_like(rewards)
        # import pdb

        # pdb.set_trace()
        for i, info in enumerate([infos]):
            costs = info["constraint_cost"]

        self.episode_costs += costs
        if not self.is_vector_env:
            infos = [infos]
            dones = [dones]
        else:
            infos = list(infos)  # Convert infos to mutable type
        for i in range(len(dones)):
            if dones[i]:
                infos[i] = infos[i].copy()
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_cost = self.episode_costs[i]

                episode_info = {
                    "r": episode_return,
                    "l": episode_length,
                    "c": episode_cost,
                    "t": round(time.perf_counter() - self.t0, 6),
                }
                infos[i]["episode"] = episode_info
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.cost_queue.append(episode_cost)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        if self.is_vector_env:
            infos = tuple(infos)
        return (
            observations,
            rewards,
            dones if self.is_vector_env else dones[0],
            infos if self.is_vector_env else infos[0],
        )


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = RecordEpisodeStatisticsC(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        print(idx)
        return env

    return thunk


envs = gym.vector.SyncVectorEnv(
    [make_env("MyEnv-v0", 420 + i, i, False, "test") for i in range(16)]
)
print(envs.reset().shape)


cpo = CPO(
    policy,
    value_fun,
    cost_fun,
    envs,
    model_name=model_name,
    bias_red_cost=bias_red_cost,
    max_constraint_val=max_constraint_val,
    num_envs=16,
)

print(f"Training policy {model_name} on {env_name} environment...\n")
print("2")
cpo.train(2 * n_episodes)
