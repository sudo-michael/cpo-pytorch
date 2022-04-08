from datetime import datetime as dt, timedelta
import numpy as np
import os
import torch
from torch.nn import MSELoss
from torch.optim import LBFGS

from autoassign import autoassign
from optimization_utils.conjugate_gradient import cg_solver
from torch_utils.distribution_utils import mean_kl_first_fixed
from optimization_utils.hvp import get_Hvp_fun
from optimization_utils.line_search import line_search
from torch_utils.torch_utils import (
    flat_grad,
    get_device,
    get_flat_params,
    normalize,
    set_params,
)

save_dir = "save-dir"


def discount(vals, discount_term):
    n = vals.size(0)
    disc_pows = torch.pow(discount_term, torch.arange(n).float())
    reverse_indxs = torch.arange(n - 1, -1, -1)

    discounted = (
        torch.cumsum((vals * disc_pows)[reverse_indxs], dim=-1)[reverse_indxs]
        / disc_pows
    )

    return discounted


def compute_advs(actual_vals, exp_vals, discount_term, bias_red_param):
    exp_vals_next = torch.cat([exp_vals[1:], torch.tensor([0.0])])
    td_res = actual_vals + discount_term * exp_vals_next - exp_vals
    advs = discount(td_res, discount_term * bias_red_param)

    return advs


class CPO:
    @autoassign
    def __init__(
        self,
        policy,
        value_fun,
        cost_fun,
        envs,
        max_kl=1e-2,
        max_val_step=1e-2,
        max_cost_step=1e-2,
        max_constraint_val=0.1,
        val_iters=1,
        cost_iters=2,
        val_l2_reg=1e-3,
        cost_l2_reg=1e-3,
        discount_val=0.99,
        discount_cost=1.0,
        bias_red_val=0.98,
        bias_red_cost=0.98,
        cg_damping=1e-3,
        cg_max_iters=10,
        line_search_coef=0.9,
        line_search_max_iter=10,
        line_search_accept_ratio=0.1,
        model_name=None,
        continue_from_file=False,
        save_every=5,
        print_updates=True,
        num_envs=8,
    ):
        self.mse_loss = MSELoss(reduction="mean")
        self.value_optimizer = LBFGS(
            self.value_fun.parameters(), lr=max_val_step, max_iter=25
        )
        self.cost_optimizer = LBFGS(
            self.cost_fun.parameters(), lr=max_cost_step, max_iter=25
        )
        self.episode_num = 0
        self.elapsed_time = timedelta(0)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and True else "cpu"
        )

        self.policy = policy
        self.value_fn = value_fun
        self.cost_fn = cost_fun
        self.envs = envs
        self.max_kl = max_kl
        self.max_val_step = max_val_step
        self.max_cost_step = max_cost_step
        self.max_constraint_val = max_constraint_val
        self.val_iters = val_iters
        self.cost_iters = cost_iters
        self.val_l2_reg = val_l2_reg
        self.cost_l2_reg = cost_l2_reg
        self.discount_val = discount_val
        self.discount_cost = discount_cost
        self.bias_red_val = bias_red_val
        self.bias_red_cost = bias_red_cost
        self.cg_damping = cg_damping
        self.cg_max_iters = cg_max_iters
        self.line_search_coef = line_search_coef
        self.line_search_max_iter = line_search_max_iter
        self.line_search_accept_ratio = line_search_accept_ratio
        self.model_name = model_name
        self.continue_from_file = continue_from_file
        self.save_every = save_every
        self.print_updates = print_updates
        self.num_envs = num_envs

        self.mean_rewards = []
        self.mean_costs = []

        if not model_name and continue_from_file:
            raise Exception(
                "Argument continue_from_file to __init__ method of "
                "CPO case was set to True but model_name was not "
                "specified."
            )

        if not model_name and save_every:
            raise Exception(
                "Argument save_every to __init__ method of CPO "
                "was set to a value greater than 0 but model_name "
                "was not specified."
            )

        if continue_from_file:
            self.load_session()

    def train(self, n_updates):
        # states_w_time_prev = None
        # disc_rewards_prev = None
        # disc_costs_prev = None

        # while self.episode_num < n_episodes:
        #     start_time = dt.now()
        #     self.episode_num += 1

        #     memory = self.simulator.run_sim()
        #     observations, actions, rewards, costs = memory.sample()

        #     trajectory_sizes = torch.tensor([len(trajectory) for trajectory in memory])
        #     trajectory_limits = torch.cat(
        #         [torch.tensor([0]), torch.cumsum(trajectory_sizes, dim=-1)]
        #     )
        #     N = np.sum([len(trajectory) for trajectory in memory])
        #     T = self.simulator.trajectory_len
        #     time = torch.cat([torch.arange(size).float() for size in trajectory_sizes])
        #     time = torch.unsqueeze(time, dim=1) / T
        #     states_w_time = torch.cat([observations, time], dim=1)

        #     disc_rewards = torch.zeros(N)
        #     disc_costs = torch.zeros(N)
        #     reward_advs = torch.zeros(N)
        #     cost_advs = torch.zeros(N)

        #     with torch.no_grad():
        #         state_vals = (
        #             self.value_fun(states_w_time.to(self.device)).view(-1).cpu()
        #         )
        #         state_costs = (
        #             self.cost_fun(states_w_time.to(self.device)).view(-1).cpu()
        #         )

        #     for start, end in zip(trajectory_limits[:-1], trajectory_limits[1:]):
        #         disc_rewards[start:end] = discount(
        #             rewards[start:end], self.discount_val
        #         )
        #         disc_costs[start:end] = discount(costs[start:end], self.discount_cost)
        #         reward_advs[start:end] = compute_advs(
        #             rewards[start:end],
        #             state_vals[start:end],
        #             self.discount_val,
        #             self.bias_red_val,
        #         )
        #         cost_advs[start:end] = compute_advs(
        #             costs[start:end],
        #             state_costs[start:end],
        #             self.discount_cost,
        #             self.bias_red_cost,
        #         )

        #     reward_advs -= reward_advs.mean()
        #     reward_advs /= reward_advs.std()
        #     cost_advs -= reward_advs.mean()
        #     cost_advs /= cost_advs.std()

        #     if states_w_time_prev is not None:
        #         states_w_time_train = torch.cat([states_w_time, states_w_time_prev])
        #         disc_rewards_train = torch.cat([disc_rewards, disc_rewards_prev])
        #         disc_costs_train = torch.cat([disc_costs, disc_costs_prev])
        #     else:
        #         states_w_time_train = states_w_time
        #         disc_rewards_train = disc_rewards
        #         disc_costs_train = disc_costs

        #     states_w_time_prev = states_w_time
        #     disc_rewards_prev = disc_rewards
        #     disc_costs_prev = disc_costs

        #             constraint_cost = torch.mean(torch.tensor([disc_costs[start] for start in trajectory_limits[:-1]]))

        num_steps = 1_000
        obs = torch.zeros(
            (num_steps, self.num_envs) + self.envs.single_observation_space.shape
        ).to(self.device)
        actions = torch.zeros(
            (num_steps, self.num_envs) + self.envs.single_action_space.shape
        ).to(self.device)
        logprobs = torch.zeros((num_steps, self.num_envs)).to(self.device)
        rewards = torch.zeros((num_steps, self.num_envs)).to(self.device)
        costs = torch.zeros((num_steps, self.num_envs)).to(self.device)
        dones = torch.zeros((num_steps, self.num_envs)).to(self.device)
        values = torch.zeros((num_steps, self.num_envs)).to(self.device)

        global_step = 0
        next_obs = torch.Tensor(self.envs.reset()).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        num_updates = 100
        for update in range(1, num_updates + 1):
            # # Annealing the rate if instructed to do so.
            # if args.anneal_lr:
            #     frac = 1.0 - (update - 1.0) / num_updates
            #     lrnow = frac * args.learning_rate
            #     optimizer.param_groups[0]["lr"] = lrnow

            done_0 = False
            for step in range(0, num_steps):
                global_step += 1 * self.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action_dists = self.policy(next_obs)
                    action = action_dists.sample()
                    logprob = action_dists.log_prob(action)
                    value = self.value_fun(next_obs).flatten()
                    values[step] = value

                # import pdb

                # pdb.set_trace()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, infos = self.envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                costs[step] = torch.tensor(
                    np.array([info["constraint_cost"] for info in infos])
                ).to(self.device)
                next_obs, next_done = (
                    torch.Tensor(next_obs).to(self.device),
                    torch.Tensor(done).to(self.device),
                )

                for item in infos:
                    _r = 0
                    _c = 0
                    if "episode" in item.keys():
                        _r += item["episode"]["r"]
                        _c += item["episode"]["c"]
                    print(
                        f"global_step={global_step}, episodic_return={_r / self.num_envs}  episodic_cost={_c / self.num_envs}"
                    )

                #         # writer.add_scalar(
                #         #     "charts/episodic_return", item["episode"]["r"], global_step
                #         # )
                #         # writer.add_scalar(
                #         #     "charts/episodic_length", item["episode"]["l"], global_step
                #         # )
                #         break

            # bootstrap value if not done
            with torch.no_grad():
                # next_value = agent.get_value(next_obs).reshape(1, -1)
                next_value = self.value_fn(next_obs).reshape(1, -1)
                if True:
                    advantages = torch.zeros_like(rewards).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(num_steps)):
                        if t == num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + self.discount_val * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + self.discount_val
                            # * self.gae
                            * 0.95 * nextnonterminal * lastgaelam
                        )
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = (
                            rewards[t] + args.gamma * nextnonterminal * next_return
                        )
                    advantages = returns - values

                next_cost = self.cost_fn(next_obs).reshape(1, -1)
                if True:
                    advantages_costs = torch.zeros_like(costs).to(self.device)
                    lastgaelam = 0
                    for t in reversed(range(num_steps)):
                        if t == num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextcosts = next_cost
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextcosts = costs[t + 1]
                        delta = (
                            rewards[t]
                            + self.discount_cost * nextcosts * nextnonterminal
                            - costs[t]
                        )
                        advantages_costs[t] = lastgaelam = (
                            delta
                            + self.discount_cost
                            # * self.gae
                            * 0.95 * nextnonterminal * lastgaelam
                        )
                    returns_cost = advantages_costs + values

                # if True
                #     constraint_costs = torch.zeros_like(rewards).to(self.device)
                #     for t in reversed(range(num_steps)):
                #         if t == num_steps - 1:
                #             nextnonterminal = 1.0 - next_done
                #             nextcosts = next_cost
                #         else:
                #             nextnonterminal = 1.0 - dones[t + 1]
                #             nextcosts = costs[t + 1]
                #         constraint_costs[t] = (
                #             costs[t] + self.discount_cost * nextnonterminal * nextcosts
                #         )

            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_advantages_costs = advantages_costs.reshape(-1)
            b_returns = returns.reshape(-1)
            b_returns_costs = returns_cost.reshape(-1)
            b_values = values.reshape(-1)

            b_advantages = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + 1e-8
            )

            b_advantages_costs = (b_advantages_costs - b_advantages_costs.mean()) / (
                b_advantages_costs.std() + 1e-8
            )

            J_c = torch.mean(costs)

            self.update_policy(b_obs, b_actions, b_advantages, b_advantages_costs, J_c)
            self.update_nn_regressor(
                self.value_fun,
                self.value_optimizer,
                # states_w_time_train,
                # disc_rewards_train,
                b_obs,
                b_returns,
                self.val_l2_reg,
                self.val_iters,
            )
            self.update_nn_regressor(
                self.cost_fun,
                self.cost_optimizer,
                # states_w_time_train,
                # disc_costs_train,
                b_obs,
                b_returns_costs,
                self.cost_l2_reg,
                self.cost_iters,
            )

            # reward_sums = [np.sum(trajectory.rewards) for trajectory in memory]
            # cost_sums = [np.sum(trajectory.costs) for trajectory in memory]
            # self.mean_rewards.append(np.mean(reward_sums))
            # self.mean_costs.append(np.mean(cost_sums))
            # self.elapsed_time += dt.now() - start_time

            # if self.print_updates:
            #     self.print_update()

            if self.save_every and not self.episode_num % self.save_every:
                self.save_session()

    def update_policy(self, observations, actions, reward_advs, constraint_advs, J_c):
        self.policy.train()

        # observations = observations.to(self.device)
        # actions = actions.to(self.device)
        # reward_advs = reward_advs.to(self.device)
        # constraint_advs = constraint_advs.to(self.device)

        action_dists = self.policy(observations)
        log_action_probs = action_dists.log_prob(actions)

        imp_sampling = torch.exp(log_action_probs - log_action_probs.detach())
        # Change to torch.matmul
        reward_loss = -torch.mean(imp_sampling * reward_advs)
        reward_grad = flat_grad(
            reward_loss, self.policy.parameters(), retain_graph=True
        )
        # Change to torch.matmul
        constraint_loss = torch.sum(imp_sampling * constraint_advs) / self.num_envs
        constraint_grad = flat_grad(
            constraint_loss, self.policy.parameters(), retain_graph=True
        )

        mean_kl = mean_kl_first_fixed(action_dists, action_dists)
        Fvp_fun = get_Hvp_fun(mean_kl, self.policy.parameters())

        F_inv_g = cg_solver(Fvp_fun, reward_grad)
        F_inv_b = cg_solver(Fvp_fun, constraint_grad)

        q = torch.matmul(reward_grad, F_inv_g)
        r = torch.matmul(reward_grad, F_inv_b)
        s = torch.matmul(constraint_grad, F_inv_b)
        c = (J_c - self.max_constraint_val).to(self.device)

        is_feasible = False if c > 0 and c ** 2 / s - 2 * self.max_kl > 0 else True

        if is_feasible:
            lam, nu = self.calc_dual_vars(q, r, s, c)
            search_dir = -(lam ** -1) * (F_inv_g + nu * F_inv_b)
        else:
            search_dir = -torch.sqrt(2 * self.max_kl / s) * F_inv_b

        # Should be positive
        exp_loss_improv = torch.matmul(reward_grad, search_dir)
        current_policy = get_flat_params(self.policy)

        def line_search_criterion(search_dir, step_len):
            test_policy = current_policy + step_len * search_dir
            set_params(self.policy, test_policy)

            with torch.no_grad():
                # Test if conditions are satisfied
                test_dists = self.policy(observations)
                test_probs = test_dists.log_prob(actions)

                imp_sampling = torch.exp(test_probs - log_action_probs.detach())

                test_loss = -torch.mean(imp_sampling * reward_advs)
                test_cost = torch.sum(imp_sampling * constraint_advs) / self.num_envs
                test_kl = mean_kl_first_fixed(action_dists, test_dists)

                loss_improv_cond = (test_loss - reward_loss) / (
                    step_len * exp_loss_improv
                ) >= self.line_search_accept_ratio
                cost_cond = step_len * torch.matmul(constraint_grad, search_dir) <= max(
                    -c, 0.0
                )
                kl_cond = test_kl <= self.max_kl

            set_params(self.policy, current_policy)

            if is_feasible:
                return loss_improv_cond and cost_cond and kl_cond

            return cost_cond and kl_cond

        step_len = line_search(
            search_dir, 1.0, line_search_criterion, self.line_search_coef
        )
        print("Step Len.:", step_len, "\n")
        new_policy = current_policy + step_len * search_dir
        set_params(self.policy, new_policy)

    def update_nn_regressor(
        self, nn_regressor, optimizer, states, targets, l2_reg_coef, n_iters=1
    ):
        nn_regressor.train()

        # states = states.to(self.device)
        # targets = targets.to(self.device)

        for _ in range(n_iters):

            def mse():
                optimizer.zero_grad()

                predictions = nn_regressor(states).view(-1)
                loss = self.mse_loss(predictions, targets)

                flat_params = get_flat_params(nn_regressor)
                l2_loss = l2_reg_coef * torch.sum(torch.pow(flat_params, 2))
                loss += l2_loss

                loss.backward()

                return loss

            optimizer.step(mse)

    def calc_dual_vars(self, q, r, s, c):
        if c < 0.0 and c ** 2 / s - 2 * self.max_kl > 0.0:
            lam = torch.sqrt(q / (2 * self.max_kl))
            nu = 0.0

            return lam, nu

        A = q - r ** 2 / s
        B = 2 * self.max_kl - c ** 2 / s

        lam_mid = r / c
        lam_a = torch.sqrt(A / B)
        lam_b = torch.sqrt(q / (2 * self.max_kl))

        f_mid = -0.5 * (q / lam_mid + 2 * lam_mid * self.max_kl)
        f_a = -torch.sqrt(A * B) - r * c / s
        f_b = -torch.sqrt(2 * q * self.max_kl)

        if lam_mid > 0:
            if c < 0:
                if lam_a > lam_mid:
                    lam_a = lam_mid
                    f_a = f_mid
                if lam_b < lam_mid:
                    lam_b = lam_mid
                    f_b = f_mid
            else:
                if lam_a < lam_mid:
                    lam_a = lam_mid
                    f_a = f_mid
                if lam_b > lam_mid:
                    lam_b = lam_mid
                    f_b = f_mid
        else:
            if c < 0:
                lam = lam_b
            else:
                lam = lam_a

        lam = lam_a if f_a >= f_b else lam_b
        nu = max(0.0, (lam * c - r) / s)

        return lam, nu

    def save_session(self):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_path = os.path.join(save_dir, self.model_name + ".pt")

        ckpt = dict(
            policy_state_dict=self.policy.state_dict(),
            value_state_dict=self.value_fun.state_dict(),
            cost_state_dict=self.cost_fun.state_dict(),
            mean_rewards=self.mean_rewards,
            mean_costs=self.mean_costs,
            episode_num=self.episode_num,
            elapsed_time=self.elapsed_time,
        )

        # if self.simulator.obs_filter:
        #     ckpt["obs_filter"] = self.simulator.obs_filter

        torch.save(ckpt, save_path)

    def load_session(self):
        load_path = os.path.join(save_dir, self.model_name + ".pt")
        ckpt = torch.load(load_path)

        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.value_fun.load_state_dict(ckpt["value_state_dict"])
        self.cost_fun.load_state_dict(ckpt["cost_state_dict"])
        # self.mean_rewards = ckpt["mean_rewards"]
        # self.mean_costs = ckpt["mean_costs"]
        # self.episode_num = ckpt["episode_num"]
        # self.elapsed_time = ckpt["elapsed_time"]

        # try:
        #     self.simulator.obs_filter = ckpt["obs_filter"]
        # except KeyError:
        #     pass

    def print_update(self):
        update_message = "[Episode]: {0} | [Avg. Reward]: {1} | [Avg. Cost]: {2} | [Elapsed Time]: {3}"
        elapsed_time_str = "".join(str(self.elapsed_time)).split(".")[0]
        format_args = (
            self.episode_num,
            self.mean_rewards[-1],
            self.mean_costs[-1],
            elapsed_time_str,
        )
        print(update_message.format(*format_args))
