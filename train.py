import sys
from typing import final
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from IEEE_env import myenv 
from model import GraphPolicy, Critic
from tqdm import tqdm
import pickle
import pdb
import wandb
import time
import os
from tqdm.contrib.concurrent import process_map
import copy
from concurrent.futures import ProcessPoolExecutor
import ray
import argparse
import copy
import yaml
import json
import setproctitle
setproctitle.setproctitle("PV-planning")

class RolloutBuffer:
    ## Buffer for 1 episode
    def __init__(self, max_steps, num_agents, obs_dim, act_dim):
        self.obs = torch.zeros((max_steps, num_agents, obs_dim))
        self.actions = torch.zeros((max_steps, num_agents, act_dim))
        self.log_probs = torch.zeros((max_steps, num_agents))
        self.rewards = torch.zeros((max_steps, num_agents))
        self.dones = torch.zeros((max_steps, num_agents))
        self.values = torch.zeros((max_steps, num_agents))
        self.stds = torch.zeros((max_steps))
        self.logging_info = []
        self.step = 0
        self.ret = None
        self.adv = None

    def insert(
        self, obs, action, log_prob, reward, done, value, logging_info=None, std=0
    ):
        if self.step == self.obs.shape[0]:
            raise RuntimeError("RolloutBuffer is full")
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.stds[self.step] = std
        self.step += 1
        if logging_info is not None:
            self.logging_info.append(logging_info)

    def compute_returns_and_advantages(self, last_value, gamma=0.95, lam=0.95):
        if self.step != self.obs.shape[0]:
            raise RuntimeError("RolloutBuffer is not full")
        returns = torch.zeros_like(self.rewards)
        advantages = torch.zeros_like(self.rewards)
        gae = 0
        for t in reversed(range(self.step)):
            delta = (
                self.rewards[t]
                + gamma * last_value * (1 - self.dones[t])
                - self.values[t]
            )
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + self.values[t]
            last_value = self.values[t]
        self.ret = returns
        self.adv = advantages


@ray.remote
def sample_one(args):
    i, actor_state_dict, critic_state_dict, sample_step, num_agents = args
    env = myenv(
        num_agents=num_agents,
        max_step=system_args.sample_step,
        lambda1=system_args.lambda1,
        lambda2=system_args.lambda2,
        case=system_args.case
    )
    global edge_index
    actor = GraphPolicy(obs_dim=obs_dim, act_dim=1, hidden_dim=hidden_dim, edge_index=edge_index, num_agents=num_agents)

    critic = Critic(
        global_obs_dim=obs_dim, hidden_dim=hidden_dim, edge_index=edge_index
    )
    actor.load_state_dict(actor_state_dict)
    critic.load_state_dict(critic_state_dict)

    with torch.no_grad():
        obs = env.reset()
        buffer = RolloutBuffer(
            max_steps=sample_step,
            num_agents=num_agents,
            obs_dim=obs.shape[1],
            act_dim=1,
        )
        done = False
        
        for _ in range(system_args.sample_step):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            h = actor.get_graph_embedding(obs_tensor)
            idx = torch.arange(num_agents, dtype=torch.long)
            idx = idx.unsqueeze(0)
            mean, std = actor.act(h, idx)
            mean = mean.squeeze(0)
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.sample()
            tanh_action = torch.tanh(raw_action)

            final_action = tanh_action / 16
            l = dist.log_prob(raw_action) - torch.log(1 - tanh_action**2 + 1e-6)
            l = l.sum(-1)

            h_critic = critic.get_graph_embedding(obs_tensor)
            value = critic(h_critic, idx).squeeze(0)
            next_obs, rewards, done, info = env.step(final_action)
            buffer.insert(obs_tensor, raw_action, l, rewards, done, value, info, std)
            obs = next_obs

        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        h_last = critic.get_graph_embedding(obs_tensor)
        value_last = critic(h_last, idx).squeeze(0)
        buffer.compute_returns_and_advantages(value_last)
        env = None
        return buffer


edge_index = None


class MAPPO:
    def __init__(self, obs_dim, act_dim, hidden_dim, num_agents, system_args):
        global edge_index
        self.num_agents = num_agents
        self.env = myenv(
            num_agents=num_agents,
            max_step=system_args.sample_step,
            lambda1=system_args.lambda1,
            lambda2=system_args.lambda2,
            case=system_args.case
        )
        self.edge_index = self.env.get_graph()
        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        edge_index = self.edge_index
        print("Edge index shape:", self.edge_index.shape)
        self.actor = GraphPolicy(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim, edge_index=self.edge_index, num_agents=num_agents).to(device)

        self.critic = Critic(
            global_obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_agents=num_agents,
            edge_index=self.edge_index,
        ).to(device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=system_args.lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=system_args.lr_critic)
        self.Meta_buffer = []
        self.sample_step = system_args.sample_step
        self.n_episodes = system_args.n_episode
        self.system_args = system_args

    def sample_and_store(self, N_episodes=10):
        self.actor.eval()
        self.critic.eval()
        actor_state_dict = self.actor.state_dict().copy()
        for k, v in actor_state_dict.items():
            actor_state_dict[k] = v.to("cpu")
        critic_state_dict = self.critic.state_dict().copy()
        for k, v in critic_state_dict.items():
            critic_state_dict[k] = v.to("cpu")

        args = []
        for i in range(N_episodes):
            args.append(
                (
                    i,
                    actor_state_dict,
                    critic_state_dict,
                    self.sample_step,
                    self.num_agents,
                )
            )

        futures = [sample_one.remote(arg) for arg in args]
        result = ray.get(futures)
        self.Meta_buffer.extend(result)

    def train(self, meta_buffer, clip_eps=0.2, epochs=4, batch_size=64, n_iter=0):
        self.actor.train()
        self.critic.train()
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)

        obs = torch.cat([buffer.obs for buffer in meta_buffer], dim=0)
        actions = torch.cat([buffer.actions for buffer in meta_buffer], dim=0)
        old_log_probs = torch.cat([buffer.log_probs for buffer in meta_buffer], dim=0)
        old_values = torch.cat([buffer.values for buffer in meta_buffer], dim=0)

        adv = torch.cat([buffer.adv for buffer in meta_buffer], dim=0)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.cat([buffer.ret for buffer in meta_buffer], dim=0)
      
        obs = obs.view(-1, self.num_agents, obs.shape[-1])
        actions = actions.view(-1, self.num_agents, actions.shape[-1])
        old_log_probs = old_log_probs.view(-1, self.num_agents)
    

        shuffle_idx = np.random.permutation(len(obs))
        obs = obs[shuffle_idx]
        actions = actions[shuffle_idx]
        old_log_probs = old_log_probs[shuffle_idx]
        old_values = old_values[shuffle_idx]
        adv = adv[shuffle_idx]
        ret = ret[shuffle_idx]

        act_loss = []
        crit_loss = []
        for _ in range(self.system_args.n_epoch):
            idx = np.random.permutation(len(obs))
            for i in range(0, len(obs), batch_size):
                b_idx = idx[i : i + batch_size]
                b_obs = obs[b_idx].detach()
                b_actions = actions[b_idx].detach()
                b_actions = b_actions.to(device)
                b_old_logp = old_log_probs[b_idx].detach().to(device)
                b_adv = adv[b_idx].detach().to(device)
                b_ret = ret[b_idx].detach().to(device)
                b_old_values = old_values[b_idx].detach().to(device)

                # actor update
                b_obs = b_obs.to(device)
                h_actor = self.actor.get_graph_embedding(b_obs)

                agent_idx = torch.arange(
                    self.num_agents, dtype=torch.long, device=device
                )
                agent_idx = agent_idx.unsqueeze(0).repeat(b_obs.shape[0], 1)
                mean, std = self.actor.act(h_actor, agent_idx)
                new_dist = torch.distributions.Normal(mean, std)

                entropy = new_dist.entropy().mean()
                new_logp = new_dist.log_prob(b_actions) - torch.log(
                    1 - torch.tanh(b_actions) ** 2 + 1e-6
                )
                new_logp = new_logp.sum(-1)

                ratio = (new_logp - b_old_logp).exp()
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                actor_loss = (
                    -torch.min(ratio * b_adv, clipped * b_adv).mean()
                    - self.system_args.entropy_weight
                    * entropy
                    * n_iter
                    / system_args.n_iter
                )
                act_loss.append(actor_loss.item())
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                # critic update
                h_critic = self.critic.get_graph_embedding(b_obs)
                value_est = self.critic(h_critic, agent_idx)
                v_pred_clipped = b_old_values + (value_est - b_old_values).clamp(
                    -clip_eps, clip_eps
                )
                v_loss_unclipped = (value_est - b_ret) ** 2
                v_loss_clipped = (v_pred_clipped - b_ret) ** 2
                critic_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()

                crit_loss.append(critic_loss.item())
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

        return np.mean(act_loss), np.mean(crit_loss)

    def test(self):
        self.actor.eval()
        self.critic.eval()
        self.actor = self.actor.to("cpu")
        self.critic = self.critic.to("cpu")
        with torch.no_grad():
            obs = self.env.reset()
            done = False
            test_buffer = RolloutBuffer(
                self.system_args.sample_step, self.num_agents, obs.shape[1], 1
            )
            # while not done:
            for _ in range(self.system_args.sample_step):
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                h = self.actor.get_graph_embedding(obs_tensor)
                # h = obs_tensor
                idx = torch.arange(self.num_agents, dtype=torch.long)
                idx = idx.unsqueeze(0)
                mean, std = self.actor.act(h, idx)
                mean = mean.squeeze(0)
                dist = torch.distributions.Normal(mean, std)
                raw_action = mean
                tanh_action = torch.tanh(raw_action)

                final_action = tanh_action * 1 / 16
                l = dist.log_prob(raw_action) - torch.log(1 - tanh_action**2 + 1e-6)
                l = l.sum(-1)

                h_critic = self.critic.get_graph_embedding(obs_tensor)
                value = self.critic(h_critic, idx).squeeze(0)
                next_obs, rewards, done, info = self.env.step(final_action)

                test_buffer.insert(
                    obs_tensor, raw_action, l, rewards, done, value, info, std
                )
                obs = next_obs

            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            h_last = self.critic.get_graph_embedding(obs_tensor)
            value_last = self.critic(h_last, idx).squeeze(0)
            test_buffer.compute_returns_and_advantages(value_last)

        return self.env.solar_percentage, test_buffer, mean

    def load_ckpt(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        print("Model loaded.")

    def run(self, N_times=300):
        for n_iter in tqdm(range(N_times)):
            self.sample_and_store(N_episodes=self.n_episodes)
            mean_reward = 0
            for buffer in self.Meta_buffer:
                mean_reward += buffer.rewards.sum().item()
            mean_reward /= len(self.Meta_buffer)
            mean_reward /= self.num_agents
            print("Mean reward:", mean_reward)

            mean_action = 0
            for buffer in self.Meta_buffer:
                mean_action += buffer.actions.mean().item()
            mean_action /= len(self.Meta_buffer)
            print("Mean action:", mean_action)

            mean_logp = 0
            for buffer in self.Meta_buffer:
                mean_logp += buffer.log_probs.mean().item()
            mean_logp /= len(self.Meta_buffer)
            print("Mean logp:", mean_logp)

            std = 0
            for buffer in self.Meta_buffer:
                std += buffer.stds.mean().item()
            std /= len(self.Meta_buffer)
            print("Std:", std)

            info = [x.logging_info[-1] for x in self.Meta_buffer]
            mean_solar = np.mean([x[0] for x in info])
            mean_power_negative = np.mean([x[1] for x in info])
            mean_ramping = np.mean([x[2] for x in info])
            ramping_cost = np.mean([x[3] for x in info])
            violation = np.mean([x[4] for x in info])
            print(
                "Mean solar:",
                mean_solar,
                "Mean power negative:",
                mean_power_negative,
                "Mean ramping:",
                mean_ramping,
                "Ramping cost: $",
                (ramping_cost) * 5e4,
                "Violation:",
                violation
            )

            act_loss, crit_loss = self.train(
                self.Meta_buffer, batch_size=32, n_iter=n_iter
            )
            if not os.path.exists(f"./ckpt/{now}"):
                os.makedirs(f"./ckpt/{now}")
            with open(f"./ckpt/{now}/buffer.pkl", "wb") as f:
                pickle.dump(self.Meta_buffer, f)

            self.Meta_buffer = []
            if system_args.use_wandb:
                logger.log(
                    {
                        "mean_reward": mean_reward,
                        "mean_action": mean_action,
                        "mean_solar": mean_solar,
                        "mean_power_negative": mean_power_negative,
                        "mean_ramping": mean_ramping,
                        "mean_logp": mean_logp,
                        "act_loss": act_loss,
                        "crit_loss": crit_loss,
                        "std": std,
                        "ramping_cost": (ramping_cost) * 5e4,
                        "violation": violation
                    }
                )

            if (n_iter % 20 == 0 or n_iter == N_times - 1) and system_args.use_wandb:
                torch.save(self.actor.state_dict(), f"./ckpt/{now}/actor_{n_iter}.pth")
                torch.save(
                    self.critic.state_dict(), f"./ckpt/{now}/critic_{n_iter}.pth"
                )
                print("Model saved.")
                print(self.system_args)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_agents", type=int, default=300)
    argparser.add_argument("--n_episode", type=int, default=128)
    argparser.add_argument("--sample_step", type=int, default=16)
    argparser.add_argument("--n_epoch", type=int, default=10)
    argparser.add_argument("--n_iter", type=int, default=100)
    argparser.add_argument("--lr_actor", type=float, default=1e-4)
    argparser.add_argument("--lr_critic", type=float, default=3e-4)
    argparser.add_argument("--lambda1", type=float, default=1)
    argparser.add_argument("--lambda2", type=float, default=1)
    argparser.add_argument("--use_wandb", action="store_true")
    argparser.add_argument("--entropy_weight", type=float, default=0.01)
    argparser.add_argument("--case", type=str, default="./data/case300.mat")
    argparser.add_argument("--n_cpu", type=int, default=64)
    system_args = argparser.parse_args()
    print(system_args)

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    ray.init(num_cpus=system_args.n_cpu)
    save_folder = f"./ckpt/{now}"

    if system_args.use_wandb:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        import wandb

        wandb.require("core")
        wandb.login()
        logger = wandb.init(project="MAPPO", name=f"IEEE_MAPPO_{now}")
    obs_dim = 192 + 4 + 16
    act_dim = 1
    num_agents = system_args.num_agents
    hidden_dim = 128
    # hidden_dim = 192
    mappo = MAPPO(obs_dim, act_dim, hidden_dim, num_agents, system_args=system_args)
    print("actor parameters:", sum(p.numel() for p in mappo.actor.parameters()))
    print("critic parameters:", sum(p.numel() for p in mappo.critic.parameters()))
    mappo.run(system_args.n_iter)
