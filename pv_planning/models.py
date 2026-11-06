import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# ==== Actor-Critic ====

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GNNEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=128, out_dim=128):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        # x: [N, in_dim], edge_index: [2, E]
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x

class SharedActor(nn.Module):
    def __init__(self, emb_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, h):
        mean = self.net(h)
        std = torch.exp(self.log_std)
        return mean, std

class GraphPolicy(nn.Module):
    def __init__(self, obs_dim=2, act_dim=1, hidden_dim=32, edge_index=None, num_agents=39):
        super().__init__()
        self.gnn = GNNEncoder(obs_dim, hidden_dim, hidden_dim)
        self.actor = SharedActor(hidden_dim + 32, act_dim)
        self.agent_id_embedding = nn.Embedding(num_agents, 32)
        self.edge_index = edge_index
        self.max_edge_index = int(edge_index.max()) + 1

    def get_graph_embedding(self, obs_batch):
        # obs_batch: [(B), N, obs_dim]
        if len(obs_batch.shape) == 2:
            obs_batch = obs_batch.unsqueeze(0)
        B = obs_batch.shape[0]
        N = obs_batch.shape[1]
        MAX_INDEX = max(self.max_edge_index, N)
        zero_pad = torch.zeros((B, MAX_INDEX - N, obs_batch.shape[2]), dtype=obs_batch.dtype, device=obs_batch.device)
        obs_batch = torch.cat([obs_batch, zero_pad], dim=1)  # [B, MAX_INDEX, obs_dim]
        self.edge_index = self.edge_index.to(obs_batch.device)
        h = self.gnn(obs_batch, self.edge_index)
        return h  # [B, MAX_INDEX, hidden_dim]

    def act(self, h, index):
        # obs_batch: [B, N, obs_dim]
        # h: [B, N, hidden_dim]
        # index: [B]
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        index_emb = self.agent_id_embedding(index)  # [B, N, 32]
        h = h[:, :index.shape[-1], :]
        h = torch.cat([h, index_emb], dim=-1)  # [B, N, hidden_dim + 32]
        mean, std = self.actor(h)
        return mean, std


class Critic(nn.Module):
    def __init__(self, global_obs_dim, hidden_dim=128, num_agents=1, edge_index=None):
        super().__init__()
        self.gnn = GNNEncoder(global_obs_dim, hidden_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.edge_index = edge_index
        self.max_edge_index = int(edge_index.max()) + 1

    def get_graph_embedding(self, obs_batch):
        # obs_batch: [(B), N, obs_dim]
        if len(obs_batch.shape) == 2:
            obs_batch = obs_batch.unsqueeze(0)
        B = obs_batch.shape[0]
        N = obs_batch.shape[1]
        MAX_INDEX = max(self.max_edge_index, N)
        zero_pad = torch.zeros((B, MAX_INDEX - N, obs_batch.shape[2]), dtype=obs_batch.dtype, device=obs_batch.device)
        obs_batch = torch.cat([obs_batch, zero_pad], dim=1)  # [B, MAX_INDEX, obs_dim]
        self.edge_index = self.edge_index.to(obs_batch.device)
        h = self.gnn(obs_batch, self.edge_index)
        return h  # [B, MAX_INDEX, hidden_dim]

    def forward(self, h, index=None):
        # h: [B, N, hidden_dim]
        h = h[:, :index.shape[-1], :]  # [B, hidden_dim]
        return self.net(h).squeeze(-1)  # [B]
