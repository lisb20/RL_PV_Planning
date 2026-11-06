"""Regression checks for terminal returns, padded graphs, and packaged data."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from pv_planning.models import GraphPolicy, Critic
from pv_planning.train import compute_gae


class CoreTests(unittest.TestCase):
    def test_terminal_return_does_not_bootstrap(self):
        rewards = torch.tensor([1., 2.])
        values = torch.zeros(2, 3)
        adv, ret = compute_gae(rewards, values, torch.tensor([0., 1.]), torch.full((3,), 999.), 1., 1.)
        torch.testing.assert_close(ret, torch.tensor([[3., 3., 3.], [2., 2., 2.]]))
        torch.testing.assert_close(adv, ret)

    def test_extra_electrical_nodes_are_not_action_agents(self):
        edge = torch.tensor([[0, 1, 1, 3, 3, 0], [1, 0, 3, 1, 0, 3]])
        actor = GraphPolicy(212, 1, 16, edge, 3)
        critic = Critic(212, 16, 3, edge)
        obs = torch.randn(2, 3, 212)
        ids = torch.arange(3).expand(2, -1)
        mean, std = actor.act(actor.get_graph_embedding(obs), ids)
        value = critic(critic.get_graph_embedding(obs), ids)
        self.assertEqual(mean.shape, (2, 3, 1))
        self.assertEqual(value.shape, (2, 3))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue((std > 0).all())
        (mean.sum() + value.sum()).backward()
        self.assertTrue(all(p.grad is not None for p in actor.parameters() if p is not actor.actor.log_std))

    def test_data_checksums(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / 'pv_planning/data/manifest.json').read_text())
        for name, expected in manifest.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected, name)
        power = np.load(root / 'pv_planning/data/beijing/power_map0221.npy')
        self.assertEqual(power.shape, (803, 96))


if __name__ == '__main__':
    unittest.main()
