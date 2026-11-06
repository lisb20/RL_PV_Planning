"""Train the shared GCN/IPPO planning policy on IEEE-300 or Beijing.

Rollouts retain full graphs; minibatches consist of joint planning steps.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import random

import numpy as np
import torch
from torch.distributions import Normal

from .models import GraphPolicy, Critic
from .runtime import make_env, metrics

_ENV = None
_ACTOR = None
_CRITIC = None


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_worker(config, edge):
    global _ENV, _ACTOR, _CRITIC
    torch.set_num_threads(1)
    _ENV = make_env(config["grid"], config["steps"], config["cost_weight"], config["action_penalty"])
    _ACTOR, _CRITIC = build_models(_ENV.N_agent, config["hidden_dim"], edge)


def build_models(n, hidden, edge):
    actor = GraphPolicy(212, 1, hidden, edge, n)
    critic = Critic(212, hidden, n, edge)
    return actor, critic


def rollout(task):
    actor_state, critic_state, seed, grid = task
    seed_all(seed)
    _ACTOR.load_state_dict(actor_state)
    _CRITIC.load_state_dict(critic_state)
    _ACTOR.eval()
    _CRITIC.eval()
    env = _ENV
    obs = env.reset()
    initial_plan = env.solar_percentage.copy()
    index = torch.arange(env.N_agent).unsqueeze(0)
    data = {k: [] for k in ["obs", "actions", "logp", "values", "rewards", "dones"]}
    with torch.no_grad():
        for _ in range(env.max_step):
            x = torch.as_tensor(obs, dtype=torch.float32)
            mean, std = _ACTOR.act(_ACTOR.get_graph_embedding(x), index)
            dist = Normal(mean.squeeze(0), std)
            raw = dist.sample()
            squashed = raw.tanh()
            logp = (dist.log_prob(raw) - torch.log(1 - squashed.square() + 1e-6)).sum(-1)
            value = _CRITIC(_CRITIC.get_graph_embedding(x), index).squeeze(0)
            next_obs, reward, done, info = env.step(squashed.numpy() / 16)
            for key, item in zip(data, [x, raw, logp, value, float(reward), float(done)]):
                data[key].append(item)
            obs = next_obs
        last = _CRITIC(_CRITIC.get_graph_embedding(torch.as_tensor(obs, dtype=torch.float32)), index).squeeze(0)
    data = {k: torch.stack(v) if torch.is_tensor(v[0]) else torch.tensor(v) for k, v in data.items()}
    data.update(last_value=last, initial_plan=initial_plan,
                plan=env.solar_percentage.copy(), metrics=metrics(env, info, grid),
                objective=float(env.last_reward), seed=seed)
    return data


def compute_gae(rewards, values, dones, last_value, gamma=0.95, lam=0.95):
    """Return advantages/returns for T joint steps and N agents."""
    advantages = torch.zeros_like(values)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        live = 1 - dones[t]
        delta = rewards[t] + gamma * last_value * live - values[t]
        gae = delta + gamma * lam * live * gae
        advantages[t] = gae
        last_value = values[t]
    return advantages, advantages + values


def update(actor, critic, optim_actor, optim_critic, batch, args, iteration):
    for b in batch:
        b["adv"], b["returns"] = compute_gae(b["rewards"], b["values"], b["dones"],
                                             b["last_value"], args.gamma, args.gae_lambda)
    data = {k: torch.cat([b[k] for b in batch]) for k in ["obs", "actions", "logp", "values", "adv", "returns"]}
    data["adv"] = (data["adv"] - data["adv"].mean()) / (data["adv"].std() + 1e-8)
    device = next(actor.parameters()).device
    n = data["obs"].shape[1]
    losses = []
    for _ in range(args.epochs):
        order = torch.randperm(len(data["obs"]))
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            b = {k: v[idx].to(device) for k, v in data.items()}
            agent_idx = torch.arange(n, device=device).expand(len(idx), -1)
            mean, std = actor.act(actor.get_graph_embedding(b["obs"]), agent_idx)
            dist = Normal(mean, std)
            logp = (dist.log_prob(b["actions"]) - torch.log(1 - b["actions"].tanh().square() + 1e-6)).sum(-1)
            ratio = (logp - b["logp"]).exp()
            surrogate = torch.minimum(ratio * b["adv"], ratio.clamp(0.8, 1.2) * b["adv"])
            loss_actor = -surrogate.mean()
            # Preserve the historical linear entropy warm-up.
            loss_actor -= args.entropy_weight * iteration / args.iterations * dist.entropy().mean()
            optim_actor.zero_grad()
            loss_actor.backward()
            optim_actor.step()
            value = critic(critic.get_graph_embedding(b["obs"]), agent_idx)
            clipped = b["values"] + (value - b["values"]).clamp(-0.2, 0.2)
            loss_critic = torch.maximum((value - b["returns"]).square(), (clipped - b["returns"]).square()).mean()
            optim_critic.zero_grad()
            loss_critic.backward()
            optim_critic.step()
            losses.append((loss_actor.item(), loss_critic.item()))
    return np.mean(losses, axis=0).tolist()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", choices=["ieee", "beijing"], default="ieee")
    p.add_argument("--iterations", "--n_iter", type=int, default=300)
    p.add_argument("--rollouts", "--n_episode", type=int, default=128)
    p.add_argument("--steps", "--sample_step", type=int, default=16)
    p.add_argument("--epochs", "--n_epoch", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", "--n_cpu", type=int, default=1)
    p.add_argument("--cost-weight", "--lambda2", type=float, default=None)
    p.add_argument("--action-penalty", "--lambda1", type=float, default=0)
    p.add_argument("--lr-actor", "--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr-critic", "--lr_critic", type=float, default=3e-4)
    p.add_argument("--entropy-weight", "--entropy_weight", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cpu", help="cpu, cuda:0, etc.; rollout workers always use CPU")
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--torch-threads", type=int, default=1)
    return p


def main():
    args = parser().parse_args()
    if args.cost_weight is None:
        args.cost_weight = 5.0 if args.grid == "ieee" else 0.4
    for key in ["iterations", "rollouts", "steps", "epochs", "batch_size", "workers", "torch_threads"]:
        if getattr(args, key) < 1:
            raise ValueError(f"{key} must be positive")
    torch.set_num_threads(args.torch_threads)
    seed_all(args.seed)
    env = make_env(args.grid, args.steps, args.cost_weight, args.action_penalty)
    edge = torch.as_tensor(env.get_graph(), dtype=torch.long)
    actor, critic = build_models(env.N_agent, args.hidden_dim, edge)
    actor.to(args.device)
    critic.to(args.device)
    optim_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    optim_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
    config = {"algorithm": "ippo", **vars(args)}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    out = args.output or (args.resume.parent if args.resume else Path("runs") / f"{args.grid}_ippo_{datetime.now():%Y%m%d_%H%M%S}")
    if args.resume and out.resolve() != args.resume.resolve().parent:
        raise ValueError("Resume in the original output folder to retain candidate provenance")
    if (out / "config.json").exists() and not args.resume:
        raise FileExistsError(f"Run already exists: {out}; choose a new --output or use --resume")
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates").mkdir(exist_ok=True)
    start, best = 0, -float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ["grid", "algorithm", "steps", "rollouts", "hidden_dim", "cost_weight", "action_penalty"]:
            if ckpt["config"][key] != config[key]:
                raise ValueError(f"Resume configuration mismatch: {key}")
        actor.load_state_dict(ckpt["actor"])
        critic.load_state_dict(ckpt["critic"])
        optim_actor.load_state_dict(ckpt["optim_actor"])
        optim_critic.load_state_dict(ckpt["optim_critic"])
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        if args.device.startswith("cuda") and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng"], args.device)
        start, best = ckpt["iteration"] + 1, ckpt["best_objective"]
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    training_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    if args.workers == 1:
        init_worker(config, edge)
        pool = None
    else:
        pool = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"),
                                   initializer=init_worker, initargs=(config, edge))
    torch.set_rng_state(training_rng[0])
    np.random.set_state(training_rng[1])
    random.setstate(training_rng[2])
    try:
        for iteration in range(start, args.iterations):
            actor_state, critic_state = cpu_state(actor), cpu_state(critic)
            training_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
            tasks = [(actor_state, critic_state, args.seed + iteration * 1009 + i, args.grid)
                     for i in range(args.rollouts)]
            batch = list(map(rollout, tasks) if pool is None else pool.map(rollout, tasks))
            torch.set_rng_state(training_rng[0])
            np.random.set_state(training_rng[1])
            random.setstate(training_rng[2])
            plans = np.stack([b["plan"] for b in batch])
            objectives = np.array([b["objective"] for b in batch])
            np.savez_compressed(out / "candidates" / f"{iteration:06d}.npz", plans=plans,
                                initial_plans=np.stack([b["initial_plan"] for b in batch]),
                                objectives=objectives, seeds=[b["seed"] for b in batch])
            pick = int(objectives.argmax())
            if objectives[pick] > best:
                best = float(objectives[pick])
                np.save(out / "best_plan.npy", plans[pick])
                (out / "best_plan.json").write_text(json.dumps({"iteration": iteration, "candidate": pick,
                    "seed": batch[pick]["seed"], "objective": best, **batch[pick]["metrics"]}, indent=2) + "\n")
            actor_loss, critic_loss = update(actor, critic, optim_actor, optim_critic, batch, args, iteration)
            record = {"iteration": iteration, "mean_return": float(np.mean([b["rewards"].sum().item() for b in batch])),
                      "best_objective": best, "actor_loss": actor_loss, "critic_loss": critic_loss,
                      "full_day_evaluations": (iteration + 1) * args.rollouts * (args.steps + 1),
                      "mean_ramping_cost_usd": float(np.mean([b["metrics"]["ramping_cost_usd"] for b in batch]))}
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            checkpoint = {"actor": cpu_state(actor), "critic": cpu_state(critic), "config": config,
                          "optim_actor": optim_actor.state_dict(), "optim_critic": optim_critic.state_dict(),
                          "iteration": iteration, "best_objective": best, "torch_rng": torch.get_rng_state(),
                          "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                          "cuda_rng": torch.cuda.get_rng_state(args.device) if args.device.startswith("cuda") else None}
            torch.save(checkpoint, out / "last.tmp")
            (out / "last.tmp").replace(out / "last.pt")
            print(json.dumps(record), flush=True)
    finally:
        if pool is not None:
            pool.shutdown()


if __name__ == "__main__":
    main()
