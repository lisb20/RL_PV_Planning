"""Evaluate a saved plan, or sample planning trajectories from a checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .runtime import make_env, metrics
from .train import build_models, seed_all


def evaluate_plan(plan, grid, cost_weight=None, seed=2026):
    seed_all(seed)
    env = make_env(grid, steps=1, cost_weight=cost_weight)
    plan = np.asarray(plan, dtype=float)
    if plan.shape != (env.N_agent,) or not np.isfinite(plan).all():
        raise ValueError(f"Expected {env.N_agent} finite deployment fractions")
    if np.any((plan < 0) | (plan > 1)):
        raise ValueError("Deployment fractions must be in [0, 1]")
    env.reset()
    _, _, _, info = env.step((plan - env.solar_percentage).reshape(-1, 1))
    return metrics(env, info, grid)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", type=Path)
    source.add_argument("--checkpoint", type=Path)
    p.add_argument("--grid", choices=["ieee", "beijing"], default="ieee")
    p.add_argument("--candidates", type=int, default=128)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, default=Path("evaluation"))
    args = p.parse_args()
    if args.candidates < 1:
        p.error("--candidates must be positive")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.plan:
        result = evaluate_plan(np.load(args.plan, allow_pickle=False), args.grid, seed=args.seed)
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        env = make_env(config["grid"], config["steps"], config["cost_weight"], config["action_penalty"])
        edge = torch.as_tensor(env.get_graph(), dtype=torch.long)
        actor, _ = build_models(env.N_agent, config["hidden_dim"], edge)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        index = torch.arange(env.N_agent).unsqueeze(0)
        plans, scores, rows = [], [], []
        with torch.no_grad():
            for i in range(args.candidates):
                seed_all(args.seed + i)
                obs = env.reset()
                for _ in range(env.max_step):
                    mean, std = actor.act(actor.get_graph_embedding(torch.as_tensor(obs, dtype=torch.float32)), index)
                    raw = mean if args.deterministic else torch.distributions.Normal(mean, std).sample()
                    obs, _, _, info = env.step(raw.squeeze(0).tanh().numpy() / 16)
                plans.append(env.solar_percentage.copy())
                scores.append(float(env.last_reward))
                rows.append(metrics(env, info, config["grid"]))
        pick = int(np.argmax(scores))
        np.savez_compressed(args.output / "candidates.npz", plans=plans, objectives=scores)
        np.save(args.output / "best_plan.npy", plans[pick])
        result = {"candidate": pick, "objective": scores[pick], "grid": config["grid"], **rows[pick]}
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
