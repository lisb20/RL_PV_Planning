"""Uniform allocation, demand-based allocation and simulator-evaluated GA references."""
import argparse
import json
from pathlib import Path
import numpy as np
from .runtime import make_env, metrics
from .train import seed_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid', choices=['ieee', 'beijing'], default='ieee')
    p.add_argument('--method', choices=['uniform', 'demand', 'ga'], default='uniform')
    p.add_argument('--fraction', type=float, default=0.5)
    p.add_argument('--cost-weight', type=float)
    p.add_argument('--population', type=int, default=50)
    p.add_argument('--generations', type=int, default=100)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--output', type=Path, default=Path('runs/baseline'))
    args = p.parse_args()
    if not 0 <= args.fraction <= 1 or args.population < 3 or args.generations < 1:
        p.error('Require fraction in [0,1], population >= 3, generations >= 1')
    seed_all(args.seed)
    env = make_env(args.grid, steps=1, cost_weight=args.cost_weight)
    env.reset()
    calls = 1

    def evaluate(plan):
        nonlocal calls
        # Set deployment directly before stepping, so all candidates are scored
        # without a random-start action penalty (action penalty is zero here).
        env.solar_percentage = plan.copy()
        _, _, _, info = env.step(np.zeros((env.N_agent, 1)))
        calls += 1
        return float(env.last_reward), metrics(env, info, args.grid)

    plan = np.full(env.N_agent, args.fraction)
    if args.method == 'demand':
        demand = env.P if args.grid == 'ieee' else env.power_demand_ref
        weight = demand[:, 48].clip(0)
        # Allocate target area by noon demand, capping individual roof fractions.
        ratio = np.divide(weight, env.area_vec, out=np.zeros_like(weight), where=env.area_vec > 0)
        if args.fraction == 1:
            plan = np.ones(env.N_agent)
        elif args.fraction > 0:
            target = args.fraction * env.area_vec.sum()
            if env.area_vec[ratio > 0].sum() < target:
                raise ValueError('Demand-positive nodes cannot supply the target PV area')
            lo, hi = 0., 1.
            while np.dot(np.minimum(hi * ratio, 1), env.area_vec) < target:
                hi *= 2
            for _ in range(80):
                mid = (lo + hi) / 2
                if np.dot(np.minimum(mid * ratio, 1), env.area_vec) < target:
                    lo = mid
                else:
                    hi = mid
            plan = np.minimum(hi * ratio, 1)
    args.output.mkdir(parents=True, exist_ok=True)
    best_score, best_metrics = evaluate(plan)
    if args.method == 'ga':
        # Blend crossover, Gaussian mutation, and tournament selection. This
        # reference optimizes the same PV-minus-ramping objective as the policy.
        population = np.random.uniform(0, 1, (args.population, env.N_agent))
        for generation in range(args.generations):
            scored = [evaluate(candidate) for candidate in population]
            fitness = np.array([row[0] for row in scored])
            pick = int(fitness.argmax())
            if fitness[pick] > best_score:
                best_score, best_metrics = scored[pick]
                plan = population[pick].copy()
            parents = []
            for _ in range(args.population):
                tournament = np.random.choice(args.population, 3, replace=False)
                parents.append(population[tournament[fitness[tournament].argmax()]].copy())
            children = np.stack(parents)
            for i in range(0, args.population - 1, 2):
                if np.random.random() < 0.5:
                    blend = np.random.uniform(-0.5, 1.5, env.N_agent)
                    first, second = children[i].copy(), children[i+1].copy()
                    children[i] = blend * first + (1-blend) * second
                    children[i+1] = blend * second + (1-blend) * first
            for child in children:
                if np.random.random() < 0.2:
                    mask = np.random.random(env.N_agent) < 0.2
                    child[mask] += np.random.normal(0, 0.1, mask.sum())
            population = np.clip(children, 0, 1)
            with (args.output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps({'generation': generation, 'objective': best_score}) + '\n')
    np.save(args.output / 'best_plan.npy', plan)
    result = {'method': args.method, 'grid': args.grid, 'seed': args.seed,
              'objective': best_score, 'full_day_evaluations': calls, **best_metrics}
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
