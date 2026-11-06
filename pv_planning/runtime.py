"""Shared environment construction and metric conversion."""
import numpy as np


def make_env(grid="ieee", steps=16, cost_weight=None, action_penalty=0.0):
    if grid == "ieee":
        from .envs.ieee import myenv
        return myenv(max_step=steps, lambda1=action_penalty,
                     lambda2=5.0 if cost_weight is None else cost_weight)
    if grid == "beijing":
        from .envs.beijing import myenv
        return myenv(max_step=steps, lambda1=action_penalty,
                     lambda2=0.4 if cost_weight is None else cost_weight)
    raise ValueError(f"Unknown grid: {grid}")


def metrics(env, info, grid):
    """Convert legacy simulator outputs to named units (96 quarter-hour samples)."""
    raw_ramping = (float(info[3]) + (4 if grid == "beijing" else 0)) * 50000
    return {
        "pv_mwh": float(info[0]) * 0.25,
        "curtailment_proxy_mwh": -float(info[1]) * 0.25,
        "ramping_cost_usd": raw_ramping * 0.25,
        "ramping_cost_raw": raw_ramping,
        "mean_max_ramping_rate": float(info[2]),
        "action_bound_excess": float(info[4]),
        "installed_area_fraction": float(np.dot(env.solar_percentage, env.area_vec) / env.area_vec.sum()),
    }
