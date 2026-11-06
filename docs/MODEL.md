# Planning and simulator conventions

This release keeps the historical environments' numerical behavior. Path handling, process management, model output shapes, and artifact saving have been made portable.

## Planning and observations

Each planning step modifies each consumer's PV allocation fraction by `tanh(raw_action)/16`, followed by clipping to `[0,1]`. The default horizon is 16 steps. The action scale remains `1/16` even when using a shorter smoke-test horizon.

There are 300 agents on IEEE-300 and 803 on Beijing. Beijing's electrical graph additionally contains upstream nodes. Their observations are zero-padded for graph propagation; only consumer nodes produce actions and values. Each agent has 212 observation features: 96 normalized net-demand samples, 96 normalized PV samples, the current allocation fraction, its upper and lower adjustment headroom, normalized available area, and a 16-dimensional step embedding.

The actor uses two GCN layers, a shared MLP, and a 32-dimensional learned agent-ID embedding. The critic uses two GCN layers and a shared node-level value head. Parameters are shared across agents and optimized with clipped PPO.

## AC power flow

A full-day evaluation solves 96 quarter-hour AC power flows. Generation-side reference nodes have fixed voltage phasors; the power flow determines their active and reactive injections. The planning objective uses ramping costs computed from these generator power trajectories.

The IEEE environment clips negative node net demand to zero and adjusts reactive load using its original power-factor rule. Beijing allows negative net demand and uses its original network construction and power factor. These grid-specific historical rules have been retained.

## Objective and reporting

The potential combines PV generation and ramping cost. Training receives the difference in potential between consecutive planning steps, including the original action-bound penalty if enabled. In command-line/code notation `--lambda2` is the grid-cost weight and `--lambda1` is the action-bound penalty. Default weights are 5 and 0.4 for IEEE and Beijing, respectively; the default action penalty is zero.

The historical ramp function uses `P[:,4:] - P[:,:-4]`: one-hour differences sampled every 15 minutes, without wraparound. Generator classes and ramp thresholds depend on the generator's peak in that evaluated day. The original piecewise coefficients and whole-ramp surcharge rule are retained.

Reported fields:

- `pv_mwh`: sum of the 96 PV power samples multiplied by 0.25 hours.
- `ramping_cost_raw`: the unconverted output of the historical ramp-cost function.
- `ramping_cost_usd`: `ramping_cost_raw * 0.25`, following the reporting convention used for the deployment comparisons. Divide by 1,000 for the paper-style table scale. Training still uses the original unconverted potential.
- `curtailment_proxy_mwh`: the sum of generator deficits below 40% of each generator's own evaluated daily peak, multiplied by 0.25.
- `installed_area_fraction`: area-weighted PV allocation fraction.

The Beijing potential subtracts a constant 4 from normalized ramping cost. It cancels in reward differences; metric conversion removes it.

## Candidate selection

Training stores every actual terminal deployment. `best_plan.npy` is selected using the terminal objective, including any configured action-bound penalty, across all retained training batches. `last.pt` contains the policy after the update. Use the saved deployment for exact-plan evaluation.
