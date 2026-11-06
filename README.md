# Grid-aware rooftop PV planning with multi-agent reinforcement learning

Research code for **Learning to Plan City-Scale Distributed Photovoltaics with Grid-Aware Multi-Agent Reinforcement Learning**.

The planner allocates rooftop PV capacity across grid nodes. A shared two-layer GCN actor and node-level critic learn from full-day AC power-flow feedback. This repository provides portable IEEE-300 and synthetic Beijing environments, training, and saved-plan evaluation.

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the PyTorch build appropriate for your CUDA version before `pip install -e .` if using a GPU. CPU execution is supported. `requirements-tested.txt` records the environment used for local validation.

## Quick start

Run one small update to check the complete training path:

```bash
python -m pv_planning.train --grid ieee --iterations 1 --rollouts 2 --steps 2 --epochs 1 --output runs/ieee_smoke
python -m pv_planning.train --grid beijing --iterations 1 --rollouts 2 --steps 2 --epochs 1 --workers 2 --output runs/beijing_smoke
```

Evaluate the supplied deployment examples:

```bash
python -m pv_planning.evaluate --grid ieee --plan examples/ieee_plan.npy --output runs/ieee_example
python -m pv_planning.evaluate --grid beijing --plan examples/beijing_plan.npy --output runs/beijing_example
```

Pretrained policy checkpoints are also included:

```bash
python -m pv_planning.evaluate --checkpoint examples/ieee_policy.pt --candidates 128 --output runs/ieee_pretrained
python -m pv_planning.evaluate --checkpoint examples/beijing_policy.pt --candidates 128 --output runs/beijing_pretrained
```

For training continuation, use the `last.pt` checkpoint produced by the training command below.

## Training

Use 128 joint trajectories, 16 planning steps, and 10 PPO epochs per update:

```bash
python -m pv_planning.train --grid ieee --cost-weight 5 --action-penalty 0 --iterations 300 --rollouts 128 --steps 16 --epochs 10 --workers 16 --device cuda:0 --output runs/ieee
python -m pv_planning.train --grid beijing --cost-weight 0.4 --action-penalty 0 --iterations 300 --rollouts 128 --steps 16 --epochs 10 --workers 16 --device cuda:0 --output runs/beijing
```

Use `--device cpu` without a GPU. Rollout workers always run on CPU; `--workers` controls the number of independent simulator processes. Set this to the CPU/memory resources available. The two default cost weights match the selected historical configurations. Use the learning-curve script below to monitor training.

Key options:

|Option|Default|Meaning|
|---|---:|---|
|`--seed`|2026|Initialization and deterministic rollout-seed schedule|
|`--hidden-dim`|128|GCN and critic hidden dimension|
|`--lr-actor`|0.0001|Actor learning rate|
|`--lr-critic`|0.0003|Critic learning rate|
|`--batch-size`|32|Joint planning steps per minibatch|
|`--entropy-weight`|0.01|Gaussian entropy coefficient; IPPO uses historical linear warm-up|
|`--gamma` / `--gae-lambda`|0.95 / 0.95|Return and advantage estimation|

Resume with the same configuration and output folder. `--iterations` specifies the total number of updates:

```bash
python -m pv_planning.train --grid ieee --iterations 300 --rollouts 128 --steps 16 --workers 16 --device cuda:0 --output runs/ieee --resume runs/ieee/last.pt
```

Each run writes:

- `config.json`: resolved run configuration.
- `metrics.jsonl`: learning curve, losses, and full-day simulator evaluation count; each full-day evaluation solves 96 AC power flows.
- `last.pt`: actor, critic, optimizer state, random-generator states, and completed iteration.
- `candidates/000000.npz`, etc.: the actual initial and final deployment vectors, objective scores, and seeds from each training batch.
- `best_plan.npy` and `best_plan.json`: the best terminal deployment encountered in training, with its metrics and source candidate.

A deployment vector contains 300 or 803 fractions in `[0,1]`. Candidate archives retain the initial and final deployments from every training batch.

## Evaluation

Evaluate the saved best training deployment directly:

```bash
python -m pv_planning.evaluate --grid ieee --plan runs/ieee/best_plan.npy --output runs/ieee/evaluation
```

Alternatively, generate 128 **new** candidate trajectories from a saved policy:

```bash
python -m pv_planning.evaluate --checkpoint runs/ieee/last.pt --candidates 128 --output runs/ieee/new_candidates
```

Policy evaluation samples stochastic actions by default; `--deterministic` uses mean actions. The checkpoint determines the grid and planning horizon.

Plot the training curves with the optional plotting dependency:

```bash
pip install -e ".[plot]"
python scripts/plot_learning_curve.py runs/ieee
```

## Data and simulation conventions

All required simulation inputs are bundled under `pv_planning/data/`; no original server paths are needed. See [data provenance](docs/DATA.md) and [model conventions](docs/MODEL.md) for file descriptions, units, and the historical objective.

Deployment origins and clipping records are provided in `examples/provenance.json`.

## Repository layout

```text
pv_planning/
  envs/            IEEE and Beijing simulators and Beijing network construction
  data/            Packaged simulation inputs and SHA-256 manifest
  models.py        Shared GCN actor, agent embeddings, and node-level critic
  train.py         IPPO training and candidate storage
  evaluate.py      Exact-plan evaluation and policy sampling
examples/          Historical deployment vectors and provenance
scripts/           Learning-curve plotting
tests/            Core regression checks
```

The root `train.py`, `IEEE_env.py`, and `model.py` retain compatibility entry points for the original IEEE release. The data now live inside the package so installations work outside the repository directory.

## Validation

```bash
python -m unittest discover -s tests -v
python -m compileall -q pv_planning
```

Local release checks and their results are recorded in [VALIDATION.md](docs/VALIDATION.md). GitHub Actions runs core checks and an IEEE end-to-end smoke run.
