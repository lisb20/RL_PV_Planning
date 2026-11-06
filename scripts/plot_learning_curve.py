"""Plot the training log: pip install matplotlib; python scripts/plot_learning_curve.py runs/ieee."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('--output', type=Path)
args = p.parse_args()
rows = [json.loads(line) for line in (args.run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
if not rows:
    p.error('The training log has no completed updates')
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
for ax, (key, title) in zip(axes, [('mean_return', 'Mean episode return'),
                                  ('best_objective', 'Best terminal objective'),
                                  ('mean_ramping_cost_usd', 'Mean ramping cost (USD)')]):
    ax.plot([r['iteration'] for r in rows], [r[key] for r in rows])
    ax.set(xlabel='Training update', ylabel=title)
    ax.grid(alpha=0.25)
fig.savefig(args.output or args.run / 'learning_curve.png', dpi=180)
