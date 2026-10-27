"""Plot episode scores of a Dreamer-only run against a Dreamer-then-PPO run.

  python scripts/plot_dreamer_vs_ppo.py OUTDIR_DREAMER OUTDIR_PPO [--out plot.png]

Both arguments are run output dirs as made by crafter_dreamer_then_ppo.sh.
  OUTDIR_DREAMER: scores from OUTDIR_DREAMER/dreamer
  OUTDIR_PPO:     scores from OUTDIR_PPO/dreamer, then OUTDIR_PPO/ppo
Each logdir holds scores.jsonl (or metrics.jsonl). The PPO run's step counter
restarts at 0, so its steps are shifted by the last step of OUTDIR_PPO/dreamer
to put it on one continuous axis. A vertical line marks where PPO starts.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

KEY = 'episode/score'
DREAMER_COLOR = '#0072B2'
PPO_RUN_DREAMER_COLOR = '#009E73'
PPO_COLOR = '#D55E00'


def load_scores(logdir):
  logdir = pathlib.Path(logdir)
  for name in ('scores.jsonl', 'metrics.jsonl'):
    path = logdir / name
    if path.exists():
      break
  else:
    raise FileNotFoundError(f'No scores.jsonl or metrics.jsonl in {logdir}')
  steps, scores = [], []
  with path.open() as f:
    for line in f:
      if not line.strip():
        continue
      row = json.loads(line)
      if KEY in row:
        steps.append(row['step'])
        scores.append(row[KEY])
  if not steps:
    raise ValueError(f'No {KEY} entries in {path}')
  return np.array(steps, float), np.array(scores, float)


def smooth(y, k):
  k = min(k, len(y))
  return np.convolve(y, np.ones(k) / k, mode='valid'), k


def draw(ax, x, y, color, label, k):
  ax.plot(x, y, color=color, alpha=0.2, linewidth=1)
  ys, k = smooth(y, k)
  ax.plot(x[k - 1:], ys, color=color, linewidth=2, label=f'{label} ({k}-ep mean)')


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('dreamer_outdir', help='Dreamer-only run outdir')
  parser.add_argument('ppo_outdir', help='Dreamer-then-PPO run outdir')
  parser.add_argument('--smooth', type=int, default=20,
                      help='moving-average window in episodes (default 20)')
  parser.add_argument('--out', help='save to this file instead of showing')
  args = parser.parse_args()

  dreamer = pathlib.Path(args.dreamer_outdir)
  both = pathlib.Path(args.ppo_outdir)
  dx, dy = load_scores(dreamer / 'dreamer')
  bx, by = load_scores(both / 'dreamer')
  px, py = load_scores(both / 'ppo')
  start = bx.max()  # PPO continues from where its Dreamer stage stopped
  px = px + start

  fig, ax = plt.subplots(figsize=(10, 5))
  draw(ax, dx, dy, DREAMER_COLOR, 'Dreamer only', args.smooth)
  draw(ax, bx, by, PPO_RUN_DREAMER_COLOR, 'Dreamer (PPO run)', args.smooth)
  draw(ax, px, py, PPO_COLOR, 'PPO fine-tune', args.smooth)
  ax.axvline(start, color='gray', linestyle='--', linewidth=1)
  ax.annotate('PPO starts', (start, 1), xycoords=('data', 'axes fraction'),
              xytext=(4, -12), textcoords='offset points', color='dimgray')
  ax.set_xlabel('env step')
  ax.set_ylabel('episode score')
  ax.set_title('Episode score: Dreamer only vs Dreamer then PPO')
  ax.grid(alpha=0.2)
  for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
  ax.legend(frameon=False, loc='upper left')
  fig.tight_layout()

  if args.out:
    fig.savefig(args.out, dpi=150)
    print('Saved', args.out)
  else:
    plt.show()


if __name__ == '__main__':
  main()
