"""PPO fine-tune every checkpoint of a DreamerV3 run for one checkpoint interval.

Takes a DreamerV3 run directory (the logdir containing `ckpt/` and, ideally,
`config.yaml`) or its `ckpt/` directory. Checkpoints are named
`<timestamp>-<step>`. For each one, oldest first, the frozen world model, value
head and initial policy are loaded, and the policy is fine-tuned with PPO for
about as many environment steps as lie between that checkpoint and the next
one. The last checkpoint has no successor, so it reuses the previous interval.

Each fine-tuned model is written to its own directory:

  <out>/<checkpoint name>/
    ckpt/          fine-tuned PPO checkpoint
    metrics.jsonl  all metrics of that fine-tuning run
    scores.jsonl   episode scores
    config.yaml    the config used

Example:

  python scripts/ppo_finetune_checkpoints.py \
    --dreamer ~/logdir/dreamer/20250101T000000 \
    --out ~/logdir/ppo_sweep \
    --run.envs 16 --jax.platform cuda

Unrecognized flags (e.g. --ppo.epochs, --agent.opt.lr, --run.envs) are passed
through to ppo_finetune. The model config is read from the Dreamer run's
config.yaml, so model flags need not be repeated; if there is no config.yaml,
pass --configs as for ppo_finetune.py. Fine-tuned checkpoints that already
have a `done` marker in <out> are skipped, so an interrupted sweep can resume.
"""

import argparse
import gc
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import elements
import ruamel.yaml as yaml

from dreamerv3 import ppo_finetune


def list_checkpoints(directory):
  """Completed checkpoints as (step, path), sorted by step."""
  directory = elements.Path(directory)
  if (directory / 'ckpt').exists():
    directory = directory / 'ckpt'
  found = []
  for path in directory.glob('*'):
    if path.name == 'latest' or not (path / 'done').exists():
      continue
    found.append((int(path.name.rsplit('-', 1)[1]), path))
  found.sort(key=lambda x: x[0])
  if not found:
    raise FileNotFoundError(f'No completed checkpoints in {directory}')
  return found


def interval_steps(ckpts):
  """Env steps from each checkpoint to the next; the last reuses the prior."""
  steps = [b[0] - a[0] for a, b in zip(ckpts[:-1], ckpts[1:])]
  steps.append(steps[-1] if steps else 0)
  return steps


def main():
  ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  ap.add_argument('--dreamer', required=True, help='Dreamer run dir or ckpt dir')
  ap.add_argument('--out', required=True, help='Output directory')
  args, other = ap.parse_known_args()

  dreamer = elements.Path(str(pathlib.Path(args.dreamer).expanduser()))
  out = elements.Path(str(pathlib.Path(args.out).expanduser()))
  ckpts = list_checkpoints(dreamer)
  steps = interval_steps(ckpts)

  base = None
  cfgfile = dreamer / 'config.yaml'
  if not cfgfile.exists() and (dreamer.parent / 'config.yaml').exists():
    cfgfile = dreamer.parent / 'config.yaml'
  if cfgfile.exists():
    base = yaml.YAML(typ='safe').load(cfgfile.read())
    print('Model config from', cfgfile)

  print(f'{len(ckpts)} checkpoints:')
  for (step, path), n in zip(ckpts, steps):
    print(f'  {path.name}: fine-tune for ~{n} steps')

  for (step, path), n in zip(ckpts, steps):
    logdir = out / path.name
    if (logdir / 'ckpt').exists() and elements.Checkpoint(
        logdir / 'ckpt').latest():
      print(f'Skipping {path.name}: already fine-tuned')
      continue
    if n <= 0:
      print(f'Skipping {path.name}: only one checkpoint, no interval')
      continue

    flags = ['--ckpt', str(path), '--logdir', str(logdir), *other]
    # Parse once with a placeholder iteration count to learn the number of env
    # steps per iteration (envs * batch_length), which depends on the flags.
    config = ppo_finetune.parse_config(flags, base)
    if base is not None and not any(
        f.startswith('--logger.outputs') for f in other):
      config = config.update(logger=dict(outputs=['jsonl']))
    per_iter = config.run.envs * config.batch_length
    iters = max(1, round(n / per_iter))
    # Sync saves to the iteration count: save_every saves mid-run, and the
    # final save at the end of run() is the model we keep.
    config = config.update(ppo=dict(iters=iters))
    print(f'=== {path.name}: {iters} iterations x {per_iter} steps '
          f'= {iters * per_iter} steps (target {n}) ===')
    ppo_finetune.run(config)
    gc.collect()


if __name__ == '__main__':
  main()
