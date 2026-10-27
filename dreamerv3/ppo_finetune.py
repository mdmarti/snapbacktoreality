"""Fine-tune the policy of a trained DreamerV3 agent with PPO on real rewards.

Everything except the policy head is frozen: the encoder, RSSM, decoder, reward
head, continue head and value head are loaded from a trained DreamerV3
checkpoint and never updated. The agent acts in the real environment, the
frozen world model turns the observation stream into recurrent latent features,
and the policy is optimized with PPO on advantages computed from the rewards
the environment actually returned (GAE, using the frozen value head as the
baseline). No imagined rollouts are used.

Example:

  python dreamerv3/ppo_finetune.py \
    --ckpt ~/logdir/dreamer/20250101T000000/ckpt \
    --configs crafter \
    --batch_size 16 --run.envs 16 \
    --ppo.iters 500

The model flags (--configs, --agent.*) must match the ones used for training
the checkpoint. Each iteration collects `batch_length` steps from every one of
`run.envs` environments, and `run.envs` must be a multiple of `batch_size`;
each PPO minibatch consists of `batch_size` environments.
"""

import collections
import importlib
import os
import pathlib
import sys
from functools import partial as bind

folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder.parent))
sys.path.insert(1, str(folder.parent.parent))
__package__ = folder.name

import elements
import embodied
import embodied.jax
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import portal
import ruamel.yaml as yaml
from embodied.jax import internal

from . import agent as dreamer
from . import main as dmain

f32 = jnp.float32
sg = jax.lax.stop_gradient
sample = lambda xs: jax.tree.map(lambda x: x.sample(nj.seed()), xs)

# Parameters that are loaded from the Dreamer checkpoint. The optimizer state
# (opt/) is deliberately excluded because the PPO optimizer only tracks the
# policy and starts from scratch.
LOAD_REGEX = (
    r'^(enc|dyn|dec|rew|con|pol|val|slowval|retnorm|valnorm|advnorm)/')


class PPOAgent(dreamer.Agent):
  """DreamerV3 agent whose train step is PPO on the policy head only."""

  def __init__(self, obs_space, act_space, config):
    super().__init__(obs_space, act_space, config)
    assert not config.replay_context, 'PPO fine-tuning needs replay_context=0'
    self.ppo = config.ppo
    # The optimizer holds all modules by default. Restricting it to the policy
    # is what freezes the rest: the other parameters get no gradient and no
    # update, and the value/slow-value targets are not touched at all.
    self.modules = [self.pol]
    self.opt.modules = self.modules

  @property
  def policy_keys(self):
    # The actor needs the value head to produce baselines while acting.
    return '^(enc|dyn|dec|pol|val|valnorm)/'

  @property
  def ext_space(self):
    rssm = self.config.dyn.rssm
    size = rssm.deter + rssm.stoch * rssm.classes
    return {
        'ppo_feat': elements.Space(np.float32, (size,)),
        'ppo_logp': elements.Space(np.float32),
        'ppo_adv': elements.Space(np.float32),
    }

  def policy(self, carry, obs, mode='train'):
    (enc_carry, dyn_carry, dec_carry, prevact) = carry
    kw = dict(training=False, single=True)
    reset = obs['is_first']
    enc_carry, _, tokens = self.enc(enc_carry, obs, reset, **kw)
    dyn_carry, _, feat = self.dyn.observe(
        dyn_carry, tokens, prevact, reset, **kw)
    if dec_carry:
      dec_carry, _, _ = self.dec(dec_carry, feat, reset, **kw)
    x = self.feat2tensor(feat)
    dist = self.pol(x, bdims=1)
    act = sample(dist)
    logp = sum([dist[k].logp(act[k]) for k in act])
    voffset, vscale = self.valnorm.stats()
    val = self.val(x, 1).pred() * vscale + voffset
    out = {'ppo_feat': x, 'ppo_logp': logp, 'ppo_val': val}
    out['finite'] = elements.tree.flatdict(jax.tree.map(
        lambda x: jnp.isfinite(x).all(range(1, x.ndim)),
        dict(obs=obs, carry=carry, tokens=tokens, feat=feat, act=act)))
    carry = (enc_carry, dyn_carry, dec_carry, act)
    return carry, act, out

  def train(self, carry, data):
    if nj.creating():
      # Parameter initialization only traces the code that runs, so touch the
      # frozen modules once here. Otherwise they would not exist as parameters
      # and the checkpoint could not be loaded into them.
      enc_carry, dyn_carry, dec_carry, lastact = carry
      obs = {k: data[k] for k in self.obs_space}
      prepend = lambda x, y: jnp.concatenate([x[:, None], y[:, :-1]], 1)
      prevact = {k: prepend(lastact[k], data[k]) for k in self.act_space}
      self.loss(
          (enc_carry, dyn_carry, dec_carry), obs, prevact, training=False)
    metrics, mets = self.opt(self.ppo_loss, data, has_aux=True)
    metrics.update(mets)
    return carry, {}, metrics

  def ppo_loss(self, data):
    cfg = self.ppo
    dist = self.pol(data['ppo_feat'], 2)
    logp = f32(sum([dist[k].logp(sg(data[k])) for k in self.act_space]))
    ent = f32(sum([dist[k].entropy() for k in self.act_space]))
    adv = sg(data['ppo_adv'])
    # Steps with is_last have no action (the env resets), so they are masked.
    valid = f32(~data['is_last'])
    count = jnp.maximum(valid.sum(), 1.0)
    mean = lambda x: (x * valid).sum() / count

    logratio = logp - sg(data['ppo_logp'])
    ratio = jnp.exp(logratio)
    clipped = jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)
    policy_loss = -jnp.minimum(ratio * adv, clipped * adv)
    loss = mean(policy_loss - cfg.ent * ent)

    mets = {
        'ppo/ratio': sg(mean(ratio)),
        'ppo/clipfrac': sg(mean(f32(jnp.abs(ratio - 1) > cfg.clip))),
        'ppo/approx_kl': sg(mean(ratio - 1 - logratio)),
        'ppo/entropy': sg(mean(ent)),
    }
    return loss, mets

  def report(self, carry, data):
    return carry, {}


def make_agent(config):
  env = dmain.make_env(config, 0)
  notlog = lambda k: not k.startswith('log/')
  obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  env.close()
  return PPOAgent(obs_space, act_space, elements.Config(
      **config.agent,
      ppo=dict(clip=config.ppo.clip, ent=config.ppo.ent),
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  ))


def resolve_checkpoint(path):
  """Accept a checkpoint dir, a run's ckpt dir, or a run logdir."""
  path = elements.Path(path)
  if (path / 'done').exists():
    return path
  if (path / 'ckpt').exists():
    path = path / 'ckpt'
  latest = elements.Checkpoint(path).latest()
  if not latest:
    raise FileNotFoundError(f'No checkpoint found in {path}')
  return latest


def gae(rew, last, term, val, gamma, lam):
  """Advantages from real rewards and the frozen value baseline.

  All inputs have shape (envs, T + 1). Entry t holds the observation at step t
  (including the reward received on arriving there), so the reward for the
  action taken at step t is rew[:, t + 1]. Returns advantages and returns for
  steps 0..T-1. Steps with last[:, t] set have no action and get garbage that
  must be masked by the caller.
  """
  T = rew.shape[1] - 1
  adv = np.zeros(rew[:, :T].shape, np.float32)
  running = np.zeros(rew.shape[0], np.float32)
  for t in reversed(range(T)):
    nonterm = 1.0 - term[:, t + 1]
    delta = rew[:, t + 1] + gamma * nonterm * val[:, t + 1] - val[:, t]
    # Cut the trace at episode boundaries but still bootstrap on time limits.
    cont = 1.0 - last[:, t + 1]
    running = delta + gamma * lam * cont * running
    adv[:, t] = running
  return adv, adv + val[:, :T]


def parse_config(argv=None, base=None):
  """Build the PPO config from flags. `base` (a mapping, e.g. the config.yaml
  saved by the Dreamer run) replaces the --configs blocks when given."""
  configs = elements.Path(folder / 'configs.yaml').read()
  configs = yaml.YAML(typ='safe').load(configs)
  parsed, other = elements.Flags(
      configs=['defaults'], ckpt='',
      ppo=dict(
          iters=1000, epochs=4, clip=0.2, ent=3e-4, gamma=0.0, lam=0.95,
          save_every=50),
  ).parse_known(argv)
  if base is not None:
    config = elements.Config(base)
  else:
    config = elements.Config(configs['defaults'])
    for name in parsed.configs:
      config = config.update(configs[name])
  config = config.update(
      replay_context=0,
      logger=dict(filter='score|length|fps|ppo/|train/opt/loss'))
  config = elements.Config(**config, ppo=parsed.ppo, ckpt=parsed.ckpt)
  config = elements.Flags(config).parse(other)
  config = config.update(logdir=(
      config.logdir.format(timestamp=elements.timestamp())))
  assert config.ckpt, 'Pass the trained Dreamer checkpoint with --ckpt'
  return config


def main(argv=None):
  run(parse_config(argv))


def run(config):
  args = config.run
  B, T = config.batch_size, config.batch_length
  assert args.envs % B == 0, (
      f'run.envs ({args.envs}) must be a multiple of batch_size ({B})')
  gamma = config.ppo.gamma or 1 - 1 / config.agent.horizon

  logdir = elements.Path(config.logdir)
  print('Logdir:', logdir)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')

  def init():
    elements.timer.global_timer.enabled = config.logger.timer
  portal.setup(
      errfile=config.errfile and logdir / 'error',
      clientkw=dict(logging_color='cyan'),
      serverkw=dict(logging_color='cyan'),
      initfns=[init],
      ipv6=config.ipv6,
  )

  agent = make_agent(config)
  logger = dmain.make_logger(config)
  step = logger.step
  rng = np.random.default_rng(config.seed)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  source = resolve_checkpoint(config.ckpt)
  print(f'Loading Dreamer checkpoint (frozen parts + policy): {source}')
  elements.checkpoint.load(source, dict(
      agent=bind(agent.load, regex=LOAD_REGEX)))

  episodes = collections.defaultdict(elements.Agg)
  fns = [bind(dmain.make_env, config, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())

  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
  driver.on_step(logfn)

  # Rolling per-env buffers. The last transition of an iteration is kept as
  # the first one of the next, so every iteration has T + 1 transitions: T
  # to train on plus the final one for bootstrapping.
  bufs = [[] for _ in range(args.envs)]
  driver.on_step(lambda tran, worker: bufs[worker].append(
      {k: v for k, v in tran.items() if not k.startswith('log/')}))

  def make_batch(data):
    # Mirrors what agent.stream does for batches coming from replay.
    data = internal.device_put(data, agent.train_sharded)
    with agent.n_batches.lock:
      counter = agent.n_batches.value
      agent.n_batches.value += 1
    seed = agent._seeds(counter, agent.train_mirrored)
    return {**data, 'seed': seed}

  carry = agent.init_train(B)
  train_agg = elements.Agg()
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)

  print('Start PPO loop')
  for it in range(config.ppo.iters):
    rounds = T + 1 - len(bufs[0])
    driver(policy, steps=rounds * args.envs)
    assert all(len(b) == T + 1 for b in bufs)
    roll = {
        k: np.stack([np.stack([t[k] for t in b]) for b in bufs])
        for k in bufs[0][0]}
    for b in bufs:
      del b[:-1]

    adv, ret = gae(
        roll['reward'], f32np(roll['is_last']), f32np(roll['is_terminal']),
        roll['ppo_val'], gamma, config.ppo.lam)
    valid = ~roll['is_last'][:, :T]
    mean, std = adv[valid].mean(), adv[valid].std()
    adv = np.where(valid, (adv - mean) / (std + 1e-8), 0.0)

    data = {k: roll[k][:, :T] for k in agent.spaces if k != 'ppo_adv'}
    data['ppo_adv'] = adv
    data = {k: np.asarray(v, agent.spaces[k].dtype) for k, v in data.items()}

    for _ in range(config.ppo.epochs):
      perm = rng.permutation(args.envs)
      for start in range(0, args.envs, B):
        idx = perm[start: start + B]
        batch = make_batch({k: v[idx] for k, v in data.items()})
        carry, _, mets = agent.train(carry, batch)
        train_agg.add(mets, prefix='train')

    val = roll['ppo_val'][:, :T][valid]
    retv = ret[valid]
    logger.add(train_agg.result())
    logger.add({
        'iteration': it,
        'return_mean': retv.mean(),
        'value_mean': val.mean(),
        'explained_var': 1 - np.var(retv - val) / max(np.var(retv), 1e-8),
        'adv_raw_std': std,
    }, prefix='ppo')
    logger.write()

    if (it + 1) % config.ppo.save_every == 0:
      cp.save()

  cp.save()
  logger.close()


def f32np(x):
  return np.asarray(x, np.float32)


if __name__ == '__main__':
  main()
