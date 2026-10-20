"""Smallest runnable task: a random agent acting in the Dummy env."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import embodied
from embodied.envs.dummy import Dummy

env_fns = [lambda: Dummy('disc', length=10) for _ in range(2)]
driver = embodied.Driver(env_fns, parallel=False)
agent = embodied.RandomAgent(env_fns[0]().obs_space, env_fns[0]().act_space)

totals = []
driver.on_step(lambda tran, worker: totals.append(float(tran['reward'])))
driver(agent.policy, steps=50)
driver.close()
print(f'Ran {len(totals)} steps, total reward {sum(totals):.0f}')
