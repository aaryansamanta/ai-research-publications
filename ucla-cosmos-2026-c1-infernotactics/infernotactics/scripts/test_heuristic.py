import sys
sys.path.insert(0, r'A:\AI\InfernoTactics\infernotactics\src')
import numpy as np
from env.inferno_env import InfernoEnv, RESOURCE_TYPES
from train.heuristic_policy import HeuristicPolicy

env = InfernoEnv(seed=0)
obs = env.reset(ignition_point=(207, 222), seed=0, use_real_weather=True)

policy = HeuristicPolicy(env)

obs = env.reset(ignition_point=(207, 222), seed=0, use_real_weather=True)
actions = policy.decide_actions(obs['grid'][-1], np.array([obs['scalars'][f'{r}_available'] for r in RESOURCE_TYPES], dtype=np.float32))
print('actions:', actions)
obs, reward, done, info = env.step(actions)
print(f'Tick {info["tick"]}: reward={reward:.1f}, destroyed={info["buildings_destroyed"]}, contained={info["contained"]}, action={actions}')
if done:
    print('Done')
else:
    print('Not done')
print(f'Final: destroyed={info["buildings_destroyed"]}, contained={info["contained"]}, timeout={info.get("timeout")}')