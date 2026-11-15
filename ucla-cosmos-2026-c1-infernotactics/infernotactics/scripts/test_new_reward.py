import sys
sys.path.insert(0, r'A:\AI\InfernoTactics\infernotactics\src')
import numpy as np
from env.inferno_env import (
    InfernoEnv, TRAINING_IGNITION_POINT, RESOURCE_TYPES,
    FIRE_PENALTY_PER_CELL_PER_TICK, DISPATCH_COST, TRENCH_BREAK_HOLD_BONUS,
)

# Test 1: Verify per-tick fire penalty
print("=" * 60)
print("TEST 1: Per-tick fire penalty")
print("=" * 60)
env = InfernoEnv(seed=42)
obs = env.reset(ignition_point=(207, 222), seed=42, use_real_weather=True)
done = False
tick_count = 0
fire_penalties = []
active_cells_history = []
while not done and tick_count < 20:
    next_obs, reward, done, info = env.step([])
    rc = info.get('reward_components', {})
    fire_pen = rc.get('fire_penalty', 0)
    active = info.get('active_fire_cells', 0)
    fire_penalties.append(fire_pen)
    active_cells_history.append(active)
    print(f"Tick {info['tick']}: active_fire={active}, fire_penalty={fire_pen}, "
          f"expected={-FIRE_PENALTY_PER_CELL_PER_TICK * active}")
    tick_count += 1

# Verify the penalty equals -0.5 * active_cells each tick
correct = all(abs(p - (-FIRE_PENALTY_PER_CELL_PER_TICK * a)) < 1e-6
              for p, a in zip(fire_penalties, active_cells_history))
print(f"\nFire penalty formula correct: {correct}")
print(f"Max fire penalty observed: {min(fire_penalties):.1f}")

# Test 2: Verify dispatch costs
print("\n" + "=" * 60)
print("TEST 2: Per-dispatch costs")
print("=" * 60)
env = InfernoEnv(seed=42)
obs = env.reset(ignition_point=(207, 222), seed=42, use_real_weather=True)
# Dispatch one of each resource type
dispatches = [
    ("water_team", 18),
    ("trench_crew", 19),
    ("rescue_vehicle", 18),
    ("helicopter", 18),
]
next_obs, reward, done, info = env.step(dispatches)
rc = info.get('reward_components', {})
print(f"Dispatch cost: {rc.get('dispatch_cost', 'N/A')}")
expected_cost = -sum(DISPATCH_COST[rtype] for rtype, _ in dispatches)
print(f"Expected cost: {expected_cost}")
print(f"Match: {abs(rc.get('dispatch_cost', 0) - expected_cost) < 1e-6}")

# Test 3: Verify multi-dispatch is supported
print("\n" + "=" * 60)
print("TEST 3: Multi-dispatch in single tick")
print("=" * 60)
env = InfernoEnv(seed=42)
obs = env.reset(ignition_point=(207, 222), seed=42, use_real_weather=True)
# Dispatch 5 helicopters in one tick
dispatches = [("helicopter", 18)] * 5
next_obs, reward, done, info = env.step(dispatches)
dispatch_results = info.get('dispatch', [])
print(f"Attempted 5 helicopter dispatches, got {len(dispatch_results)} results")
for d in dispatch_results:
    print(f"  Status: {d.get('status')}, station: {d.get('station_id', 'N/A')}")

# Test 4: Verify total reward with no actions
print("\n" + "=" * 60)
print("TEST 4: Empty step (no dispatches)")
print("=" * 60)
env = InfernoEnv(seed=42)
obs = env.reset(ignition_point=(207, 222), seed=42, use_real_weather=True)
total_reward = 0
for i in range(5):
    next_obs, reward, done, info = env.step([])
    total_reward += reward
    print(f"Tick {info['tick']}: reward={reward:.1f}, done={done}")
print(f"Total reward after 5 ticks (no actions): {total_reward:.1f}")
print("Negative expected due to fire penalty, no extinguish reward, no building loss (yet)")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)