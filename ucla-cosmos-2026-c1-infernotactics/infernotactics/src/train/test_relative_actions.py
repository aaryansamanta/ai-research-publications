"""Focused tests for the v8 semantic action boundary."""

import unittest

import numpy as np

from env.inferno_env import InfernoEnv, RESOURCE_TYPES
from train.relative_actions import TARGET_TYPES, decode_action, resolve_relative_targets


class RelativeActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = InfernoEnv(seed=17)
        cls.env.reset(seed=17)

    def test_same_semantic_action_resolves_to_current_fire(self):
        first = self.env.reset(ignition_point=(207, 222), seed=1)
        first_zones, first_features = resolve_relative_targets(self.env, first)
        second = self.env.reset(ignition_point=(57, 371), seed=1)
        second_zones, second_features = resolve_relative_targets(self.env, second)

        active_idx = TARGET_TYPES.index("active_fire")
        self.assertNotEqual(first_zones[0, active_idx], second_zones[0, active_idx])
        # The anchor zone is helicopter-only in the real road graph.  Use the
        # air resource to test the semantic target rather than treating a
        # legitimate ground-routing infinity as a resolver failure.
        helicopter_idx = RESOURCE_TYPES.index("helicopter")
        self.assertGreater(first_features[helicopter_idx, active_idx, 1], 0.0)
        self.assertGreater(second_features[helicopter_idx, active_idx, 1], 0.0)

    def test_noop_is_a_real_decodable_action(self):
        zones = np.full((len(RESOURCE_TYPES), len(TARGET_TYPES)), -1, dtype=np.int64)
        noop = TARGET_TYPES.index("noop")
        zones[:, noop] = -1
        self.assertIsNone(decode_action(0, noop, zones))

    def test_targets_have_stable_shapes_and_valid_noop(self):
        obs = self.env.reset(ignition_point=(207, 222), seed=2)
        zones, features = resolve_relative_targets(self.env, obs)
        self.assertEqual(zones.shape, (len(RESOURCE_TYPES), len(TARGET_TYPES)))
        self.assertEqual(features.shape[:2], zones.shape)
        self.assertEqual(features.shape[2], 10)
        self.assertTrue(np.all(zones[:, -1] == -1))


if __name__ == "__main__":
    unittest.main()
