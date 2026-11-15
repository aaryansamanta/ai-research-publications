"""Focused tests for the default synthetic traffic/delay lifecycle."""

import math
import unittest

from env.inferno_env import (
    GROUND_RESOURCE_TYPES,
    RESOURCE_DELAY_CONFIG,
    RESOURCE_TYPES,
    InfernoEnv,
)


class SyntheticTrafficTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = InfernoEnv(seed=77)

    def test_default_mode_and_configured_delays(self):
        self.assertEqual(self.env.traffic_mode, "synthetic")
        self.assertEqual(set(RESOURCE_DELAY_CONFIG), set(RESOURCE_TYPES))
        for rtype in RESOURCE_TYPES:
            self.assertGreaterEqual(self.env.delay_config[rtype]["dispatch_delay_ticks"], 0)
            self.assertGreaterEqual(self.env.delay_config[rtype]["arrival_setup_delay_ticks"], 0)

    def test_ground_resource_reserves_route_after_preparation(self):
        obs = self.env.reset(seed=77)
        zone = next(
            z for z in range(self.env.n_zones)
            if math.isfinite(self.env.zone_travel_time_s["water_team"][z])
        )
        _obs, _reward, _done, info = self.env.step([("water_team", zone)])
        self.assertEqual(info["dispatch"][0]["status"], "dispatched")
        self.assertEqual(sum(self.env._edge_resource_load.values()), 0.0)
        self.env.step([])
        self.assertGreater(sum(self.env._edge_resource_load.values()), 0.0)
        self.assertTrue(any(unit["state"] == "traveling" for unit in self.env.resources["water_team"]))

    def test_helicopter_has_preparation_delay_but_no_road_load(self):
        self.env.reset(ignition_point=(207, 222), seed=78)
        _obs, _reward, _done, info = self.env.step([("helicopter", 18)])
        self.assertEqual(info["dispatch"][0]["status"], "dispatched")
        self.assertTrue(any(unit["state"] == "preparing" for unit in self.env.resources["helicopter"]))
        self.assertEqual(sum(self.env._edge_resource_load.values()), 0.0)

    def test_legacy_mode_skips_new_preparation_phase(self):
        legacy = InfernoEnv(seed=79, traffic_mode="legacy")
        legacy.reset(seed=79)
        zone = next(
            z for z in range(legacy.n_zones)
            if math.isfinite(legacy.zone_travel_time_s["water_team"][z])
        )
        _obs, _reward, _done, info = legacy.step([("water_team", zone)])
        self.assertEqual(info["dispatch"][0]["status"], "dispatched")
        self.assertTrue(any(unit["state"] == "traveling" for unit in legacy.resources["water_team"]))


if __name__ == "__main__":
    unittest.main()
