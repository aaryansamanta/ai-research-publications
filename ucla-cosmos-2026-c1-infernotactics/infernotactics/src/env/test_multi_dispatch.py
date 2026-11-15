"""Tests for list-only, same-tick multi-dispatch semantics."""

import math
import unittest

from env.inferno_env import InfernoEnv


class MultiDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = InfernoEnv(seed=101)

    def test_step_requires_a_list(self):
        self.env.reset(seed=101)
        with self.assertRaises(ValueError):
            self.env.step(("water_team", 0))

    def test_two_dispatches_share_one_simulation_tick(self):
        self.env.reset(seed=102)
        zones = [
            z for z in range(self.env.n_zones)
            if math.isfinite(self.env.zone_travel_time_s["water_team"][z])
        ]
        _obs, _reward, _done, info = self.env.step([
            ("water_team", zones[0]),
            ("water_team", zones[1]),
        ])
        self.assertEqual(info["tick"], 1)
        self.assertEqual(len(info["dispatch"]), 2)
        self.assertTrue(all(item["status"] == "dispatched" for item in info["dispatch"]))

    def test_empty_list_advances_fire_without_dispatch(self):
        self.env.reset(seed=103)
        _obs, _reward, _done, info = self.env.step([])
        self.assertEqual(info["tick"], 1)
        self.assertEqual(info["dispatch"], [])

    def test_same_tick_dispatches_consume_available_units(self):
        self.env.reset(seed=104)
        zone = next(
            z for z in range(self.env.n_zones)
            if math.isfinite(self.env.zone_travel_time_s["water_team"][z])
        )
        self.env.step([("water_team", zone), ("water_team", zone)])
        available = sum(u["state"] == "available" for u in self.env.resources["water_team"])
        self.assertEqual(available, 1)


if __name__ == "__main__":
    unittest.main()
