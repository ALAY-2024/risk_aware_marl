import unittest
from dataclasses import dataclass

import numpy as np

from sat_net.traffic_stress import TrafficStressScenario


@dataclass
class Region:
    name: str


class TrafficStressScenarioTest(unittest.TestCase):
    def setUp(self):
        self.regions = [Region("A"), Region("B"), Region("C")]
        self.weights = np.array([0.5, 0.3, 0.2], dtype=float)

    def test_normal_scenario_preserves_distribution(self):
        scenario = TrafficStressScenario()
        probs, scale = scenario.source_distribution(self.regions, self.weights, 1000)
        np.testing.assert_allclose(probs, self.weights)
        self.assertAlmostEqual(scale, 1.0)

    def test_active_hotspot_increases_probability_and_total_load(self):
        scenario = TrafficStressScenario(
            name="stress",
            hotspot_regions=("A",),
            multiplier=2.0,
            start_ms=100.0,
            duration_ms=500.0,
        )
        probs, scale = scenario.source_distribution(self.regions, self.weights, 200.0)
        np.testing.assert_allclose(probs, [1.0 / 1.5, 0.3 / 1.5, 0.2 / 1.5])
        self.assertAlmostEqual(scale, 1.5)

    def test_outside_window_returns_baseline(self):
        scenario = TrafficStressScenario(
            hotspot_regions=("A",), multiplier=3.0, start_ms=100.0, duration_ms=100.0
        )
        for timestamp in (0.0, 99.9, 200.0, 1000.0):
            probs, scale = scenario.source_distribution(
                self.regions, self.weights, timestamp
            )
            np.testing.assert_allclose(probs, self.weights)
            self.assertAlmostEqual(scale, 1.0)

    def test_unknown_hotspot_region_rejected_when_active(self):
        scenario = TrafficStressScenario(
            hotspot_regions=("missing",),
            multiplier=2.0,
            start_ms=0.0,
            duration_ms=1000.0,
        )
        with self.assertRaises(ValueError):
            scenario.source_distribution(self.regions, self.weights, 1.0)

    def test_invalid_multiplier_rejected(self):
        with self.assertRaises(ValueError):
            TrafficStressScenario.from_config({"multiplier": 0.5})


if __name__ == "__main__":
    unittest.main()
