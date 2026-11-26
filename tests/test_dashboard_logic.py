import unittest

import numpy as np

from dashboard.logic import (
    SCENARIOS,
    grid_svg,
    impact_summary,
    make_perturbation,
    validate_api_url,
)


class DashboardLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.latitudes = np.linspace(6.5, 37.5, 32)
        cls.longitudes = np.linspace(65.5, 100.5, 36)
        cls.states = np.zeros((3, 6, 32, 36), dtype=np.float32)
        cls.states[:, 0] = 290.0
        cls.states[:, 1] = 5.0
        cls.states[:, 2] = 2.0
        cls.states[:, 3] = 1.0
        cls.states[:, 4] = 101000.0
        cls.states[:, 5] = 30.0

    def test_only_local_http_servers_are_allowed(self):
        self.assertEqual(validate_api_url("http://localhost:8000"), "http://localhost:8000")
        self.assertEqual(validate_api_url("http://127.0.0.1:8000/"), "http://127.0.0.1:8000")
        for url in (
            "https://127.0.0.1:8000",
            "http://example.com",
            "http://127.0.0.1:8000/remote",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_api_url(url)

    def test_zero_shock_is_exactly_zero(self):
        perturbation = make_perturbation(
            SCENARIOS[3], self.states, self.latitudes, self.longitudes
        )
        self.assertEqual(perturbation.shape, (6, 32, 36))
        self.assertEqual(np.count_nonzero(perturbation), 0)

    def test_warm_ocean_shock_is_local_and_changes_only_requested_channels(self):
        perturbation = make_perturbation(
            SCENARIOS[1], self.states, self.latitudes, self.longitudes
        )
        self.assertEqual(np.count_nonzero(perturbation[0]), 144)
        self.assertTrue(np.all(perturbation[0][perturbation[0] != 0] == 3.0))
        self.assertTrue(
            np.all(perturbation[5][perturbation[5] != 0] == np.float32(4.5))
        )
        for channel in (1, 2, 3, 4):
            self.assertEqual(np.count_nonzero(perturbation[channel]), 0)

    def test_wind_shock_preserves_signed_wind_physics(self):
        states = self.states.copy()
        states[-1, 2] = -2.0
        states[-1, 3] = 3.0
        perturbation = make_perturbation(
            SCENARIOS[2], states, self.latitudes, self.longitudes
        )
        self.assertEqual(np.count_nonzero(perturbation[2]), 120)
        self.assertTrue(np.allclose(perturbation[2][perturbation[2] != 0], 0.8))
        self.assertTrue(np.allclose(perturbation[3][perturbation[3] != 0], -1.2))

    def test_invalid_inputs_and_shocks_fail_loudly(self):
        with self.assertRaises(ValueError):
            make_perturbation(
                SCENARIOS[0], self.states[:, :, :, :-1], self.latitudes, self.longitudes
            )
        with self.assertRaises(ValueError):
            make_perturbation(
                "unknown", self.states, self.latitudes, self.longitudes
            )
        with self.assertRaises(ValueError):
            make_perturbation(
                SCENARIOS[4],
                self.states,
                self.latitudes,
                self.longitudes,
                custom_latitude=90.0,
            )

    def test_impacts_and_map_are_finite_and_grid_shaped(self):
        rollout = np.repeat(self.states[-1][None], 4, axis=0)
        impacts = impact_summary(
            rollout, self.states[-1], self.latitudes, self.longitudes
        )
        self.assertEqual(impacts["crop_deficit_days"], 0)
        self.assertEqual(impacts["peak_rain_mm_day"], 5.0)
        svg = grid_svg(
            rollout[0, 1],
            self.latitudes,
            self.longitudes,
            "Test precipitation",
        )
        self.assertIn("<svg", svg)
        self.assertIn("Test precipitation", svg)
        with self.assertRaises(ValueError):
            grid_svg(np.full((32, 36), np.nan), self.latitudes, self.longitudes, "bad")


if __name__ == "__main__":
    unittest.main()
