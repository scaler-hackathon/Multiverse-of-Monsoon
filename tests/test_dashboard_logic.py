import unittest
import re

import numpy as np

from dashboard.logic import (
    SCENARIOS,
    grid_cell_area_weights,
    grid_svg,
    impact_summary,
    make_perturbation,
    response_diagnostics,
    uploaded_dataset_key,
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

    def test_uploaded_dataset_key_detects_replacement_with_same_name_and_size(self):
        first = uploaded_dataset_key("climate.nc", b"first")
        replacement = uploaded_dataset_key("climate.nc", b"other")
        self.assertNotEqual(first, replacement)
        self.assertEqual(first, uploaded_dataset_key("climate.nc", b"first"))

    def test_impact_coverage_and_regional_means_are_area_weighted(self):
        rollout = np.repeat(self.states[-1][None], 2, axis=0)
        row_values = np.arange(32, dtype=np.float32)
        rollout[:, 0] = (300.0 + row_values)[None, :, None]
        rollout[:, 1] = row_values[None, :, None]
        weights = grid_cell_area_weights(self.latitudes)
        central = (
            (self.latitudes[:, None] >= 16.0)
            & (self.latitudes[:, None] <= 23.0)
            & (self.longitudes[None, :] >= 73.0)
            & (self.longitudes[None, :] <= 82.0)
        )

        impacts = impact_summary(
            rollout, self.states[-1], self.latitudes, self.longitudes
        )
        expected_central_rain = np.average(
            rollout[0, 1, central], weights=weights[central]
        )
        expected_heavy_rain_area = weights[row_values >= 10].sum() / weights.sum()
        expected_hot_area = weights[row_values + 300.0 > 308.15].sum() / weights.sum()
        self.assertAlmostEqual(
            impacts["crop_rain_mm_day"], float(expected_central_rain), places=6
        )
        self.assertAlmostEqual(
            impacts["flood_exceedance_fraction"],
            float(expected_heavy_rain_area),
            places=6,
        )
        self.assertAlmostEqual(
            impacts["hot_area_day_fraction"], float(expected_hot_area), places=6
        )

    def test_shared_map_limits_preserve_colors_and_reject_invalid_bounds(self):
        first = np.zeros((32, 36), dtype=np.float32)
        second = np.zeros((32, 36), dtype=np.float32)
        first[-1, 0] = second[-1, 0] = 2.0
        second[-1, 1] = 9.0
        first_svg = grid_svg(
            first,
            self.latitudes,
            self.longitudes,
            "Baseline",
            scale_limits=(0.0, 10.0),
        )
        second_svg = grid_svg(
            second,
            self.latitudes,
            self.longitudes,
            "Alternate",
            scale_limits=(0.0, 10.0),
        )
        first_color = re.search(
            r'<rect x="54" y="18"[^>]*fill="(#[0-9a-f]{6})"', first_svg
        ).group(1)
        second_color = re.search(
            r'<rect x="54" y="18"[^>]*fill="(#[0-9a-f]{6})"', second_svg
        ).group(1)
        self.assertEqual(first_color, second_color)
        self.assertIn("10", first_svg)
        with self.assertRaises(ValueError):
            grid_svg(
                first,
                self.latitudes,
                self.longitudes,
                "Invalid scale",
                scale_limits=(10.0, 0.0),
            )

    def test_spatial_response_magnitude_does_not_cancel_signed_cells(self):
        baseline = np.zeros((2, 6, 32, 36), dtype=np.float32)
        alternate = baseline.copy()
        alternate[0, 0, 0, 0] = 2.0
        alternate[0, 0, 0, 1] = -2.0

        difference, mean, rms, peak = response_diagnostics(
            baseline, alternate, 0
        )
        self.assertAlmostEqual(float(mean[0]), 0.0, places=7)
        self.assertGreater(float(rms[0]), 0.0)
        self.assertAlmostEqual(float(peak[0]), 2.0)
        self.assertEqual(float(rms[1]), 0.0)
        self.assertEqual(int(np.count_nonzero(difference[1])), 0)
        with self.assertRaises(ValueError):
            response_diagnostics(baseline, alternate, 6)


if __name__ == "__main__":
    unittest.main()
