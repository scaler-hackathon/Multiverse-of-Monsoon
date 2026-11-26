import unittest

import numpy as np

import serve


class InferenceEdgeCaseTests(unittest.TestCase):
    def setUp(self):
        self.initial = np.zeros((3, 6, 32, 36), dtype=np.float32)
        self.phase = (0.5, 0.5)

    def test_zero_initialization_rollout_stays_in_physical_bounds(self):
        forecast = serve.simulate(self.initial, self.phase, 10, None)
        self.assertEqual(forecast.shape, (10, 6, 32, 36))
        self.assertTrue(np.isfinite(forecast).all())
        self.assertGreaterEqual(float(forecast[:, 1].min()), 0.0)
        self.assertGreaterEqual(float(forecast[:, 0].min()), 233.15)
        self.assertLessEqual(float(forecast[:, 0].max()), 333.15)

    def test_zero_perturbation_is_exact_identity(self):
        zero = np.zeros((6, 32, 36), dtype=np.float32)
        without = serve.simulate(self.initial, self.phase, 2, None)
        with_zero = serve.simulate(self.initial, self.phase, 2, zero)
        np.testing.assert_array_equal(without, with_zero)

    def test_invalid_shapes_phases_steps_and_nonfinite_values_are_rejected(self):
        cases = (
            lambda: serve.simulate(np.zeros((1, 6, 32, 36)), self.phase, 1, None),
            lambda: serve.simulate(self.initial, (0.0, 0.0), 1, None),
            lambda: serve.simulate(self.initial, self.phase, True, None),
            lambda: serve.simulate(
                np.full_like(self.initial, np.nan), self.phase, 1, None
            ),
            lambda: serve.simulate(
                self.initial, self.phase, 1, np.zeros((6, 31, 36))
            ),
        )
        for invalid_call in cases:
            with self.subTest(call=invalid_call), self.assertRaises(ValueError):
                invalid_call()


if __name__ == "__main__":
    unittest.main()
