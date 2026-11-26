from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import serve
from training.train_varsha import VarshaNet


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

    def test_training_model_matches_deployed_checkpoint_and_forward(self):
        training_model = VarshaNet(rain_mode=serve.model.rain_mode).eval()
        serve.load_model_weights(training_model, serve.weights_path)
        sample = torch.zeros((1, 22, 32, 36), dtype=torch.float32)
        with torch.no_grad():
            training_output = training_model(sample).numpy()
            inference_output = serve.model(sample).numpy()
        np.testing.assert_array_equal(training_output, inference_output)

    def test_signed_rain_head_can_predict_a_decrease(self):
        model = serve.VarshaNet(rain_mode=serve.SIGNED_RAIN_MODE)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        with torch.no_grad():
            model.rain_scale.fill_(1.0)
            model.out_proj.bias[1] = -1.0
        output = model(torch.zeros((1, 22, 32, 36)))
        self.assertTrue(torch.all(output[:, 1] < 0))

    def test_new_checkpoint_metadata_selects_signed_rain_mode(self):
        source = VarshaNet(rain_mode=serve.SIGNED_RAIN_MODE)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            torch.save(
                {
                    "state_dict": source.state_dict(),
                    "rain_mode": serve.SIGNED_RAIN_MODE,
                    "training_years": [2005, 2018],
                },
                path,
            )
            target = VarshaNet()
            metadata = serve.load_model_weights(target, path)
        self.assertEqual(target.rain_mode, serve.SIGNED_RAIN_MODE)
        self.assertEqual(metadata["training_years"], [2005, 2018])

    def test_temperature_climatology_metadata_anchors_forecasts(self):
        coefficients = torch.zeros((5, 32, 36), dtype=torch.float32)
        coefficients[0].fill_(300.0)
        checkpoint = {
            "state_dict": serve.model.state_dict(),
            "rain_mode": serve.SIGNED_RAIN_MODE,
            "t2m_climatology_coefficients": coefficients,
            "t2m_anomaly_retention": 0.0,
            "t2m_anomaly_decay": 0.18,
        }
        initial = np.zeros((3, 6, 32, 36), dtype=np.float32)
        initial[:, 0] = 290.0
        initial[:, 1] = 5.0
        zero_delta = torch.zeros((1, 6, 32, 36), dtype=torch.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "climatology.pt"
            torch.save(checkpoint, path)
            try:
                serve.load_model_weights(serve.model, path)
                with patch.object(serve.model, "forward", return_value=zero_delta):
                    forecast = serve.simulate(initial, self.phase, 2, None)
                np.testing.assert_allclose(forecast[:, 0], 300.0)
            finally:
                serve.load_model_weights(serve.model, serve.weights_path)

    def test_invalid_temperature_climatology_metadata_is_rejected(self):
        checkpoint = {
            "state_dict": serve.model.state_dict(),
            "rain_mode": serve.SIGNED_RAIN_MODE,
            "t2m_climatology_coefficients": torch.zeros((4, 32, 36)),
            "t2m_anomaly_retention": 0.9,
            "t2m_anomaly_decay": 0.18,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-climatology.pt"
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                serve.load_model_weights(serve.VarshaNet(), path)

    def test_signed_rain_forecast_can_fall_without_becoming_negative(self):
        initial = np.zeros((3, 6, 32, 36), dtype=np.float32)
        initial[:, 0] = 290.0
        initial[:, 1] = 20.0
        negative_rain_delta = torch.zeros((1, 6, 32, 36))
        negative_rain_delta[:, 1] = -1.0
        with patch.object(serve.model, "forward", return_value=negative_rain_delta):
            forecast = serve.simulate(initial, self.phase, 1, None)
        self.assertTrue(np.all(forecast[:, 1] < initial[-1, 1]))
        self.assertGreaterEqual(float(forecast[:, 1].min()), 0.0)


if __name__ == "__main__":
    unittest.main()
