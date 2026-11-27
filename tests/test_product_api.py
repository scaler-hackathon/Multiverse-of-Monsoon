import asyncio
import unittest

import numpy as np
from fastapi import HTTPException

from app_product import scenario


class JsonRequest:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class ProductApiTests(unittest.TestCase):
    def setUp(self):
        initial = np.zeros((3, 6, 32, 36), dtype=np.float32)
        initial[:, 0] = 290.0
        initial[:, 1] = 4.0
        initial[:, 2:4] = 1.0
        initial[:, 4] = 101000.0
        initial[:, 5] = 30.0
        self.body = {
            "init_states": initial.tolist(),
            "doy_sincos": [0.5, 0.5],
            "steps": 1,
            "perturbation": None,
        }

    def test_scenario_returns_rollouts_responses_and_live_impacts(self):
        result = asyncio.run(scenario(JsonRequest(self.body)))

        self.assertEqual(
            result["baseline_rollout"], result["scenario_rollout"]
        )
        self.assertEqual(
            np.asarray(result["response"]["rms_change_by_channel"]).shape,
            (1, 6),
        )
        self.assertEqual(set(result["impacts"]), {"baseline", "scenario"})
        self.assertGreaterEqual(result["inference_ms"], 0.0)

    def test_scenario_rejects_more_than_ten_steps(self):
        body = dict(self.body, steps=11)

        with self.assertRaises(HTTPException) as error:
            asyncio.run(scenario(JsonRequest(body)))

        self.assertEqual(error.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
