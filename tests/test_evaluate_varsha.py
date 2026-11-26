import unittest

import numpy as np

from training.evaluate_varsha import _window_skill


class WindowSkillTests(unittest.TestCase):
    def test_skill_is_computed_against_the_better_baseline_per_window(self):
        skill = _window_skill(
            np.array([1.0, 4.0]),
            np.array([2.0, 2.0]),
            np.array([3.0, 8.0]),
        )
        np.testing.assert_allclose(skill, [0.5, -1.0])


if __name__ == "__main__":
    unittest.main()
