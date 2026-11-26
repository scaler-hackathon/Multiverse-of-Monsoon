import unittest

import numpy as np

from training.data_utils import year_split_bounds


class YearSplitTests(unittest.TestCase):
    def test_bounds_come_from_year_axis_and_include_only_selected_year(self):
        years = np.array([2019, 2020, 2021, 2021, 2022], dtype=np.int32)
        self.assertEqual(year_split_bounds(years, 2021), (2, 4))

    def test_holdout_requires_prior_data_and_matching_year(self):
        with self.assertRaises(ValueError):
            year_split_bounds(np.array([2021, 2022]), 2021)
        with self.assertRaises(ValueError):
            year_split_bounds(np.array([2019, 2020]), 2021)

    def test_year_axis_must_be_sorted(self):
        with self.assertRaises(ValueError):
            year_split_bounds(np.array([2020, 2019, 2021]), 2021)


if __name__ == "__main__":
    unittest.main()
