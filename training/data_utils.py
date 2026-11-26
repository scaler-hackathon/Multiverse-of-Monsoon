from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILENAME = "india_climate_train.nc"


def resolve_dataset_path(path=None):
    if path is not None:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Climate dataset not found at {candidate}.")
        return candidate

    candidates = (
        PROJECT_ROOT.parent / DATA_FILENAME,
        PROJECT_ROOT / DATA_FILENAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Missing {DATA_FILENAME}. Download it from the official team-kit release "
        f"or pass --data PATH. Checked: {expected}"
    )


def year_split_bounds(years, holdout_year):
    years = np.asarray(years)
    if (
        years.ndim != 1
        or years.size == 0
        or not np.issubdtype(years.dtype, np.integer)
        or isinstance(holdout_year, (bool, np.bool_))
        or not isinstance(holdout_year, (int, np.integer))
    ):
        raise ValueError("Time-axis years and holdout year must be one-dimensional integers.")
    if np.any(np.diff(years) < 0):
        raise ValueError("Time-axis years must be sorted in ascending order.")

    start = int(np.searchsorted(years, holdout_year, side="left"))
    end = int(np.searchsorted(years, holdout_year + 1, side="left"))
    if start == 0 or end <= start:
        raise ValueError(
            f"Holdout year {holdout_year} must have data and earlier training years."
        )
    return start, end
