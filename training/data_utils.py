from pathlib import Path

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
