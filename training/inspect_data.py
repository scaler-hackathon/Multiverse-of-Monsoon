
import json
from pathlib import Path

import numpy as np
import xarray as xr
from data_utils import resolve_dataset_path

ROOT = Path(__file__).resolve().parents[1]

nc_path = resolve_dataset_path()
stats_path = ROOT / "stats.json"

print("\n========== VARSHA DATA INSPECTION ==========\n")

print("Python dataset:", nc_path)
print("Dataset exists:", nc_path.exists())
print("Stats exist:", stats_path.exists())

if not nc_path.exists():
    raise FileNotFoundError(nc_path)

if not stats_path.exists():
    raise FileNotFoundError(stats_path)

ds = xr.open_dataset(nc_path)

print("\n1. DATASET OVERVIEW")
print(ds)
print("\nDimensions:", dict(ds.sizes))

print("\n2. VARIABLES")
for name, variable in ds.data_vars.items():
    print(
        f"{name}: shape={variable.shape}, "
        f"dtype={variable.dtype}, dims={variable.dims}"
    )

print("\n3. COORDINATES")
for name, coordinate in ds.coords.items():
    print(f"{name}: shape={coordinate.shape}")

print("\n4. NUMERICAL STATISTICS")
for name, variable in ds.data_vars.items():
    if np.issubdtype(variable.dtype, np.number):
        values = np.asarray(variable.values)
        finite = values[np.isfinite(values)]

        if finite.size:
            print(f"\n{name}")
            print("  Min:", float(finite.min()))
            print("  Max:", float(finite.max()))
            print("  Mean:", float(finite.mean()))
            print("  NaN count:", int(np.isnan(values).sum()))

print("\n5. NORMALIZATION STATISTICS")
with open(stats_path, "r", encoding="utf-8") as f:
    stats = json.load(f)

print(json.dumps(stats, indent=2)[:6000])

ds.close()

print("\n========== INSPECTION COMPLETE ==========\n")
