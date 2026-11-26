from pathlib import Path
import json
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
ds = xr.open_dataset(ROOT / "india_climate_train.nc")
data = ds["data"].values  # shape: (6209, 6, 32, 36)

with open(ROOT / "stats.json") as f:
    stats = json.load(f)

print(f"Total timesteps loaded: {data.shape[0]}")

# Use the last 365 days (year 2021) as a local validation set
val_start_idx = 6209 - 365
lead_days = [1, 4, 8]

# Calculate Persistence baseline errors across the validation set
# Input: 3 consecutive days -> init_states[-1] is day t
# Target: day t + k (where k is lead day)
persistence_rmse = {lead: [] for lead in lead_days}

for t in range(val_start_idx, data.shape[0] - 8):
    init_state = data[t]  # day t
    for k in lead_days:
        target = data[t + k]
        sq_err = (init_state - target) ** 2
        # Mean across lat/lon per channel
        persistence_rmse[k].append(np.sqrt(np.mean(sq_err, axis=(1, 2))))

print("\n--- PERSISTENCE BASELINE RMSE (Physical Units) ---")
channels = stats["channels"]
for k in lead_days:
    mean_rmse = np.mean(persistence_rmse[k], axis=0)
    print(f"\nLead Day +{k}:")
    for ch_name, rmse_val in zip(channels, mean_rmse):
        print(f"  {ch_name:6s}: {rmse_val:.4f}")