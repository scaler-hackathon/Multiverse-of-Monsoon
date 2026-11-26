import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class VarshaNet(nn.Module):
    def __init__(self, in_channels=22, out_channels=6, base_dim=48):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.blocks = nn.Sequential(
            ResBlock(base_dim),
            ResBlock(base_dim),
            ResBlock(base_dim),
            ResBlock(base_dim)
        )
        self.out_proj = nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.in_proj(x)
        feat = self.blocks(feat)
        delta = self.out_proj(feat)
        return delta


if __name__ == "__main__":
    print("Loading dataset, normalization stats, and trained weights...")
    ds = xr.open_dataset(ROOT / "india_climate_train.nc")
    data = ds["data"].values

    with open(ROOT / "stats.json") as f:
        stats = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
    std = np.array(stats["std"], dtype=np.float32).reshape(1, 6, 1, 1)
    norm_static = np.load(ROOT / "static_norm.npy")

    model = VarshaNet()
    weights_path = ROOT / "varsha_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file at {weights_path}. Run train_varsha.py first.")

    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()

    val_start = 6209 - 365
    lead_days = [1, 4, 8]
    varsha_rmse = {k: [] for k in lead_days}

    print("Running 10-step autoregressive rollout on validation year...")
    H, W = 32, 36

    with torch.no_grad():
        for t in range(val_start, len(data) - 10, 5):
            history = list((data[t : t + 3] - mean) / std)
            doy = (t % 365) + 1
            
            forecasts = []
            for step in range(10):
                sin_doy = math.sin(2 * math.pi * (doy + step) / 365.0)
                cos_doy = math.cos(2 * math.pi * (doy + step) / 365.0)
                doy_planes = np.stack([
                    np.full((H, W), sin_doy, dtype=np.float32),
                    np.full((H, W), cos_doy, dtype=np.float32)
                ], axis=0)
                
                x_in = np.concatenate([
                    np.concatenate(history[-3:], axis=0),
                    norm_static,
                    doy_planes
                ], axis=0)
                
                x_tensor = torch.tensor(x_in[None], dtype=torch.float32)
                delta = model(x_tensor).numpy()[0]
                
                next_day = history[-1] + delta
                history.append(next_day)
                
                next_phys = next_day * std[0] + mean[0]
                next_phys[1] = np.clip(next_phys[1], 0, None)
                next_phys[0] = np.clip(next_phys[0], 233.15, 333.15)
                forecasts.append(next_phys)
                
            for k in lead_days:
                target = data[t + 2 + k]
                pred = forecasts[k - 1]
                sq_err = (pred - target) ** 2
                varsha_rmse[k].append(np.sqrt(np.mean(sq_err, axis=(1, 2))))

    print("\n--- VARSHA MODEL VALIDATION RMSE (Physical Units) ---")
    for k in lead_days:
        mean_v = np.mean(varsha_rmse[k], axis=0)
        print(f"\nLead Day +{k}:")
        for name, v in zip(stats["channels"], mean_v):
            print(f"  {name:6s}: {v:.4f}")