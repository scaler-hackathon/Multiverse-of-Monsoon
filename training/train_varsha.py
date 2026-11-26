import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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


def get_batch(data_slice, norm_static, batch_size=32):
    max_idx = len(data_slice) - 4
    indices = np.random.randint(0, max_idx, size=batch_size)
    
    batch_inputs = []
    batch_targets = []
    
    H, W = 32, 36
    for idx in indices:
        hist = data_slice[idx : idx + 3].reshape(18, H, W)
        
        doy = (idx % 365) + 1
        sin_doy = math.sin(2 * math.pi * doy / 365.0)
        cos_doy = math.cos(2 * math.pi * doy / 365.0)
        doy_planes = np.stack([
            np.full((H, W), sin_doy, dtype=np.float32),
            np.full((H, W), cos_doy, dtype=np.float32)
        ], axis=0)
        
        x_in = np.concatenate([hist, norm_static, doy_planes], axis=0)
        
        last_day = data_slice[idx + 2]
        target_day = data_slice[idx + 3]
        delta_target = target_day - last_day
        
        batch_inputs.append(x_in)
        batch_targets.append(delta_target)
        
    return torch.tensor(np.stack(batch_inputs), dtype=torch.float32), torch.tensor(np.stack(batch_targets), dtype=torch.float32)


if __name__ == "__main__":
    print("Loading dataset and normalization statistics...")
    ds = xr.open_dataset(ROOT / "india_climate_train.nc")
    raw_data = ds["data"].values
    raw_static = ds["static"].values

    with open(ROOT / "stats.json") as f:
        stats = json.load(f)

    mean = np.array(stats["mean"], dtype=np.float32).reshape(1, 6, 1, 1)
    std = np.array(stats["std"], dtype=np.float32).reshape(1, 6, 1, 1)

    norm_data = (raw_data - mean) / std

    static_mean = raw_static.mean(axis=(1, 2), keepdims=True)
    static_std = raw_static.std(axis=(1, 2), keepdims=True) + 1e-6
    norm_static = (raw_static - static_mean) / static_std

    val_split = 6209 - 365
    train_data = norm_data[:val_split]
    val_data = norm_data[val_split:]

    print(f"Train steps: {len(train_data)}, Val steps: {len(val_data)}")

    model = VarshaNet()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    print("\nStarting training (fast CPU pass)...")
    model.train()
    for step in range(1, 1501):
        x_batch, y_batch = get_batch(train_data, norm_static, batch_size=32)
        optimizer.zero_grad()
        pred_delta = model(x_batch)
        loss = criterion(pred_delta, y_batch)
        loss.backward()
        optimizer.step()
        
        if step % 250 == 0:
            print(f"Step [{step:4d}/1500] - Loss: {loss.item():.5f}")

    output_path = ROOT / "varsha_weights.pt"
    static_path = ROOT / "static_norm.npy"

    torch.save(model.state_dict(), output_path)
    np.save(static_path, norm_static)

    print(f"\nModel weights saved to: {output_path}")
    print(f"Static channel map saved to: {static_path}")