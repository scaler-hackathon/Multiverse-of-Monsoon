import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xarray as xr

if __package__:
    from .data_utils import resolve_dataset_path, year_split_bounds
else:
    from data_utils import resolve_dataset_path, year_split_bounds

ROOT = Path(__file__).resolve().parents[1]
HEIGHT, WIDTH = 32, 36
SIGNED_RAIN_MODE = "signed_increment"


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class VarshaNet(nn.Module):
    def __init__(
        self,
        in_channels=22,
        out_channels=6,
        base_dim=64,
        rain_mode=SIGNED_RAIN_MODE,
    ):
        super().__init__()
        self.rain_mode = rain_mode
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(base_dim) for _ in range(6)])
        self.out_proj = nn.Conv2d(base_dim, out_channels, kernel_size=1)
        self.rain_scale = nn.Parameter(torch.ones(1))
        self.t2m_climatology_coefficients = None
        self.t2m_anomaly_retention = 1.0
        self.t2m_anomaly_decay = 0.0

    def forward(self, x):
        delta = self.out_proj(self.blocks(self.in_proj(x)))
        rain = delta[:, 1:2] * self.rain_scale
        if self.rain_mode == "softplus_increment":
            rain = F.softplus(rain)
        return torch.cat((delta[:, :1], rain, delta[:, 2:]), dim=1)


def _forecast_doy(last_input_doy, lead):
    base_doy = ((int(last_input_doy) - 1) % 365) + 1
    return ((base_doy + int(lead) - 1) % 365) + 1


def _fit_t2m_harmonic_climatology(data, day_of_year):
    if (
        data.ndim != 4
        or data.shape[1:] != (6, HEIGHT, WIDTH)
        or day_of_year.shape != (len(data),)
        or len(data) < 365
        or not np.isfinite(data[:, 0]).all()
    ):
        raise ValueError("Temperature climatology fit received invalid training data.")
    phase = 2.0 * np.pi * (day_of_year.astype(np.float64) - 1.0) / 365.0
    design = np.stack(
        (
            np.ones_like(phase),
            np.sin(phase),
            np.cos(phase),
            np.sin(2.0 * phase),
            np.cos(2.0 * phase),
        ),
        axis=1,
    )
    targets = data[:, 0].reshape(len(data), -1).astype(np.float64)
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    return coefficients.reshape(5, HEIGHT, WIDTH).astype(np.float32)


def _rollout_batch(
    model,
    histories,
    targets,
    input_doys,
    norm_static,
    mean,
    std,
    heavy_rain_weight,
    late_temperature_weight,
):
    batch_size, horizon = targets.shape[:2]
    device = histories.device
    static = torch.as_tensor(norm_static, dtype=torch.float32, device=device)
    static = static.unsqueeze(0).expand(batch_size, -1, -1, -1)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    rain_floor = -mean_t[1] / std_t[1]
    temp_min = (233.15 - mean_t[0]) / std_t[0]
    temp_max = (333.15 - mean_t[0]) / std_t[0]
    history = [histories[:, day] for day in range(3)]
    loss = torch.zeros((), dtype=torch.float32, device=device)

    for step in range(horizon):
        forecast_doys = [_forecast_doy(doy, step + 1) for doy in input_doys]
        phases = [2.0 * math.pi * doy / 365.0 for doy in forecast_doys]
        phase_tensor = torch.tensor(
            [[math.sin(phase), math.cos(phase)] for phase in phases],
            dtype=torch.float32,
            device=device,
        )
        doy_planes = phase_tensor[:, :, None, None].expand(
            -1, -1, HEIGHT, WIDTH
        )
        dynamic = torch.cat(history[-3:], dim=1)
        prediction = history[-1] + model(
            torch.cat((dynamic, static, doy_planes), dim=1)
        )
        coefficients = model.t2m_climatology_coefficients
        if coefficients is not None:
            climate_phases = [
                2.0 * math.pi * (doy - 1.0) / 365.0
                for doy in forecast_doys
            ]
            climate_basis = torch.tensor(
                [
                    [
                        1.0,
                        math.sin(phase),
                        math.cos(phase),
                        math.sin(2.0 * phase),
                        math.cos(2.0 * phase),
                    ]
                    for phase in climate_phases
                ],
                dtype=torch.float32,
                device=device,
            )
            coefficient_tensor = torch.as_tensor(
                coefficients, dtype=torch.float32, device=device
            )
            climate_t2m = torch.einsum(
                "bk,khw->bhw", climate_basis, coefficient_tensor
            )
            climate_norm = (climate_t2m - mean_t[0]) / std_t[0]
            retention = model.t2m_anomaly_retention * math.exp(
                -model.t2m_anomaly_decay * step
            )
            predicted_t2m = climate_norm + retention * (
                prediction[:, 0] - climate_norm
            )
            prediction = torch.cat(
                (predicted_t2m[:, None], prediction[:, 1:]), dim=1
            )
        physical = prediction * std_t[None, :, None, None] + mean_t[None, :, None, None]
        guarded = torch.cat(
            (
                physical[:, :1].clamp(233.15, 333.15),
                physical[:, 1:2].clamp_min(0.0),
                physical[:, 2:],
            ),
            dim=1,
        )
        next_state = (
            guarded - mean_t[None, :, None, None]
        ) / std_t[None, :, None, None]
        squared_error = (next_state - targets[:, step]).square()
        loss_weights = torch.ones_like(squared_error)
        target_rain = targets[:, step, 1] * std_t[1] + mean_t[1]
        loss_weights[:, 1] = 1.0 + heavy_rain_weight * (
            target_rain >= 10.0
        ).to(dtype=squared_error.dtype)
        if step >= horizon // 2:
            loss_weights[:, 0] = late_temperature_weight
        loss = loss + (squared_error * loss_weights).mean()
        history.append(next_state)

    return loss / horizon


def main():
    parser = argparse.ArgumentParser(description="Train VARSHA with calendar-aligned rollouts.")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--rollout-horizon", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--heavy-rain-weight", type=float, default=3.0)
    parser.add_argument("--late-temperature-weight", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout-year",
        type=int,
        required=True,
        help="Year in the dataset time axis to exclude from training.",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "varsha_weights.pt")
    parser.add_argument(
        "--initial-weights", type=Path, default=ROOT / "varsha_weights.pt"
    )
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()

    if (
        args.steps < 1
        or not 1 <= args.rollout_horizon <= 8
        or args.batch_size < 1
        or not math.isfinite(args.heavy_rain_weight)
        or args.heavy_rain_weight < 0
        or not math.isfinite(args.late_temperature_weight)
        or args.late_temperature_weight <= 0
    ):
        parser.error("steps and batch size must be positive; rollout horizon must be 1..8")

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    data_path = resolve_dataset_path(args.data)

    print("Loading training data and calendar coordinates...")
    with xr.open_dataset(data_path) as ds:
        if "data" not in ds or "static" not in ds or "time" not in ds.coords:
            raise ValueError("Dataset must provide data, static, and time.")
        raw_data = np.asarray(ds["data"].values, dtype=np.float32)
        raw_static = np.asarray(ds["static"].values, dtype=np.float32)
        years = np.asarray(ds.time.dt.year.values, dtype=np.int32)
        day_of_year = np.asarray(ds.time.dt.dayofyear.values, dtype=np.int16)

    if raw_data.ndim != 4 or raw_data.shape[1:] != (6, HEIGHT, WIDTH):
        raise ValueError(f"Unexpected data shape: {raw_data.shape}")
    if raw_static.shape != (2, HEIGHT, WIDTH):
        raise ValueError(f"Unexpected static shape: {raw_static.shape}")
    if not np.isfinite(raw_data).all() or not np.isfinite(raw_static).all():
        raise ValueError("Training data contains NaN or Inf.")

    with (ROOT / "stats.json").open(encoding="utf-8") as file:
        stats = json.load(file)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if (
        mean.shape != (6,)
        or std.shape != (6,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("stats.json must contain six finite means and positive standard deviations.")

    val_start, _ = year_split_bounds(years, args.holdout_year)
    train_data = (raw_data[:val_start] - mean[None, :, None, None]) / std[
        None, :, None, None
    ]
    norm_static = (raw_static - raw_static.mean(axis=(1, 2), keepdims=True)) / (
        raw_static.std(axis=(1, 2), keepdims=True) + 1e-6
    )
    max_start = len(train_data) - args.rollout_horizon - 2
    if max_start <= 0:
        raise ValueError("Training split is shorter than the requested rollout horizon.")

    t2m_climatology_coefficients = _fit_t2m_harmonic_climatology(
        raw_data[:val_start], day_of_year[:val_start]
    )
    model = VarshaNet()
    model.t2m_climatology_coefficients = t2m_climatology_coefficients
    model.t2m_anomaly_retention = 0.92
    model.t2m_anomaly_decay = 0.18
    initial_checkpoint_sha256 = None
    if args.from_scratch:
        print("Training from a fresh initialization")
    elif args.initial_weights.exists():
        initial_checkpoint_sha256 = hashlib.sha256(
            args.initial_weights.read_bytes()
        ).hexdigest()
        checkpoint = torch.load(
            args.initial_weights, map_location="cpu", weights_only=True
        )
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(
            state_dict
        )
        print(
            f"Fine-tuning weights from {args.initial_weights} "
            "with signed rainfall increments"
        )
    else:
        raise FileNotFoundError(
            f"Missing initial checkpoint {args.initial_weights}; pass --from-scratch to start fresh."
        )

    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    model.train()
    for step in range(1, args.steps + 1):
        indices = rng.integers(0, max_start, size=args.batch_size)
        histories = torch.from_numpy(
            np.stack([train_data[i : i + 3] for i in indices])
        )
        targets = torch.from_numpy(
            np.stack(
                [
                    train_data[i + 3 : i + 3 + args.rollout_horizon]
                    for i in indices
                ]
            )
        )
        input_doys = [int(day_of_year[i + 2]) for i in indices]

        optimizer.zero_grad(set_to_none=True)
        loss = _rollout_batch(
            model,
            histories,
            targets,
            input_doys,
            norm_static,
            mean,
            std,
            args.heavy_rain_weight,
            args.late_temperature_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"Step [{step}/{args.steps}] rollout loss: {loss.item():.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_name(args.output.name + ".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "rain_mode": SIGNED_RAIN_MODE,
            "training_years": f"{years[0]}-{years[val_start - 1]}",
            "validation_year": int(args.holdout_year),
            "holdout_year": int(args.holdout_year),
            "training_steps": int(args.steps),
            "training_batch_size": int(args.batch_size),
            "rollout_horizon": int(args.rollout_horizon),
            "learning_rate": float(args.learning_rate),
            "heavy_rain_weight": float(args.heavy_rain_weight),
            "late_temperature_weight": float(args.late_temperature_weight),
            "training_mode": "from_scratch" if args.from_scratch else "fine_tune",
            "initial_checkpoint_sha256": initial_checkpoint_sha256,
            "t2m_climatology_coefficients": torch.from_numpy(
                t2m_climatology_coefficients
            ),
            "t2m_anomaly_retention": model.t2m_anomaly_retention,
            "t2m_anomaly_decay": model.t2m_anomaly_decay,
            "climatology_method": "two_harmonic_t2m",
        },
        temporary_path,
    )
    os.replace(temporary_path, args.output)

    print(f"Candidate weights saved to {args.output}")
    print(f"Holdout year {args.holdout_year} was excluded from training.")


if __name__ == "__main__":
    main()
