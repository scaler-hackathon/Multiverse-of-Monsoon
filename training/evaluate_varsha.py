import argparse
from collections import defaultdict
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from data_utils import resolve_dataset_path

ROOT = Path(__file__).resolve().parents[1]
HEIGHT, WIDTH = 32, 36
LEADS = (1, 4, 8)
METHODS = ("model", "persistence", "climatology")


def load_validation_data(data_path=None):
    data_path = resolve_dataset_path(data_path)
    with xr.open_dataset(data_path) as ds:
        data = np.asarray(ds["data"].values, dtype=np.float32)
        times = np.asarray(ds.time.values)
        latitudes = np.asarray(ds.lat.values, dtype=np.float64)
        day_of_year = np.asarray(ds.time.dt.dayofyear.values, dtype=np.int16)

    val_start = int(np.searchsorted(times, np.datetime64("2021-01-01")))
    if val_start <= 0 or val_start >= len(data):
        raise ValueError("Expected a training period before the 2021 holdout.")
    if data.shape[1:] != (6, HEIGHT, WIDTH) or not np.isfinite(data).all():
        raise ValueError(f"Unexpected or non-finite climate data: {data.shape}")
    if latitudes.shape != (HEIGHT,):
        raise ValueError(f"Unexpected latitude coordinates: {latitudes.shape}")

    climatology_samples = defaultdict(list)
    for index in range(val_start):
        key = np.datetime_as_string(times[index], unit="D")[5:]
        climatology_samples[key].append(data[index])
    climatology = {
        key: np.mean(samples, axis=0, dtype=np.float64).astype(np.float32)
        for key, samples in climatology_samples.items()
    }

    with (ROOT / "stats.json").open(encoding="utf-8") as file:
        stats = json.load(file)
    if len(stats.get("channels", [])) != 6:
        raise ValueError("stats.json must define the six API channels.")
    return data, times, day_of_year, climatology, np.cos(
        np.deg2rad(latitudes)
    ).astype(np.float64), stats


def _accumulate(metrics, scores, lead, forecast, truth, weights):
    error = forecast.astype(np.float64) - truth.astype(np.float64)
    metrics[lead] += np.sum(error * error * weights[None, :, None], axis=(1, 2))
    predicted_rain = forecast[1] >= 10.0
    observed_rain = truth[1] >= 10.0
    scores[lead] += np.array(
        [
            np.sum((predicted_rain & observed_rain) * weights[:, None]),
            np.sum((predicted_rain & ~observed_rain) * weights[:, None]),
            np.sum((~predicted_rain & observed_rain) * weights[:, None]),
        ],
        dtype=np.float64,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare autoregressive forecasts with aligned persistence and climatology baselines."
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument("--weights", type=Path, default=ROOT / "varsha_weights.pt")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be positive")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    data, times, day_of_year, climatology, lat_weights, stats = (
        load_validation_data(args.data)
    )
    weights = lat_weights
    denominator = float(weights.sum() * WIDTH)
    validation_start = int(np.searchsorted(times, np.datetime64("2021-01-01")))
    channels = stats["channels"]

    sums = {
        method: {lead: np.zeros(6, dtype=np.float64) for lead in LEADS}
        for method in METHODS
    }
    rain_scores = {
        method: {lead: np.zeros(3, dtype=np.float64) for lead in LEADS}
        for method in METHODS
    }
    variance_ratios = []

    model = None
    if not args.baselines_only:
        sys.path.insert(0, str(ROOT))
        import serve

        model = serve
        if not args.weights.is_file():
            raise FileNotFoundError(f"Missing model weights: {args.weights}")
        model.model.load_state_dict(
            torch.load(args.weights, map_location="cpu", weights_only=True)
        )
        torch.set_num_threads(min(4, torch.get_num_threads()))

    starts = range(validation_start + 2, len(data) - max(LEADS), args.stride)
    windows = list(starts)
    if not windows:
        raise ValueError("Validation split contains no complete forecast windows.")

    for number, index in enumerate(windows, start=1):
        forecasts = None
        if model is not None:
            doy = int(day_of_year[index])
            phase = (
                math.sin(2.0 * math.pi * doy / 365.0),
                math.cos(2.0 * math.pi * doy / 365.0),
            )
            forecasts = model.simulate(
                data[index - 2 : index + 1], phase, max(LEADS), None
            )
            if forecasts.shape != (max(LEADS), 6, HEIGHT, WIDTH):
                raise ValueError(f"Unexpected rollout shape: {forecasts.shape}")
            if not np.isfinite(forecasts).all():
                raise FloatingPointError(f"Non-finite rollout at validation index {index}.")
            if forecasts[:, 1].min() < 0:
                raise ValueError(f"Negative precipitation at validation index {index}.")
            if forecasts[:, 0].min() < 233.15 or forecasts[:, 0].max() > 333.15:
                raise ValueError(f"Temperature guard failed at validation index {index}.")
            variance_ratios.append(
                np.var(forecasts[max(LEADS) - 1], axis=(1, 2))
                / (np.var(data[index + max(LEADS)], axis=(1, 2)) + 1e-12)
            )

        for lead in LEADS:
            truth = data[index + lead]
            key = np.datetime_as_string(times[index + lead], unit="D")[5:]
            forecasts_at_lead = {
                "persistence": data[index],
                "climatology": climatology[key],
            }
            if forecasts is not None:
                forecasts_at_lead["model"] = forecasts[lead - 1]
            for name, forecast in forecasts_at_lead.items():
                _accumulate(sums[name], rain_scores[name], lead, forecast, truth, weights)

        if number % 20 == 0 or number == len(windows):
            print(f"Evaluated {number}/{len(windows)} held-out windows")

    results = {
        "holdout_year": 2021,
        "windows": len(windows),
        "stride": args.stride,
        "channels": channels,
        "leads": {},
    }
    print(f"\nHeld-out year: 2021 | windows: {len(windows)} | stride: {args.stride}")
    for lead in LEADS:
        rmse = {
            method: np.sqrt(sums[method][lead] / (len(windows) * denominator))
            for method in METHODS
            if (method == "model" and not args.baselines_only)
            or method != "model"
        }
        best_baseline = np.minimum(rmse["persistence"], rmse["climatology"])
        lead_results = {"rmse": {}, "rain_csi_10mm_day": {}}
        for method, values in rmse.items():
            lead_results["rmse"][method] = values.tolist()
        print(f"\nLead day +{lead} RMSE (model / persistence / climatology):")
        for channel_index, channel in enumerate(channels):
            if "model" in rmse:
                skill = 1.0 - rmse["model"][channel_index] / max(
                    best_baseline[channel_index], 1e-12
                )
                print(
                    f"  {channel:5s} {rmse['model'][channel_index]:.4f} / "
                    f"{rmse['persistence'][channel_index]:.4f} / "
                    f"{rmse['climatology'][channel_index]:.4f} "
                    f"(skill vs best: {skill:+.1%})"
                )
            else:
                print(
                    f"  {channel:5s} -- / {rmse['persistence'][channel_index]:.4f} / "
                    f"{rmse['climatology'][channel_index]:.4f}"
                )
        print("  Heavy-rain CSI (threshold 10 mm/day):")
        for method in METHODS:
            if method == "model" and args.baselines_only:
                continue
            tp, fp, fn = rain_scores[method][lead]
            csi = tp / max(tp + fp + fn, 1e-12)
            lead_results["rain_csi_10mm_day"][method] = float(csi)
            print(f"    {method:12s} {csi:.4f}")
        if "model" in rmse:
            lead_results["skill_vs_best_baseline"] = (
                1.0 - rmse["model"] / np.maximum(best_baseline, 1e-12)
            ).tolist()
        results["leads"][str(lead)] = lead_results

    if variance_ratios:
        ratios = np.mean(variance_ratios, axis=0)
        results["day8_spatial_variance_ratio"] = ratios.tolist()
        print("\nDay +8 spatial variance ratio (model / verifying observations):")
        for channel, ratio in zip(channels, ratios):
            print(f"  {channel:5s} {ratio:.3f}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = args.json_out.with_name(args.json_out.name + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
            file.write("\n")
        os.replace(temporary_path, args.json_out)
        print(f"\nStructured metrics saved to {args.json_out}")


if __name__ == "__main__":
    main()
