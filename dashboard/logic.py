import colorsys
import html
import ipaddress
import math
from urllib.parse import urlsplit

import numpy as np

GRID_HEIGHT = 32
GRID_WIDTH = 36
CHANNELS = {
    "t2m": 0,
    "tp": 1,
    "u10": 2,
    "v10": 3,
    "msl": 4,
    "tcwv": 5,
}
SCENARIOS = (
    "Timeline Prime (unperturbed)",
    "Warm Bay of Bengal (+3 K, +15% water vapour)",
    "Weaker Arabian Sea winds (-40%)",
    "Zero perturbation (identity check)",
    "Custom localized shock",
)


def validate_api_url(raw_url):
    parts = urlsplit(raw_url.strip())
    if (
        parts.scheme != "http"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Use a plain local HTTP URL, e.g. http://127.0.0.1:8000.")
    hostname = parts.hostname.lower()
    if hostname != "localhost":
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError as exc:
            raise ValueError("The dashboard only connects to a local inference server.") from exc
        if not is_loopback:
            raise ValueError("The dashboard only connects to a local inference server.")
    return f"http://{parts.netloc}"


def _validate_grid_inputs(init_states, latitudes, longitudes):
    states = np.asarray(init_states, dtype=np.float32)
    lats = np.asarray(latitudes, dtype=np.float64)
    lons = np.asarray(longitudes, dtype=np.float64)
    if states.shape != (3, 6, GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(f"Expected initialization shape (3, 6, 32, 36), got {states.shape}.")
    if lats.shape != (GRID_HEIGHT,) or lons.shape != (GRID_WIDTH,):
        raise ValueError("Latitude and longitude coordinates must match the 32 x 36 grid.")
    if not np.isfinite(states).all() or not np.isfinite(lats).all() or not np.isfinite(lons).all():
        raise ValueError("Initialization states and coordinates must be finite.")
    return states, lats, lons


def make_perturbation(
    scenario,
    init_states,
    latitudes,
    longitudes,
    *,
    custom_channel="t2m",
    custom_delta=1.0,
    custom_latitude=20.0,
    custom_longitude=80.0,
    custom_radius=3.0,
):
    states, lats, lons = _validate_grid_inputs(init_states, latitudes, longitudes)
    if scenario == SCENARIOS[0]:
        return None

    perturbation = np.zeros((6, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    if scenario == SCENARIOS[1]:
        box = (
            (lats[:, None] >= 8.0)
            & (lats[:, None] <= 20.0)
            & (lons[None, :] >= 82.0)
            & (lons[None, :] <= 94.0)
        )
        if not box.any():
            raise ValueError("The grid does not contain any Bay of Bengal cells.")
        perturbation[0, box] = 3.0
        perturbation[5, box] = states[-1, 5, box] * 0.15
    elif scenario == SCENARIOS[2]:
        box = (
            (lats[:, None] >= 8.0)
            & (lats[:, None] <= 20.0)
            & (lons[None, :] >= 65.0)
            & (lons[None, :] <= 75.0)
        )
        if not box.any():
            raise ValueError("The grid does not contain any Arabian Sea cells.")
        perturbation[2, box] = -states[-1, 2, box] * 0.40
        perturbation[3, box] = -states[-1, 3, box] * 0.40
    elif scenario == SCENARIOS[3]:
        return perturbation
    elif scenario == SCENARIOS[4]:
        if custom_channel not in CHANNELS:
            raise ValueError(f"Unknown shock channel: {custom_channel}")
        values = (custom_delta, custom_latitude, custom_longitude, custom_radius)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Custom shock settings must be finite.")
        if not 0.25 <= float(custom_radius) <= 15.0:
            raise ValueError("Custom shock radius must be between 0.25 and 15 degrees.")
        distance = np.hypot(
            lats[:, None] - float(custom_latitude),
            (lons[None, :] - float(custom_longitude))
            * math.cos(math.radians(float(custom_latitude))),
        )
        mask = distance <= float(custom_radius)
        if not mask.any():
            raise ValueError("Custom shock does not intersect the grid; move its center or enlarge the radius.")
        perturbation[CHANNELS[custom_channel], mask] = float(custom_delta)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    if not np.isfinite(perturbation).all():
        raise ValueError("Perturbation calculation produced a non-finite value.")
    return perturbation


def impact_summary(rollout, init_state, latitudes, longitudes):
    forecast = np.asarray(rollout, dtype=np.float64)
    previous = np.asarray(init_state, dtype=np.float64)
    lats = np.asarray(latitudes, dtype=np.float64)
    lons = np.asarray(longitudes, dtype=np.float64)
    if (
        forecast.ndim != 4
        or forecast.shape[1:] != (6, GRID_HEIGHT, GRID_WIDTH)
        or forecast.shape[0] < 1
        or previous.shape != (6, GRID_HEIGHT, GRID_WIDTH)
        or lats.shape != (GRID_HEIGHT,)
        or lons.shape != (GRID_WIDTH,)
    ):
        raise ValueError("Impact summary received an invalid rollout or grid shape.")
    if not np.isfinite(forecast).all() or not np.isfinite(previous).all():
        raise ValueError("Impact summary requires finite physical-unit weather fields.")

    central = (
        (lats[:, None] >= 16.0)
        & (lats[:, None] <= 23.0)
        & (lons[None, :] >= 73.0)
        & (lons[None, :] <= 82.0)
    )
    land = lats[:, None] >= 8.0
    if not central.any():
        central = land
    rain = forecast[:, 1]
    crop_mean = float(rain[:, central].mean())
    crop_deficit_days = int(np.count_nonzero(rain[:, central].mean(axis=1) < 4.0))
    flood_fraction = float(np.mean(rain >= 10.0))
    peak_rain = float(rain.max())
    wind = np.hypot(forecast[:, 2], forecast[:, 3])
    baseline_wind_power = float(np.mean(np.hypot(previous[2], previous[3]) ** 3))
    forecast_wind_power = float(np.mean(wind**3))
    wind_index = (
        forecast_wind_power / baseline_wind_power
        if baseline_wind_power > 1e-8
        else None
    )
    return {
        "crop_rain_mm_day": crop_mean,
        "crop_deficit_days": crop_deficit_days,
        "flood_exceedance_fraction": flood_fraction,
        "peak_rain_mm_day": peak_rain,
        "wind_power_index": wind_index,
        "wind_power_relative_pct": (
            (wind_index - 1.0) * 100.0 if wind_index is not None else None
        ),
    }


def grid_svg(values, latitudes, longitudes, title, *, diverging=False):
    grid = np.asarray(values, dtype=np.float64)
    lats = np.asarray(latitudes, dtype=np.float64)
    lons = np.asarray(longitudes, dtype=np.float64)
    if grid.shape != (GRID_HEIGHT, GRID_WIDTH):
        raise ValueError(f"Map layer must have shape (32, 36), got {grid.shape}.")
    if lats.shape != (GRID_HEIGHT,) or lons.shape != (GRID_WIDTH,):
        raise ValueError("Map coordinates must match the 32 x 36 grid.")
    if not np.isfinite(grid).all():
        raise ValueError("Map layer contains NaN or Inf.")

    if diverging:
        scale = float(np.quantile(np.abs(grid), 0.98))
        scale = max(scale, 1e-8)
        normalized = np.clip(grid / scale, -1.0, 1.0)
    else:
        low, high = (float(value) for value in np.quantile(grid, (0.02, 0.98)))
        if high - low < 1e-10:
            low, high = float(grid.min()), float(grid.max())
        if high - low < 1e-10:
            high = low + 1.0
        normalized = np.clip((grid - low) / (high - low), 0.0, 1.0)

    cell = 18
    margin_left, margin_top = 54, 18
    map_width, map_height = GRID_WIDTH * cell, GRID_HEIGHT * cell
    svg_width, svg_height = margin_left + map_width + 12, margin_top + map_height + 40
    labels = []
    cells = []
    for row in range(GRID_HEIGHT):
        source_row = GRID_HEIGHT - 1 - row
        if row % 8 == 0:
            labels.append(
                f'<text x="4" y="{margin_top + row * cell + 13}" '
                f'font-size="11" fill="#334155">{html.escape(f"{lats[source_row]:.1f}°N")}</text>'
            )
        for col in range(GRID_WIDTH):
            value = float(normalized[source_row, col])
            if diverging:
                hue = 0.58 - 0.58 * ((value + 1.0) / 2.0)
                saturation = 0.72
                lightness = 0.50 + 0.08 * (1.0 - abs(value))
            else:
                hue = 0.60 - 0.60 * value
                saturation = 0.72
                lightness = 0.50 + 0.15 * (1.0 - value)
            red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
            color = f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"
            x, y = margin_left + col * cell, margin_top + row * cell
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="0.6">'
                f'<title>{lats[source_row]:.1f}°N, {lons[col]:.1f}°E: '
                f'{grid[source_row, col]:.3g}</title></rect>'
            )
    for col in range(0, GRID_WIDTH, 8):
        labels.append(
            f'<text x="{margin_left + col * cell}" y="{margin_top + map_height + 16}" '
            f'font-size="11" fill="#334155">{html.escape(f"{lons[col]:.1f}°E")}</text>'
        )
    safe_title = html.escape(str(title))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{safe_title}" '
        f'viewBox="0 0 {svg_width} {svg_height}" style="width:100%;height:auto">'
        f'<title>{safe_title}</title>{"".join(labels)}{"".join(cells)}</svg>'
    )
