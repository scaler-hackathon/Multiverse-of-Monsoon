import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import requests
import streamlit as st
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.logic import (
    CHANNELS,
    SCENARIOS,
    grid_svg,
    impact_summary,
    make_perturbation,
    validate_api_url,
)

DATA_PATH = Path(
    os.environ.get(
        "VARSHA_DATA",
        str(
            ROOT.parent / "india_climate_train.nc"
            if (ROOT.parent / "india_climate_train.nc").is_file()
            else ROOT / "india_climate_train.nc"
        ),
    )
).expanduser()
SIMULATE_PATH = "/sim" + "ulate"


@st.cache_data(show_spinner=False, max_entries=2)
def load_dataset(path, upload_bytes):
    source = io.BytesIO(upload_bytes) if upload_bytes is not None else path
    with xr.open_dataset(source) as dataset:
        for variable in ("data",):
            if variable not in dataset:
                raise ValueError(f"Dataset is missing required variable {variable!r}.")
        if not {"time", "channel", "lat", "lon"}.issubset(dataset["data"].dims):
            raise ValueError("Dataset must have time, channel, lat, and lon dimensions.")
        expected_channels = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
        if tuple(str(value) for value in dataset.channel.values) != expected_channels:
            raise ValueError("Dataset channel names or order do not match the simulation API.")
        data = np.asarray(dataset["data"].values, dtype=np.float32)
        times = np.asarray(dataset.time.values)
        latitudes = np.asarray(dataset.lat.values, dtype=np.float64)
        longitudes = np.asarray(dataset.lon.values, dtype=np.float64)

    if data.ndim != 4 or data.shape[1:] != (6, 32, 36):
        raise ValueError(f"Expected data with shape [days, 6, 32, 36], got {data.shape}.")
    if len(times) != data.shape[0] or len(times) < 3 or not np.issubdtype(
        times.dtype, np.datetime64
    ):
        raise ValueError("The dataset must contain at least three timestamped daily states.")
    if not np.isfinite(data).all():
        raise ValueError("The dataset contains NaN or Inf values.")
    if (
        not np.isfinite(latitudes).all()
        or not np.isfinite(longitudes).all()
        or np.any(np.diff(latitudes) <= 0)
        or np.any(np.diff(longitudes) <= 0)
    ):
        raise ValueError("Latitude and longitude coordinates must be finite and increasing.")
    if np.any(np.diff(times).astype("timedelta64[D]").astype(np.int64) != 1):
        raise ValueError("The dataset timestamps must be consecutive daily samples.")
    return data, times, latitudes, longitudes


def _validate_rollout(payload, steps):
    if not isinstance(payload, dict) or "rollout" not in payload:
        raise ValueError("The simulation server response is missing its rollout.")
    rollout = np.asarray(payload["rollout"], dtype=np.float32)
    if rollout.shape != (steps, 6, 32, 36):
        raise ValueError(f"Server returned invalid rollout shape {rollout.shape}.")
    if not np.isfinite(rollout).all():
        raise ValueError("Server returned NaN or Inf values.")
    if np.any(rollout[:, 1] < 0):
        raise ValueError("Server returned negative precipitation.")
    if np.any((rollout[:, 0] < 233.15) | (rollout[:, 0] > 333.15)):
        raise ValueError("Server returned temperature outside the contract limits.")
    inference_ms = float(payload.get("inference_ms", float("nan")))
    if not math.isfinite(inference_ms) or inference_ms < 0:
        raise ValueError("Server returned an invalid inference time.")
    return rollout, inference_ms


def _request_simulation(url, initial_states, phase, steps, perturbation):
    payload = {
        "init_states": initial_states.tolist(),
        "doy_sincos": list(phase),
        "steps": steps,
        "perturbation": None if perturbation is None else perturbation.tolist(),
    }
    response = requests.post(f"{url}{SIMULATE_PATH}", json=payload, timeout=(2, 20))
    response.raise_for_status()
    return _validate_rollout(response.json(), steps)


st.set_page_config(
    page_title="Code Blooded | VARSHA",
    page_icon="🌧️",
    layout="wide",
)
st.title("VARSHA | India's Monsoon Multiverse")
st.caption(
    "Code Blooded · Live, offline-first weather rollouts with transparent local shocks "
    "and screening-level climate impacts."
)

with st.sidebar:
    st.header("Simulation setup")
    raw_api_url = st.text_input("Local inference server", "http://127.0.0.1:8000")
    try:
        api_url = validate_api_url(raw_api_url)
    except ValueError as exc:
        api_url = None
        st.error(str(exc))

    st.subheader("Starting weather")
    upload = st.file_uploader(
        "Use the kit's NetCDF file (or your own compatible copy)",
        type=("nc", "nc4"),
        help="The climate file is a local training/demo input and is deliberately not bundled in the repo.",
    )
    if upload is None and not DATA_PATH.is_file():
        st.info("Upload india_climate_train.nc from the official team-kit release to start a live simulation.")

if upload is not None or DATA_PATH.is_file():
    upload_bytes = upload.getvalue() if upload is not None else None
    try:
        data, times, latitudes, longitudes = load_dataset(str(DATA_PATH), upload_bytes)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not load climate input: {exc}")
        st.stop()

    if upload is not None:
        st.sidebar.caption(f"Loaded uploaded file: {upload.name}")
    else:
        st.sidebar.caption(f"Loaded local kit data: {DATA_PATH.name}")

    last_available = len(data) - 1
    initial_index = max(2, last_available - 1)
    input_day_index = st.sidebar.slider(
        "Last input day",
        min_value=2,
        max_value=last_available,
        value=initial_index,
        help="The preceding two consecutive days are sent with this day as the model context.",
    )
    input_date = np.datetime_as_string(times[input_day_index], unit="D")
    steps = st.sidebar.slider("Forecast horizon (days)", 1, 10, 8)

    scenario = st.sidebar.selectbox("Counterfactual branch", SCENARIOS)
    custom_options = {}
    if scenario == SCENARIOS[4]:
        custom_options["custom_channel"] = st.sidebar.selectbox(
            "Shock variable", tuple(CHANNELS)
        )
        custom_options["custom_delta"] = st.sidebar.number_input(
            "Local physical change", value=1.0, format="%.2f"
        )
        custom_options["custom_latitude"] = st.sidebar.slider(
            "Shock center latitude", 6.5, 37.5, 20.0
        )
        custom_options["custom_longitude"] = st.sidebar.slider(
            "Shock center longitude", 65.5, 100.5, 82.0
        )
        custom_options["custom_radius"] = st.sidebar.slider(
            "Shock radius (degrees)", 0.25, 15.0, 3.0
        )

    if upload is not None:
        dataset_key = ("upload", upload.name, len(upload_bytes))
    else:
        file_stat = DATA_PATH.stat()
        dataset_key = ("local", file_stat.st_mtime_ns, file_stat.st_size)
    selection_key = (
        dataset_key,
        input_day_index,
        scenario,
        steps,
        tuple(sorted(custom_options.items())),
    )
    saved_result = st.session_state.get("varsha_result")
    if saved_result is not None and saved_result.get("selection_key") != selection_key:
        st.session_state.pop("varsha_result", None)

    if api_url:
        try:
            health_response = requests.post(
                f"{api_url}/health", json={}, timeout=(2, 3)
            )
            health_response.raise_for_status()
            health = health_response.json()
            if isinstance(health, dict) and health.get("status") == "ready":
                st.sidebar.success(f"Ready · {health.get('model_name', 'Climate model')}")
            else:
                st.sidebar.warning("The inference server has not reported ready.")
                api_url = None
        except (requests.RequestException, ValueError) as exc:
            st.sidebar.error(f"Local inference server unavailable: {exc}")
            api_url = None

    st.markdown(f"**Initial conditions:** 3 daily states ending {input_date}.")
    channels = {
        "Temperature (K)": 0,
        "Precipitation (mm/day)": 1,
        "Eastward wind (m/s)": 2,
        "Northward wind (m/s)": 3,
        "Sea-level pressure (Pa)": 4,
        "Water vapour (kg/m²)": 5,
    }
    selected_label = st.selectbox("Map layer", tuple(channels))
    channel_index = channels[selected_label]
    lead = st.slider("Map forecast day", 1, steps, min(steps, 4))

    if st.button(
        "Run live simulation",
        type="primary",
        disabled=api_url is None,
        use_container_width=True,
    ):
        initial_states = data[input_day_index - 2 : input_day_index + 1].copy()
        input_doy = int(
            xr.DataArray(times[input_day_index]).dt.dayofyear.item()
        )
        phase = (
            math.sin(2.0 * math.pi * input_doy / 365.0),
            math.cos(2.0 * math.pi * input_doy / 365.0),
        )
        try:
            perturbation = make_perturbation(
                scenario,
                initial_states,
                latitudes,
                longitudes,
                **custom_options,
            )
            with st.spinner("Running the model on the local inference server..."):
                baseline, baseline_ms = _request_simulation(
                    api_url, initial_states, phase, steps, None
                )
                if scenario == SCENARIOS[0]:
                    result = baseline
                    scenario_ms = baseline_ms
                else:
                    result, scenario_ms = _request_simulation(
                        api_url, initial_states, phase, steps, perturbation
                    )

            st.session_state["varsha_result"] = {
                "baseline": baseline,
                "scenario": result,
                "scenario_name": scenario,
                "scenario_ms": scenario_ms,
                "baseline_ms": baseline_ms,
                "input_date": input_date,
                "input_state": initial_states[-1],
                "latitudes": latitudes,
                "longitudes": longitudes,
                "selection_key": selection_key,
            }
        except (requests.RequestException, ValueError, TypeError) as exc:
            st.error(f"Simulation failed: {exc}")

result = st.session_state.get("varsha_result")
if result is not None:
    forecast = result["scenario"]
    baseline = result["baseline"]
    step_data = forecast[lead - 1]
    counterfactual = step_data - baseline[lead - 1]
    metric_grid = step_data[channel_index]
    left, right = st.columns((3, 2))
    with left:
        st.subheader(f"{selected_label} · forecast day +{lead}")
        st.html(
            grid_svg(
                metric_grid,
                result["latitudes"],
                result["longitudes"],
                f"{selected_label}, forecast day plus {lead}",
            )
        )
        st.caption(
            "Each colored cell is one kit grid cell; coordinates are grid centers. "
            "This is a gridded field, not an administrative-boundary map."
        )
    with right:
        st.subheader("Live impact screening")
        impacts = impact_summary(
            forecast,
            result["input_state"],
            result["latitudes"],
            result["longitudes"],
        )
        st.metric(
            "Central-India mean rain",
            f"{impacts['crop_rain_mm_day']:.1f} mm/day",
            help="A broad-region mean; not a calibrated crop-yield model.",
        )
        st.metric(
            "Rain-deficit days",
            f"{impacts['crop_deficit_days']} / {len(forecast)}",
            help="Days with broad central-India mean rainfall below a simple 4 mm/day screening threshold.",
        )
        st.metric(
            "Cells above 10 mm/day",
            f"{100 * impacts['flood_exceedance_fraction']:.1f}%",
            help="Fraction of forecast grid-cell days crossing the kit's heavy-rain threshold; not a flood inundation prediction.",
        )
        st.metric("Peak grid-cell rain", f"{impacts['peak_rain_mm_day']:.1f} mm/day")
        if impacts["wind_power_relative_pct"] is None:
            st.metric("Wind-energy proxy", "Unavailable (near-zero input wind)")
        else:
            st.metric(
                "Wind-energy proxy",
                f"{impacts['wind_power_index']:.2f}× input",
                f"{impacts['wind_power_relative_pct']:+.1f}%",
                help="Relative mean wind-speed-cubed index; not a calibrated turbine-generation forecast.",
            )

    st.subheader("Counterfactual response")
    map_column, details_column = st.columns((3, 2))
    with map_column:
        st.html(
            grid_svg(
                counterfactual[channel_index],
                result["latitudes"],
                result["longitudes"],
                f"Scenario minus baseline {selected_label}, forecast day plus {lead}",
                diverging=True,
            )
        )
        st.caption("Scenario minus unperturbed model output, in the selected channel's physical units.")
    with details_column:
        differences = forecast[:, channel_index].mean(axis=(1, 2)) - baseline[
            :, channel_index
        ].mean(axis=(1, 2))
        st.line_chart(
            {
                "Scenario minus baseline": differences,
            }
        )
        st.write(f"**Branch:** {result['scenario_name']}")
        st.write(
            f"**Inference:** baseline {result['baseline_ms']:.1f} ms; "
            f"branch {result['scenario_ms']:.1f} ms."
        )
        st.caption(
            "Impact indicators are transparent screening proxies derived only from this live rollout; "
            "they are not crop-yield, flood-depth, or power-generation forecasts."
        )
else:
    st.info("Choose an initialization day and scenario, then run a live simulation to see maps and impacts.")
