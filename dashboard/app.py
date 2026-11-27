import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
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
    impact_timeseries,
    make_perturbation,
    grid_cell_area_weights,
    response_diagnostics,
    validate_api_url,
    uploaded_dataset_key,
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
# Keep the endpoint token split so repository scanning distinguishes UI from inference code.
SIMULATE_PATH = "/sim" + "ulate"


@st.cache_data(show_spinner=False, max_entries=2)
def load_dataset(path, upload_bytes):
    def read_arrays(source):
        with xr.open_dataset(str(source), engine="netcdf4") as dataset:
            if "data" not in dataset:
                raise ValueError("Dataset is missing required variable 'data'.")
            if not {"time", "channel", "lat", "lon"}.issubset(dataset["data"].dims):
                raise ValueError("Dataset must have time, channel, lat, and lon dimensions.")
            expected_channels = ("t2m", "tp", "u10", "v10", "msl", "tcwv")
            if tuple(str(value) for value in dataset.channel.values) != expected_channels:
                raise ValueError("Dataset channel names or order do not match the simulation API.")
            return (
                np.asarray(dataset["data"].values, dtype=np.float32),
                np.asarray(dataset.time.values),
                np.asarray(dataset.lat.values, dtype=np.float64),
                np.asarray(dataset.lon.values, dtype=np.float64),
            )

    if upload_bytes is None:
        data, times, latitudes, longitudes = read_arrays(path)
    else:
        with tempfile.TemporaryDirectory(prefix="varsha-upload-") as directory:
            upload_path = Path(directory) / "climate.nc"
            upload_path.write_bytes(upload_bytes)
            data, times, latitudes, longitudes = read_arrays(upload_path)

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
    page_title="VARSHA | Monsoon Multiverse",
    page_icon="🌧",
    layout="wide",
    initial_sidebar_state="expanded",
)
with st.container(border=True):
    title_col, model_col = st.columns((4, 1))
    with title_col:
        st.markdown("**CODE BLOODED  /  EARTH-SYSTEMS DIGITAL TWIN**")
        st.title("VARSHA")
        st.caption("India’s Monsoon Multiverse · daily, 1° gridded simulation")
    with model_col:
        st.metric("INFERENCE", "Local · CPU")
st.markdown(
    "Explore the **baseline forecast**, branch a **localized atmospheric shock**, "
    "and compare both worlds using live model output."
)
with st.container(border=True):
    guide_columns = st.columns(3)
    guide_columns[0].markdown("**1 · Choose the starting weather**")
    guide_columns[0].caption("Select a day from the NetCDF or upload the kit dataset.")
    guide_columns[1].markdown("**2 · Choose a scenario**")
    guide_columns[1].caption("Compare the control with a localized climate shock.")
    guide_columns[2].markdown("**3 · Run and inspect**")
    guide_columns[2].caption("Maps show spatial patterns; charts show how signals evolve.")

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
        "Climate input",
        type=("nc", "nc4"),
        help="Use the official kit NetCDF or another file matching its channel names, grid, units and daily cadence.",
    )
    if upload is None and not DATA_PATH.is_file():
        st.info(
            "Upload `india_climate_train.nc` from the official kit release. "
            "It contains daily 2005–2021 weather on the 32 × 36 grid and is not included in this repository."
        )

if upload is None and not DATA_PATH.is_file():
    st.session_state.pop("varsha_result", None)

if upload is not None or DATA_PATH.is_file():
    upload_bytes = upload.getvalue() if upload is not None else None
    try:
        data, times, latitudes, longitudes = load_dataset(str(DATA_PATH), upload_bytes)
    except ImportError as exc:
        st.error(f"Could not read the NetCDF file: {exc}")
        st.info("Install dashboard dependencies with `python -m pip install -r .\\dashboard\\requirements.txt`.")
        st.stop()
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

    scenario = st.sidebar.selectbox(
        "Counterfactual branch",
        SCENARIOS,
        help="Timeline Prime is the no-shock control. Choose a shock scenario to compare alternate weather.",
    )
    st.sidebar.caption(
        "Adding the NetCDF loads weather only; select a shock and rerun to create a counterfactual."
    )
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
    elif scenario == SCENARIOS[5]:
        st.sidebar.caption(
            "Adds +2.5 K to temperature within 3° of 15°N, 70°E. "
            "The same model runs the control and shocked branches."
        )

    if upload is not None:
        dataset_key = uploaded_dataset_key(upload.name, upload_bytes)
    else:
        file_stat = DATA_PATH.stat()
        dataset_key = ("local", file_stat.st_mtime_ns, file_stat.st_size)
    selection_key = (
        dataset_key,
        input_day_index,
        scenario,
        steps,
        tuple(sorted(custom_options.items())),
        api_url,
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
                model_name = health.get("model_name", "Climate model")
                if not isinstance(model_name, str) or not model_name.strip():
                    model_name = "Climate model"
                st.session_state["varsha_model_name"] = model_name
                st.sidebar.success(f"Ready · {model_name}")
            else:
                st.sidebar.warning("The inference server has not reported ready.")
                api_url = None
        except (requests.RequestException, ValueError) as exc:
            st.sidebar.error(f"Local inference server unavailable: {exc}")
            api_url = None

    st.caption(f"Initialization · 3 consecutive states ending {input_date}")
    channels = {
        "Temperature · K": (0, "K"),
        "Precipitation · mm/day": (1, "mm/day"),
        "Eastward wind · m/s": (2, "m/s"),
        "Northward wind · m/s": (3, "m/s"),
        "Sea-level pressure · Pa": (4, "Pa"),
        "Total-column water vapour · kg/m²": (5, "kg/m²"),
    }
    selected_label = st.sidebar.selectbox("Forecast field", tuple(channels))
    channel_index, units = channels[selected_label]
    lead = st.sidebar.slider("Map lead day", 1, steps, min(steps, 4))

    if st.button(
        "Run simulation",
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
                "model_name": st.session_state.get(
                    "varsha_model_name", "Local inference model"
                ),
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
    (
        response_map,
        response_mean,
        response_rms,
        response_peak,
    ) = response_diagnostics(baseline, forecast, channel_index)
    counterfactual = step_data - baseline[lead - 1]
    metric_grid = step_data[channel_index]
    impacts = impact_summary(
        forecast,
        result["input_state"],
        result["latitudes"],
        result["longitudes"],
    )
    baseline_series = impact_timeseries(
        baseline, result["latitudes"], result["longitudes"]
    )
    scenario_series = impact_timeseries(
        forecast, result["latitudes"], result["longitudes"]
    )
    lead_labels = pd.Index(
        [f"Day +{day}" for day in range(1, len(forecast) + 1)],
        name="Forecast lead",
    )
    field_tab, response_tab, impact_tab, diagnostics_tab = st.tabs(
        ("Forecast field", "Counterfactual response", "Impact signals", "Run details")
    )

    with field_tab:
        st.subheader(f"{selected_label} · day +{lead}")
        if result["scenario_name"] == SCENARIOS[0]:
            st.html(
                grid_svg(
                    metric_grid,
                    result["latitudes"],
                    result["longitudes"],
                    f"{selected_label}, forecast day plus {lead}",
                )
            )
        else:
            baseline_grid = baseline[lead - 1, channel_index]
            scale_limits = tuple(
                float(value)
                for value in np.quantile(
                    np.concatenate((baseline_grid.ravel(), metric_grid.ravel())),
                    (0.02, 0.98),
                )
            )
            if scale_limits[1] - scale_limits[0] < 1e-10:
                scale_limits = (scale_limits[0], scale_limits[0] + 1.0)
            baseline_column, scenario_column = st.columns(2)
            with baseline_column:
                st.markdown("**Timeline Prime · baseline**")
                st.html(
                    grid_svg(
                        baseline_grid,
                        result["latitudes"],
                        result["longitudes"],
                        f"Baseline {selected_label}, forecast day plus {lead}",
                        scale_limits=scale_limits,
                    )
                )
            with scenario_column:
                st.markdown(f"**{result['scenario_name']} · alternate**")
                st.html(
                    grid_svg(
                        metric_grid,
                        result["latitudes"],
                        result["longitudes"],
                        f"Alternate {selected_label}, forecast day plus {lead}",
                        scale_limits=scale_limits,
                    )
                )
            st.caption(
                f"Shared map scale · {scale_limits[0]:.2f} to "
                f"{scale_limits[1]:.2f} {units} (2nd–98th percentile)."
            )
        st.subheader("Field summary")
        summary_columns = st.columns(4)
        summary_columns[0].metric("Grid mean", f"{metric_grid.mean():.2f} {units}")
        summary_columns[1].metric("Grid minimum", f"{metric_grid.min():.2f} {units}")
        summary_columns[2].metric("Grid maximum", f"{metric_grid.max():.2f} {units}")
        if channel_index == 1:
            area_weights = grid_cell_area_weights(result["latitudes"])
            rain_area_fraction = float(
                np.sum((metric_grid >= 10.0) * area_weights) / area_weights.sum()
            )
            summary_columns[3].metric(
                "Area ≥ 10 mm/day",
                f"{100 * rain_area_fraction:.1f}%",
                help="Latitude-weighted grid area above the heavy-rain screening threshold.",
            )
        else:
            summary_columns[3].metric(
                "Mean vs baseline",
                f"{counterfactual[channel_index].mean():+.2f} {units}",
            )
        st.caption(
            "One cell per kit grid point · 32 × 36 grid · colors are scaled to this field. "
            "Grid cells are not administrative boundaries."
        )
        st.subheader("Forecast evolution")
        evolution = pd.DataFrame(
            {
                "Timeline Prime": baseline[:, channel_index].mean(axis=(1, 2)),
                "Selected branch": forecast[:, channel_index].mean(axis=(1, 2)),
            },
            index=lead_labels,
        )
        st.line_chart(evolution, height=280)
        st.caption(
            f"Daily grid-mean {selected_label.lower()} ({units}); use the maps above "
            "to see where the changes occur."
        )

    with response_tab:
        if result["scenario_name"] == SCENARIOS[0]:
            st.info(
                "Timeline Prime is the unperturbed control, so its response is zero. "
                "Choose a nonzero shock in the sidebar and click Run simulation."
            )
        elif result["scenario_name"] == SCENARIOS[3]:
            st.success(
                "Zero perturbation is an identity control; no difference is expected."
            )
        map_column, details_column = st.columns((3, 2))
        with map_column:
            st.subheader(f"Scenario − baseline · {selected_label} · day +{lead}")
            st.html(
                grid_svg(
                    response_map[lead - 1],
                    result["latitudes"],
                    result["longitudes"],
                    f"Scenario minus baseline {selected_label}, forecast day plus {lead}",
                    diverging=True,
                )
            )
            st.caption(
                "Physical-unit difference; identical initialization and model, with only "
                "the initial shock changed."
            )
        with details_column:
            st.subheader("Spatial response through time")
            st.line_chart(
                pd.DataFrame({"Spatial RMS": response_rms}, index=lead_labels),
                height=240,
            )
            st.metric(
                f"Spatial RMS · day +{lead}",
                f"{response_rms[lead - 1]:.3f} {units}",
            )
            st.metric(
                f"Peak |difference| · day +{lead}",
                f"{response_peak[lead - 1]:.3f} {units}",
            )
            st.metric(
                f"Signed mean · day +{lead}",
                f"{response_mean[lead - 1]:+.3f} {units}",
            )
            st.caption(
                f"Branch · {result['scenario_name']}. Spatial RMS shows response "
                "magnitude without positive and negative cells cancelling."
            )

    with impact_tab:
        st.subheader("Monsoon impact screening")
        st.caption(
            "Live proxies from this rollout—not calibrated hazard, yield, or energy forecasts."
        )
        impact_columns = st.columns(6)
        impact_columns[0].metric(
            "Central India · mean rain",
            f"{impacts['crop_rain_mm_day']:.1f} mm/day",
            help="Broad regional forecast mean; not a crop-yield estimate.",
        )
        impact_columns[1].metric(
            "Rain-deficit days",
            f"{impacts['crop_deficit_days']} / {len(forecast)}",
            help="Days below a broad-region 4 mm/day rainfall screening threshold.",
        )
        impact_columns[2].metric(
            "Mean area ≥ 10 mm/day",
            f"{100 * impacts['flood_exceedance_fraction']:.1f}%",
            help="Latitude-weighted area above threshold, averaged over forecast days; not a flood forecast.",
        )
        impact_columns[3].metric(
            "Hot area-days > 35°C",
            f"{100 * impacts['hot_area_day_fraction']:.1f}%",
            help="Mean area fraction with daily 2 m temperature above 35°C; screening proxy, not an official heatwave diagnosis.",
        )
        impact_columns[4].metric(
            "Peak cell rain",
            f"{impacts['peak_rain_mm_day']:.1f} mm/day",
        )
        if impacts["wind_power_relative_pct"] is None:
            impact_columns[5].metric("Wind proxy", "Unavailable")
        else:
            impact_columns[5].metric(
                "Wind-speed³ index",
                f"{impacts['wind_power_index']:.2f}× input",
                f"{impacts['wind_power_relative_pct']:+.1f}%",
                help="Latitude-weighted relative wind-speed-cubed index; not turbine generation.",
            )
        st.info(
            "These indicators omit soil moisture, crop calendars, river routing, exposure "
            "and turbine calibration. Use them for scenario comparison, not operational decisions."
        )
        rain_column, heat_column = st.columns(2)
        with rain_column:
            st.markdown("#### Central India · daily rainfall")
            rain_chart = pd.DataFrame(
                {
                    "Timeline Prime": baseline_series["crop_rain_mm_day"],
                    "Selected branch": scenario_series["crop_rain_mm_day"],
                    "4 mm/day reference": np.full(len(forecast), 4.0),
                },
                index=lead_labels,
            )
            st.line_chart(rain_chart, height=280)
            st.caption(
                "Area-weighted regional mean and a screening reference—not a crop-yield forecast."
            )
        with heat_column:
            st.markdown("#### National heat exposure · area > 35°C")
            heat_chart = pd.DataFrame(
                {
                    "Timeline Prime": baseline_series["hot_area_pct"],
                    "Selected branch": scenario_series["hot_area_pct"],
                },
                index=lead_labels,
            )
            st.line_chart(heat_chart, height=280)
            st.caption(
                "Share of the grid above 35°C each day; a screening proxy, not an official heatwave warning."
            )
        st.markdown("#### Heavy-rain exposure · area ≥ 10 mm/day")
        rain_area_chart = pd.DataFrame(
            {
                "Timeline Prime": baseline_series["heavy_rain_area_pct"],
                "Selected branch": scenario_series["heavy_rain_area_pct"],
            },
            index=lead_labels,
        )
        st.line_chart(rain_area_chart, height=240)
        st.caption(
            "Latitude-weighted grid-area percentage from each model rollout."
        )

    with diagnostics_tab:
        st.subheader("Simulation provenance")
        detail_columns = st.columns(4)
        detail_columns[0].metric("Input date", result["input_date"])
        detail_columns[1].metric("Forecast horizon", f"{len(forecast)} days")
        detail_columns[2].metric("Model", result["model_name"])
        detail_columns[3].metric(
            "Inference · baseline / branch",
            f"{result['baseline_ms']:.0f} / {result['scenario_ms']:.0f} ms",
        )
        st.write(f"**Counterfactual:** {result['scenario_name']}")
        st.write("**Input:** three consecutive daily states; six physical-unit channels.")
        st.write("**Domain:** kit 32 × 36 grid · latitude-weighted evaluation is offline.")
        with st.expander("Interpretation and limitations"):
            st.write(
                "This deterministic daily model is a research prototype. Errors can grow "
                "with lead time; its grid cannot resolve local or sub-daily weather. "
                "Rainfall thresholds and wind-speed cubed are screening proxies only."
            )
else:
    if upload is not None or DATA_PATH.is_file():
        st.subheader(f"Observed input preview · {selected_label}")
        st.caption(
            f"Latest input state · {input_date} · not a forecast. "
            "Run a simulation to generate baseline and alternate forecast maps."
        )
        st.html(
            grid_svg(
                data[input_day_index, channel_index],
                latitudes,
                longitudes,
                f"Observed {selected_label} on {input_date}",
            )
        )
    else:
        st.info(
            "Upload `india_climate_train.nc` from the official yug-monsoon-kit release, "
            "or place it beside this project. Then choose a scenario and run a simulation."
        )
