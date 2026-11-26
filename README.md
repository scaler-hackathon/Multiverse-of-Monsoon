# Code Blooded | VARSHA

**Multiverse of Monsoon — Track 2: Earth-System Modelling, Spatiotemporal Deep Learning & Climate Simulation**

VARSHA is an offline-capable, autoregressive weather model and live counterfactual dashboard for the Indian 32 x 36 daily climate grid. It predicts one day at a time from the previous three states and accepts physical-unit perturbations to the latest state.

## Team and design

**Team:** Code Blooded

**Model:** VARSHA-v1-ResNet

The server normalizes the six dynamic channels with the kit's `stats.json`, concatenates the three input days with the two bundled normalized geography fields and day-of-year sine/cosine planes, and predicts the next-state residual with a four-block 2D ResNet. It autoregressively feeds forecasts back into the model. Forecast rain is floored at zero and air temperature is bounded to the API's -40 C to +60 C validity range. These guards prevent invalid outputs; they are not a substitute for atmospheric conservation laws.

The API always accepts and returns physical units. Static geography, normalization data and trained weights are local files. The inference server makes no network requests and uses no future dates or weather-data lookups. The dashboard only calls the inference server on localhost.

## Repository layout

```text
serve.py                         Offline FastAPI inference server
varsha_weights.pt                Included CPU model checkpoint
static_norm.npy                  Included normalized geography maps
stats.json                       Kit channel statistics
API_CONTRACT.md                  Official request/response contract
training/
  train_varsha.py                Reproducible, multi-day autoregressive training
  evaluate_varsha.py             Aligned skill, rain-CSI and variance evaluation
  evaluate_baseline.py           Persistence/climatology-only benchmark
  inspect_data.py                Local dataset inspection
dashboard/
  app.py                         Streamlit UI backed by the live /simulate API
  logic.py                       Local shock, map and impact calculations
  requirements.txt               Dashboard-only dependencies
tests/
  test_dashboard_logic.py        Shock, offline URL, and impact edge cases
submission/
  create_brief.py                Builds the required two-page PDF without extra packages
```

The multi-year NetCDF and kit wheel are intentionally not committed. This avoids shipping a climate archive; the training and dashboard scripts use a local copy downloaded from the official [yug-monsoon-kit v1.0.0 release](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0). By default, put the training file next to (not inside) the repository folder; training commands also accept `--data PATH`, and the dashboard accepts `VARSHA_DATA`. The offline evaluator needs the included weights and static assets, not the training archive.

## Run offline inference

Use the provided organizer environment (Python, CPU PyTorch, NumPy, FastAPI and Uvicorn), then start the server from the repository root:

```powershell
python .\serve.py --port 8000
```

It binds to `0.0.0.0` and exposes the kit's `POST /health` and `POST /simulate` endpoints. Requests use the canonical nested JSON lists from `API_CONTRACT.md`; the server validates dimensions, finite values, seasonal phase, perturbations and step limits, returning clear 4xx responses for malformed requests.

The kit validator can be used when the team-kit package is installed:

```powershell
python -m monsoon_r1.schema_validator http://127.0.0.1:8000
```

Run the local shock, dashboard-logic and inference-boundary tests with:

```powershell
python -m unittest discover -s tests -v
```

## Reproduce training and evaluation

Download `india_climate_train.nc` from the official kit release into the parent directory of the repository, or specify another local path with `--data`. Keep the validation year 2021 out of training. Install training dependencies in the development environment if they are not already present:

```powershell
python -m pip install torch numpy xarray
```

Train a candidate checkpoint without overwriting the checked-in model:

```powershell
python .\training\train_varsha.py --steps 1200 --rollout-horizon 8 --batch-size 6 --learning-rate 0.00001 --output .\varsha_candidate.pt
```

Compare it against the current persistence and day-of-year climatology baselines on the same held-out 2021 dates. The evaluator reports latitude-weighted RMSE at +1, +4 and +8 days, CSI for rain >= 10 mm/day, finite/physical checks, and day-8 spatial variance:

```powershell
python .\training\evaluate_varsha.py --weights .\varsha_candidate.pt --json-out .\submission\validation_metrics.json
python .\training\evaluate_baseline.py
```

Evaluation is a local historical holdout, not a guarantee of performance on the hidden 2023-24 weather windows. Re-run it for every candidate and only replace the included weights after comparing the complete report; the training script writes checkpoints atomically.

## Run the live dashboard

Install the UI dependencies in a development environment:

```powershell
python -m pip install -r .\dashboard\requirements.txt
```

Keep the inference server running in one terminal and launch the dashboard in another:

```powershell
python -m streamlit run .\dashboard\app.py
```

Upload the official kit NetCDF or place your local copy in the repository root, choose the last input day, a forecast lead and a scenario, then run the simulation. The dashboard shows an actual forecast grid, scenario-minus-baseline response, inference time and live rainfall/wind screening metrics. Available branches are the unperturbed baseline, kit-style warm Bay of Bengal, Arabian Sea wind slowdown, exact zero perturbation and a configurable local shock. Calls are restricted to `localhost`/loopback HTTP; the UI has no weather API or external service integration.

Impact outputs are explicitly screening proxies: broad central-India rainfall deficit, grid cells above the 10 mm/day heavy-rain threshold, peak grid rainfall and a relative wind-speed-cubed index. They are not calibrated crop-yield, flood-depth or power-generation forecasts.

## Two-page desk-review brief

After running the evaluator to create structured metrics, build the PDF:

```powershell
python .\submission\create_brief.py
```

This writes `submission/Code-Blooded-VARSHA-Brief.pdf`, covering the model, rollout safeguards and failure modes on page one, and live visualization plus the impact extension on page two.

## Limits

The model is deterministic, uses daily 1-degree grids and has no explicit mass/energy-conservation solver or ensemble uncertainty. Autoregressive errors grow with lead time. The supplied 2005-2021 data cannot verify hidden post-2023 event skill. Dashboard impact indicators are educational screening tools only.
