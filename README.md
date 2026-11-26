# VARSHA | Multiverse of Monsoon

**Team:** Code Blooded · **Track 2:** Earth-System Modelling, Spatiotemporal Deep Learning & Climate Simulation

VARSHA is an offline, autoregressive Indian weather digital twin on the kit’s 32 × 36 daily grid. Given the previous three states, it predicts one physical-unit state per day. A counterfactual branch applies a localized delta to the latest input state and rolls the same model forward.

## Model and safeguards

The six-block, 64-channel 2D ResNet predicts normalized daily increments for six dynamic fields, conditioned on day-of-year sine/cosine planes and two bundled geography fields. `stats.json` handles normalization internally; the API boundary remains in physical units. Each forecast state is autoregressive and clamps precipitation at zero and temperature to the contract range. The current checkpoint uses signed rainfall increments and a bundled two-harmonic temperature climatology; lead-decaying anomaly retention reduces long-range temperature drift without shipping a daily lookup table. For a nonzero shock, inference evolves a matched control and shocked trajectory together, applying seasonal anchoring to the control and preserving the model-produced shock-minus-control temperature response. Zero perturbations remain an exact identity. Legacy checkpoints retain their original rain head and do not use the seasonal correction.

The bundled checkpoint is a 1,200-step CPU fine-tune on 2005–2020, using 8-day rollouts, heavy-rain-weighted loss, and late-lead temperature weighting; it is not the full Colab curriculum. Candidate selection used 2021 metrics in `submission/validation_metrics.json`, so those results are validation—not an untouched test or a guarantee of skill on hidden 2023–24 cases.

The server loads only repository weights and assets. It makes no network requests and uses no future-weather lookup. The dashboard accepts only localhost/loopback HTTP, visualizes live model output, and compares baseline and alternate maps on shared scales. Regional rain, hot-area-days, heavy-rain coverage, and wind-speed-cubed indicators use latitude-weighted grid areas and are screening proxies—not calibrated hazard, crop-yield, or energy forecasts.

## Run locally

Use the organizer-compatible Python environment with PyTorch, NumPy, FastAPI and Uvicorn. From the repository root, start the inference server:

```powershell
python .\serve.py --port 8000
```

In a second terminal, install the dashboard dependencies and launch the UI:

```powershell
python -m pip install -r .\dashboard\requirements.txt
python -m streamlit run .\dashboard\app.py
```

The dashboard loads a local `india_climate_train.nc` from the repository or its parent directory, or accepts an upload. To use another local file, set `VARSHA_DATA` before launching Streamlit. The NetCDF is not included in the submission; obtain it from the official [yug-monsoon-kit v1.0.0 release](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0).

Once the dataset is loaded, the dashboard shows an observed-state grid preview. Forecast grids appear after **Run simulation**; choose a counterfactual scenario to see baseline and alternate maps with their shared color scale.

Run tests and, when the kit package is installed, its API validator:

```powershell
python -m unittest discover -s tests -v
python -m monsoon_r1.schema_validator http://127.0.0.1:8000
```

## Train and evaluate

Keep the training NetCDF outside the repository. Training scripts accept `--data PATH`; by default they look beside the repository. Both scripts require `--holdout-year` and derive the split from the dataset time-axis years. The example uses 2021, which remains a used-for-selection validation year for this checkpoint. Training writes signed-rain candidate checkpoints without overwriting submitted weights:

```powershell
python .\training\train_varsha.py --holdout-year 2021 --steps 1200 --rollout-horizon 8 --batch-size 4 --learning-rate 0.000005 --heavy-rain-weight 3 --heavy-rain-lead-decay 0.25 --late-temperature-weight 1.5 --late-lead-weight 1 --spatial-variance-weight 0.2 --initial-weights .\varsha_weights.pt --output .\varsha_candidate.pt
python .\training\evaluate_varsha.py --holdout-year 2021 --weights .\varsha_candidate.pt --json-out .\submission\validation_metrics.json
python .\training\evaluate_baseline.py
```

Evaluation compares aligned holdout-year windows against persistence and day-of-year climatology. It reports both pooled-RMSE skill and arithmetic mean per-window skill, plus day-8 variance-ratio ranges and anti-blur band counts, heavy-rain CSI, and physical-validity checks. A checkpoint trained with Colab’s `--full-train` option has seen the holdout year, so those scores are not holdout evidence. The supplied 2021 split was used for candidate selection; results are optimistic validation, not hidden 2023–24 performance guarantees.

For GPU training, upload `colab_train.py`, `india_climate_train.nc`, and `stats.json` to the same mounted Drive folder. Run `python colab_train.py --holdout-year 2021 --smoke` first, then `python colab_train.py --holdout-year 2021 --phase all`. The split is derived from the time axis: the requested year is held out, and the preceding two years are used for validation. Use `--full-train` only for a final refit after model selection; that run includes the holdout year and has no internal validation holdout.

## Submission contents and limits

- `serve.py`, `varsha_weights.pt`, `static_norm.npy`, and `stats.json`: offline CPU inference.
- `training/`: training, data inspection, and baseline/model evaluation.
- `colab_train.py`: GPU curriculum training with signed-rain, calendar-aligned rollouts and localized-shock augmentation.
- `dashboard/`: Streamlit UI, local shock generation, comparable forecast maps, and area-weighted rain/heat/wind impact screening.
- `tests/`: server and dashboard-logic checks.
- `submission/Code-Blooded-VARSHA-Brief.pdf`: two-page model and impact brief.

VARSHA is deterministic and operates on daily, approximately 1° grid cells. It has no explicit mass/energy-conservation solver or ensemble uncertainty, and autoregressive errors can grow with lead time. Its impacts are exploratory signals, not operational forecasts.
