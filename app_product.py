"""Optional offline scenario and impact API for later-round demonstrations."""

import argparse
import json
import math
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

import serve
from dashboard.logic import impact_summary

app = FastAPI(title="VARSHA Product API")
LATITUDES = np.linspace(6.5, 37.5, 32, dtype=np.float64)
LONGITUDES = np.linspace(65.5, 100.5, 36, dtype=np.float64)


@app.post("/health")
async def health():
    return {
        "status": "ready",
        "service": "VARSHA Product API",
        "model_name": serve.MODEL_NAME,
    }


@app.post("/scenario")
async def scenario(request: Request):
    started = time.perf_counter()
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=400, detail="Request body must be valid JSON."
        ) from exc
    if not isinstance(body, dict):
        raise HTTPException(
            status_code=422, detail="Request body must be a JSON object."
        )

    required = ("init_states", "doy_sincos", "steps")
    missing = [key for key in required if key not in body]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields: {', '.join(missing)}.",
        )
    steps = body["steps"]
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 1 <= steps <= serve.MAX_STEPS
    ):
        raise HTTPException(
            status_code=422,
            detail=f"steps must be an integer between 1 and {serve.MAX_STEPS}.",
        )

    try:
        initial = serve._arr(
            body["init_states"], (3, 6, 32, 36), "init_states"
        )
        phase = np.asarray(body["doy_sincos"], dtype=np.float64)
        if (
            phase.shape != (2,)
            or not np.isfinite(phase).all()
            or float(np.hypot(*phase)) < 1e-8
        ):
            raise ValueError(
                "doy_sincos must contain two finite values defining a non-zero phase."
            )
        perturbation = (
            None
            if body.get("perturbation") is None
            else serve._arr(body["perturbation"], (6, 32, 36), "perturbation")
        )

        control = serve.simulate(initial, phase, steps, None)
        branch = serve.simulate(initial, phase, steps, perturbation)
        baseline_impact = impact_summary(
            control, initial[-1], LATITUDES, LONGITUDES
        )
        branch_impact = impact_summary(
            branch, initial[-1], LATITUDES, LONGITUDES
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    difference = branch.astype(np.float64) - control.astype(np.float64)
    return {
        "model_name": serve.MODEL_NAME,
        "baseline_rollout": control.tolist(),
        "scenario_rollout": branch.tolist(),
        "impacts": {
            "baseline": baseline_impact,
            "scenario": branch_impact,
        },
        "response": {
            "mean_change_by_channel": difference.mean(axis=(2, 3)).tolist(),
            "rms_change_by_channel": np.sqrt(
                np.mean(difference**2, axis=(2, 3))
            ).tolist(),
        },
        "inference_ms": (time.perf_counter() - started) * 1000.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the optional offline VARSHA product API."
    )
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
