import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()
LEGACY_RAIN_MODE = "softplus_increment"
SIGNED_RAIN_MODE = "signed_increment"

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(min(4, torch.get_num_threads()))

# --- 1. Model Architecture (Exact Match to Trained Colab Checkpoint) ---
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
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
        rain_mode=LEGACY_RAIN_MODE,
    ):
        super().__init__()
        if rain_mode not in (LEGACY_RAIN_MODE, SIGNED_RAIN_MODE):
            raise ValueError(f"Unsupported rain increment mode: {rain_mode}")
        self.rain_mode = rain_mode
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(base_dim) for _ in range(6)])
        self.out_proj = nn.Conv2d(base_dim, out_channels, kernel_size=1)  # 1x1 pointwise conv
        self.rain_scale = nn.Parameter(torch.ones(1))
        self.t2m_climatology_coefficients = None
        self.t2m_anomaly_retention = 1.0
        self.t2m_anomaly_decay = 0.0

    def forward(self, x):
        feat = self.in_proj(x)
        feat = self.blocks(feat)
        delta = self.out_proj(feat)
        rain = delta[:, 1:2] * self.rain_scale
        if self.rain_mode == LEGACY_RAIN_MODE:
            rain = F.softplus(rain)
        return torch.cat([delta[:, :1], rain, delta[:, 2:]], dim=1)


def load_model_weights(target_model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        rain_mode = checkpoint.get("rain_mode")
        if not isinstance(rain_mode, str):
            raise ValueError("Checkpoint metadata must declare a rain_mode.")
        metadata = {
            key: value
            for key, value in checkpoint.items()
            if key not in ("state_dict", "rain_mode")
        }
    else:
        state_dict = checkpoint
        rain_mode = LEGACY_RAIN_MODE
        metadata = {"checkpoint_format": "legacy"}

    if rain_mode not in (LEGACY_RAIN_MODE, SIGNED_RAIN_MODE):
        raise ValueError(f"Unsupported checkpoint rain_mode: {rain_mode}")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint must contain a model state dictionary.")
    climatology = metadata.get("t2m_climatology_coefficients")
    if climatology is None:
        target_model.t2m_climatology_coefficients = None
        target_model.t2m_anomaly_retention = 1.0
        target_model.t2m_anomaly_decay = 0.0
    else:
        if not isinstance(climatology, torch.Tensor):
            raise ValueError("Temperature climatology coefficients must be a tensor.")
        climatology = climatology.detach().to(device="cpu", dtype=torch.float32)
        if tuple(climatology.shape) != (5, 32, 36) or not torch.isfinite(climatology).all():
            raise ValueError("Temperature climatology coefficients must be finite with shape (5, 32, 36).")
        try:
            retention = float(metadata["t2m_anomaly_retention"])
            decay = float(metadata["t2m_anomaly_decay"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Checkpoint must define finite temperature anomaly blending settings.") from exc
        if not np.isfinite((retention, decay)).all() or not 0.0 <= retention <= 1.0 or decay < 0.0:
            raise ValueError("Temperature anomaly blending settings are outside valid ranges.")
        target_model.t2m_climatology_coefficients = climatology.numpy()
        target_model.t2m_anomaly_retention = retention
        target_model.t2m_anomaly_decay = decay
    target_model.rain_mode = rain_mode
    target_model.load_state_dict(state_dict)
    metadata["rain_mode"] = rain_mode
    return metadata


# --- 2. Load Weights and Static Assets ---
with open(ROOT / "stats.json") as f:
    stats = json.load(f)

# Shape (6, 1, 1) for broadcasting with (6, 32, 36)
CH_MEAN = np.array(stats["mean"], dtype=np.float32).reshape(6, 1, 1)
CH_STD = np.array(stats["std"], dtype=np.float32).reshape(6, 1, 1)
NORM_STATIC = np.load(ROOT / "static_norm.npy").astype(np.float32)
TEMP_MIN = float(np.nextafter(np.float32(233.15), np.float32(np.inf)))
TEMP_MAX = float(np.float32(333.15))

model = VarshaNet(in_channels=22, out_channels=6, base_dim=64)
weights_path = ROOT / "varsha_weights.pt"
checkpoint_metadata = load_model_weights(model, weights_path)
if model.t2m_climatology_coefficients is not None:
    MODEL_NAME = "VARSHA-v4-ResNet64-Seasonal"
elif model.rain_mode == LEGACY_RAIN_MODE:
    MODEL_NAME = "VARSHA-v2-ResNet64"
else:
    MODEL_NAME = "VARSHA-v3-ResNet64"
model.eval()


# --- 3. Request Parsing ---
def _arr(x, expected_shape, name):
    """Accept nested lists (canonical contract) or base64 dicts."""
    if isinstance(x, dict):
        import base64
        try:
            if list(x["shape"]) != list(expected_shape):
                raise ValueError(f"{name} must have shape {expected_shape}.")
            raw = base64.b64decode(x["b64"], validate=True)
            expected_bytes = int(np.prod(expected_shape)) * np.dtype("<f4").itemsize
            if len(raw) != expected_bytes:
                raise ValueError(f"{name} has an invalid encoded length.")
            result = np.frombuffer(raw, dtype="<f4").reshape(expected_shape).copy()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {name}: {exc}") from exc
    else:
        try:
            result = np.asarray(x, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain numeric values.") from exc
        if result.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return result.astype(np.float32, copy=False)


# --- 4. Simulation Engine ---
def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """
    init_states: [3, 6, 32, 36] in physical units.
    perturbation: [6, 32, 36] delta or None.
    doy_sincos: [sin, cos] of day-of-year for init_states[-1].
    returns: [steps, 6, 32, 36] physical units.
    """
    states = _arr(init_states, (3, 6, 32, 36), "init_states").copy()
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or not 1 <= steps <= 30:
        raise ValueError("steps must be an integer between 1 and 30.")
    phase = np.asarray(doy_sincos, dtype=np.float64)
    if phase.shape != (2,) or not np.isfinite(phase).all():
        raise ValueError("doy_sincos must contain two finite numbers.")
    if float(np.hypot(phase[0], phase[1])) < 1e-8:
        raise ValueError("doy_sincos must define a non-zero seasonal phase.")
    if perturbation is not None:
        states[-1] = states[-1] + _arr(perturbation, (6, 32, 36), "perturbation")
        if not np.isfinite(states[-1]).all():
            raise ValueError("Perturbed initialization must remain finite.")

    # Normalize input states: (3, 6, 32, 36) with (6, 1, 1)
    norm_states = (states - CH_MEAN) / CH_STD
    history = [norm_states[i] for i in range(3)]

    # Infer base day of year from [sin, cos]
    sin_val, cos_val = float(phase[0]), float(phase[1])
    base_doy = math.atan2(sin_val, cos_val) * (365.0 / (2.0 * math.pi))
    if base_doy < 0:
        base_doy += 365.0

    H, W = 32, 36
    forecasts = []

    with torch.no_grad():
        for step in range(steps):
            curr_doy = (base_doy + 1 + step) % 365.0
            sin_d = math.sin(2.0 * math.pi * curr_doy / 365.0)
            cos_d = math.cos(2.0 * math.pi * curr_doy / 365.0)

            doy_planes = np.stack([
                np.full((H, W), sin_d, dtype=np.float32),
                np.full((H, W), cos_d, dtype=np.float32)
            ], axis=0)

            # Assemble 22 channels: 18 dynamic + 2 static + 2 doy
            x_in = np.concatenate([
                np.concatenate(history[-3:], axis=0),
                NORM_STATIC,
                doy_planes
            ], axis=0)

            x_tensor = torch.from_numpy(x_in[None])
            delta = model(x_tensor).numpy()[0]

            # Autoregressive update
            next_norm = history[-1] + delta

            # De-normalize to physical units
            next_phys = next_norm * CH_STD + CH_MEAN
            climatology = model.t2m_climatology_coefficients
            if climatology is not None:
                climate_phase = 2.0 * math.pi * ((curr_doy - 1.0) % 365.0) / 365.0
                climate_basis = np.asarray(
                    (
                        1.0,
                        math.sin(climate_phase),
                        math.cos(climate_phase),
                        math.sin(2.0 * climate_phase),
                        math.cos(2.0 * climate_phase),
                    ),
                    dtype=np.float32,
                )
                climate_t2m = np.tensordot(
                    climate_basis, climatology, axes=(0, 0)
                )
                retention = model.t2m_anomaly_retention * math.exp(
                    -model.t2m_anomaly_decay * step
                )
                next_phys[0] = climate_t2m + retention * (
                    next_phys[0] - climate_t2m
                )

            # Physical guards per API Contract
            next_phys[1] = np.clip(next_phys[1], 0.0, None)         # rain >= 0
            next_phys[0] = np.clip(next_phys[0], TEMP_MIN, TEMP_MAX)
            if not np.isfinite(next_phys).all():
                raise FloatingPointError(f"Model produced a non-finite forecast at lead {step + 1}.")

            next_norm = (next_phys - CH_MEAN) / CH_STD
            history.append(next_norm)

            forecasts.append(next_phys)

    return np.stack(forecasts, axis=0).astype(np.float32)


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    import time
    t0 = time.perf_counter()
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=422, detail="Request body must be a JSON object.")

    required = ("init_states", "doy_sincos", "steps")
    missing = [key for key in required if key not in body]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required fields: {', '.join(missing)}.")
    steps = body["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 30:
        raise HTTPException(status_code=422, detail="steps must be an integer between 1 and 30.")
    try:
        init = _arr(body["init_states"], (3, 6, 32, 36), "init_states")
        phase = np.asarray(body["doy_sincos"], dtype=np.float64)
        if phase.shape != (2,) or not np.isfinite(phase).all() or np.hypot(*phase) < 1e-8:
            raise ValueError("doy_sincos must contain two finite values defining a non-zero phase.")
        pert = (
            None
            if body.get("perturbation") is None
            else _arr(body["perturbation"], (6, 32, 36), "perturbation")
        )
        out = simulate(init, phase, steps, pert)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FloatingPointError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    return {
        "rollout": np.asarray(out, dtype="f4").tolist(),
        "inference_ms": (time.perf_counter() - t0) * 1000.0
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")