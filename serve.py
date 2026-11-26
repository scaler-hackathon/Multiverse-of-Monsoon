import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()
MODEL_NAME = "VARSHA-v1-ResNet"

ROOT = Path(__file__).resolve().parent

# --- 1. Model Architecture ---
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
        return self.out_proj(feat)


# --- 2. Load Weights and Static Assets ---
with open(ROOT / "stats.json") as f:
    stats = json.load(f)

# Shape (6, 1, 1) for broadcasting with (6, 32, 36)
CH_MEAN = np.array(stats["mean"], dtype=np.float32).reshape(6, 1, 1)
CH_STD = np.array(stats["std"], dtype=np.float32).reshape(6, 1, 1)
NORM_STATIC = np.load(ROOT / "static_norm.npy").astype(np.float32)

model = VarshaNet()
weights_path = ROOT / "varsha_weights.pt"
model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
model.eval()


# --- 3. Request Parsing ---
def _arr(x):
    """Accept nested lists (canonical contract) or base64 dicts."""
    if isinstance(x, dict):
        import base64
        return np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4").reshape(x["shape"]).astype("f4")
    return np.asarray(x, dtype="f4")


# --- 4. Simulation Engine ---
def simulate(init_states: np.ndarray, doy_sincos, steps: int, perturbation) -> np.ndarray:
    """
    init_states: [3, 6, 32, 36] in physical units.
    perturbation: [6, 32, 36] delta or None.
    doy_sincos: [sin, cos] of day-of-year for init_states[-1].
    returns: [steps, 6, 32, 36] physical units.
    """
    states = init_states.copy()
    if perturbation is not None:
        states[-1] = states[-1] + perturbation

    # Normalize input states: (3, 6, 32, 36) with (6, 1, 1)
    norm_states = (states - CH_MEAN) / CH_STD
    history = [norm_states[i] for i in range(3)]

    # Infer base day of year from [sin, cos]
    sin_val, cos_val = float(doy_sincos[0]), float(doy_sincos[1])
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
            history.append(next_norm)

            # De-normalize to physical units
            next_phys = next_norm * CH_STD + CH_MEAN

            # Physical guards per API Contract
            next_phys[1] = np.clip(next_phys[1], 0.0, None)         # rain >= 0
            next_phys[0] = np.clip(next_phys[0], 233.15, 333.15)    # -40C to 60C

            forecasts.append(next_phys)

    return np.stack(forecasts, axis=0).astype(np.float32)


@app.post("/health")
def health():
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    import time
    t0 = time.perf_counter()
    body = await request.json()
    
    init = _arr(body["init_states"])
    pert = None if body.get("perturbation") is None else _arr(body["perturbation"])
    
    out = simulate(init, body["doy_sincos"], int(body["steps"]), pert)
    
    return {
        "rollout": np.asarray(out, dtype="f4").tolist(),
        "inference_ms": (time.perf_counter() - t0) * 1000.0
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, log_level="warning")