# VARSHA: Multiverse of Monsoon Climate Digital Twin
**Track 2: Earth-System Modelling, Spatiotemporal Deep Learning & Climate Simulation**  
*Inspired by ISRO's Bharatiya Antariksh Hackathon PS-5 (AI-Powered Digital Twin of India's Climate)*

---

## 1. Mission Overview & System Architecture

VARSHA (**V**ariational **A**tmospheric **R**ecursive **S**imulation for **H**eterogeneous **A**nalysis) is an offline-capable, physics-constrained spatiotemporal deep learning digital twin of the Indian subcontinent's weather dynamics.

The system is designed to fulfill two foundational operational capabilities:
1. **Timeline Prime (Autoregressive Rollout):** Given a 3-day initialization sequence over a $32 \times 36$ grid at $1^\circ$ resolution, VARSHA rolls forward 10 or more days across 6 dynamic physical atmospheric channels without spatial dissipation or blurring into climatological means.
2. **Multiverse Branching (Causal Counterfactual Shocks):** Evaluates localized thermodynamic and kinematic atmospheric perturbations (e.g., Arabian Sea wind stalls, Bay of Bengal sea-surface heating) and computes realistic non-linear spatiotemporal propagation over downstream terrestrial India.

```
Input Sequence [3, 6, 32, 36] (Physical Units)
  │
  ├── Physical Normalization (stats.json) ──> [3, 6, 32, 36]
  │                                                  │
  ├── Static Geography (Land-Sea, Orography) ───────┼──> Concatenation [22, 32, 36]
  │                                                  │
  └── Day-of-Year Harmonics [sin, cos] ─────────────┘
                                                     │
                                                     ▼
                                          ┌──────────────────────┐
                                          │  VARSHA ResNet-2D    │
                                          │   4x Residual Blocks │
                                          │   GroupNorm + GELU   │
                                          └──────────────────────┘
                                                     │
                                                     ▼
                                           Predicted Residual Delta
                                                ΔX_t [6, 32, 36]
                                                     │
                                                     ▼
                                        Autoregressive State Update
                                            X_{t+1} = X_t + ΔX_t
                                                     │
                                                     ▼
                                          Physical Boundary Guard
                                          - tp >= 0.0 mm/day
                                          - -40°C <= t2m <= 60°C
                                                     │
                                                     ▼
                                      Output Rollout [10, 6, 32, 36]
```

### Dynamic Channels ($32 \times 36$ Grid, $1^\circ$ Cells: Lat 6.5–37.5°N, Lon 65.5–100.5°E)
| Channel Index | Variable | Description | Physical Unit |
|---|---|---|---|
| 0 | `t2m` | 2-meter air temperature | Kelvin ($K$) |
| 1 | `tp` | 24-hour total precipitation | $mm / \text{day}$ |
| 2 | `u10` | 10-meter zonal wind speed | $m/s$ |
| 3 | `v10` | 10-meter meridional wind speed | $m/s$ |
| 4 | `msl` | Mean sea-level pressure | Pascals ($Pa$) |
| 5 | `tcwv` | Total column water vapour | $kg/m^2$ |

### Bundled Static Variables
- `land_sea_mask`: Fractional land-ocean boundary.
- `orography_m`: Surface elevation above sea level ($m$) across the Western Ghats and Himalayan barrier.

---

## 2. API Contract & Offline Sandbox Execution

The server operates completely offline without external internet access, adhering to the container sandbox constraints:
- **Compute constraints:** Maximum 4 CPU cores, 6 GB RAM, no GPU acceleration.
- **Latency constraint:** 10-step simulation response completed in under 5.0 seconds.
- **Payload format:** Pure nested JSON lists with physical units on the wire.

### Starting the Server
```bash
python serve.py --port 8000
```

### Endpoints
#### 1. `POST /health`
Verifies model readiness and preloaded memory state.
- **Response:**
  ```json
  {
    "status": "ready",
    "model_name": "VARSHA-v1-ResNet"
  }
  ```

#### 2. `POST /simulate`
Computes forward simulation rollouts with optional shock injection.
- **Request Body:**
  ```json
  {
    "init_states": [[[[...]]]], // Shape: [3, 6, 32, 36] in physical units
    "doy_sincos": [0.42, -0.91], // Day-of-year harmonics for init_states[-1]
    "steps": 10,
    "perturbation": null        // null OR [6, 32, 36] physical delta on init_states[-1]
  }
  ```
- **Response Body:**
  ```json
  {
    "rollout": [[[[...]]]],      // Shape: [10, 6, 32, 36] in physical units
    "inference_ms": 138.4
  }
  ```

---

## 3. Repository Structure

```
multiverse-of-monsoon/
├── API_CONTRACT.md              # Official API specification
├── README.md                    # Project documentation & system specification
├── serve.py                     # Offline API server (<5s execution on 4-core CPU)
├── varsha_weights.pt            # Pretrained VARSHA neural network weights
├── static_norm.npy              # Normalized static orography & land-sea mask
├── stats.json                   # ERA5 normalization channel parameters
├── training/
│   ├── train_varsha.py          # Complete reproducible training pipeline
│   ├── evaluate_varsha.py       # 10-step autoregressive validation engine
│   ├── evaluate_baseline.py     # Persistence & climatology benchmark calculator
│   └── inspect_data.py          # Dataset schema and variable verifier
└── dashboard/
    └── app.py                   # Decoupled visualization dashboard & impact engine
```

---

## 4. Model Training & Validation

### Reproduce Training
The training script fits the residual mapping $\Delta X_t$ using PyTorch CPU with GroupNorm to maintain stability without batch dependency:
```bash
python .\training\train_varsha.py
```

### Benchmark Validation vs Persistence
Evaluate the trained model against persistence across lead days +1, +4, and +8:
```bash
python .\training\evaluate_varsha.py
```

#### Performance Comparison (Root Mean Square Error in Physical Units)
| Lead Time | Variable | Persistence Baseline | VARSHA Model | Relative Error Delta |
|---|---|---|---|---|
| **Day +1** | `tp` (Precipitation) | 5.3727 mm/day | **4.2862 mm/day** | **-20.2%** |
| | `u10` (Zonal Wind) | 1.0513 m/s | **0.8801 m/s** | **-16.3%** |
| | `v10` (Meridional Wind) | 1.0612 m/s | **0.8267 m/s** | **-22.1%** |
| | `msl` (Pressure) | 194.48 Pa | **142.38 Pa** | **-26.8%** |
| | `tcwv` (Water Vapor) | 3.5463 kg/m² | **2.8819 kg/m²** | **-18.7%** |
| **Day +4** | `tp` (Precipitation) | 7.9002 mm/day | **6.6823 mm/day** | **-15.4%** |
| | `u10` (Zonal Wind) | 1.9442 m/s | **1.6608 m/s** | **-14.6%** |
| | `v10` (Meridional Wind) | 1.8404 m/s | **1.4799 m/s** | **-19.6%** |
| | `msl` (Pressure) | 341.40 Pa | **285.74 Pa** | **-16.3%** |
| | `tcwv` (Water Vapor) | 7.1598 kg/m² | **6.5413 kg/m²** | **-8.6%** |

---

## 5. Automated Verification & Sanity Checks

### 1. Schema & Latency Validation
Run the official competition test suite against the active inference server:
```bash
python -m monsoon_r1.schema_validator http://localhost:8000
```
- Validates canonical nested lists and base64 payloads.
- Tests unperturbed, `warm_ocean`, and zero shock perturbations.
- Asserts strict execution latency well below the 5.0-second timeout.

### 2. Static Code Health & Anti-Cheat Scan
Verify repository hygiene and compliance with the evaluation criteria:
```bash
python -c "import monsoon_r1.repo_scan as r; print(r.scan('.'))"
```

---

## 6. Real-World Impact Extension

VARSHA couples forward digital twin rollouts to downstream socio-economic and agricultural risk engines:
1. **Monsoon Stall Agricultural Stress:** Quantifies continuous rain deficits over central rainfed agrarian belts (Vidarbha, Marathwada) when Arabian Sea low-level jet streams weaken.
2. **Extreme Flood Risk & CSI:** Tracks spatial moisture advection along the Western Ghats and Bay of Bengal coastal boundaries for localized heavy-rainfall thresholds ($>10\text{ mm/day}$).
3. **Renewable Energy Forecasting:** Computes boundary layer kinetic energy flux from combined $u_{10}$ and $v_{10}$ wind vector fields to predict offshore wind turbine energy yields.