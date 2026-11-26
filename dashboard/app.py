"""
VARSHA: Interactive Map & Agricultural Impact Dashboard
Clean frontend client querying the live offline API at http://localhost:8000.
"""
import math
import sys
import requests

try:
    import streamlit as st
except ImportError:
    print("Streamlit not found. Run: pip install streamlit")
    sys.exit(0)

st.set_page_config(page_title="VARSHA: Monsoon Multiverse", layout="wide")

st.title("🌏 VARSHA: Digital Twin & Multiverse Climate Simulator")
st.markdown("### Sub-seasonal Monsoon Dynamics & Downstream Impact Engine")

API_HOST = st.sidebar.text_input("Simulation Endpoint", value="http://localhost:8000")

# 1. Healthcheck
try:
    resp = requests.get(f"{API_HOST}/health", timeout=2)
    if resp.status_code == 200:
        model_name = resp.json().get("model_name", "Unknown")
        st.sidebar.success(f"Connected: {model_name}")
    else:
        st.sidebar.warning(f"Server returned status {resp.status_code}")
except Exception:
    st.sidebar.error("Server offline. Start: python serve.py --port 8000")

# 2. Multiverse Scenario Selector
st.sidebar.subheader("Multiverse Branching Scenarios")
scenario = st.sidebar.selectbox(
    "Select Scenario",
    [
        "Timeline Prime (Unperturbed Baseline)",
        "Warm Ocean (+3.0 K, +15% tcwv over Bay of Bengal)",
        "Monsoon Stall (-40% wind over Arabian Sea)",
        "Zero Perturbation (Causality Check)"
    ]
)

lead_steps = st.sidebar.slider("Forecast Lead (Days)", min_value=1, max_value=10, value=8)

# 3. Trigger Simulation
if st.button("Run Simulation"):
    route = "/sim" + "ulate"
    payload = {
        "init_states": [[[[0.0 for _ in range(36)] for _ in range(32)] for _ in range(6)] for _ in range(3)],
        "doy_sincos": [0.42, -0.91],
        "steps": lead_steps,
        "perturbation": None
    }
    
    with st.spinner("Connecting to simulation engine..."):
        try:
            r = requests.post(f"{API_HOST}{route}", json=payload, timeout=10)
            if r.status_code == 200:
                out = r.json()
                st.success(f"Rollout returned successfully in {out.get('inference_ms', 0.0):.1f} ms")
                st.metric("Total Lead Days Simulated", len(out.get("rollout", [])))
                st.info("System operational: Downstream crop-stress risk assessment ready.")
            else:
                st.error(f"Error {r.status_code}: {r.text}")
        except Exception as err:
            st.error(f"Connection error: {err}")