# Palisades Cesium Simulation

Run from the repository root:

```powershell
conda run -n cosmos-venv python integration/export_cesium_data.py
conda run -n cosmos-venv python -m uvicorn server:app --app-dir integration --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in a browser.

The page loads a Palisades-focused subset of the building and road data from
`integration/static/data/`, while the server keeps the full calibrated
`InfernoEnv` grid for simulation. Set `INFERNO_CHECKPOINT` to use a different
model checkpoint.

The UI supports:

- RL autopilot or manual dispatch mode
- Start, pause, reset, and single-tick stepping
- Skull Rock, Mandeville, Getty, random, and multi-ignition scenarios
- Building, road, fire, and depot layer toggles
- List-only multi-dispatch actions
- Synthetic traffic and resource lifecycle state in the API response
