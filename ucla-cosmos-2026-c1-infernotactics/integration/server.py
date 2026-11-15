"""Local FastAPI bridge between Cesium and the v10 InfernoEnv."""

import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pyproj import Transformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "infernotactics" / "src"
sys.path.insert(0, str(SRC))

from data_pipeline.config import REAL_DEPOTS_PATH  # noqa: E402
from env.inferno_env import (  # noqa: E402
    RESOURCE_TYPES, TRAINING_IGNITION_POINT, VALIDATION_IGNITION_POINTS,
    InfernoEnv,
)
from models.relative_model import RelativeInfernoModel  # noqa: E402
from train.relative_actions import decode_action  # noqa: E402
from train.train_relative import MAX_DISPATCH_SLOTS, _forward  # noqa: E402
from train.heuristic_policy import HeuristicPolicy  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = STATIC_DIR / "data"
CHECKPOINT = os.environ.get(
    "INFERNO_CHECKPOINT",
    str(PROJECT_ROOT / "infernotactics" / "models" / "checkpoints_relative_v10_multi_dispatch_100" / "latest.pt"),
)

app = FastAPI(title="InfernoTactics Simulation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ResetRequest(BaseModel):
    ignition_point: list[int] | None = None
    scenario: str = "single"


class StepRequest(BaseModel):
    actions: list[list] | None = None
    mode: str = "manual"


class Session:
    def __init__(self):
        self.device = torch.device("cpu")
        self.env = InfernoEnv(seed=9100)
        obs = self.env.reset(seed=9100)
        self.model = RelativeInfernoModel(obs["grid"].shape[0], len(obs["scalars"]), len(RESOURCE_TYPES), self.env.n_zones).to(self.device)
        if os.path.exists(CHECKPOINT):
            self.model.load_state_dict(torch.load(CHECKPOINT, map_location=self.device, weights_only=True))
        self.model.eval()
        self.heuristic = HeuristicPolicy(self.env)
        self.transformer = Transformer.from_crs(self.env.meta["crs"], "EPSG:4326", always_xy=True)
        self.obs = obs
        self.total_reward = 0.0
        self.events = []
        self.done = False
        self.contained = False
        self.timeout = False
        self.last_buildings_destroyed = 0
        self.total_buildings_destroyed = 0
        self.lock = threading.Lock()
        self.station_points = {
            station["station_id"]: {"lat": station["lat"], "lon": station["lon"]}
            for station in self.env.stations
        }

    def reset(self, request):
        named = {"anchor": TRAINING_IGNITION_POINT, **VALIDATION_IGNITION_POINTS}
        if request.ignition_point:
            self.obs = self.env.reset(ignition_point=tuple(request.ignition_point), seed=9100)
        elif request.scenario in named:
            self.obs = self.env.reset(ignition_point=named[request.scenario], seed=9100)
        else:
            self.obs = self.env.reset(scenario=request.scenario if request.scenario == "multi" else "single", seed=9100)
        self.total_reward = 0.0
        self.events = []
        self.done = False
        self.contained = False
        self.timeout = False
        self.last_buildings_destroyed = 0
        self.total_buildings_destroyed = 0
        return self.state()

    def autopilot_actions(self, mode="autopilot"):
        if mode == "heuristic":
            grid = self.obs["grid"]
            scalars = np.asarray(list(self.obs["scalars"].values()), dtype=np.float32)
            return self.heuristic.decide_actions(grid, scalars)
        local_available = {r: int(self.obs["scalars"][f"{r}_available"]) for r in RESOURCE_TYPES}
        actions = []
        with torch.no_grad():
            for _ in range(MAX_DISPATCH_SLOTS):
                logits, _, _, zones = _forward(self.model, self.obs, self.env, self.device)
                resource_logits = logits["resource_type"][0].clone()
                mask = torch.tensor([local_available[r] > 0 for r in RESOURCE_TYPES], dtype=torch.bool)
                resource_logits[~mask] = -1e9
                if not bool(mask.any()):
                    break
                ri = int(torch.argmax(resource_logits))
                ti = int(torch.argmax(logits["target"][0, ri]))
                action = decode_action(ri, ti, zones)
                if action is None:
                    break
                actions.append(action)
                local_available[action[0]] -= 1
        return actions

    def step(self, actions):
        with self.lock:
            if self.done:
                return self.state() | {"last_step": None, "done": True}
            self.obs, reward, done, info = self.env.step(actions)
        self.total_reward += reward
        self.done = bool(done)
        self.contained = bool(info["contained"])
        self.timeout = bool(info["timeout"])
        self.last_buildings_destroyed = int(info["buildings_destroyed"])
        self.total_buildings_destroyed += self.last_buildings_destroyed
        event = {"tick": info["tick"], "dispatch": info["dispatch"], "resource_events": info["resource_events"], "buildings_destroyed": info["buildings_destroyed"], "reward": reward}
        self.events.append(event)
        self.events = self.events[-50:]
        return self.state() | {"last_step": event, "done": done}

    def point(self, row, col):
        a, b, c, d, e, f = self.env.meta["transform"]
        x = a * (col + 0.5) + b * (row + 0.5) + c
        y = d * (col + 0.5) + e * (row + 0.5) + f
        lon, lat = self.transformer.transform(x, y)
        return {"lat": lat, "lon": lon}

    def resource_position(self, unit):
        start = self.station_points.get(unit["station_id"], {"lat": 34.0725, "lon": -118.5425})
        target = start
        if unit.get("target_zone") is not None:
            zone = self.env.zones[unit["target_zone"]]
            target = self.point(zone["centroid_row"], zone["centroid_col"])
        if unit["state"] in ("traveling", "preparing") and unit.get("pending_travel_ticks"):
            total = max(1, int(unit["pending_travel_ticks"]))
            remaining = min(total, max(0, int(unit.get("remaining_ticks", total))))
            progress = 1.0 - remaining / total
            return {"lat": start["lat"] + (target["lat"] - start["lat"]) * progress,
                    "lon": start["lon"] + (target["lon"] - start["lon"]) * progress,
                    "height_m": 350 if unit.get("station_id") == "114" else 15}
        return {"lat": target["lat"], "lon": target["lon"], "height_m": 350 if unit.get("station_id") == "114" else 15}

    def state(self):
        fire = self.env.sim.state
        active = np.argwhere(np.isin(fire, (2, 3)))
        fire_cells = [{**self.point(int(r), int(c)), "row": int(r), "col": int(c), "state": int(fire[r, c])} for r, c in active]
        resources = []
        for rtype in RESOURCE_TYPES:
            for index, unit in enumerate(self.env.resources[rtype]):
                resources.append({"id": f"{rtype}_{index}", "resource_type": rtype, "station_id": unit["station_id"], "state": unit["state"], "remaining_ticks": unit["remaining_ticks"], "target_zone": unit["target_zone"], "position": self.resource_position(unit)})
        return {
            "tick": self.env.tick_count, "total_reward": self.total_reward,
            "done": self.done, "contained": self.contained,
            "fire_cells": fire_cells, "state_counts": self.env.sim.state_counts(),
            "buildings_destroyed": self.last_buildings_destroyed,
            "buildings_destroyed_total": self.total_buildings_destroyed,
            "timeout": self.timeout, "resources": resources,
            "weather": {"wind_speed_mph": self.obs["scalars"]["wind_speed_mph"], "wind_direction_deg": self.obs["scalars"]["wind_direction_deg"], "humidity_pct": self.obs["scalars"]["humidity_pct"]},
            "available": {r: self.obs["scalars"][f"{r}_available"] for r in RESOURCE_TYPES},
            "traffic": {"mean_load": self.obs["scalars"]["traffic_mean_load"], "max_load": self.obs["scalars"]["traffic_max_load"]},
            "events": self.events[-10:],
        }


session = Session()


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "Simulation.html")


@app.get("/api/health")
def health():
    return {"ok": True, "checkpoint": CHECKPOINT, "traffic_mode": session.env.traffic_mode}


@app.get("/api/config")
def config():
    return {"simulation_bbox": {"north": 34.150, "south": 34.030, "east": -118.440, "west": -118.605}, "display_bbox": {"north": 34.105, "south": 34.030, "east": -118.485, "west": -118.605}, "grid": session.env.meta, "roster": {r: len(session.env.resources[r]) for r in RESOURCE_TYPES}}


@app.get("/api/static/{name}")
def static_data(name: str):
    allowed = {"buildings": "palisades_buildings.geojson", "roads": "palisades_roads.geojson", "depots": "palisades_depots.json", "config": "display_config.json"}
    if name not in allowed:
        raise HTTPException(404, "Unknown static dataset")
    return FileResponse(DATA_DIR / allowed[name])


@app.post("/api/reset")
def reset(request: ResetRequest = ResetRequest()):
    return session.reset(request)


@app.post("/api/step")
def step(request: StepRequest):
    if request.mode in ("autopilot", "heuristic"):
        actions = session.autopilot_actions(request.mode)
    else:
        actions = [tuple(a) for a in (request.actions or [])]
    return session.step(actions)


@app.get("/api/state")
def state():
    return session.state()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
