"""
app.py — FastAPI Inference Server
==================================
Serves trained HVAC RL policies via REST API.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env import (
    ACTION_LABELS,
    HOUR_BINS,
    TEMP_IN_BINS,
    TEMP_OUT_BINS,
    QLearningAgent,
    SARSAAgent,
)
from monitoring.drift_evidently import run_drift_analysis
from monitoring.monitor import monitor as pred_monitor


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("hvac-api")


# ---------------- App ----------------
app = FastAPI(
    title="HVAC RL API",
    version="1.0.0",
    description="REST API for HVAC Reinforcement Learning inference",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Config ----------------
MODELS = {
    "qlearning": "models/best_qlearning.pkl",
    "sarsa": "models/best_sarsa.pkl",
}

LOG_FILE = Path("logs/inference.jsonl")
AUDIT_LOG = Path("logs/audit.jsonl")

_cache = {}


# ---------------- Utilities ----------------
def available_models():
    return [k for k, v in MODELS.items() if os.path.isfile(v)]


def load_model(name: str):
    if name in _cache:
        return _cache[name]

    path = MODELS.get(name)

    if not path or not os.path.isfile(path):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' not found. Available: {available_models()}",
        )

    with open(path, "rb") as f:
        data = pickle.load(f)

    cls = SARSAAgent if data.get("agent_type") == "sarsa" else QLearningAgent

    agent = cls(
        data["lr"],
        data["gamma"],
        data["epsilon"],
        data["eps_min"],
        data["eps_decay"],
    )

    agent.q_table = data["q_table"]
    _cache[name] = agent

    log.info("Loaded model: %s", name)
    return agent


def safe_digitize(value, bins):
    idx = int(np.digitize(value, bins))
    return max(0, min(idx, len(bins)))


def discretise(indoor_temp, outdoor_temp, occupancy, hour):
    return (
        safe_digitize(indoor_temp, TEMP_IN_BINS),
        safe_digitize(outdoor_temp, TEMP_OUT_BINS),
        int(occupancy),
        safe_digitize(hour, HOUR_BINS),
    )


def log_inference(req, resp):
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

    with open(AUDIT_LOG, "a") as f:
        f.write(
            json.dumps(
                {
                    "ts": datetime.utcnow().isoformat(),
                    "request": req,
                    "response": resp,
                }
            )
            + "\n"
        )


# ---------------- Schemas ----------------
class PredictRequest(BaseModel):
    indoor_temp: float = Field(..., ge=10.0, le=45.0)
    outdoor_temp: float = Field(..., ge=-10.0, le=50.0)
    occupancy: int = Field(..., ge=0, le=1)
    hour_of_day: float = Field(..., ge=0.0, le=24.0)
    model: str = Field("qlearning")


class PredictResponse(BaseModel):
    action: int
    action_label: str
    state: list[int]
    q_values: Optional[list[float]]
    model: str


# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": list(_cache.keys()),
        "models_available": available_models(),
    }


@app.get("/models")
def get_models():
    return [
        {"name": k, "available": os.path.isfile(v), "path": v}
        for k, v in MODELS.items()
    ]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    agent = load_model(req.model)

    state = discretise(
        req.indoor_temp,
        req.outdoor_temp,
        req.occupancy,
        req.hour_of_day,
    )

    q_vals = agent.q_table.get(
        state,
        np.zeros(len(ACTION_LABELS)),
    ).tolist()

    action = int(np.argmax(q_vals))

    resp = {
        "action": action,
        "action_label": ACTION_LABELS[action],
        "state": list(state),
        "q_values": q_vals,
        "model": req.model,
    }

    log_inference(req.model_dump(), resp)

    try:
        pred_monitor.log(state, action, model=req.model)
    except Exception as e:
        log.warning("monitoring failed: %s", e)

    log.info("predict | model=%s state=%s action=%s", req.model, state, action)

    return resp


@app.get("/metrics")
def metrics():
    count = 0

    if LOG_FILE.exists():
        with open(LOG_FILE) as f:
            count = sum(1 for _ in f)

    results = []

    if os.path.isfile("experiments/evaluation.csv"):
        import csv

        with open("experiments/evaluation.csv") as f:
            results = list(csv.DictReader(f))

    return {
        "total_inferences": count,
        "latest_evaluation": results,
    }


# ---------------- Monitoring ----------------
@app.get("/monitoring/drift")
def drift_report():
    return pred_monitor.drift_report()


@app.get("/monitoring/summary")
def monitoring_summary():
    return pred_monitor.summary()


@app.get("/monitoring/evidently")
def evidently_drift_report(n: int = 500):
    return run_drift_analysis(n_recent=n)