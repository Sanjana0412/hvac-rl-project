"""
app.py — FastAPI Inference Server
==================================
Serves trained HVAC RL policies via REST API.

Endpoints:
  GET  /health       — liveness check
  GET  /models       — list available models
  POST /predict      — get recommended HVAC action
  GET  /metrics      — latest evaluation results

Run:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
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
    HVACEnv,
    QLearningAgent,
    SARSAAgent,
)
from monitoring.drift_evidently import run_drift_analysis
from monitoring.monitor import monitor as pred_monitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hvac-api")

app = FastAPI(
    title="HVAC RL API",
    version="1.0.0",
    description="REST API for HVAC Reinforcement Learning policy inference",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

MODELS = {
    "qlearning": "models/best_qlearning.pkl",
    "sarsa": "models/best_sarsa.pkl",
}
INFERENCE_LOG = Path("logs/inference.jsonl")
_cache: dict = {}


def load_model(name):
    if name in _cache:
        return _cache[name]
    path = MODELS.get(name)
    if not path or not os.path.isfile(path):
        raise HTTPException(
            404, f"Model '{name}' not found. Available: {available_models()}"
        )
    with open(path, "rb") as f:
        d = pickle.load(f)
    cls = SARSAAgent if d.get("agent_type") == "sarsa" else QLearningAgent
    agent = cls(d["lr"], d["gamma"], d["epsilon"], d["eps_min"], d["eps_decay"])
    agent.q_table = d["q_table"]
    _cache[name] = agent
    log.info(f"Loaded model: {name}")
    return agent


def available_models():
    return [k for k, v in MODELS.items() if os.path.isfile(v)]


def discretise(indoor_temp, outdoor_temp, occupancy, hour):
    return (
        int(np.digitize(indoor_temp, TEMP_IN_BINS)),
        int(np.digitize(outdoor_temp, TEMP_OUT_BINS)),
        int(occupancy),
        int(np.digitize(hour, HOUR_BINS)),
    )


def log_inference(req, resp):
    # NOTE: detailed request/response audit log kept separately from the
    # monitor's inference.jsonl so formats don't collide.
    audit_log = INFERENCE_LOG.parent / "audit.jsonl"
    audit_log.parent.mkdir(exist_ok=True)
    with open(audit_log, "a") as f:
        f.write(
            json.dumps(
                {"ts": datetime.utcnow().isoformat(), "request": req, "response": resp}
            )
            + "\n"
        )


# ── Schemas ──────────────────────────────────────────────────
class PredictRequest(BaseModel):
    indoor_temp: float = Field(..., ge=10.0, le=45.0, description="Indoor °C")
    outdoor_temp: float = Field(..., ge=-10.0, le=50.0, description="Outdoor °C")
    occupancy: int = Field(..., ge=0, le=1, description="1=occupied, 0=empty")
    hour_of_day: float = Field(..., ge=0.0, le=24.0, description="Hour (0-24)")
    model: str = Field("qlearning", description="qlearning or sarsa")


class PredictResponse(BaseModel):
    action: int
    action_label: str
    state: list[int]
    q_values: Optional[list[float]]
    model: str


# ── Endpoints ────────────────────────────────────────────────
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
        req.indoor_temp, req.outdoor_temp, req.occupancy, req.hour_of_day
    )
    q_vals = agent.q_table[state].tolist()
    action = int(np.argmax(q_vals))
    resp = {
        "action": action,
        "action_label": ACTION_LABELS[action],
        "state": list(state),
        "q_values": q_vals,
        "model": req.model,
    }
    log_inference(req.model_dump(), resp)
    pred_monitor.log(state, action, model=req.model)
    log.info(f"predict | model={req.model} state={state} → action={action}")
    return resp


@app.get("/metrics")
def metrics():
    count = 0
    if INFERENCE_LOG.exists():
        with open(INFERENCE_LOG) as f:
            count = sum(1 for _ in f)
    results = []
    if os.path.isfile("experiments/evaluation.csv"):
        import csv

        with open("experiments/evaluation.csv") as f:
            results = list(csv.DictReader(f))
    return {"total_inferences": count, "latest_evaluation": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


@app.get("/monitoring/drift", tags=["Monitoring"])
def drift_report():
    """PSI-based drift report over the last 500 predictions."""
    return pred_monitor.drift_report()


@app.get("/monitoring/summary", tags=["Monitoring"])
def monitoring_summary():
    """Prediction count, action distribution, and drift status."""
    return pred_monitor.summary()


@app.get("/monitoring/evidently", tags=["Monitoring"])
def evidently_drift_report(n: int = 500):
    """
    Full EvidentlyAI drift report comparing reference distribution
    against the last `n` inference requests.
    Returns per-feature drift detection + HTML report path.
    Falls back to PSI if evidently is not installed.
    """
    return run_drift_analysis(n_recent=n)
