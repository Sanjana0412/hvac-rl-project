"""
monitoring/monitor.py
======================
Sliding-window prediction monitor with PSI-based drift detection.

Usage (standalone):
    python -m monitoring.monitor --report

Usage (in app.py):
    from monitoring.monitor import monitor
    monitor.log(state, action, model="qlearning")
    report = monitor.drift_report()
"""

import argparse
import json
import logging
import math
import os
from collections import Counter, deque
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "..", "logs")
INFERENCE_LOG = os.path.join(LOG_DIR, "inference.jsonl")
DRIFT_LOG = os.path.join(LOG_DIR, "drift_alerts.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

# ── Thresholds ────────────────────────────────────────────────
WINDOW = 500  # sliding window size
PSI_WARN = 0.10  # PSI ≥ 0.10 → moderate shift, warn
PSI_ALERT = 0.20  # PSI ≥ 0.20 → significant shift, alert

# FIX: Reference distributions are no longer hardcoded guesses.
# They are loaded from logs/baseline_dist.json when available (written by
# calibrate_reference() below), and fall back to safe defaults only on a
# fresh install before any training has run.
#
# To regenerate after retraining:
#   python -m monitoring.monitor --calibrate
_BASELINE_PATH = os.path.join(LOG_DIR, "baseline_dist.json")

_DEFAULT_REF_ACTION_DIST = {0: 0.30, 1: 0.25, 2: 0.25, 3: 0.20}
_DEFAULT_REF_T_IN_DIST = {i: 1 / 7 for i in range(7)}


def _load_reference_dists():
    """Load reference distributions from file, or return defaults."""
    if os.path.exists(_BASELINE_PATH):
        try:
            with open(_BASELINE_PATH) as f:
                d = json.load(f)
            # JSON keys are strings; cast back to int
            action_dist = {int(k): v for k, v in d["action_dist"].items()}
            t_in_dist = {int(k): v for k, v in d["t_in_dist"].items()}
            return action_dist, t_in_dist
        except Exception:
            pass
    return dict(_DEFAULT_REF_ACTION_DIST), dict(_DEFAULT_REF_T_IN_DIST)


REF_ACTION_DIST, REF_T_IN_DIST = _load_reference_dists()


def calibrate_reference(inference_log: str = None, n: int = 2000) -> dict:
    """
    Compute reference distributions from the first `n` lines of the
    inference log (representing stable, post-training baseline behaviour)
    and write them to logs/baseline_dist.json.

    Call this once after an initial training run before going live so that
    PSI drift detection compares against what the model *actually* does,
    not a hardcoded guess.

    Usage:
        python -m monitoring.monitor --calibrate
    """
    path = inference_log or INFERENCE_LOG
    if not os.path.exists(path):
        raise FileNotFoundError(f"Inference log not found: {path}")

    actions, t_ins = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                r = json.loads(line)
                actions.append(r["action"])
                t_ins.append(r["state"][0])
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    if len(actions) < 50:
        raise ValueError(f"Need ≥ 50 records to calibrate; found {len(actions)}")

    total = len(actions)
    action_dist = {}
    for a in set(actions):
        action_dist[a] = actions.count(a) / total

    t_total = len(t_ins)
    t_in_dist = {}
    for b in set(t_ins):
        t_in_dist[b] = t_ins.count(b) / t_total

    baseline = {
        "action_dist": action_dist,
        "t_in_dist": t_in_dist,
        "calibrated_from_n": total,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(_BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)

    # Reload module-level refs
    global REF_ACTION_DIST, REF_T_IN_DIST
    REF_ACTION_DIST = {int(k): v for k, v in action_dist.items()}
    REF_T_IN_DIST = {int(k): v for k, v in t_in_dist.items()}

    log.info(f"[Calibration] baseline_dist.json written ({total} records)")
    return baseline


# ── PSI helper ────────────────────────────────────────────────
def _psi(ref: dict, actual: dict, bins: list) -> float:
    """
    Population Stability Index.
    PSI < 0.10  → stable
    PSI 0.10–0.20 → moderate shift (warn)
    PSI > 0.20  → significant shift (alert)
    """
    total_ref = sum(ref.values()) or 1
    total_act = sum(actual.values()) or 1
    psi = 0.0
    for b in bins:
        p_ref = (ref.get(b, 0) / total_ref) + 1e-6
        p_act = (actual.get(b, 0) / total_act) + 1e-6
        psi += (p_act - p_ref) * math.log(p_act / p_ref)
    return round(psi, 5)


# ── Monitor ───────────────────────────────────────────────────
class PredictionMonitor:
    """
    Sliding-window monitor for HVAC RL inference.

    Tracks:
      - every prediction (state, action, model, timestamp)
      - action-distribution drift via PSI
      - indoor-temperature input drift via PSI
    """

    def __init__(self, window: int = WINDOW):
        self.window = window
        self._records = deque(maxlen=window)  # full records
        self._actions = deque(maxlen=window)  # action ints only

    # ── Log a prediction ──────────────────────────────────────
    def log(
        self,
        state: tuple,
        action: int,
        model: str = "unknown",
        extra: Optional[dict] = None,
    ) -> None:
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "model": model,
            "state": list(state),
            "action": action,
            **(extra or {}),
        }
        self._records.append(record)
        self._actions.append(action)

        # Persist to disk (append-only JSONL)
        with open(INFERENCE_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ── Drift report ─────────────────────────────────────────
    def drift_report(self) -> dict:
        n = len(self._actions)
        if n < 50:
            return {"status": "insufficient_data", "n": n, "min_required": 50}

        # Action distribution drift
        action_counts = dict(Counter(self._actions))
        action_psi = _psi(REF_ACTION_DIST, action_counts, [0, 1, 2, 3])

        # Indoor-temp bin distribution drift (state[0])
        t_in_counts = dict(Counter(r["state"][0] for r in self._records))
        t_in_psi = _psi(REF_T_IN_DIST, t_in_counts, list(range(7)))

        severity = "ok"
        if action_psi >= PSI_ALERT or t_in_psi >= PSI_ALERT:
            severity = "alert"
        elif action_psi >= PSI_WARN or t_in_psi >= PSI_WARN:
            severity = "warn"

        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "window_n": n,
            "action_psi": action_psi,
            "t_in_psi": t_in_psi,
            "severity": severity,
            "drift": severity != "ok",
            "action_dist": action_counts,
            "t_in_dist": t_in_counts,
            "thresholds": {"warn": PSI_WARN, "alert": PSI_ALERT},
        }

        if severity == "alert":
            log.warning(
                f"[DRIFT ALERT] action_psi={action_psi:.4f} t_in_psi={t_in_psi:.4f}"
            )
            with open(DRIFT_LOG, "a") as f:
                f.write(json.dumps(report) + "\n")
        elif severity == "warn":
            log.warning(
                f"[DRIFT WARN]  action_psi={action_psi:.4f} t_in_psi={t_in_psi:.4f}"
            )

        return report

    # ── Summary ───────────────────────────────────────────────
    def summary(self) -> dict:
        n = len(self._actions)
        if n == 0:
            return {"status": "empty"}
        action_pct = {k: round(v / n, 3) for k, v in Counter(self._actions).items()}
        return {
            "n_predictions": n,
            "action_dist_pct": action_pct,
            "drift": self.drift_report(),
        }

    # ── Load history from disk ────────────────────────────────
    def load_history(self, path: str = INFERENCE_LOG) -> int:
        """Replay the last `window` lines of the inference log into memory."""
        if not os.path.exists(path):
            return 0
        loaded = 0
        with open(path) as f:
            lines = f.readlines()
        for line in lines[-self.window :]:
            try:
                r = json.loads(line)
                self._records.append(r)
                self._actions.append(r["action"])
                loaded += 1
            except (json.JSONDecodeError, KeyError):
                continue
        return loaded


# ── Singleton ─────────────────────────────────────────────────
monitor = PredictionMonitor()


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HVAC RL — Drift Monitor")
    parser.add_argument(
        "--report", action="store_true", help="Print drift report from log"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print prediction summary"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Compute reference distributions from inference log and save to baseline_dist.json",
    )
    parser.add_argument(
        "--calib-n",
        type=int,
        default=2000,
        help="Number of inference records to use for calibration (default: 2000)",
    )
    args = parser.parse_args()

    if args.calibrate:
        import pprint

        result = calibrate_reference(n=args.calib_n)
        print("Calibration complete — baseline_dist.json written:")
        pprint.pprint(result)
    else:
        m = PredictionMonitor()
        n = m.load_history()
        print(f"Loaded {n} records from {INFERENCE_LOG}\n")

        if args.report or args.summary:
            import pprint

            pprint.pprint(m.summary() if args.summary else m.drift_report())
        else:
            parser.print_help()
