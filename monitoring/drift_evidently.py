"""
monitoring/drift_evidently.py — EvidentlyAI Drift Reports
===========================================================
Generates full HTML drift reports and JSON metric summaries using
EvidentlyAI, comparing a reference dataset (training distribution)
against recent inference traffic.

Works standalone (CLI) or is called from app.py.

Usage:
    python -m monitoring.drift_evidently               # full HTML report
    python -m monitoring.drift_evidently --json        # JSON metrics only
    python -m monitoring.drift_evidently --compare     # reference vs recent
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
PROJECT_ROOT = _HERE.parent
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports" / "drift"
INFERENCE_LOG = LOG_DIR / "inference.jsonl"

LOG_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Column schema ──────────────────────────────────────────────────────────
# State tuple: (t_in_bin, t_out_bin, occupancy, hour_bin)
FEATURE_COLS = ["t_in_bin", "t_out_bin", "occupancy", "hour_bin", "action"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build reference dataset from clean sensor data / synthetic baseline
# ─────────────────────────────────────────────────────────────────────────────


def build_reference(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Build a reference DataFrame that represents the expected distribution
    of (state, action) tuples during normal operation.

    If data/clean_sensor_data.csv exists we derive bins from it;
    otherwise we generate a synthetic reference.
    """
    clean_path = DATA_DIR / "clean_sensor_data.csv"
    if clean_path.exists():
        log.info(f"Loading reference from {clean_path}")
        df = pd.read_csv(clean_path)
        ref = _derive_bins(df, n=n, seed=seed)
    else:
        log.info("No clean data found — generating synthetic reference")
        ref = _synthetic_reference(n=n, seed=seed)

    return ref


def _derive_bins(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Derive discretised feature bins from the preprocessed sensor CSV."""
    rng = np.random.default_rng(seed)

    # Bin boundaries mirror env.py
    t_in_bins = [20, 22, 24, 26, 28, 30]
    t_out_bins = [20, 25, 30, 35]
    hour_bins = [6, 12, 18]

    sub = df.sample(min(n, len(df)), random_state=seed).copy()

    # Use actual column names from preprocess.py output
    sub["t_in_bin"] = np.digitize(sub["indoor_temp"], t_in_bins).astype(int)
    sub["t_out_bin"] = np.digitize(sub["outdoor_temp"], t_out_bins).astype(int)
    sub["hour_bin"] = np.digitize(sub["hour"], hour_bins).astype(int)
    sub["occupancy"] = sub["occupancy"].astype(int)

    if "action" not in sub.columns:
        sub["action"] = rng.integers(0, 4, size=len(sub))

    return sub[FEATURE_COLS].reset_index(drop=True)


def _synthetic_reference(n: int, seed: int) -> pd.DataFrame:
    """Generate a synthetic reference distribution matching training priors."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "t_in_bin": rng.integers(0, 7, n),
            "t_out_bin": rng.integers(0, 5, n),
            "occupancy": rng.integers(0, 2, n),
            "hour_bin": rng.integers(0, 4, n),
            "action": rng.choice([0, 1, 2, 3], p=[0.30, 0.25, 0.25, 0.20], size=n),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load recent inference traffic from JSONL log
# ─────────────────────────────────────────────────────────────────────────────


def load_recent_inference(n: int = 500) -> Optional[pd.DataFrame]:
    """
    Load the last `n` records from logs/inference.jsonl and return
    a DataFrame with FEATURE_COLS.  Returns None if log is too small.
    """
    if not INFERENCE_LOG.exists():
        log.warning(f"No inference log at {INFERENCE_LOG}")
        return None

    rows = []
    with open(INFERENCE_LOG) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                state = r.get("state") or r.get("request", {}).get(
                    "state", [0, 0, 0, 0]
                )
                if len(state) < 4:
                    continue
                rows.append(
                    {
                        "t_in_bin": int(state[0]),
                        "t_out_bin": int(state[1]),
                        "occupancy": int(state[2]),
                        "hour_bin": int(state[3]),
                        "action": int(
                            r.get("action", r.get("response", {}).get("action", 0))
                        ),
                    }
                )
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    if len(rows) < 50:
        log.warning(f"Only {len(rows)} inference records — need ≥ 50 for drift report")
        return None

    df = pd.DataFrame(rows[-n:])  # last n records
    log.info(f"Loaded {len(df)} inference records for drift analysis")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. EvidentlyAI report generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_dir: Path = REPORTS_DIR,
) -> dict:
    """
    Generate an EvidentlyAI HTML drift report + JSON metrics.

    Returns a summary dict with drift status for each feature.
    """
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report
    except ImportError:
        log.warning("evidently not installed — pip install evidently>=0.4")
        return _fallback_psi_report(reference, current)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    html_path = output_dir / f"drift_report_{timestamp}.html"
    json_path = output_dir / f"drift_metrics_{timestamp}.json"

    col_mapping = ColumnMapping(
        target=None,
        prediction="action",
        numerical_features=["t_in_bin", "t_out_bin", "hour_bin"],
        categorical_features=["occupancy", "action"],
    )

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(
        reference_data=reference, current_data=current, column_mapping=col_mapping
    )

    # Save HTML
    report.save_html(str(html_path))
    log.info(f"HTML drift report saved → {html_path}")

    # Extract JSON metrics
    report_dict = report.as_dict()
    summary = _extract_summary(report_dict, html_path, json_path, timestamp)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"JSON drift metrics saved → {json_path}")

    return summary


def _extract_summary(
    report_dict: dict, html_path: Path, json_path: Path, timestamp: str
) -> dict:
    """Pull the key drift metrics out of the Evidently report dict."""
    summary: dict = {
        "timestamp": timestamp,
        "html_report": str(html_path),
        "json_metrics": str(json_path),
        "dataset_drift": False,
        "features": {},
    }

    try:
        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})

            # DatasetDriftMetric
            if "dataset_drift" in result:
                summary["dataset_drift"] = result["dataset_drift"]
                summary["drift_share"] = result.get("share_of_drifted_columns", 0.0)
                summary["n_drifted"] = result.get("number_of_drifted_columns", 0)

            # ColumnDriftMetric per feature
            col = result.get("column_name")
            if col and "drift_detected" in result:
                summary["features"][col] = {
                    "drift_detected": result["drift_detected"],
                    "stattest": result.get("stattest_name", ""),
                    "p_value": result.get("drift_score"),
                }
    except Exception as exc:
        log.warning(f"Could not extract Evidently summary: {exc}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 4. PSI fallback (no Evidently installed)
# ─────────────────────────────────────────────────────────────────────────────


def _psi(ref_col: pd.Series, cur_col: pd.Series) -> float:
    """Population Stability Index between two discrete series."""
    bins = sorted(set(ref_col.unique()) | set(cur_col.unique()))
    total_r = len(ref_col)
    total_c = len(cur_col)
    psi = 0.0
    for b in bins:
        p_r = (ref_col == b).sum() / total_r + 1e-6
        p_c = (cur_col == b).sum() / total_c + 1e-6
        psi += (p_c - p_r) * np.log(p_c / p_r)
    return round(float(psi), 5)


def _fallback_psi_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Pure-PSI drift report used when Evidently is not installed."""
    PSI_WARN = 0.10
    PSI_ALERT = 0.20

    features = {}
    for col in FEATURE_COLS:
        psi = _psi(reference[col], current[col])
        severity = "ok"
        if psi >= PSI_ALERT:
            severity = "alert"
        elif psi >= PSI_WARN:
            severity = "warn"
        features[col] = {
            "psi": psi,
            "severity": severity,
            "drift_detected": psi >= PSI_WARN,
        }

    dataset_drift = any(v["drift_detected"] for v in features.values())
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "method": "psi_fallback",
        "dataset_drift": dataset_drift,
        "features": features,
        "note": (
            "Install evidently>=0.4 for full HTML reports with "
            "statistical tests and visualisations."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Public API used by app.py
# ─────────────────────────────────────────────────────────────────────────────


def run_drift_analysis(n_recent: int = 500) -> dict:
    """
    Entry point called by the FastAPI endpoint /monitoring/evidently.

    1. Loads reference distribution
    2. Loads recent inference logs
    3. Generates Evidently (or PSI fallback) report
    4. Returns summary dict
    """
    reference = build_reference()
    current = load_recent_inference(n=n_recent)
    if current is None:
        return {
            "status": "insufficient_data",
            "message": "Need ≥ 50 inference records. Run /predict first.",
        }
    return generate_evidently_report(reference, current)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="HVAC RL — EvidentlyAI Drift Reporter")
    parser.add_argument(
        "--json", action="store_true", help="Print JSON summary to stdout"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare reference vs recent inference traffic",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Number of recent inference records to analyse",
    )
    args = parser.parse_args()

    result = run_drift_analysis(n_recent=args.n)

    if args.json or args.compare:
        import pprint

        pprint.pprint(result)
    else:
        drift = result.get("dataset_drift", False)
        print(f"\n{'='*55}")
        print(f"  Drift detected: {'⚠ YES' if drift else '✅ NO'}")
        if "features" in result:
            print(f"\n  Per-feature breakdown:")
            for col, info in result["features"].items():
                flag = "⚠" if info.get("drift_detected") else "✅"
                val = info.get("psi") or info.get("p_value") or "—"
                print(f"    {flag} {col:<15} {val}")
        if "html_report" in result:
            print(f"\n  HTML report → {result['html_report']}")
        print(f"{'='*55}\n")
