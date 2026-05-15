"""
simulate_hvac_traffic.py — Realistic HVAC Traffic Simulator
=============================================================
Generates realistic inference requests to the HVAC RL API, mimicking
how a real building would send sensor readings throughout the day.

Key design decisions (see comments marked DESIGN):
  - Temperature ranges match the training distribution from env.py so
    drift monitoring reflects *real* distribution shift, not simulator noise.
  - Three named scenarios cover normal, heat-wave, and fault conditions.
  - A --drift flag intentionally injects out-of-distribution data so you
    can verify drift alerts actually fire.

Usage:
    # 200 normal requests against a running API
    python simulate_hvac_traffic.py --n 200

    # Heat-wave scenario
    python simulate_hvac_traffic.py --n 200 --scenario heatwave

    # Intentional drift (for testing monitoring)
    python simulate_hvac_traffic.py --n 200 --drift

    # Dry-run (no HTTP — just print sample rows)
    python simulate_hvac_traffic.py --n 10 --dry-run

    # Save generated requests to CSV
    python simulate_hvac_traffic.py --n 500 --save traffic_log.csv
"""

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from typing import List, Optional

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


# ── API config ────────────────────────────────────────────────
DEFAULT_URL   = "http://localhost:8000/predict"
DEFAULT_MODEL = "qlearning"

# ── Scenario definitions ──────────────────────────────────────
#
# DESIGN: Temperature ranges here deliberately mirror the training
# simulator in env.py:
#   _outdoor():  base 30°C ± amplitude 10°C, clipped to [15, 45]
#                → realistic outdoor range: ~20–40°C normal, up to 45°C heatwave
#   reset():     indoor starts in [22, 30]
#   step():      indoor drifts with physics, clipped to [10, 45]
#
# Keeping simulator ranges consistent with training means PSI drift
# detection reports REAL distribution shift, not artificial noise
# introduced by an overly wide random generator.

SCENARIOS = {
    "normal": {
        "description": "Typical office building — temperate climate",
        # Outdoor: follows same sinusoidal pattern as env._outdoor()
        "outdoor_base":   30.0,
        "outdoor_amp":    10.0,   # peak swing ±10°C → 20–40°C range
        "outdoor_noise":   1.5,
        # Indoor: comfortable range with realistic drift
        "indoor_base":    24.0,
        "indoor_noise":    1.5,
        "indoor_min":     20.0,
        "indoor_max":     30.0,
    },
    "heatwave": {
        "description": "Summer heat-wave — elevated outdoor temps",
        "outdoor_base":   38.0,
        "outdoor_amp":     6.0,   # still sinusoidal, but shifted up → 32–44°C
        "outdoor_noise":   1.0,
        "indoor_base":    27.0,
        "indoor_noise":    2.0,
        "indoor_min":     24.0,
        "indoor_max":     36.0,
    },
    "fault": {
        "description": "Sensor fault — erratic indoor readings",
        "outdoor_base":   30.0,
        "outdoor_amp":    10.0,
        "outdoor_noise":   1.5,
        "indoor_base":    24.0,
        "indoor_noise":    6.0,   # high noise simulates faulty sensor
        "indoor_min":     10.0,
        "indoor_max":     42.0,
    },
    # DESIGN: The 'drift' scenario (enabled by --drift flag) uses ranges
    # that are intentionally *outside* the training distribution to verify
    # that PSI monitoring correctly fires an alert.
    "_drift": {
        "description": "Out-of-distribution data (for testing drift detection)",
        "outdoor_base":   10.0,   # ← well outside training range
        "outdoor_amp":     3.0,
        "outdoor_noise":   0.5,
        "indoor_base":    10.0,   # ← well outside training range
        "indoor_noise":    1.0,
        "indoor_min":      8.0,
        "indoor_max":     15.0,
    },
}


# ── Temperature generators ────────────────────────────────────

def _sinusoidal_outdoor(hour: float, sc: dict, rng: random.Random) -> float:
    """
    Mirror env.py's _outdoor() shape: base + amplitude * sin(2π(h-6)/24) + noise
    Clipped to [15, 45] same as training simulator.
    """
    val = (sc["outdoor_base"]
           + sc["outdoor_amp"] * math.sin(2 * math.pi * (hour - 6) / 24)
           + rng.gauss(0, sc["outdoor_noise"]))
    return round(max(15.0, min(45.0, val)), 2)


def _indoor_temp(hour: float, sc: dict, rng: random.Random) -> float:
    """
    Indoor temperature follows a mild daily curve plus noise.
    Clipped to scenario bounds so we stay in the intended regime.
    """
    # Small sinusoidal drift: warmer mid-afternoon from solar/occupancy gain
    drift = 1.5 * math.sin(2 * math.pi * (hour - 4) / 24)
    val   = sc["indoor_base"] + drift + rng.gauss(0, sc["indoor_noise"])
    return round(max(sc["indoor_min"], min(sc["indoor_max"], val)), 2)


def _occupancy(hour: float, rng: random.Random) -> int:
    """
    Mirror env.py's _occupancy() logic so the distribution matches training.
      09:00–18:00 → occupied (with lunch-hour dip 13–14)
      19:00–21:00 → occasionally occupied
      otherwise   → empty
    """
    h = int(hour) % 24
    if 9 <= h <= 18:
        if 13 <= h <= 14:
            return int(rng.random() > 0.3)
        return 1
    if 19 <= h <= 21:
        return int(rng.random() > 0.75)
    return 0


# ── Request builder ───────────────────────────────────────────

def build_request(
    hour: float,
    sc: dict,
    rng: random.Random,
    model: str = DEFAULT_MODEL,
) -> dict:
    return {
        "indoor_temp":  _indoor_temp(hour, sc, rng),
        "outdoor_temp": _sinusoidal_outdoor(hour, sc, rng),
        "occupancy":    _occupancy(hour, rng),
        "hour_of_day":  round(hour % 24, 2),
        "model":        model,
    }


# ── HTTP sender ───────────────────────────────────────────────

def send_request(payload: dict, url: str) -> Optional[dict]:
    if not _HAS_REQUESTS:
        print("[WARNING] requests not installed — run: pip install requests")
        return None
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return None


# ── Main simulation loop ──────────────────────────────────────

def simulate(
    n: int,
    scenario: str = "normal",
    drift: bool = False,
    url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
    delay: float = 0.05,
    dry_run: bool = False,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Run the simulation and return a list of result dicts.

    Parameters
    ----------
    n         : number of requests to send
    scenario  : one of 'normal', 'heatwave', 'fault'
    drift     : if True, use out-of-distribution ranges (for monitoring tests)
    url       : API endpoint
    model     : which RL model to query
    delay     : seconds between requests (0 = as fast as possible)
    dry_run   : if True, build payloads but do not send HTTP requests
    save_path : CSV path to save results (None = don't save)
    seed      : random seed for reproducibility
    """
    sc_key = "_drift" if drift else scenario
    if sc_key not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{sc_key}'. Choose: {list(SCENARIOS.keys())}")

    sc  = SCENARIOS[sc_key]
    rng = random.Random(seed)

    print(f"\n{'='*60}")
    print(f"  HVAC Traffic Simulator")
    print(f"  Scenario : {sc_key} — {sc['description']}")
    print(f"  Requests : {n}")
    print(f"  Target   : {url if not dry_run else 'DRY RUN (no HTTP)'}")
    print(f"{'='*60}\n")

    # DESIGN: Time advances 15 min per request (matching env.py's dt_min=15),
    # starting at a random hour so we cover the full occupancy cycle across runs.
    start_hour = rng.uniform(0, 24)
    results    = []

    for i in range(n):
        hour    = (start_hour + i * 0.25) % 24   # 15-min increments
        payload = build_request(hour, sc, rng, model)

        if dry_run:
            response = {"dry_run": True, "would_send": payload}
        else:
            response = send_request(payload, url) or {}

        record = {
            "i":           i + 1,
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            "hour":        round(hour, 2),
            "scenario":    sc_key,
            **payload,
            "action":      response.get("action", ""),
            "action_label":response.get("action_label", ""),
        }
        results.append(record)

        if dry_run or (i < 5) or ((i + 1) % max(1, n // 10) == 0):
            tag = "DRY" if dry_run else "→"
            print(f"  [{i+1:>4}/{n}] {tag} "
                  f"t_in={payload['indoor_temp']:5.1f}°C  "
                  f"t_out={payload['outdoor_temp']:5.1f}°C  "
                  f"occ={payload['occupancy']}  "
                  f"h={hour:5.2f}  "
                  f"action={record['action_label'] or '?'}")

        if delay > 0 and not dry_run:
            time.sleep(delay)

    # ── Save to CSV ───────────────────────────────────────────
    if save_path and results:
        with open(save_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)
        print(f"\n  [Saved] {len(results)} records → {save_path}")

    # ── Summary ───────────────────────────────────────────────
    actions = [r["action"] for r in results if r["action"] != ""]
    if actions:
        from collections import Counter
        dist = Counter(actions)
        total = len(actions)
        print(f"\n  Action distribution ({total} predictions):")
        for act, cnt in sorted(dist.items()):
            bar = "█" * int(cnt / total * 40)
            print(f"    action {act} : {cnt:>4} ({cnt/total*100:5.1f}%)  {bar}")

    print(f"\n  Done. {len(results)} requests {'built' if dry_run else 'sent'}.\n")
    return results


# ── CLI ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HVAC RL — Traffic Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulate_hvac_traffic.py --n 200
  python simulate_hvac_traffic.py --n 200 --scenario heatwave
  python simulate_hvac_traffic.py --n 500 --drift --save drift_traffic.csv
  python simulate_hvac_traffic.py --n 10 --dry-run
        """,
    )
    parser.add_argument("--n",        type=int,   default=200,          help="Number of requests (default: 200)")
    parser.add_argument("--scenario", choices=["normal", "heatwave", "fault"], default="normal")
    parser.add_argument("--drift",    action="store_true",              help="Use out-of-distribution data (tests drift alerts)")
    parser.add_argument("--url",      default=DEFAULT_URL,              help=f"API endpoint (default: {DEFAULT_URL})")
    parser.add_argument("--model",    default=DEFAULT_MODEL,            help="RL model: qlearning or sarsa (default: qlearning)")
    parser.add_argument("--delay",    type=float, default=0.05,         help="Seconds between requests (default: 0.05)")
    parser.add_argument("--dry-run",  action="store_true",              help="Build payloads but do not send HTTP")
    parser.add_argument("--save",     default=None,                     help="Save results to CSV path")
    parser.add_argument("--seed",     type=int,   default=42,           help="Random seed (default: 42)")
    args = parser.parse_args()

    simulate(
        n=args.n,
        scenario=args.scenario,
        drift=args.drift,
        url=args.url,
        model=args.model,
        delay=args.delay,
        dry_run=args.dry_run,
        save_path=args.save,
        seed=args.seed,
    )
