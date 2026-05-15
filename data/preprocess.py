"""
data/preprocess.py — Synthetic Sensor Data Cleaning & Feature Engineering
==========================================================================
Generates a synthetic HVAC sensor dataset, applies cleaning steps,
engineers occupancy-aware features, and saves a normalised CSV ready
for exploratory analysis or supervised-learning baselines.

Usage:
    python data/preprocess.py                  # generate + clean + save
    python data/preprocess.py --rows 5000      # custom row count
    python data/preprocess.py --input raw.csv  # clean an existing CSV
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Output paths ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__))
RAW_PATH = os.path.join(DATA_DIR, "raw_sensor_data.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "clean_sensor_data.csv")

COMFORT_LOW = 20.0
COMFORT_HIGH = 26.0
TARGET_TEMP = 23.0
CARBON_INTENSITY = 0.82  # kg CO₂ / kWh (grid average)


# ─────────────────────────────────────────────────────────────
# 1. Synthetic data generation
# ─────────────────────────────────────────────────────────────


def generate_raw_data(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate 15-min interval HVAC sensor readings with realistic noise,
    missing values, and outliers so the cleaning pipeline has something
    to do.
    """
    rng = np.random.default_rng(seed)
    rows = []

    indoor_temp = float(rng.uniform(22.0, 26.0))
    for step in range(n_rows):
        hour = (step % 96) / 4  # 96 steps per day, 15-min each
        day = step // 96

        # Outdoor temperature: sinusoidal daily cycle + inter-day variation
        outdoor_temp = (
            30.0
            + 6.5 * np.sin(2 * np.pi * (hour - 6) / 24)
            + rng.normal(0, 0.8)
            + rng.normal(0, 0.3) * (day % 7)  # weekly drift
        )

        # Occupancy: business hours with lunch dip
        if 9 <= hour <= 18:
            occupancy = int(rng.random() > 0.3) if 13 <= hour <= 14 else 1
        elif 19 <= hour <= 21:
            occupancy = int(rng.random() > 0.75)
        else:
            occupancy = 0

        # Action chosen (simulate a mixed policy)
        action = int(rng.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.30, 0.20]))
        power_map = {0: 0.0, 1: 0.5, 2: 1.1, 3: 1.9}
        cool_map = {0: 0.0, 1: 0.6, 2: 1.4, 3: 2.3}
        power = power_map[action]
        cool = cool_map[action]

        # Indoor temperature dynamics
        delta = (
            0.12 * (outdoor_temp - indoor_temp)
            + (0.35 if 8 <= hour <= 17 else 0.0)
            + 0.6 * occupancy
            - cool
            + rng.normal(0, 0.08)
        )
        indoor_temp = float(np.clip(indoor_temp + delta, 10.0, 45.0))

        energy = power * (15 / 60)  # kWh per 15-min slot
        carbon = energy * CARBON_INTENSITY

        rows.append(
            {
                "step": step,
                "day": day,
                "hour": round(hour, 2),
                "indoor_temp": round(indoor_temp, 2),
                "outdoor_temp": round(outdoor_temp, 2),
                "occupancy": occupancy,
                "action": action,
                "energy_kwh": round(energy, 4),
                "carbon_kg": round(carbon, 5),
            }
        )

    df = pd.DataFrame(rows)

    # ── Inject realistic dirt ────────────────────────────────
    mask_missing = rng.random(len(df)) < 0.04  # 4 % missing
    df.loc[mask_missing, "indoor_temp"] = np.nan
    df.loc[rng.random(len(df)) < 0.03, "outdoor_temp"] = np.nan
    df.loc[rng.random(len(df)) < 0.02, "occupancy"] = np.nan

    # Hard outliers (sensor glitch)
    outlier_idx = rng.choice(len(df), size=15, replace=False)
    df.loc[outlier_idx[:8], "indoor_temp"] = rng.uniform(50, 80, 8)
    df.loc[outlier_idx[8:], "outdoor_temp"] = rng.uniform(-30, -10, 7)

    return df


# ─────────────────────────────────────────────────────────────
# 2. Cleaning
# ─────────────────────────────────────────────────────────────


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning steps and return a clean copy.
    Steps:
      1. Drop exact duplicate rows
      2. Clamp temperature outliers to physically plausible range
      3. Impute missing values (median for temps, mode for occupancy)
      4. Enforce integer types for categorical columns
    """
    df = df.copy()
    n_before = len(df)

    # Step 1 — duplicates
    df.drop_duplicates(inplace=True)
    print(f"  [clean] dropped {n_before - len(df)} duplicate rows")

    # Step 2 — outlier clamping
    df["indoor_temp"] = df["indoor_temp"].clip(10.0, 45.0)
    df["outdoor_temp"] = df["outdoor_temp"].clip(-10.0, 50.0)

    # Step 3 — imputation
    for col in ["indoor_temp", "outdoor_temp"]:
        n_null = df[col].isna().sum()
        if n_null:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  [clean] imputed {n_null} missing values in '{col}' with median")

    for col in ["occupancy"]:
        n_null = df[col].isna().sum()
        if n_null:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  [clean] imputed {n_null} missing values in '{col}' with mode")

    # Step 4 — type enforcement
    df["occupancy"] = df["occupancy"].astype(int)
    df["action"] = df["action"].astype(int)

    return df


# ─────────────────────────────────────────────────────────────
# 3. Feature engineering
# ─────────────────────────────────────────────────────────────


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive occupancy-aware and temporal features used downstream.

    New columns:
      temp_deviation      — |indoor_temp - TARGET_TEMP|
      comfort_violation   — 1 if outside comfort band while occupied
      temp_delta          — indoor – outdoor (heat-load proxy)
      hour_sin / hour_cos — cyclical encoding of hour-of-day
      is_business_hours   — 1 during 09:00–18:00
      cumulative_energy   — running sum of energy per day
      comfort_cost        — occupancy-weighted temperature deviation
    """
    df = df.copy()

    df["temp_deviation"] = (df["indoor_temp"] - TARGET_TEMP).abs()
    df["comfort_violation"] = (
        (df["indoor_temp"] < COMFORT_LOW) | (df["indoor_temp"] > COMFORT_HIGH)
    ).astype(int) * df["occupancy"]

    df["temp_delta"] = df["indoor_temp"] - df["outdoor_temp"]
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 18)).astype(int)
    df["cumulative_energy"] = df.groupby("day")["energy_kwh"].cumsum()
    df["comfort_cost"] = df["temp_deviation"] * df["occupancy"]

    return df


# ─────────────────────────────────────────────────────────────
# 4. Normalisation
# ─────────────────────────────────────────────────────────────

SCALE_COLS = [
    "indoor_temp",
    "outdoor_temp",
    "temp_deviation",
    "temp_delta",
    "energy_kwh",
    "carbon_kg",
    "comfort_cost",
    "cumulative_energy",
]


def normalise(df: pd.DataFrame, scaler: MinMaxScaler = None):
    """
    Min-max scale numeric columns.
    Returns (df_scaled, fitted_scaler).
    """
    df = df.copy()
    cols = [c for c in SCALE_COLS if c in df.columns]
    if scaler is None:
        scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])
    return df, scaler


# ─────────────────────────────────────────────────────────────
# 5. Full pipeline
# ─────────────────────────────────────────────────────────────


def run_pipeline(input_path: str = None, n_rows: int = 2000, seed: int = 42):
    print("=" * 55)
    print("  HVAC Data Preprocessing Pipeline")
    print("=" * 55)

    # Generate or load raw data
    if input_path and os.path.isfile(input_path):
        print(f"  [load] reading {input_path}")
        df_raw = pd.read_csv(input_path)
    else:
        print(f"  [generate] {n_rows} synthetic sensor readings (seed={seed})")
        df_raw = generate_raw_data(n_rows=n_rows, seed=seed)
        os.makedirs(DATA_DIR, exist_ok=True)
        df_raw.to_csv(RAW_PATH, index=False)
        print(f"  [saved] raw data → {RAW_PATH}")

    print(f"\n  Raw shape : {df_raw.shape}")
    print(f"  Missing   : {df_raw.isna().sum().sum()} total NaNs")

    # Clean
    df_clean = clean(df_raw)

    # Feature engineering
    df_feat = engineer_features(df_clean)

    # Normalise
    df_norm, scaler = normalise(df_feat)

    # Save
    df_norm.to_csv(CLEAN_PATH, index=False)
    print(f"\n  [saved] clean+normalised → {CLEAN_PATH}")
    print(f"  Final shape : {df_norm.shape}")
    print(f"  Features    : {list(df_norm.columns)}\n")

    # Quick stats
    print("  ── Summary statistics (selected columns) ──")
    print(
        df_feat[
            [
                "indoor_temp",
                "outdoor_temp",
                "energy_kwh",
                "comfort_violation",
                "temp_deviation",
            ]
        ]
        .describe()
        .round(3)
        .to_string()
    )
    print()
    return df_norm, scaler


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HVAC sensor data preprocessing")
    parser.add_argument("--input", default=None, help="Path to an existing raw CSV")
    parser.add_argument("--rows", type=int, default=2000, help="Rows to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_pipeline(input_path=args.input, n_rows=args.rows, seed=args.seed)
