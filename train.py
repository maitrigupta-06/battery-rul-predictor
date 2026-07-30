"""
train.py – Feature extraction and Random Forest model training for Li-ion Battery RUL prediction.
Dataset: NASA Battery Prognostics dataset (B0005, B0006, B0007, B0018)
"""

import os
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
EOL_CAPACITY  = 1.4          # 30% fade from rated 2 Ahr → End-of-Life threshold
RATED_CAPACITY = 2.0
BATTERY_IDS   = ["B0005", "B0006", "B0007", "B0018"]
FEATURES      = [
    "cycle_num", "mean_voltage", "min_voltage",
    "std_voltage", "max_temp", "mean_temp", "discharge_duration",
]
TARGET = "rul"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────
def load_battery(mat_path: str, batt_id: str) -> list[dict]:
    """Load a .mat file and extract per-cycle features from discharge cycles."""
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    cycles = mat[batt_id].cycle
    discharge_cycles = [c for c in cycles if c.type == "discharge"]
    total = len(discharge_cycles)
    capacities = [float(dc.data.Capacity) for dc in discharge_cycles]

    records = []
    for i, dc in enumerate(discharge_cycles):
        d = dc.data
        cap = capacities[i]

        # RUL = number of remaining cycles still above EOL threshold
        rul = sum(1 for c in capacities[i:] if c >= EOL_CAPACITY)

        v    = np.array(d.Voltage_measured, dtype=float)
        t    = np.array(d.Temperature_measured, dtype=float)
        time = np.array(d.Time, dtype=float)

        records.append({
            "battery"           : batt_id,
            "cycle_num"         : i + 1,
            "cycle_pct"         : (i + 1) / total,
            "mean_voltage"      : np.mean(v),
            "min_voltage"       : np.min(v),
            "std_voltage"       : np.std(v),
            "max_temp"          : np.max(t),
            "mean_temp"         : np.mean(t),
            "discharge_duration": time[-1] - time[0],
            "capacity"          : cap,
            "rul"               : rul,
        })
    return records


def build_dataset(data_dir: str) -> pd.DataFrame:
    """Build the full feature DataFrame from all four batteries."""
    all_records = []
    for bid in BATTERY_IDS:
        path = os.path.join(data_dir, f"{bid}.mat")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find {path}. Place the .mat files in {data_dir}/")
        all_records.extend(load_battery(path, bid))
    return pd.DataFrame(all_records)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def lobo_evaluate(df: pd.DataFrame) -> dict:
    """
    Leave-One-Battery-Out cross-validation.
    Train on 3 batteries, test on the held-out one. Repeat 4 times.
    Returns per-battery and aggregate metrics.
    """
    batteries = df["battery"].unique()
    per_battery, all_preds, all_true = [], [], []

    for test_batt in batteries:
        train_df = df[df["battery"] != test_batt]
        test_df  = df[df["battery"] == test_batt]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[FEATURES].values)
        X_test  = scaler.transform(test_df[FEATURES].values)
        y_train = train_df[TARGET].values
        y_test  = test_df[TARGET].values

        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)

        all_preds.extend(preds.tolist())
        all_true.extend(y_test.tolist())

        per_battery.append({
            "battery": test_batt,
            "MAE"    : mean_absolute_error(y_test, preds),
            "RMSE"   : mean_squared_error(y_test, preds) ** 0.5,
            "R2"     : r2_score(y_test, preds),
        })

    overall = {
        "MAE" : mean_absolute_error(all_true, all_preds),
        "RMSE": mean_squared_error(all_true, all_preds) ** 0.5,
        "R2"  : r2_score(all_true, all_preds),
    }
    return {"per_battery": per_battery, "overall": overall}


# ──────────────────────────────────────────────────────────────────────────────
# Training (full dataset)
# ──────────────────────────────────────────────────────────────────────────────
def train_final_model(df: pd.DataFrame, models_dir: str):
    """Train Random Forest on the full dataset and save model + scaler."""
    os.makedirs(models_dir, exist_ok=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].values)
    y = df[TARGET].values

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    joblib.dump(rf,     os.path.join(models_dir, "rf_rul.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    print(f"✓ Model saved to {models_dir}/rf_rul.pkl")
    print(f"✓ Scaler saved to {models_dir}/scaler.pkl")
    return rf, scaler


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train RUL predictor for Li-ion batteries.")
    parser.add_argument("--data_dir",   default="data",   help="Directory containing .mat files")
    parser.add_argument("--models_dir", default="models", help="Directory to save trained model")
    args = parser.parse_args()

    print("=== Battery RUL Predictor — Training ===\n")
    print(f"Loading data from: {args.data_dir}")
    df = build_dataset(args.data_dir)
    df.to_csv(os.path.join(args.data_dir, "battery_features.csv"), index=False)
    print(f"Dataset: {len(df)} discharge cycles across {df['battery'].nunique()} batteries\n")

    print("Running Leave-One-Battery-Out Cross-Validation …")
    eval_res = lobo_evaluate(df)

    print("\n── Per-Battery Results ──")
    for r in eval_res["per_battery"]:
        print(f"  {r['battery']}: MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  R²={r['R2']:.4f}")

    ov = eval_res["overall"]
    print(f"\n── Overall ──")
    print(f"  MAE={ov['MAE']:.2f} cycles  |  RMSE={ov['RMSE']:.2f} cycles  |  R²={ov['R2']:.4f}\n")

    print("Training final model on all data …")
    train_final_model(df, args.models_dir)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
