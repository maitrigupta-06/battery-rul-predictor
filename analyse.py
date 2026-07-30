"""
analyse.py – Run LOBO cross-validation and generate analysis plots.
Usage: python src/analyse.py --data_dir data --plots_dir plots
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train import build_dataset, FEATURES, TARGET, EOL_CAPACITY

COLORS = ["#2563eb", "#16a34a", "#dc2626", "#d97706"]
PALETTE_BG = "#f8fafc"


def plot_capacity_fade(df: pd.DataFrame, out_dir: str):
    batteries = df["battery"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=PALETTE_BG)
    fig.suptitle("Battery Capacity Fade Over Discharge Cycles",
                 fontsize=14, fontweight="bold", y=0.98)
    for ax, batt, col in zip(axes.flatten(), batteries, COLORS):
        bdf = df[df["battery"] == batt]
        ax.plot(bdf["cycle_num"], bdf["capacity"], color=col, linewidth=1.6)
        ax.axhline(EOL_CAPACITY, color="red", linestyle="--", linewidth=1.2,
                   label=f"EOL = {EOL_CAPACITY} Ahr")
        ax.set_title(batt, fontsize=11, fontweight="bold")
        ax.set_xlabel("Cycle Number", fontsize=9)
        ax.set_ylabel("Capacity (Ahr)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(PALETTE_BG)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_capacity_fade.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_true_vs_pred(eval_res: dict, out_dir: str):
    per_battery = eval_res["per_battery"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=PALETTE_BG)
    fig.suptitle("LOBO Cross-Validation: True vs Predicted RUL (Random Forest)",
                 fontsize=14, fontweight="bold", y=0.98)
    for ax, pbr, col in zip(axes.flatten(), per_battery, COLORS):
        true, pred = pbr["true"], pbr["preds"]
        ax.scatter(true, pred, color=col, alpha=0.55, s=18,
                   label=f"MAE={pbr['MAE']:.1f}, R²={pbr['R2']:.3f}")
        lim = max(max(true), max(pred)) + 5
        ax.plot([0, lim], [0, lim], "k--", linewidth=1.0, label="Perfect")
        ax.set_title(pbr["battery"], fontsize=11, fontweight="bold")
        ax.set_xlabel("True RUL (cycles)", fontsize=9)
        ax.set_ylabel("Predicted RUL (cycles)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(PALETTE_BG)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_true_vs_pred.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_feature_importance(df: pd.DataFrame, out_dir: str):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES].values)
    y = df[TARGET].values
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    fi = dict(zip(FEATURES, rf.feature_importances_))

    feat_labels = {
        "mean_voltage": "Mean Voltage", "min_voltage": "Min Voltage",
        "cycle_num": "Cycle No.", "discharge_duration": "Discharge Dur.",
        "std_voltage": "Voltage StDev", "mean_temp": "Mean Temp",
        "max_temp": "Max Temp",
    }
    pairs = sorted([(feat_labels[k], v) for k, v in fi.items()], key=lambda x: x[1])
    names, vals = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=PALETTE_BG)
    bars = ax.barh(names, vals, color="#2563eb", edgecolor="white", height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#1e293b")
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title("Random Forest Feature Importances (trained on all data)",
                 fontsize=12, fontweight="bold")
    ax.set_facecolor(PALETTE_BG)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    return fi


def plot_rul_timeline(eval_res: dict, out_dir: str):
    per_battery = eval_res["per_battery"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=PALETTE_BG)
    fig.suptitle("Predicted vs Actual RUL Over Discharge Cycles",
                 fontsize=14, fontweight="bold", y=0.98)
    for ax, pbr, col in zip(axes.flatten(), per_battery, COLORS):
        true, pred = pbr["true"], pbr["preds"]
        cycles = list(range(1, len(true) + 1))
        ax.plot(cycles, true, color="gray", linewidth=1.5, label="True RUL", alpha=0.8)
        ax.plot(cycles, pred, color=col, linewidth=1.5, linestyle="--", label="Predicted RUL")
        ax.fill_between(cycles, true, pred, alpha=0.1, color=col)
        ax.set_title(pbr["battery"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Discharge Cycle Index", fontsize=9)
        ax.set_ylabel("RUL (cycles)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(PALETTE_BG)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_rul_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def lobo_with_preds(df: pd.DataFrame) -> dict:
    """LOBO evaluation that also stores per-cycle predictions for plotting."""
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
            "true"   : y_test,
            "preds"  : preds,
        })

    return {
        "per_battery": per_battery,
        "overall": {
            "MAE" : mean_absolute_error(all_true, all_preds),
            "RMSE": mean_squared_error(all_true, all_preds) ** 0.5,
            "R2"  : r2_score(all_true, all_preds),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate RUL analysis plots.")
    parser.add_argument("--data_dir",  default="data",  help="Directory with .mat files")
    parser.add_argument("--plots_dir", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.plots_dir, exist_ok=True)

    print("Loading dataset …")
    df = build_dataset(args.data_dir)
    print(f"  {len(df)} discharge cycles across {df['battery'].nunique()} batteries\n")

    print("Running LOBO evaluation …")
    eval_res = lobo_with_preds(df)

    print("\n── Metrics ──")
    for r in eval_res["per_battery"]:
        print(f"  {r['battery']}: MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  R²={r['R2']:.4f}")
    ov = eval_res["overall"]
    print(f"  Overall:  MAE={ov['MAE']:.2f}  RMSE={ov['RMSE']:.2f}  R²={ov['R2']:.4f}\n")

    print("Generating plots …")
    plot_capacity_fade(df, args.plots_dir)
    plot_true_vs_pred(eval_res, args.plots_dir)
    plot_feature_importance(df, args.plots_dir)
    plot_rul_timeline(eval_res, args.plots_dir)
    print("\n✓ All plots saved to", args.plots_dir)


if __name__ == "__main__":
    main()
