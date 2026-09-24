import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering, TREND_SENSORS_FD002
from utils import stratified_engine_split, clip_rul

RUL_CLIP = 125
EWMA_SPANS = [5, 20]

BASELINE_RMSE = 16.996
BASELINE_MAE = 11.494
BASELINE_PER_BIN = {
    "[0, 25)": {"rmse": 7.175, "mae": 5.296, "n": 1352},
    "[25, 50)": {"rmse": 17.110, "mae": 13.163, "n": 1300},
    "[50, 75)": {"rmse": 20.129, "mae": 16.695, "n": 1300},
    "[75, 100)": {"rmse": 18.736, "mae": 15.505, "n": 1300},
    "[100, 125)": {"rmse": 17.442, "mae": 10.436, "n": 5447},
}

PARAM_COMBINATIONS = [
    {"n_estimators": 134, "max_depth": 23, "min_samples_split": 24, "min_samples_leaf": 2, "max_features": None},
    {"n_estimators": 215, "max_depth": 34, "min_samples_split": 28, "min_samples_leaf": 13, "max_features": None},
    {"n_estimators": 398, "max_depth": 14, "min_samples_split": 17, "min_samples_leaf": 1, "max_features": None},
    {"n_estimators": 330, "max_depth": 12, "min_samples_split": 25, "min_samples_leaf": 10, "max_features": None},
    {"n_estimators": 405, "max_depth": 12, "min_samples_split": 23, "min_samples_leaf": 11, "max_features": None},
    {"n_estimators": 221, "max_depth": 27, "min_samples_split": 26, "min_samples_leaf": 10, "max_features": None},
    {"n_estimators": 368, "max_depth": None, "min_samples_split": 6, "min_samples_leaf": 1, "max_features": "log2"},
    {"n_estimators": 180, "max_depth": None, "min_samples_split": 23, "min_samples_leaf": 4, "max_features": None},
]


def compute_ewma_features(df, sensors, spans=[5, 20]):
    for span in spans:
        for col in sensors:
            if col in df.columns:
                df[f"{col}_ewma_{span}"] = (
                    df.groupby("unit_id")[col]
                    .transform(lambda s: s.ewm(span=span, min_periods=1).mean())
                )
    return df


def drop_rolling_features(df):
    rolling_cols = [c for c in df.columns if "_roll_mean_" in c or "_roll_std_" in c]
    df = df.drop(columns=rolling_cols)
    return df


def compute_second_diff_ewma(df, sensors, spans=[5, 20]):
    for span in spans:
        for col in sensors:
            ewma_col = f"{col}_ewma_{span}"
            if ewma_col in df.columns:
                diff2_col = f"{col}_ewma_{span}_diff2"
                df[diff2_col] = (
                    df.groupby("unit_id")[ewma_col]
                    .transform(lambda s: s.diff().diff())
                )
    return df


def evaluate_per_bin(y_true, y_pred, n_bins=5, clip_value=125):
    bin_edges = np.linspace(0, clip_value, n_bins + 1)
    y_true_binned = pd.cut(y_true, bins=bin_edges, labels=False, include_lowest=True)

    per_bin_metrics = {}
    for bin_idx in sorted(y_true_binned.unique()):
        bin_mask = y_true_binned == bin_idx
        y_true_bin = y_true[bin_mask]
        y_pred_bin = y_pred[bin_mask]

        if len(y_true_bin) > 0:
            bin_rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin))
            bin_mae = mean_absolute_error(y_true_bin, y_pred_bin)
        else:
            bin_rmse = float("nan")
            bin_mae = float("nan")

        lower = int(bin_idx * clip_value / n_bins)
        upper = int((bin_idx + 1) * clip_value / n_bins)
        bin_label = f"[{lower}, {upper})"
        per_bin_metrics[bin_label] = {"rmse": bin_rmse, "mae": bin_mae, "n": len(y_true_bin)}

    return per_bin_metrics


def compute_pct_diff(new_val, baseline_val):
    """Compute percentage difference: positive means worse, negative means better"""
    if baseline_val == 0:
        return 0
    return ((new_val - baseline_val) / baseline_val) * 100


def prepare_data():
    print("Loading and preparing data...")
    df_train, df_test, rul = load_fd_data("FD002")

    df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(
        df_train, df_test, dataset_name="FD002", n_conditions=6
    )

    df_train_fe = drop_rolling_features(df_train_fe)
    df_test_fe = drop_rolling_features(df_test_fe)

    df_train_fe = compute_ewma_features(df_train_fe, TREND_SENSORS_FD002, spans=EWMA_SPANS)
    df_test_fe = compute_ewma_features(df_test_fe, TREND_SENSORS_FD002, spans=EWMA_SPANS)

    df_train_fe = compute_second_diff_ewma(df_train_fe, TREND_SENSORS_FD002, spans=EWMA_SPANS)
    df_test_fe = compute_second_diff_ewma(df_test_fe, TREND_SENSORS_FD002, spans=EWMA_SPANS)

    df_train_final, df_val = stratified_engine_split(
        df_train_fe,
        test_size=0.2,
        n_bins=4,
        random_state=42,
    )

    exclude_cols = ["unit_id", "RUL"]
    feature_cols = [c for c in df_train_final.columns if c not in exclude_cols]

    X_train = df_train_final[feature_cols]
    y_train = df_train_final["RUL"]
    y_train_clipped = clip_rul(y_train, RUL_CLIP)

    X_val = df_val[feature_cols]
    y_val = df_val["RUL"]
    y_val_clipped = clip_rul(y_val, RUL_CLIP)

    print(f"Number of features used: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    return X_train, y_train_clipped, X_val, y_val_clipped, feature_cols


def test_combination(params, X_train, y_train, X_val, y_val, combo_idx):
    print(f"\n{'='*60}")
    print(f"Testing combination {combo_idx + 1}/{len(PARAM_COMBINATIONS)}")
    print(f"Params: {params}")
    print(f"{'='*60}")

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    per_bin = evaluate_per_bin(y_val, y_pred, n_bins=5, clip_value=RUL_CLIP)

    rmse_pct = compute_pct_diff(rmse, BASELINE_RMSE)
    mae_pct = compute_pct_diff(mae, BASELINE_MAE)

    print(f"\nOverall Metrics:")
    print(f"  RMSE: {rmse:.3f} ({rmse_pct:+.2f}% vs baseline)")
    print(f"  MAE:  {mae:.3f} ({mae_pct:+.2f}% vs baseline)")

    print(f"\nPer-bin Metrics:")
    print(f"{'Bin':<15} {'RMSE':>10} {'MAE':>10} {'Samples':>10} {'RMSE%':>10} {'MAE%':>10}")
    print("-" * 65)

    bin_results = []
    for bin_label in sorted(per_bin.keys()):
        bin_rmse = per_bin[bin_label]["rmse"]
        bin_mae = per_bin[bin_label]["mae"]
        bin_n = per_bin[bin_label]["n"]

        baseline_rmse = BASELINE_PER_BIN[bin_label]["rmse"]
        baseline_mae = BASELINE_PER_BIN[bin_label]["mae"]

        rmse_pct_bin = compute_pct_diff(bin_rmse, baseline_rmse)
        mae_pct_bin = compute_pct_diff(bin_mae, baseline_mae)

        print(f"{bin_label:<15} {bin_rmse:>10.3f} {bin_mae:>10.3f} {bin_n:>10} {rmse_pct_bin:>+9.2f}% {mae_pct_bin:>+9.2f}%")

        bin_results.append({
            "bin": bin_label,
            "rmse": bin_rmse,
            "mae": bin_mae,
            "n": bin_n,
            "rmse_pct": rmse_pct_bin,
            "mae_pct": mae_pct_bin,
        })

    return {
        "params": params,
        "rmse": rmse,
        "mae": mae,
        "rmse_pct": rmse_pct,
        "mae_pct": mae_pct,
        "per_bin": per_bin,
        "bin_results": bin_results,
    }


def main():
    X_train, y_train, X_val, y_val, feature_cols = prepare_data()

    all_results = []
    for i, params in enumerate(PARAM_COMBINATIONS):
        result = test_combination(params, X_train, y_train, X_val, y_val, i)
        all_results.append(result)

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)

    print(f"\n{'Combo':<6} {'n_est':<7} {'max_d':<7} {'min_sp':<7} {'min_lf':<7} {'max_ft':<8} {'RMSE':<8} {'RMSE%':<10} {'MAE':<8} {'MAE%':<10}")
    print("-" * 85)

    for i, r in enumerate(all_results):
        p = r["params"]
        max_d = str(p["max_depth"]) if p["max_depth"] else "None"
        max_f = str(p["max_features"]) if p["max_features"] else "None"
        print(f"{i+1:<6} {p['n_estimators']:<7} {max_d:<7} {p['min_samples_split']:<7} {p['min_samples_leaf']:<7} {max_f:<8} {r['rmse']:<8.3f} {r['rmse_pct']:>+9.2f}% {r['mae']:<8.3f} {r['mae_pct']:>+9.2f}%")

    print(f"\n{'Baseline':<6} {'-':<7} {'-':<7} {'-':<7} {'-':<7} {'-':<8} {BASELINE_RMSE:<8.3f} {'0.00%':<10} {BASELINE_MAE:<8.3f} {'0.00%':<10}")

    print("\n\nPER-BIN DETAILED COMPARISON")
    print("=" * 80)

    bin_labels = ["[0, 25)", "[25, 50)", "[50, 75)", "[75, 100)", "[100, 125)"]

    for bin_label in bin_labels:
        print(f"\n--- Bin: {bin_label} ---")
        print(f"Baseline RMSE: {BASELINE_PER_BIN[bin_label]['rmse']:.3f}, MAE: {BASELINE_PER_BIN[bin_label]['mae']:.3f}")
        print(f"{'Combo':<6} {'RMSE':>8} {'RMSE%':>10} {'MAE':>8} {'MAE%':>10}")
        print("-" * 45)

        for i, r in enumerate(all_results):
            bin_data = r["per_bin"][bin_label]
            rmse_pct = compute_pct_diff(bin_data["rmse"], BASELINE_PER_BIN[bin_label]["rmse"])
            mae_pct = compute_pct_diff(bin_data["mae"], BASELINE_PER_BIN[bin_label]["mae"])
            print(f"{i+1:<6} {bin_data['rmse']:>8.3f} {rmse_pct:>+9.2f}% {bin_data['mae']:>8.3f} {mae_pct:>+9.2f}%")

    # Find best overall
    best_rmse_idx = min(range(len(all_results)), key=lambda i: all_results[i]["rmse"])
    best_mae_idx = min(range(len(all_results)), key=lambda i: all_results[i]["mae"])

    print(f"\n\nBEST OVERALL RMSE: Combo {best_rmse_idx + 1} (RMSE: {all_results[best_rmse_idx]['rmse']:.3f}, {all_results[best_rmse_idx]['rmse_pct']:+.2f}%)")
    print(f"BEST OVERALL MAE:  Combo {best_mae_idx + 1} (MAE: {all_results[best_mae_idx]['mae']:.3f}, {all_results[best_mae_idx]['mae_pct']:+.2f}%)")

    # Count how many are better than baseline
    better_rmse = sum(1 for r in all_results if r["rmse"] < BASELINE_RMSE)
    better_mae = sum(1 for r in all_results if r["mae"] < BASELINE_MAE)
    print(f"\nCombinations better than baseline: {better_rmse}/{len(all_results)} for RMSE, {better_mae}/{len(all_results)} for MAE")


if __name__ == "__main__":
    main()