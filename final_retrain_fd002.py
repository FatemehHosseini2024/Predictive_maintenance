import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering, TREND_SENSORS_FD002
from utils import clip_rul

RUL_CLIP = 125
EWMA_SPANS = [5, 20]

BEST_PARAMS = {
    "n_estimators": 221,
    "max_depth": 27,
    "min_samples_split": 26,
    "min_samples_leaf": 10,
    "max_features": None,
    "random_state": 42,
    "n_jobs": -1,
}


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
    print(f"\nPer-bin metrics (RUL bins of {int(clip_value/n_bins)}):")
    print(f"{'Bin':<20} {'RMSE':<10} {'MAE':<10} {'Samples':<10}")
    print("-" * 50)

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
        print(f"{bin_label:<20} {bin_rmse:<10.3f} {bin_mae:<10.3f} {len(y_true_bin):<10}")

    return per_bin_metrics


def main():
    print("=" * 60)
    print("Final Retrain for FD002 - Full Train Set")
    print("=" * 60)

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

    exclude_cols = ["unit_id", "RUL"]
    feature_cols = [c for c in df_train_fe.columns if c not in exclude_cols]

    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"Train samples: {len(df_train_fe)}")
    print(f"Test samples: {len(df_test_fe)}")

    X_train_full = df_train_fe[feature_cols]
    y_train_full = clip_rul(df_train_fe["RUL"], RUL_CLIP)

    print("\nTraining final model on FULL training set...")
    final_model = RandomForestRegressor(**BEST_PARAMS)
    final_model.fit(X_train_full, y_train_full)

    X_test = df_test_fe[feature_cols]
    y_test = df_test_fe["RUL"]
    y_test_clipped = clip_rul(y_test, RUL_CLIP)

    y_test_pred = final_model.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test_clipped, y_test_pred))
    test_mae = mean_absolute_error(y_test_clipped, y_test_pred)

    print(f"\n{'='*60}")
    print("FINAL TEST SET PERFORMANCE")
    print(f"{'='*60}")
    print(f"Test RMSE: {test_rmse:.3f}")
    print(f"Test MAE:  {test_mae:.3f}")
    print(f"RUL Clip:  {RUL_CLIP}")
    print(f"Params:    {BEST_PARAMS}")

    per_bin = evaluate_per_bin(y_test_clipped, y_test_pred, n_bins=5, clip_value=RUL_CLIP)

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Feature Importances:")
    print(feature_importance.head(20).to_string(index=False))

    return {
        "model": final_model,
        "feature_cols": feature_cols,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "per_bin_metrics": per_bin,
        "feature_importance": feature_importance,
    }


if __name__ == "__main__":
    results = main()