import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering, TREND_SENSORS_FD002, compute_trend_features
from utils import stratified_engine_split, clip_rul
from utils.features import compute_rolling_features


RUL_CLIP = 125
EWMA_SPANS = [5, 20]
WINDOWS = [5, 20]


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
    """Drop rolling mean and std features, keep slope features"""
    rolling_cols = [c for c in df.columns if "_roll_mean_" in c or "_roll_std_" in c]
    df = df.drop(columns=rolling_cols)
    print(f"Dropped {len(rolling_cols)} rolling features")
    return df


def evaluate_per_bin(y_true, y_pred, n_bins=5, clip_value=125):
    bin_edges = np.linspace(0, clip_value, n_bins + 1)
    y_true_binned = pd.cut(y_true, bins=bin_edges, labels=False, include_lowest=True)
    
    per_bin_metrics = {}
    print(f"\nPer-bin metrics (RUL bins of {int(clip_value/n_bins)}):")
    print(f"{'Bin':<20} {'RMSE':<10} {'MAE':<10} {'Samples':<10}")
    
    for bin_idx in sorted(y_true_binned.unique()):
        bin_mask = y_true_binned == bin_idx
        y_true_bin = y_true[bin_mask]
        y_pred_bin = y_pred[bin_mask]
        
        if len(y_true_bin) > 0:
            bin_rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin))
            bin_mae = mean_absolute_error(y_true_bin, y_pred_bin)
        else:
            bin_rmse = float('nan')
            bin_mae = float('nan')
        
        lower = int(bin_idx * clip_value / n_bins)
        upper = int((bin_idx + 1) * clip_value / n_bins)
        bin_label = f"[{lower}, {upper})"
        per_bin_metrics[bin_label] = {"rmse": bin_rmse, "mae": bin_mae, "n": len(y_true_bin)}
        print(f"{bin_label:<20} {bin_rmse:<10.3f} {bin_mae:<10.3f} {len(y_true_bin):<10}")
    
    return per_bin_metrics


def train_and_evaluate(X_train, y_train_clipped, X_val, y_val_clipped, feature_cols, label):
    model = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)
    
    print(f"\nTraining {label}...")
    model.fit(X_train, y_train_clipped)
    
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val_clipped, y_pred))
    mae = mean_absolute_error(y_val_clipped, y_pred)
    
    print(f"Number of features: {len(feature_cols)}")
    print(f"{label} - Validation RMSE: {rmse:.3f}")
    print(f"{label} - Validation MAE: {mae:.3f}")
    
    per_bin = evaluate_per_bin(y_val_clipped, y_pred, n_bins=5, clip_value=RUL_CLIP)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 15 Feature Importances ({label}):")
    print(feature_importance.head(15).to_string(index=False))
    
    return {
        "rmse": rmse,
        "mae": mae,
        "per_bin": per_bin,
        "importance": feature_importance,
        "model": model,
    }


def main():
    print("=" * 70)
    print("COMPARISON: Baseline+EWMA vs EWMA-only (replacing rolling)")
    print("=" * 70)
    
    df_train, df_test, rul = load_fd_data("FD002")
    
    # Get base features (no rolling, no slope, no EWMA yet)
    df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(
        df_train, df_test, dataset_name="FD002", n_conditions=6
    )
    
    # ---- STATE 1: Baseline + EWMA ----
    print("\n" + "=" * 70)
    print("STATE 1: Baseline features + EWMA features")
    print("=" * 70)
    
    df_train_s1 = compute_ewma_features(df_train_fe.copy(), TREND_SENSORS_FD002, spans=EWMA_SPANS)
    df_test_s1 = compute_ewma_features(df_test_fe.copy(), TREND_SENSORS_FD002, spans=EWMA_SPANS)
    
    df_train_final_s1, df_val_s1 = stratified_engine_split(df_train_s1, test_size=0.2, n_bins=4, random_state=42)
    
    exclude_cols = ["unit_id", "RUL"]
    feature_cols_s1 = [c for c in df_train_final_s1.columns if c not in exclude_cols]
    
    X_train_s1 = df_train_final_s1[feature_cols_s1]
    y_train_s1 = df_train_final_s1["RUL"]
    y_train_clipped_s1 = clip_rul(y_train_s1, RUL_CLIP)
    
    X_val_s1 = df_val_s1[feature_cols_s1]
    y_val_s1 = df_val_s1["RUL"]
    y_val_clipped_s1 = clip_rul(y_val_s1, RUL_CLIP)
    
    results_s1 = train_and_evaluate(X_train_s1, y_train_clipped_s1, X_val_s1, y_val_clipped_s1, feature_cols_s1, "State 1: Baseline + EWMA")
    
    # ---- STATE 2: EWMA instead of rolling (keep slope) ----
    print("\n" + "=" * 70)
    print("STATE 2: EWMA features INSTEAD OF rolling features (keep slope)")
    print("=" * 70)
    
    df_train_s2 = drop_rolling_features(df_train_fe.copy())
    df_test_s2 = drop_rolling_features(df_test_fe.copy())
    
    df_train_s2 = compute_ewma_features(df_train_s2, TREND_SENSORS_FD002, spans=EWMA_SPANS)
    df_test_s2 = compute_ewma_features(df_test_s2, TREND_SENSORS_FD002, spans=EWMA_SPANS)
    
    df_train_final_s2, df_val_s2 = stratified_engine_split(df_train_s2, test_size=0.2, n_bins=4, random_state=42)
    
    feature_cols_s2 = [c for c in df_train_final_s2.columns if c not in exclude_cols]
    
    X_train_s2 = df_train_final_s2[feature_cols_s2]
    y_train_s2 = df_train_final_s2["RUL"]
    y_train_clipped_s2 = clip_rul(y_train_s2, RUL_CLIP)
    
    X_val_s2 = df_val_s2[feature_cols_s2]
    y_val_s2 = df_val_s2["RUL"]
    y_val_clipped_s2 = clip_rul(y_val_s2, RUL_CLIP)
    
    results_s2 = train_and_evaluate(X_train_s2, y_train_clipped_s2, X_val_s2, y_val_clipped_s2, feature_cols_s2, "State 2: EWMA instead of rolling")
    
    # ---- SUMMARY ----
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'State 1: Baseline+EWMA':<25} {'State 2: EWMA only':<25} {'Diff':<10}")
    print(f"{'RMSE':<25} {results_s1['rmse']:<25.3f} {results_s2['rmse']:<25.3f} {results_s2['rmse'] - results_s1['rmse']:<10.3f}")
    print(f"{'MAE':<25} {results_s1['mae']:<25.3f} {results_s2['mae']:<25.3f} {results_s2['mae'] - results_s1['mae']:<10.3f}")
    print(f"{'N features':<25} {len(feature_cols_s1):<25} {len(feature_cols_s2):<25} {len(feature_cols_s2) - len(feature_cols_s1):<10}")
    
    print("\nPer-bin RMSE comparison:")
    print(f"{'Bin':<20} {'State 1':<15} {'State 2':<15} {'Diff':<10}")
    for bin_label in results_s1['per_bin']:
        rmse1 = results_s1['per_bin'][bin_label]['rmse']
        rmse2 = results_s2['per_bin'][bin_label]['rmse']
        print(f"{bin_label:<20} {rmse1:<15.3f} {rmse2:<15.3f} {rmse2 - rmse1:<10.3f}")


if __name__ == "__main__":
    main()