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


# Load data and create features
print("Loading data and creating features...")
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
    df_train_fe, test_size=0.2, n_bins=4, random_state=42
)

exclude_cols = ["unit_id", "RUL"]
feature_cols = [c for c in df_train_final.columns if c not in exclude_cols]

X_train = df_train_final[feature_cols]
y_train = df_train_final["RUL"]
y_train_clipped = clip_rul(y_train, RUL_CLIP)

X_val = df_val[feature_cols]
y_val = df_val["RUL"]
y_val_clipped = clip_rul(y_val, RUL_CLIP)

print(f"Training data: {X_train.shape}, Features: {len(feature_cols)}")
print(f"Validation data: {X_val.shape}")

# RF search on the exact final_model_fd002 validation split
param_combinations = [
    {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 20, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 20, "min_samples_split": 15, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"n_estimators": 150, "max_depth": 20, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 150, "max_depth": 20, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 150, "max_depth": 25, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 200, "max_depth": 20, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 20, "min_samples_split": 10, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"n_estimators": 100, "max_depth": 20, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": "log2"},
]

best_rmse = float("inf")
best_mae = float("inf")
best_rmse_params = None
best_mae_params = None
results = []

for i, params in enumerate(param_combinations):
    print(f"\n=== Testing combination {i+1}/{len(param_combinations)} ===")
    print(f"Params: {params}")
    
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, y_train_clipped)
    
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val_clipped, y_val_pred))
    val_mae = mean_absolute_error(y_val_clipped, y_val_pred)
    results.append({"params": params, "rmse": val_rmse, "mae": val_mae})
    
    print(f"Standard RMSE: {val_rmse:.3f}, MAE: {val_mae:.3f}")
    
    if val_rmse < best_rmse:
        best_rmse = val_rmse
        best_rmse_params = params
    if val_mae < best_mae:
        best_mae = val_mae
        best_mae_params = params

selected_result = min(results, key=lambda item: (item["rmse"], item["mae"]))
best_rmse_result = min(results, key=lambda item: item["rmse"])
best_mae_result = min(results, key=lambda item: item["mae"])
best_params = selected_result["params"]
best_model = RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
best_model.fit(X_train, y_train_clipped)

print("\n" + "=" * 70)
print("SELECTED PARAMETERS (lowest validation RMSE, then MAE):")
print(f"Params: {selected_result['params']}")
print(f"Selected RMSE: {selected_result['rmse']:.3f}")
print(f"Selected MAE: {selected_result['mae']:.3f}")
print("\nBest RMSE candidate:")
print(f"Params: {best_rmse_result['params']}")
print(f"RMSE: {best_rmse_result['rmse']:.3f}, MAE: {best_rmse_result['mae']:.3f}")
print("\nBest MAE candidate:")
print(f"Params: {best_mae_result['params']}")
print(f"RMSE: {best_mae_result['rmse']:.3f}, MAE: {best_mae_result['mae']:.3f}")

print("\nAll results:")
for result in sorted(results, key=lambda item: (item["rmse"], item["mae"])):
    print(f"RMSE: {result['rmse']:.3f}, MAE: {result['mae']:.3f}, Params: {result['params']}")

print("\nStandard metrics for selected model:")
y_val_pred_best = best_model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val_clipped, y_val_pred_best))
val_mae = mean_absolute_error(y_val_clipped, y_val_pred_best)
print(f"Standard RMSE: {val_rmse:.3f}, MAE: {val_mae:.3f}")

print("\nPer-bin metrics for selected model:")
evaluate_per_bin(y_val_clipped, y_val_pred_best, n_bins=5, clip_value=RUL_CLIP)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 20 Feature Importances:")
print(feature_importance.head(20).to_string(index=False))
