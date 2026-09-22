from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering
from utils import stratified_engine_split, clip_rul
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

RUL_CLIP = {"FD001": 125, "FD002": 125}


def get_baseline_results(dataset_name="FD001", n_conditions=None):
    rul_clip = RUL_CLIP[dataset_name]
    df_train, df_test, rul = load_fd_data(dataset_name)
    df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(
        df_train, df_test, dataset_name=dataset_name, n_conditions=n_conditions
    )

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
    y_train_clipped = clip_rul(y_train, rul_clip)

    X_val = df_val[feature_cols]
    y_val = df_val["RUL"]
    y_val_clipped = clip_rul(y_val, rul_clip)

    if dataset_name == "FD002":
        baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_label = "Random Forest"
    else:
        baseline_model = LinearRegression()
        model_label = "Linear Regression"

    baseline_model.fit(X_train, y_train_clipped)

    y_pred = baseline_model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val_clipped, y_pred))
    mae = mean_absolute_error(y_val_clipped, y_pred)

    print(f"Number of features used: {len(feature_cols)}")
    print(f"RUL Clip: {rul_clip}")
    print(f"Baseline ({model_label}) - Validation RMSE: {rmse:.3f}")
    print(f"Baseline ({model_label}) - Validation MAE: {mae:.3f}")

    bin_edges = list(range(0, rul_clip + 25, 25))
    y_val_binned = pd.cut(y_val_clipped, bins=bin_edges, labels=False, include_lowest=True)
    y_pred_series = pd.Series(y_pred, index=y_val_binned.index)

    per_bin_metrics = {}
    print(f"\nPer-bin metrics (RUL bins of 25):")
    print(f"{'Bin':<15} {'RMSE':<10} {'MAE':<10} {'Samples':<10}")
    for bin_idx in sorted(y_val_binned.unique()):
        bin_mask = y_val_binned == bin_idx
        y_true_bin = y_val_clipped[bin_mask]
        y_pred_bin = y_pred_series[bin_mask]
        bin_rmse = np.sqrt(mean_squared_error(y_true_bin, y_pred_bin)) if len(y_true_bin) > 0 else float('nan')
        bin_mae = mean_absolute_error(y_true_bin, y_pred_bin) if len(y_true_bin) > 0 else float('nan')
        bin_label = f"[{int(bin_idx * 25)}, {int(bin_idx * 25 + 25)})"
        per_bin_metrics[bin_label] = {"rmse": bin_rmse, "mae": bin_mae, "n": len(y_true_bin)}
        print(f"{bin_label:<15} {bin_rmse:<10.3f} {bin_mae:<10.3f} {len(y_true_bin):<10}")

    return {
        "df_train_final": df_train_final,
        "df_val": df_val,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "y_train_clipped": y_train_clipped,
        "X_val": X_val,
        "y_val_clipped": y_val_clipped,
        "baseline_model": baseline_model,
        "rmse": rmse,
        "mae": mae,
        "per_bin_metrics": per_bin_metrics,
    }
    
get_baseline_results(dataset_name="FD002", n_conditions=6)
