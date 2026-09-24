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
    print(f"EWMA features added for {len(sensors)} sensors with spans = {spans}")
    print(f"Number of new columns: {len(spans) * len(sensors)}")
    ewma_cols = df.filter(like="_ewma_").columns.tolist()
    print(ewma_cols)
    return df


def drop_rolling_features(df):
    """Drop rolling mean and std features, keep slope features"""
    rolling_cols = [c for c in df.columns if "_roll_mean_" in c or "_roll_std_" in c]
    df = df.drop(columns=rolling_cols)
    print(f"Dropped {len(rolling_cols)} rolling features")
    return df


def compute_second_diff_ewma(df, sensors, spans=[5, 20]):
    """Compute second difference (acceleration) of EWMA features per unit_id"""
    for span in spans:
        for col in sensors:
            ewma_col = f"{col}_ewma_{span}"
            if ewma_col in df.columns:
                diff2_col = f"{col}_ewma_{span}_diff2"
                df[diff2_col] = (
                    df.groupby("unit_id")[ewma_col]
                    .transform(lambda s: s.diff().diff())
                )
    diff2_cols = [c for c in df.columns if "_diff2" in c]
    print(f"Second-diff EWMA features added: {len(diff2_cols)} columns")
    print(diff2_cols)
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


def train_final_model():
    print("=" * 60)
    print("Final Model Training for FD002 - State 2: EWMA replaces Rolling")
    print("=" * 60)
    
    df_train, df_test, rul = load_fd_data("FD002")
    
    df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(
        df_train, df_test, dataset_name="FD002", n_conditions=6
    )
    
    # State 2: Drop rolling features, keep slope, add EWMA
    df_train_fe = drop_rolling_features(df_train_fe)
    df_test_fe = drop_rolling_features(df_test_fe)
    
    df_train_fe = compute_ewma_features(df_train_fe, TREND_SENSORS_FD002, spans=EWMA_SPANS)
    df_test_fe = compute_ewma_features(df_test_fe, TREND_SENSORS_FD002, spans=EWMA_SPANS)
    
    # Add second-difference features from EWMA
    df_train_fe = compute_second_diff_ewma(df_train_fe, TREND_SENSORS_FD002, spans=EWMA_SPANS)
    df_test_fe = compute_second_diff_ewma(df_test_fe, TREND_SENSORS_FD002, spans=EWMA_SPANS)
    
    print(f"\nTotal features after EWMA + diff2 (no rolling): {df_train_fe.shape[1]}")
    
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
    
    model = RandomForestRegressor(
        n_estimators= 221,
        max_depth= 27,
        min_samples_split= 26,
        min_samples_leaf= 10,
        max_features= None,
        
        random_state=42,
        n_jobs=-1,
    )
    
    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train_clipped)
    
    y_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val_clipped, y_pred))
    mae = mean_absolute_error(y_val_clipped, y_pred)
    
    print(f"\nNumber of features used: {len(feature_cols)}")
    print(f"RUL Clip: {RUL_CLIP}")
    print(f"Final Model - Validation RMSE: {rmse:.3f}")
    print(f"Final Model - Validation MAE: {mae:.3f}")
    
    per_bin_metrics = evaluate_per_bin(y_val_clipped, y_pred, n_bins=5, clip_value=RUL_CLIP)
    
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Feature Importances:")
    print(feature_importance.head(20).to_string(index=False))
    
    return {
        "model": model,
        "feature_cols": feature_cols,
        "rmse": rmse,
        "mae": mae,
        "per_bin_metrics": per_bin_metrics,
        "feature_importance": feature_importance,
        "df_train_final": df_train_final,
        "df_val": df_val,
        "df_test_fe": df_test_fe,
        "scaler": scaler,
    }


if __name__ == "__main__":
    results = train_final_model()