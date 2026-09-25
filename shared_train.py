import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering
from utils import stratified_engine_split, clip_rul
from config import get_config
from shared_features import apply_state2_features
from shared_eval import evaluate_per_bin, print_metrics


def prepare_data(config):
    df_train, df_test, rul = load_fd_data(config.name)

    df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(
        df_train, df_test, dataset_name=config.name, n_conditions=config.n_conditions
    )

    df_train_fe = apply_state2_features(df_train_fe, config.trend_sensors, config.ewma_spans)
    df_test_fe = apply_state2_features(df_test_fe, config.trend_sensors, config.ewma_spans)

    exclude_cols = ["unit_id", "RUL"]
    feature_cols = [c for c in df_train_fe.columns if c not in exclude_cols]

    print(f"\nNumber of features: {len(feature_cols)}")
    print(f"Train samples: {len(df_train_fe)}")
    print(f"Test samples: {len(df_test_fe)}")

    return df_train_fe, df_test_fe, feature_cols, scaler


def train_and_evaluate(X_train, y_train, X_val, y_val, config):
    model = RandomForestRegressor(**config.rf_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)

    return model, rmse, mae, y_pred


def run_validation(config):
    df_train_fe, df_test_fe, feature_cols, scaler = prepare_data(config)

    df_train_final, df_val = stratified_engine_split(
        df_train_fe,
        test_size=0.2,
        n_bins=4,
        random_state=42,
    )

    X_train = df_train_final[feature_cols]
    y_train = clip_rul(df_train_final["RUL"], config.rul_clip)
    X_val = df_val[feature_cols]
    y_val = clip_rul(df_val["RUL"], config.rul_clip)

    model, rmse, mae, y_pred = train_and_evaluate(X_train, y_train, X_val, y_val, config)

    print(f"\nNumber of features used: {len(feature_cols)}")
    print(f"RUL Clip: {config.rul_clip}")
    print(f"Final Model - Validation RMSE: {rmse:.3f}")
    print(f"Final Model - Validation MAE: {mae:.3f}")

    per_bin = evaluate_per_bin(y_val, y_pred, n_bins=5, clip_value=config.rul_clip)

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
        "per_bin_metrics": per_bin,
        "feature_importance": feature_importance,
        "df_train_final": df_train_final,
        "df_val": df_val,
        "df_test_fe": df_test_fe,
        "scaler": scaler,
    }


def run_full_retrain(config):
    df_train_fe, df_test_fe, feature_cols, scaler = prepare_data(config)

    X_train_full = df_train_fe[feature_cols]
    y_train_full = clip_rul(df_train_fe["RUL"], config.rul_clip)

    print("\nTraining final model on FULL training set...")
    model = RandomForestRegressor(**config.rf_params)
    model.fit(X_train_full, y_train_full)

    X_test = df_test_fe[feature_cols]
    y_test = df_test_fe["RUL"]
    y_test_clipped = clip_rul(y_test, config.rul_clip)

    y_test_pred = model.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test_clipped, y_test_pred))
    test_mae = mean_absolute_error(y_test_clipped, y_test_pred)

    print_metrics(test_rmse, test_mae, config.rul_clip, config.rf_params, config.name)

    per_bin = evaluate_per_bin(y_test_clipped, y_test_pred, n_bins=5, clip_value=config.rul_clip)

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Feature Importances:")
    print(feature_importance.head(20).to_string(index=False))

    return {
        "model": model,
        "feature_cols": feature_cols,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "per_bin_metrics": per_bin,
        "feature_importance": feature_importance,
    }