import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from utils import (
    load_data,
    validate_data,
    identify_constant_sensors,
    drop_columns,
    compute_rul,
    normalize_global,
    normalize_by_condition,
)
from feature_engineering import add_feature_engineering
from sklearn.cluster import KMeans
import numpy as np

TREND_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11",
    "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

WINDOWS = [5, 20]
VARIANCE_THRESHOLD = 0.005
SENSOR_EXACT_ZERO_THRESHOLD = 1e-10


def load_fd_data(dataset_name="FD001", n_conditions=None):
    paths = {
        "FD001": ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", 100),
        "FD002": ("train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt", 260),
    }

    if dataset_name not in paths:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(paths.keys())}")

    train_p, test_p, rul_p, expected_eng = paths[dataset_name]

    df_train, df_test, rul = load_data(train_p, test_p, rul_p)
    validate_data(df_train, train_p, expected_engines=expected_eng)
    validate_data(df_test, test_p, expected_engines=None)

    const_cols = identify_constant_sensors(
        df_train,
        variance_threshold=VARIANCE_THRESHOLD,
        exact_zero_threshold=SENSOR_EXACT_ZERO_THRESHOLD,
    )

    if n_conditions is not None:
        print(f"\nCreating {n_conditions} conditions via KMeans on settings:")
        kmeans = KMeans(n_clusters=n_conditions, random_state=42, n_init=10)
        df_train["condition_id"] = kmeans.fit_predict(df_train[["setting_1", "setting_2", "setting_3"]])
        df_test["condition_id"] = kmeans.predict(df_test[["setting_1", "setting_2", "setting_3"]])

    print(f"\nConstant columns to drop ({dataset_name}): {const_cols}")
    df_train = drop_columns(df_train, const_cols)
    df_test = drop_columns(df_test, const_cols)

    df_train = compute_rul(df_train, is_test=False)
    df_test = compute_rul(df_test, is_test=True, rul_last_cycles=rul["RUL"])

    print(f"\nFeature engineering ({dataset_name}):")
    df_train, df_test = add_feature_engineering(df_train, df_test, TREND_SENSORS, WINDOWS)

    sensor_cols = [c for c in df_train.columns if c.startswith("sensor_") and "_roll_" not in c and "_slope_" not in c]

    if n_conditions is not None:
        df_train, df_test, scaler = normalize_by_condition(
            df_train, df_test, sensor_cols, condition_col="condition_id", n_conditions=n_conditions
        )
    else:
        df_train, df_test, scaler = normalize_global(df_train, df_test, sensor_cols)

    print(f"\nFinal shapes ({dataset_name}):")
    print(f"  train: {df_train.shape}")
    print(f"  test: {df_test.shape}")
    print(f"  sensor cols: {len(sensor_cols)}")

    return df_train, df_test, rul, scaler, sensor_cols


df_train, df_test, rul, scaler, sensor_cols = load_fd_data("FD002",6)
