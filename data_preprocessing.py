import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from utils import (
    load_data,
    validate_data,
    identify_constant_sensors,
    drop_columns,
    compute_rul,
    compute_rolling_features,
    compute_trend_features,
    normalize_global,
    normalize_by_condition,
    clip_rul,
)
import pandas as pd
import numpy as np

TREND_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11",
    "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

RUL_CLIP = 125
WINDOWS = [5, 20]
VARIANCE_THRESHOLD = 0.005
SENSOR_EXACT_ZERO_THRESHOLD = 1e-10


def load_fd_data(dataset_name="FD001"):
    paths = {
        "FD001": ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", 100, TREND_SENSORS),
        "FD002": ("train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt", 260, TREND_SENSORS),
    }

    if dataset_name not in paths:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(paths.keys())}")

    train_p, test_p, rul_p, expected_eng, trend_sensors = paths[dataset_name]

    df_train, df_test, rul = load_data(train_p, test_p, rul_p)
    validate_data(df_train, train_p, expected_engines=expected_eng)
    validate_data(df_test, test_p, expected_engines=None)

    const_cols = identify_constant_sensors(
        df_train,
        variance_threshold=VARIANCE_THRESHOLD,
        exact_zero_threshold=SENSOR_EXACT_ZERO_THRESHOLD,
    )
    print(f"\nConstant columns to drop ({dataset_name}): {const_cols}")
    df_train = drop_columns(df_train, const_cols)
    df_test = drop_columns(df_test, const_cols)

    df_train = compute_rul(df_train, is_test=False)
    df_test = compute_rul(df_test, is_test=True, rul_last_cycles=rul["RUL"])

    df_train = compute_rolling_features(df_train, windows=WINDOWS)
    df_train = compute_trend_features(df_train, trend_sensors, windows=WINDOWS)
    df_test = compute_rolling_features(df_test, windows=WINDOWS)
    df_test = compute_trend_features(df_test, trend_sensors, windows=WINDOWS)

    sensor_cols = [c for c in df_train.columns if c.startswith("sensor_") and "_roll_" not in c and "_slope_" not in c]
    df_train, df_test, scaler = normalize_global(df_train, df_test, sensor_cols)

    print(f"\nFinal shapes ({dataset_name}):")
    print(f"  train: {df_train.shape}")
    print(f"  test: {df_test.shape}")
    print(f"  sensor cols: {len(sensor_cols)}")

    return df_train, df_test, rul, scaler, sensor_cols


# صادرات داده‌های FD001 برای سازگاری با اسکریپت‌های قبلی
df_train, df_test, rul, scaler, sensor_cols = load_fd_data("FD002")
