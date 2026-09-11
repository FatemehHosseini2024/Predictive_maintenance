import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from utils import (
    load_data,
    validate_data,
    identify_constant_sensors,
    compute_rul,
    drop_columns,
    compute_rolling_features,
    compute_trend_features,
    normalize_global,
    normalize_by_condition,
)
import pandas as pd
import numpy as np

TREND_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11",
    "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]


def load_fd001_data(
    train_path="train_FD001.txt",
    test_path="test_FD001.txt",
    rul_path="RUL_FD001.txt",
    expected_engines_train=100,
    variance_threshold=0.005,
    sensor_exact_zero_threshold=1e-10,
    windows=[5, 20],
    random_state=42,
):
    # بارگذاری و اعتبارسنجی
    df_train, df_test, rul = load_data(train_path, test_path, rul_path)
    validate_data(df_train, "train_FD001", expected_engines=expected_engines_train)
    validate_data(df_test, "test_FD001", expected_engines=None)

    # شناسایی و حذف ستون‌های ثابت
    constant_cols = identify_constant_sensors(
        df_train,
        variance_threshold=variance_threshold,
        exact_zero_threshold=sensor_exact_zero_threshold,
    )
    print(f"\nConstant columns to drop: {constant_cols}")
    df_train = drop_columns(df_train, constant_cols)
    df_test = drop_columns(df_test, constant_cols)

    # محاسبه RUL
    df_train = compute_rul(df_train, is_test=False)
    df_test = compute_rul(df_test, is_test=True, rul_last_cycles=rul["RUL"])

    print(f"\ndf_train shape after RUL: {df_train.shape}")
    print(f"df_test shape after RUL: {df_test.shape}")

    # ویژگی‌های رولینگ و شیب
    df_train = compute_rolling_features(df_train, windows=windows)
    df_train = compute_trend_features(df_train, TREND_SENSORS, windows=windows)
    df_test = compute_rolling_features(df_test, windows=windows)
    df_test = compute_trend_features(df_test, TREND_SENSORS, windows=windows)

    # استانداردسازی سراسری (بعداً قابل ارتقاء به condition-aware)
    sensor_cols = [c for c in df_train.columns if c.startswith("sensor_")]
    df_train, df_test, scaler = normalize_global(df_train, df_test, sensor_cols)

    print(f"\nFinal df_train shape: {df_train.shape}")
    print(f"Final df_test shape: {df_test.shape}")
    print(f"Number of sensor columns: {len(sensor_cols)}")

    return df_train, df_test, rul, scaler, sensor_cols


# صادرات سراسری برای سازگاری با بقیه‌ی اسکریپت‌ها
df_train, df_test, rul, scaler, sensor_cols = load_fd001_data()
