import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from utils import (
    compute_rolling_features,
    compute_trend_features,
    normalize_global,
    normalize_by_condition,
)
from sklearn.cluster import KMeans

TREND_SENSORS_FD001 = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11",
    "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

TREND_SENSORS_FD002 = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
    "sensor_12", "sensor_13", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

WINDOWS = [5, 20]


def drop_per_condition_constant_sensors(df_train, df_test, condition_col, sensor_cols):
    cols_to_drop = []
    for col in sensor_cols:
        nunique_per_group = df_train.groupby(condition_col)[col].nunique()
        if (nunique_per_group == 1).all():
            cols_to_drop.append(col)

    if cols_to_drop:
        df_train = df_train.drop(columns=cols_to_drop)
        df_test = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
        print(f"Dropped per-condition constant sensors ({len(cols_to_drop)}): {cols_to_drop}")
    else:
        print("No per-condition constant sensors to drop")

    return df_train, df_test


def add_feature_engineering(df_train, df_test, dataset_name="FD002", n_conditions=None):
    if dataset_name == "FD002":
        trend_sensors = TREND_SENSORS_FD002
    else:
        trend_sensors = TREND_SENSORS_FD001

    print(f"\nFeature engineering ({dataset_name}):")

    use_conditions = False
    if n_conditions is not None:
        setting_cols = ["setting_1", "setting_2", "setting_3"]
        available_setting_cols = [c for c in setting_cols if c in df_train.columns]
        if len(available_setting_cols) >= 2:
            print(f"Creating {n_conditions} conditions via KMeans on settings:")
            kmeans = KMeans(n_clusters=n_conditions, random_state=42, n_init=10)
            df_train["condition_id"] = kmeans.fit_predict(df_train[available_setting_cols])
            df_test["condition_id"] = kmeans.predict(df_test[available_setting_cols])

            sensor_cols = [c for c in df_train.columns if c.startswith("sensor_")]
            df_train, df_test = drop_per_condition_constant_sensors(df_train, df_test, "condition_id", sensor_cols)
            use_conditions = True
        else:
            print(f"Settings not available for KMeans, skipping condition creation")

    df_train = compute_rolling_features(df_train, windows=WINDOWS)
    df_train = compute_trend_features(df_train, trend_sensors, windows=WINDOWS)
    df_test = compute_rolling_features(df_test, windows=WINDOWS)
    df_test = compute_trend_features(df_test, trend_sensors, windows=WINDOWS)

    sensor_cols = [c for c in df_train.columns if c.startswith("sensor_") and "_roll_" not in c and "_slope_" not in c]

    if use_conditions:
        df_train, df_test, scaler = normalize_by_condition(
            df_train, df_test, sensor_cols, condition_col="condition_id", n_conditions=n_conditions
        )
    else:
        df_train, df_test, scaler = normalize_global(df_train, df_test, sensor_cols)

    print(f"\nFeature engineering complete:")
    print(f"  train: {df_train.shape}")
    print(f"  test: {df_test.shape}")
    print(f"  sensor cols: {len(sensor_cols)}")

    return df_train, df_test, scaler, sensor_cols
