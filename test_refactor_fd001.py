from utils import (
    load_data, validate_data, identify_constant_sensors,
    compute_rul, drop_columns, clip_rul,
    stratified_engine_split,
    compute_rolling_features, compute_trend_features,
    normalize_global, normalize_by_condition,
)
import numpy as np

print("=== Loading FD001 data ===")
df_train, df_test, rul = load_data("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")

print("\n=== Validating FD001 ===")
validate_data(df_train, "train_FD001")
validate_data(df_test, "test_FD001", expected_engines=None)

print("\n=== Identifying constant sensors ===")
constant_cols = identify_constant_sensors(df_train, variance_threshold=0.005)
print(f"Constant columns detected: {constant_cols}")

print("\n=== Dropping constant columns ===")
df_train = drop_columns(df_train, constant_cols)
df_test = drop_columns(df_test, constant_cols)

print("\n=== Computing RUL ===")
df_train = compute_rul(df_train, is_test=False)
df_test = compute_rul(df_test, is_test=True, rul_last_cycles=rul["RUL"])

print("\n=== Stratified engine split ===")
df_train_final, df_val = stratified_engine_split(df_train, test_size=0.2, n_bins=4, random_state=42)

print("\n=== Feature engineering (rolling + slope) ===")
trend_sensors = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11",
    "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

df_train_final = compute_rolling_features(df_train_final, windows=[5, 20])
df_train_final = compute_trend_features(df_train_final, trend_sensors, windows=[5, 20])

df_val = compute_rolling_features(df_val, windows=[5, 20])
df_val = compute_trend_features(df_val, trend_sensors, windows=[5, 20])

print("\n=== Normalization (global) ===")
sensor_cols = [c for c in df_train_final.columns if c.startswith("sensor_")]
df_train_final, df_val, scaler = normalize_global(df_train_final, df_val, sensor_cols)

print("\n=== Clipping RUL ===")
RUL_CLIP = 125
y_train_clipped = clip_rul(df_train_final["RUL"], RUL_CLIP)
y_val_clipped = clip_rul(df_val["RUL"], RUL_CLIP)

print("\n=== Results ===")
print(f"df_train_final shape: {df_train_final.shape}")
print(f"df_val shape: {df_val.shape}")
print(f"Train RUL (clipped) - mean: {y_train_clipped.mean():.2f}, max: {y_train_clipped.max():.2f}")
print(f"Val RUL (clipped) - mean: {y_val_clipped.mean():.2f}, max: {y_val_clipped.max():.2f}")
print(f"Sensor cols count: {len(sensor_cols)}")
print("Test OK")
