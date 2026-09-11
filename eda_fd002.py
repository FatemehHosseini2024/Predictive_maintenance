from utils import (
    load_data, validate_data, find_constant_features,
    dataset_summary, missing_and_duplicates, cycle_counts_statistics,
    compute_sensor_rul_correlations, compute_outlier_analysis,
    compute_setting_sensor_correlations, compute_life_progress_trend,
    analyze_life_stages,
    plot_correlation_heatmap, plot_sensor_rul_correlations, plot_engine_lifetimes,
    plot_engine_sensor_trend, plot_sensor_vs_rul, plot_sensor_distribution_by_life_stage,
    plot_sensor_across_engines, plot_life_progress_trend, plot_settings_distribution,
)
from utils.preprocessing import compute_rul
import pandas as pd

DATASET_NAME = "FD002"
TRAIN_PATH = "train_FD002.txt"
TEST_PATH = "test_FD002.txt"
RUL_PATH = "RUL_FD002.txt"

df_train, df_test, rul = load_data(TRAIN_PATH, TEST_PATH, RUL_PATH)
validate_data(df_train, TRAIN_PATH, expected_engines=None)
validate_data(df_test, TEST_PATH, expected_engines=None)

sensor_cols = [c for c in df_train.columns if c.startswith("sensor_")]
settings = ["setting_1", "setting_2", "setting_3"]

df_train = compute_rul(df_train, is_test=False)
df_test = compute_rul(df_test, is_test=True, rul_last_cycles=rul["RUL"])

dataset_summary(df_train, f"{DATASET_NAME} Train")
missing_and_duplicates(df_train)
cycle_counts_statistics(df_train)

constant_cols = find_constant_features(df_train, exclude_cols=["RUL"])
print(f"\nConstant columns to consider dropping: {constant_cols}")

compute_sensor_rul_correlations(df_train, sensor_cols, rul_col="RUL")
compute_outlier_analysis(df_train, sensor_cols)
compute_setting_sensor_correlations(df_train, settings, sensor_cols)

plot_engine_lifetimes(df_train, f"{DATASET_NAME}")
plot_correlation_heatmap(df_train, sensor_cols + ["RUL"], f"{DATASET_NAME}")
plot_sensor_rul_correlations(
    compute_sensor_rul_correlations(df_train, sensor_cols, rul_col="RUL"),
    f"{DATASET_NAME}", sensor_cols
)

engines_to_plot = df_train["unit_id"].unique()[:5]
for sensor in ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]:
    plot_sensor_across_engines(df_train, sensor, engines_to_plot, f"{DATASET_NAME}")

plot_sensor_vs_rul(df_train, "sensor_9", f"{DATASET_NAME}")

df_with_stage, _ = analyze_life_stages(df_train, sensor_cols)
plot_sensor_distribution_by_life_stage(df_with_stage, "sensor_11", f"{DATASET_NAME}")

df_with_progress, _ = compute_life_progress_trend(df_train, ["setting_1", "setting_2"], n_bins=10)
plot_life_progress_trend(df_with_progress, ["setting_1", "setting_2"], f"{DATASET_NAME}")
plot_settings_distribution(df_train, settings, f"{DATASET_NAME}")
