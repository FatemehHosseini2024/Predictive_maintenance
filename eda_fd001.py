import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils import (
    load_data, validate_data, find_constant_features,
    dataset_summary, missing_and_duplicates, cycle_counts_statistics,
)
from utils.preprocessing import compute_rul

DATASET_NAME = "FD001"
TRAIN_PATH = "train_FD001.txt"
TEST_PATH = "test_FD001.txt"
RUL_PATH = "RUL_FD001.txt"

df_train, df_test, rul = load_data(TRAIN_PATH, TEST_PATH, RUL_PATH)
validate_data(df_train, TRAIN_PATH, expected_engines=100)
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

# Sensor-RUL correlations
cols = sensor_cols + ["RUL"]
existing_cols = [c for c in cols if c in df_train.columns]
corr = df_train[existing_cols].corr()["RUL"].drop("RUL")
print(f"\n{sensor_cols} correlations with RUL:")
print(corr.sort_values())

# Outlier analysis
Q1 = df_train[sensor_cols].quantile(0.25)
Q3 = df_train[sensor_cols].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_counts = ((df_train[sensor_cols] < lower_bound) | (df_train[sensor_cols] > upper_bound)).sum()
outlier_percentage = (outlier_counts / len(df_train) * 100).sort_values(ascending=False)
print("\nOutlier counts per sensor:")
print(outlier_counts.sort_values(ascending=False))
print("\nOutlier percentages:")
print(outlier_percentage)

# Setting-Sensor correlations
cols = settings + sensor_cols
existing_cols = [c for c in cols if c in df_train.columns]
corr = df_train[existing_cols].corr()
setting_sensor_corr = corr.loc[settings, sensor_cols]
print(f"\n{settings} vs sensors correlation:")
print(setting_sensor_corr)

# Engine lifetimes distribution
engine_lifetimes = df_train.groupby("unit_id")["cycle"].max()
plt.figure(figsize=(10, 6))
sns.histplot(engine_lifetimes, bins=15, kde=True)
plt.xlabel("Engine Lifetime (Cycles)")
plt.ylabel("Number of Engines")
plt.title(f"{DATASET_NAME} - Distribution of Engine Lifetimes")
plt.show()

# Correlation heatmap
existing_cols = [c for c in sensor_cols + ["RUL"] if c in df_train.columns]
corr = df_train[existing_cols].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title(f"{DATASET_NAME} - Correlation Matrix")
plt.show()

# Sensor-RUL correlation bar plot
sensor_rul_corr = df_train[existing_cols].corr()["RUL"].drop("RUL")
plt.figure(figsize=(10, 6))
sensor_rul_corr.sort_values().plot(kind="barh")
plt.xlabel("Pearson Correlation with RUL")
plt.ylabel("Sensor")
plt.title(f"{DATASET_NAME} - Sensor-RUL Correlation")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# Sensor trends across engines
engines_to_plot = [1, 15, 55, 75, 90]
for sensor in ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]:
    plt.figure(figsize=(10, 6))
    for engine_id in engines_to_plot:
        engine_data = df_train[df_train["unit_id"] == engine_id]
        plt.plot(engine_data["cycle"], engine_data[sensor], label=f"Engine {engine_id}")
    plt.xlabel("Cycle")
    plt.ylabel(sensor)
    plt.title(f"{DATASET_NAME} - {sensor} Across Multiple Engines")
    plt.legend()
    plt.show()

# Sensor 9 vs RUL scatter
plt.figure(figsize=(8, 5))
plt.scatter(df_train["sensor_9"], df_train["RUL"], alpha=0.3)
plt.xlabel("sensor_9")
plt.ylabel("RUL")
plt.title(f"{DATASET_NAME} - sensor_9 vs RUL")
plt.grid()
plt.tight_layout()
plt.show()

# Life stage analysis
max_cycle = df_train.groupby("unit_id")["cycle"].transform("max")
life_progress = df_train["cycle"] / max_cycle
df_with_stage = df_train.copy()
df_with_stage["life_stage"] = pd.cut(life_progress, bins=3, labels=["Early-life", "Mid-life", "Late-life"], include_lowest=True)

print(f"\nLife stages value counts:")
print(df_with_stage["life_stage"].value_counts())

stage_means = df_with_stage.groupby("life_stage", observed=True)[sensor_cols].mean()
print("\nStage means:")
print(stage_means.T)

# Sensor 11 distribution by life stage
plt.figure(figsize=(10, 6))
for stage in ["Early-life", "Mid-life", "Late-life"]:
    data = df_with_stage[df_with_stage["life_stage"] == stage]["sensor_11"]
    plt.hist(data, bins=30, alpha=0.5, label=stage)
plt.xlabel("sensor_11")
plt.ylabel("Frequency")
plt.title(f"{DATASET_NAME} - sensor_11 Distribution Across Life Stages")
plt.legend()
plt.show()

# Life progress trend for settings
df_with_progress = df_train.copy()
df_with_progress["life_pct"] = df_train["cycle"] / max_cycle
bins = pd.cut(df_with_progress["life_pct"], bins=10)
trend = df_with_progress.groupby(bins, observed=True)[["setting_1", "setting_2"]].mean()
print(f"\n['setting_1', 'setting_2'] across normalized life:")
print(trend)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for i, col in enumerate(["setting_1", "setting_2"]):
    df_with_progress.groupby(bins, observed=True)[col].mean().plot(ax=axes[i], marker="o")
    axes[i].set_title(f"{DATASET_NAME} - {col} vs normalized life")
    axes[i].set_xlabel("Life percentage bin")
    axes[i].set_ylabel(col)
plt.tight_layout()
plt.show()

# Settings distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, setting in zip(axes, settings):
    sns.histplot(df_train[setting], bins=30, kde=True, ax=ax)
    ax.set_title(f"{DATASET_NAME} - {setting} Distribution")
    ax.set_xlabel(setting)
plt.tight_layout()
plt.show()