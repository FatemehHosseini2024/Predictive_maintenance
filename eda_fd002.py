from utils import (
    load_data, validate_data, find_constant_features,
    dataset_summary, missing_and_duplicates, cycle_counts_statistics,
)
from utils.preprocessing import compute_rul
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

DATASET_NAME = "FD002"
TRAIN_PATH = "train_FD002.txt"
TEST_PATH = "test_FD002.txt"
RUL_PATH = "RUL_FD002.txt"

df_train, df_test, rul = load_data(TRAIN_PATH, TEST_PATH, RUL_PATH)
validate_data(df_train, TRAIN_PATH, expected_engines=None)
validate_data(df_test, TEST_PATH, expected_engines=None)

df_train = compute_rul(df_train, is_test=False)
df_test = compute_rul(df_test, is_test=True, rul_last_cycles=rul["RUL"])

dataset_summary(df_train, f"{DATASET_NAME} Train")
missing_and_duplicates(df_train)
cycle_counts_statistics(df_train)

constant_cols = find_constant_features(df_train, exclude_cols=["RUL"])
print(f"\nConstant columns to consider dropping: {constant_cols}")

# KMeans clustering on settings to create condition_id
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df_train["condition_id"] = kmeans.fit_predict(df_train[["setting_1", "setting_2", "setting_3"]])

# Condition ID distribution
if "condition_id" in df_train.columns:
    print(f"\nCondition ID distribution:")
    cond_dist = df_train["condition_id"].value_counts().sort_index()
    print(cond_dist)
    print(f"\nEngines per condition:")
    engines_per_cond = df_train.groupby("condition_id")["unit_id"].nunique()
    print(engines_per_cond)
    
    # Plot distribution
    plt.figure(figsize=(8, 5))
    cond_dist.plot(kind="bar")
    plt.xlabel("Condition ID")
    plt.ylabel("Number of Rows")
    plt.title(f"{DATASET_NAME} - Rows per Condition ID")
    plt.show()
    
    plt.figure(figsize=(8, 5))
    engines_per_cond.plot(kind="bar")
    plt.xlabel("Condition ID")
    plt.ylabel("Number of Engines")
    plt.title(f"{DATASET_NAME} - Engines per Condition ID")
    plt.show()

# Check for 6 distinct clusters in setting space
settings = ["setting_1", "setting_2", "setting_3"]
print(f"\nUnique setting combinations:")
setting_combos = df_train[settings].drop_duplicates()
print(setting_combos)
print(f"Number of unique combinations: {len(setting_combos)}")

# Pair plot of settings (all rows)
sns.pairplot(df_train[settings], plot_kws={'alpha': 0.5, 's': 10})
plt.suptitle(f"{DATASET_NAME} - Setting Pair Plot", y=1.02)
plt.show()

# 3D scatter plot (all rows)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_train["setting_1"], df_train["setting_2"], df_train["setting_3"], alpha=0.5, s=10)
ax.set_xlabel("setting_1")
ax.set_ylabel("setting_2")
ax.set_zlabel("setting_3")
ax.set_title(f"{DATASET_NAME} - 3D Setting Space")
plt.show()

# Find sensors dependent on condition_id but stable within condition over cycles
sensor_cols = [c for c in df_train.columns if c.startswith("sensor_")]

# Between-condition variance: variance of sensor means across condition_ids
condition_means = df_train.groupby("condition_id")[sensor_cols].mean()
between_var = condition_means.var()

# Within-condition temporal variance: average variance of sensor within each engine (condition is constant per engine)
within_var_list = []
for cond in sorted(df_train["condition_id"].unique()):
    cond_data = df_train[df_train["condition_id"] == cond]
    # Variance within each engine, then average across engines in this condition
    engine_vars = cond_data.groupby("unit_id")[sensor_cols].var().mean()
    within_var_list.append(engine_vars)
within_var = pd.concat(within_var_list, axis=1).mean(axis=1)

# Ratio: high ratio = sensor differs by condition but stable within condition
ratio = between_var / (within_var + 1e-10)
ratio_sorted = ratio.sort_values(ascending=False)

print(f"\nSensor condition-dependency ratio (between_var / within_var):")
print(ratio_sorted)

# Top sensors that are condition-dependent but stable over cycles
top_sensors = ratio_sorted.head(10).index.tolist()
print(f"\nTop condition-dependent sensors (stable within condition): {top_sensors}")

# Plot: sensor means by condition for top sensors
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, sensor in enumerate(top_sensors):
    condition_means[sensor].plot(kind="bar", ax=axes[i])
    axes[i].set_title(f"{sensor}")
    axes[i].set_xlabel("Condition ID")
    axes[i].set_ylabel("Mean")
plt.tight_layout()
plt.suptitle(f"{DATASET_NAME} - Top Condition-Dependent Sensors", y=1.02)
plt.show()

# Verify low temporal trend within condition for top sensors
print(f"\nTemporal trend check (slope per cycle, averaged across engines in each condition):")
for sensor in top_sensors:
    slopes = []
    for cond in sorted(df_train["condition_id"].unique()):
        cond_data = df_train[df_train["condition_id"] == cond]
        for engine_id, g in cond_data.groupby("unit_id"):
            if len(g) > 1:
                x = g["cycle"].values
                y = g[sensor].values
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
    avg_slope = np.mean(np.abs(slopes)) if slopes else 0
    print(f"  {sensor}: avg |slope| = {avg_slope:.6f}")

# Trend of specific sensors over life cycle by condition_id
target_sensors = [
    "sensor_18", "sensor_1", "sensor_19", "sensor_5", "sensor_6", "sensor_8", 
    "sensor_13", "sensor_12", "sensor_7", "sensor_2",
    "sensor_21", "sensor_20", "sensor_10", "sensor_9", "sensor_15", 
    "sensor_17", "sensor_3", "sensor_4", "sensor_11", "sensor_14", "sensor_16"
]
max_cycle = df_train.groupby("unit_id")["cycle"].transform("max")
df_train["life_pct"] = df_train["cycle"] / max_cycle

for sensor in target_sensors:
    if sensor not in df_train.columns:
        print(f"{sensor} not found in data")
        continue
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw cycle trend
    for cond in sorted(df_train["condition_id"].unique()):
        cond_data = df_train[df_train["condition_id"] == cond]
        # Average across engines per cycle
        cycle_mean = cond_data.groupby("cycle")[sensor].mean()
        axes[0].plot(cycle_mean.index, cycle_mean.values, label=f"Cond {cond}", alpha=0.7)
    axes[0].set_xlabel("Cycle")
    axes[0].set_ylabel(sensor)
    axes[0].set_title(f"{sensor} - Mean Trend by Cycle per Condition")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Normalized life trend
    bins = pd.cut(df_train["life_pct"], bins=20)
    for cond in sorted(df_train["condition_id"].unique()):
        cond_data = df_train[df_train["condition_id"] == cond]
        life_trend = cond_data.groupby(bins, observed=True)[sensor].mean()
        axes[1].plot(range(len(life_trend)), life_trend.values, label=f"Cond {cond}", alpha=0.7, marker="o")
    axes[1].set_xlabel("Life Percentage Bin")
    axes[1].set_ylabel(sensor)
    axes[1].set_title(f"{sensor} - Mean Trend by Normalized Life per Condition")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.suptitle(f"{DATASET_NAME} - {sensor} Trends by Condition", y=1.02)
    plt.tight_layout()
    plt.show()

# Individual engine trends for specific sensors by condition_id
engine_trend_sensors = ["sensor_16", "sensor_11", "sensor_14", "sensor_4", "sensor_3"]
max_cycle = df_train.groupby("unit_id")["cycle"].transform("max")
df_train["life_pct"] = df_train["cycle"] / max_cycle

np.random.seed(42)

for sensor in engine_trend_sensors:
    if sensor not in df_train.columns:
        print(f"{sensor} not found in data")
        continue
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, cond in enumerate(sorted(df_train["condition_id"].unique())):
        ax = axes[idx]
        cond_data = df_train[df_train["condition_id"] == cond]
        
        # Sample 20 random engines per condition
        engines = cond_data["unit_id"].unique()
        n_sample = min(20, len(engines))
        sampled_engines = np.random.choice(engines, size=n_sample, replace=False)
        
        for engine_id in sampled_engines:
            engine_data = cond_data[cond_data["unit_id"] == engine_id].sort_values("cycle")
            ax.plot(engine_data["cycle"], engine_data[sensor], alpha=0.6, linewidth=1)
        
        ax.set_xlabel("Cycle")
        ax.set_ylabel(sensor)
        ax.set_title(f"{sensor} - Condition {cond} ({n_sample} sampled engines)")
        ax.grid(alpha=0.3)
    
    if len(sorted(df_train["condition_id"].unique())) < 6:
        axes[-1].set_visible(False)
    
    plt.suptitle(f"{DATASET_NAME} - {sensor} Sampled Engine Trends by Condition", y=1.02)
    plt.tight_layout()
    plt.show()

    # Also show normalized life version
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, cond in enumerate(sorted(df_train["condition_id"].unique())):
        ax = axes[idx]
        cond_data = df_train[df_train["condition_id"] == cond]
        
        engines = cond_data["unit_id"].unique()
        n_sample = min(20, len(engines))
        sampled_engines = np.random.choice(engines, size=n_sample, replace=False)
        
        for engine_id in sampled_engines:
            engine_data = cond_data[cond_data["unit_id"] == engine_id].sort_values("cycle")
            ax.plot(engine_data["life_pct"], engine_data[sensor], alpha=0.6, linewidth=1)
        
        ax.set_xlabel("Normalized Life (0-1)")
        ax.set_ylabel(sensor)
        ax.set_title(f"{sensor} - Condition {cond} ({n_sample} sampled engines)")
        ax.grid(alpha=0.3)
    
    if len(sorted(df_train["condition_id"].unique())) < 6:
        axes[-1].set_visible(False)
    
    plt.suptitle(f"{DATASET_NAME} - {sensor} Sampled Engine Trends by Normalized Life per Condition", y=1.02)
    plt.tight_layout()
    plt.show()