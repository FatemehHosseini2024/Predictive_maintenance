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

df_train = compute_rul(df_train, is_test=False)
df_test = compute_rul(df_test, is_test=True, rul_last_cycles=rul["RUL"])

sensor_cols = [c for c in df_train.columns if c.startswith("sensor_")]
settings = ["setting_1", "setting_2", "setting_3"]

max_cycle = df_train.groupby("unit_id")["cycle"].transform("max")
df_train["life_pct"] = df_train["cycle"] / max_cycle


# ============================================================
# Analysis Functions
# ============================================================

def basic_summary():
    """Basic dataset summary, missing values, duplicates, cycle counts."""
    dataset_summary(df_train, f"{DATASET_NAME} Train")
    missing_and_duplicates(df_train)
    cycle_counts_statistics(df_train)
    constant_cols = find_constant_features(df_train, exclude_cols=["RUL"])
    print(f"\nConstant columns to consider dropping: {constant_cols}")


def sensor_rul_correlations():
    """Sensor correlations with RUL."""
    cols = sensor_cols + ["RUL"]
    existing_cols = [c for c in cols if c in df_train.columns]
    corr = df_train[existing_cols].corr()["RUL"].drop("RUL")
    print(f"\n{sensor_cols} correlations with RUL:")
    print(corr.sort_values())

    # Bar plot
    plt.figure(figsize=(10, 6))
    corr.sort_values().plot(kind="barh")
    plt.xlabel("Pearson Correlation with RUL")
    plt.ylabel("Sensor")
    plt.title(f"{DATASET_NAME} - Sensor-RUL Correlation")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def outlier_analysis():
    """Outlier analysis using IQR method."""
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


def setting_sensor_correlations():
    """Setting-Sensor correlation matrix."""
    cols = settings + sensor_cols
    existing_cols = [c for c in cols if c in df_train.columns]
    corr = df_train[existing_cols].corr()
    setting_sensor_corr = corr.loc[settings, sensor_cols]
    print(f"\n{settings} vs sensors correlation:")
    print(setting_sensor_corr)


def engine_lifetime_distribution():
    """Engine lifetime distribution stats and plot."""
    engine_lifetimes = df_train.groupby("unit_id")["cycle"].max()
    print(f"\n{'='*60}")
    print(f"Engine Life Cycle Analysis")
    print(f"{'='*60}")
    print(f"Total engines: {len(engine_lifetimes)}")
    print(f"Mean lifetime: {engine_lifetimes.mean():.1f} cycles")
    print(f"Std lifetime: {engine_lifetimes.std():.1f} cycles")
    print(f"Min lifetime: {engine_lifetimes.min()} cycles")
    print(f"Max lifetime: {engine_lifetimes.max()} cycles")
    print(f"Median lifetime: {engine_lifetimes.median():.1f} cycles")
    print(f"\nFull describe():")
    print(engine_lifetimes.describe())
    print(f"\nExtended percentiles:")
    print(engine_lifetimes.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))

    plt.figure(figsize=(10, 6))
    sns.histplot(engine_lifetimes, bins=30, kde=True)
    plt.xlabel("Engine Lifetime (Cycles)")
    plt.ylabel("Number of Engines")
    plt.title(f"{DATASET_NAME} - Distribution of Engine Lifetimes")
    plt.axvline(engine_lifetimes.mean(), color='red', linestyle='--', label=f'Mean: {engine_lifetimes.mean():.1f}')
    plt.axvline(engine_lifetimes.median(), color='green', linestyle='--', label=f'Median: {engine_lifetimes.median():.1f}')
    plt.legend()
    plt.show()


def correlation_heatmap():
    """Correlation heatmap of sensors and RUL."""
    existing_cols = [c for c in sensor_cols + ["RUL"] if c in df_train.columns]
    corr = df_train[existing_cols].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title(f"{DATASET_NAME} - Correlation Matrix")
    plt.show()


def sensor_trends_across_engines(engines_to_plot=None, sensors=None):
    """Sensor trends across selected engines."""
    if engines_to_plot is None:
        engines_to_plot = [1, 15, 55, 75, 90]
    if sensors is None:
        sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21"]

    for sensor in sensors:
        plt.figure(figsize=(10, 6))
        for engine_id in engines_to_plot:
            engine_data = df_train[df_train["unit_id"] == engine_id]
            plt.plot(engine_data["cycle"], engine_data[sensor], label=f"Engine {engine_id}")
        plt.xlabel("Cycle")
        plt.ylabel(sensor)
        plt.title(f"{DATASET_NAME} - {sensor} Across Multiple Engines")
        plt.legend()
        plt.show()


def sensor_vs_rul_scatter(sensor="sensor_9"):
    """Scatter plot of sensor vs RUL."""
    if sensor not in df_train.columns:
        print(f"{sensor} not found in data")
        return

    plt.figure(figsize=(8, 5))
    plt.scatter(df_train[sensor], df_train["RUL"], alpha=0.3)
    plt.xlabel(sensor)
    plt.ylabel("RUL")
    plt.title(f"{DATASET_NAME} - {sensor} vs RUL")
    plt.grid()
    plt.tight_layout()
    plt.show()


def life_stage_analysis():
    """Life stage analysis (Early/Mid/Late life)."""
    df_with_stage = df_train.copy()
    df_with_stage["life_stage"] = pd.cut(df_train["life_pct"], bins=3, labels=["Early-life", "Mid-life", "Late-life"], include_lowest=True)

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


def life_progress_trend(trend_cols=None):
    """Settings trend over normalized life."""
    if trend_cols is None:
        trend_cols = ["setting_1", "setting_2"]

    df_with_progress = df_train.copy()
    bins = pd.cut(df_with_progress["life_pct"], bins=10)
    trend = df_with_progress.groupby(bins, observed=True)[trend_cols].mean()
    print(f"\n{trend_cols} across normalized life:")
    print(trend)

    fig, axes = plt.subplots(1, len(trend_cols), figsize=(6 * len(trend_cols), 4))
    if len(trend_cols) == 1:
        axes = [axes]

    for i, col in enumerate(trend_cols):
        df_with_progress.groupby(bins, observed=True)[col].mean().plot(ax=axes[i], marker="o")
        axes[i].set_title(f"{DATASET_NAME} - {col} vs normalized life")
        axes[i].set_xlabel("Life percentage bin")
        axes[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()


def settings_distribution():
    """Settings distribution histograms."""
    fig, axes = plt.subplots(1, len(settings), figsize=(5 * len(settings), 4))
    if len(settings) == 1:
        axes = [axes]

    for ax, setting in zip(axes, settings):
        sns.histplot(df_train[setting], bins=30, kde=True, ax=ax)
        ax.set_title(f"{DATASET_NAME} - {setting} Distribution")
        ax.set_xlabel(setting)

    plt.tight_layout()
    plt.show()


# ============================================================
# Main - Call functions you want to run
# ============================================================

if __name__ == "__main__":
    # Uncomment the functions you want to run:
    
    basic_summary()
    # sensor_rul_correlations()
    # outlier_analysis()
    # setting_sensor_correlations()
    # engine_lifetime_distribution()
    # correlation_heatmap()
    # sensor_trends_across_engines()
    # sensor_vs_rul_scatter("sensor_9")
    # life_stage_analysis()
    # life_progress_trend()
    # settings_distribution()
    pass