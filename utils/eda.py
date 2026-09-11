import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def dataset_summary(df, name):
    print(f"\n{'='*60}\n{name} Summary\n{'='*60}")
    print(f"Shape: {df.shape}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nInfo:")
    print(df.info())
    return df


def missing_and_duplicates(df):
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicates:")
    duplicates = df.duplicated()
    print(f"Number of duplicate rows: {duplicates.sum()}")
    if duplicates.sum() > 0:
        print(df[duplicates])
    return duplicates.sum()


def cycle_counts_statistics(df):
    cycle_counts = df.groupby("unit_id")["cycle"].count()
    print(f"\nCycle counts statistics:")
    print(cycle_counts.describe())
    return cycle_counts


def find_constant_features(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    nunique_df = df.nunique()
    constant_features = nunique_df[nunique_df == 1].index.tolist()
    constant_features = [c for c in constant_features if c not in exclude_cols]
    print(f"\nConstant features (nunique == 1): {constant_features}")
    return constant_features


def compute_sensor_rul_correlations(df, sensor_cols, rul_col="RUL"):
    cols = sensor_cols + [rul_col]
    existing_cols = [c for c in cols if c in df.columns]
    corr = df[existing_cols].corr()[rul_col].drop(rul_col)
    print(f"\n{sensor_cols} correlations with {rul_col}:")
    print(corr.sort_values())
    return corr


def compute_outlier_analysis(df, sensor_cols):
    Q1 = df[sensor_cols].quantile(0.25)
    Q3 = df[sensor_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_counts = ((df[sensor_cols] < lower_bound) | (df[sensor_cols] > upper_bound)).sum()
    outlier_percentage = (outlier_counts / len(df) * 100).sort_values(ascending=False)

    print("\nOutlier counts per sensor:")
    print(outlier_counts.sort_values(ascending=False))
    print("\nOutlier percentages:")
    print(outlier_percentage)

    return outlier_counts, outlier_percentage


def analyze_life_stages(df, sensor_cols, n_bins=3, labels=None):
    if labels is None:
        labels = ["Early-life", "Mid-life", "Late-life"]

    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    life_progress = df["cycle"] / max_cycle
    df = df.copy()
    df["life_stage"] = pd.cut(life_progress, bins=n_bins, labels=labels, include_lowest=True)

    print(f"\nLife stages value counts:")
    print(df["life_stage"].value_counts())

    stage_means = df.groupby("life_stage", observed=True)[sensor_cols].mean()
    print("\nStage means:")
    print(stage_means.T)

    return df, stage_means


def compute_setting_sensor_correlations(df, settings, sensor_cols):
    cols = settings + sensor_cols
    existing_cols = [c for c in cols if c in df.columns]
    corr = df[existing_cols].corr()
    setting_sensor_corr = corr.loc[settings, sensor_cols]
    print(f"\n{settings} vs sensors correlation:")
    print(setting_sensor_corr)
    return setting_sensor_corr


def compute_life_progress_trend(df, trend_cols, n_bins=10):
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df = df.copy()
    df["life_pct"] = df["cycle"] / max_cycle

    bins = pd.cut(df["life_pct"], bins=n_bins)
    trend = df.groupby(bins, observed=True)[trend_cols].mean()
    print(f"\n{trend_cols} across normalized life:")
    print(trend)
    return df, trend


# ============================================================
# Plotting functions
# ============================================================

def plot_correlation_heatmap(df, cols, dataset_name, output_path=None):
    existing_cols = [c for c in cols if c in df.columns]
    corr = df[existing_cols].corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title(f"{dataset_name} - Correlation Matrix")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_sensor_rul_correlations(corr_series, dataset_name, sensor_cols):
    plt.figure(figsize=(10, 6))
    corr_series.plot(kind="barh")
    plt.xlabel("Pearson Correlation with RUL")
    plt.ylabel("Sensor")
    plt.title(f"{dataset_name} - Sensor-RUL Correlation")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_engine_lifetimes(df, dataset_name, output_path=None):
    engine_lifetimes = df.groupby("unit_id")["cycle"].max()

    plt.figure(figsize=(10, 6))
    sns.histplot(engine_lifetimes, bins=15, kde=True)
    plt.xlabel("Engine Lifetime (Cycles)")
    plt.ylabel("Number of Engines")
    plt.title(f"{dataset_name} - Distribution of Engine Lifetimes")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_engine_sensor_trend(df, sensor, dataset_name, engine_id=1, output_path=None):
    engine_data = df[df["unit_id"] == engine_id]

    plt.figure(figsize=(10, 6))
    plt.plot(engine_data["cycle"], engine_data[sensor], marker=".", linewidth=1)
    plt.xlabel("Cycle")
    plt.ylabel(sensor)
    plt.title(f"{dataset_name} - {sensor} Trend - Engine {engine_id}")
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_sensor_vs_rul(df, sensor, dataset_name, output_path=None):
    plt.figure(figsize=(8, 5))
    plt.scatter(df[sensor], df["RUL"], alpha=0.3)
    plt.xlabel(sensor)
    plt.ylabel("RUL")
    plt.title(f"{dataset_name} - {sensor} vs RUL")
    plt.grid()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_sensor_distribution_by_life_stage(df_with_stage, sensor, dataset_name, stage_col="life_stage"):
    plt.figure(figsize=(10, 6))
    for stage in ["Early-life", "Mid-life", "Late-life"]:
        data = df_with_stage[df_with_stage[stage_col] == stage][sensor]
        plt.hist(data, bins=30, alpha=0.5, label=stage)
    plt.xlabel(sensor)
    plt.ylabel("Frequency")
    plt.title(f"{dataset_name} - {sensor} Distribution Across Life Stages")
    plt.legend()
    plt.show()


def plot_sensor_across_engines(df, sensor, engines, dataset_name, output_path=None):
    plt.figure(figsize=(10, 6))
    for engine_id in engines:
        engine_data = df[df["unit_id"] == engine_id]
        plt.plot(engine_data["cycle"], engine_data[sensor], label=f"Engine {engine_id}")
    plt.xlabel("Cycle")
    plt.ylabel(sensor)
    plt.title(f"{dataset_name} - {sensor} Across Multiple Engines")
    plt.legend()
    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_life_progress_trend(df_with_progress, trend_cols, dataset_name):
    fig, axes = plt.subplots(1, len(trend_cols), figsize=(6 * len(trend_cols), 4))
    if len(trend_cols) == 1:
        axes = [axes]

    for i, col in enumerate(trend_cols):
        bins = pd.cut(df_with_progress["life_pct"], bins=10)
        df_with_progress.groupby(bins, observed=True)[col].mean().plot(
            ax=axes[i], marker="o"
        )
        axes[i].set_title(f"{dataset_name} - {col} vs normalized life")
        axes[i].set_xlabel("Life percentage bin")
        axes[i].set_ylabel(col)

    plt.tight_layout()
    plt.show()


def plot_settings_distribution(df, settings, dataset_name):
    fig, axes = plt.subplots(1, len(settings), figsize=(5 * len(settings), 4))
    if len(settings) == 1:
        axes = [axes]

    for ax, setting in zip(axes, settings):
        sns.histplot(df[setting], bins=30, kde=True, ax=ax)
        ax.set_title(f"{dataset_name} - {setting} Distribution")
        ax.set_xlabel(setting)

    plt.tight_layout()
    plt.show()
