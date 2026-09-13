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

# KMeans clustering on settings to create condition_id
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df_train["condition_id"] = kmeans.fit_predict(df_train[["setting_1", "setting_2", "setting_3"]])

max_cycle = df_train.groupby("unit_id")["cycle"].transform("max")
df_train["life_pct"] = df_train["cycle"] / max_cycle

sensor_cols = [c for c in df_train.columns if c.startswith("sensor_")]
settings = ["setting_1", "setting_2", "setting_3"]


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


def engine_lifecycle_analysis():
    """Engine lifetime distribution stats and plots."""
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(engine_lifetimes, bins=30, kde=True, ax=axes[0])
    axes[0].set_xlabel("Engine Lifetime (Cycles)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{DATASET_NAME} - Engine Lifetime Distribution")
    axes[0].axvline(engine_lifetimes.mean(), color='red', linestyle='--', label=f'Mean: {engine_lifetimes.mean():.1f}')
    axes[0].axvline(engine_lifetimes.median(), color='green', linestyle='--', label=f'Median: {engine_lifetimes.median():.1f}')
    axes[0].legend()

    sns.boxplot(y=engine_lifetimes, ax=axes[1])
    axes[1].set_ylabel("Engine Lifetime (Cycles)")
    axes[1].set_title(f"{DATASET_NAME} - Engine Lifetime Boxplot")
    plt.tight_layout()
    plt.show()

    if "condition_id" in df_train.columns:
        print(f"\nEngine Lifetime by Condition ID:")
        cond_lifetimes = df_train.groupby("condition_id").apply(lambda x: x.groupby("unit_id")["cycle"].max())
        for cond in sorted(df_train["condition_id"].unique()):
            lifetimes = cond_lifetimes.loc[cond]
            print(f"  Condition {cond}: n={len(lifetimes)}, mean={lifetimes.mean():.1f}, std={lifetimes.std():.1f}, min={lifetimes.min()}, max={lifetimes.max()}")

        plt.figure(figsize=(10, 6))
        lifetimes_by_cond = [cond_lifetimes.loc[cond].values for cond in sorted(df_train["condition_id"].unique())]
        plt.boxplot(lifetimes_by_cond, labels=[f"Cond {c}" for c in sorted(df_train["condition_id"].unique())])
        plt.xlabel("Condition ID")
        plt.ylabel("Engine Lifetime (Cycles)")
        plt.title(f"{DATASET_NAME} - Engine Lifetime by Condition")
        plt.grid(alpha=0.3)
        plt.show()


def lifetime_vs_condition():
    """Analyze relationship between engine lifetime and condition_id."""
    # Each engine has one condition_id (settings are constant per engine)
    engine_info = df_train.groupby("unit_id").agg(
        lifetime=("cycle", "max"),
        condition_id=("condition_id", "first"),
        setting_1=("setting_1", "first"),
        setting_2=("setting_2", "first"),
        setting_3=("setting_3", "first"),
    ).reset_index()

    print(f"\n{'='*60}")
    print(f"Engine Lifetime vs Condition ID Analysis")
    print(f"{'='*60}")

    # Overall stats by condition
    print(f"\nLifetime statistics by Condition ID:")
    cond_stats = engine_info.groupby("condition_id")["lifetime"].agg(["count", "mean", "std", "min", "max", "median"])
    print(cond_stats)

    # Sort conditions by mean lifetime
    cond_mean_sorted = cond_stats.sort_values("mean", ascending=False)
    print(f"\nConditions ranked by mean lifetime (longest to shortest):")
    for cond, row in cond_mean_sorted.iterrows():
        print(f"  Condition {cond}: mean={row['mean']:.1f}, median={row['median']:.1f}, std={row['std']:.1f}, n={row['count']}")

    # Correlation between settings and lifetime
    print(f"\nCorrelation between settings and lifetime:")
    for setting in ["setting_1", "setting_2", "setting_3"]:
        corr = engine_info[setting].corr(engine_info["lifetime"])
        print(f"  {setting}: Pearson r = {corr:.4f}")

    # Boxplot
    plt.figure(figsize=(10, 6))
    lifetimes_by_cond = [engine_info[engine_info["condition_id"] == cond]["lifetime"].values 
                         for cond in sorted(engine_info["condition_id"].unique())]
    plt.boxplot(lifetimes_by_cond, labels=[f"Cond {c}" for c in sorted(engine_info["condition_id"].unique())])
    plt.xlabel("Condition ID")
    plt.ylabel("Engine Lifetime (Cycles)")
    plt.title(f"{DATASET_NAME} - Engine Lifetime Distribution by Condition")
    plt.grid(alpha=0.3)
    plt.show()

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=engine_info, x="condition_id", y="lifetime", inner="box")
    plt.xlabel("Condition ID")
    plt.ylabel("Engine Lifetime (Cycles)")
    plt.title(f"{DATASET_NAME} - Engine Lifetime by Condition (Violin)")
    plt.grid(alpha=0.3)
    plt.show()

    # Scatter: setting vs lifetime colored by condition
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, setting in enumerate(["setting_1", "setting_2", "setting_3"]):
        ax = axes[i]
        scatter = ax.scatter(engine_info[setting], engine_info["lifetime"], 
                            c=engine_info["condition_id"], cmap="tab10", alpha=0.7)
        ax.set_xlabel(setting)
        ax.set_ylabel("Lifetime (Cycles)")
        ax.set_title(f"{setting} vs Lifetime")
        ax.grid(alpha=0.3)
    plt.colorbar(scatter, ax=axes, label="Condition ID")
    plt.suptitle(f"{DATASET_NAME} - Settings vs Lifetime by Condition", y=1.02)
    plt.tight_layout()
    plt.show()

    # Statistical test: ANOVA
    from scipy import stats
    groups = [engine_info[engine_info["condition_id"] == cond]["lifetime"].values 
              for cond in sorted(engine_info["condition_id"].unique())]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"\nOne-way ANOVA (lifetime ~ condition_id): F={f_stat:.4f}, p={p_value:.6f}")
    if p_value < 0.05:
        print("  -> Significant difference in mean lifetime across conditions")
    else:
        print("  -> No significant difference in mean lifetime across conditions")

    return engine_info


def sensor_rul_correlation():
    """Global and per-condition sensor-RUL correlations with weighted average."""
    sensor_cols = [c for c in df_train.columns if c.startswith("sensor_")]
    
    print(f"\n{'='*60}")
    print(f"Sensor-RUL Correlation Analysis")
    print(f"{'='*60}")
    
    # Global correlation
    print(f"\nGlobal Sensor-RUL Correlation (all data):")
    global_corr = df_train[sensor_cols + ["RUL"]].corr()["RUL"].drop("RUL")
    global_corr_sorted = global_corr.sort_values(ascending=False)
    print(global_corr_sorted)
    
    # Top sensors globally
    top_global = global_corr_sorted.head(10)
    print(f"\nTop 10 sensors globally:")
    print(top_global)
    
    # Per-condition correlation
    print(f"\n{'='*60}")
    print(f"Per-Condition Sensor-RUL Correlations:")
    print(f"{'='*60}")
    
    cond_corrs = {}
    cond_sizes = {}
    
    for cond in sorted(df_train["condition_id"].unique()):
        cond_data = df_train[df_train["condition_id"] == cond]
        cond_sizes[cond] = len(cond_data)
        corr = cond_data[sensor_cols + ["RUL"]].corr()["RUL"].drop("RUL")
        cond_corrs[cond] = corr
        print(f"\nCondition {cond} (n={len(cond_data)}):")
        print(corr.sort_values(ascending=False))
    
    # Weighted average correlation
    print(f"\n{'='*60}")
    print(f"Weighted Average Correlation (by row count):")
    print(f"{'='*60}")
    
    total_rows = sum(cond_sizes.values())
    weighted_corr = {}
    
    for sensor in sensor_cols:
        weighted_sum = 0
        for cond in sorted(df_train["condition_id"].unique()):
            weight = cond_sizes[cond] / total_rows
            weighted_sum += weight * cond_corrs[cond][sensor]
        weighted_corr[sensor] = weighted_sum
    
    weighted_corr_series = pd.Series(weighted_corr).sort_values(ascending=False)
    print(weighted_corr_series)
    
    # Compare global vs weighted
    print(f"\n{'='*60}")
    print(f"Comparison: Global vs Weighted Average:")
    print(f"{'='*60}")
    comparison = pd.DataFrame({
        "Global": global_corr,
        "Weighted": weighted_corr_series,
        "Diff": global_corr - weighted_corr_series
    }).sort_values("Global", ascending=False)
    print(comparison)
    
    # Plot: Global correlation bar
    plt.figure(figsize=(10, 8))
    global_corr_sorted.plot(kind="barh")
    plt.xlabel("Pearson Correlation with RUL")
    plt.ylabel("Sensor")
    plt.title(f"{DATASET_NAME} - Global Sensor-RUL Correlation")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot: Weighted average correlation bar
    plt.figure(figsize=(10, 8))
    weighted_corr_series.plot(kind="barh", color="orange")
    plt.xlabel("Weighted Avg Correlation with RUL")
    plt.ylabel("Sensor")
    plt.title(f"{DATASET_NAME} - Weighted Avg Sensor-RUL Correlation (by condition size)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Heatmap: per-condition correlations
    cond_corr_df = pd.DataFrame(cond_corrs).T  # conditions x sensors
    plt.figure(figsize=(14, 8))
    sns.heatmap(cond_corr_df, cmap="coolwarm", center=0, annot=True, fmt=".3f", cbar_kws={'label': 'Correlation'})
    plt.title(f"{DATASET_NAME} - Sensor-RUL Correlation per Condition")
    plt.xlabel("Sensor")
    plt.ylabel("Condition ID")
    plt.tight_layout()
    plt.show()
    
    return global_corr_sorted, weighted_corr_series, cond_corrs


def sensor_std_by_condition(sensors=None):
    """Calculate std of sensors per condition_id."""
    if sensors is None:
        sensors = ["sensor_16", "sensor_10"]
    
    print(f"\n{'='*60}")
    print(f"Sensor Std by Condition ID")
    print(f"{'='*60}")
    
    for sensor in sensors:
        if sensor not in df_train.columns:
            print(f"{sensor} not found in data")
            continue
        
        print(f"\n{sensor} std by condition_id:")
        std_by_cond = df_train.groupby("condition_id")[sensor].std()
        print(std_by_cond)
        
        mean_by_cond = df_train.groupby("condition_id")[sensor].mean()
        print(f"\n{sensor} mean by condition_id:")
        print(mean_by_cond)
        
        # Coefficient of variation
        cv = std_by_cond / (mean_by_cond.abs() + 1e-10)
        print(f"\n{sensor} coefficient of variation (std/|mean|) by condition_id:")
        print(cv)
    
    # Combined table
    print(f"\n{'='*60}")
    print(f"Combined Std Table:")
    print(f"{'='*60}")
    std_table = df_train.groupby("condition_id")[sensors].std()
    mean_table = df_train.groupby("condition_id")[sensors].mean()
    cv_table = std_table / (mean_table.abs() + 1e-10)
    
    result = pd.DataFrame()
    for sensor in sensors:
        result[f"{sensor}_mean"] = mean_table[sensor]
        result[f"{sensor}_std"] = std_table[sensor]
        result[f"{sensor}_cv"] = cv_table[sensor]
    
    print(result)
    
    # Plot
    fig, axes = plt.subplots(1, len(sensors), figsize=(6 * len(sensors), 5))
    if len(sensors) == 1:
        axes = [axes]
    
    for i, sensor in enumerate(sensors):
        ax = axes[i]
        std_table[sensor].plot(kind="bar", ax=ax, color="skyblue")
        ax.set_xlabel("Condition ID")
        ax.set_ylabel("Std")
        ax.set_title(f"{sensor} Std by Condition")
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return std_table, mean_table, cv_table


def condition_distribution():
    """Condition ID row/engine distribution plots."""
    print(f"\nCondition ID distribution:")
    cond_dist = df_train["condition_id"].value_counts().sort_index()
    print(cond_dist)
    print(f"\nEngines per condition:")
    engines_per_cond = df_train.groupby("condition_id")["unit_id"].nunique()
    print(engines_per_cond)

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


def setting_space_analysis():
    """Setting combinations, pair plot, 3D scatter."""
    print(f"\nUnique setting combinations:")
    setting_combos = df_train[settings].drop_duplicates()
    print(setting_combos)
    print(f"Number of unique combinations: {len(setting_combos)}")

    sns.pairplot(df_train[settings], plot_kws={'alpha': 0.5, 's': 10})
    plt.suptitle(f"{DATASET_NAME} - Setting Pair Plot", y=1.02)
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_train["setting_1"], df_train["setting_2"], df_train["setting_3"], alpha=0.5, s=10)
    ax.set_xlabel("setting_1")
    ax.set_ylabel("setting_2")
    ax.set_zlabel("setting_3")
    ax.set_title(f"{DATASET_NAME} - 3D Setting Space")
    plt.show()


def condition_dependent_sensors():
    """Find sensors that differ by condition but stable within condition."""
    condition_means = df_train.groupby("condition_id")[sensor_cols].mean()
    between_var = condition_means.var()

    within_var_list = []
    for cond in sorted(df_train["condition_id"].unique()):
        cond_data = df_train[df_train["condition_id"] == cond]
        engine_vars = cond_data.groupby("unit_id")[sensor_cols].var().mean()
        within_var_list.append(engine_vars)
    within_var = pd.concat(within_var_list, axis=1).mean(axis=1)

    ratio = between_var / (within_var + 1e-10)
    ratio_sorted = ratio.sort_values(ascending=False)

    print(f"\nSensor condition-dependency ratio (between_var / within_var):")
    print(ratio_sorted)

    top_sensors = ratio_sorted.head(10).index.tolist()
    print(f"\nTop condition-dependent sensors (stable within condition): {top_sensors}")

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

    print(f"\nTemporal trend check (avg |slope| per cycle within condition):")
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

    return top_sensors, ratio_sorted


def sensor_trends_by_condition(sensors=None):
    """Mean sensor trends over cycle and normalized life by condition."""
    if sensors is None:
        sensors = [
            "sensor_18", "sensor_1", "sensor_19", "sensor_5", "sensor_6", "sensor_8",
            "sensor_13", "sensor_12", "sensor_7", "sensor_2",
            "sensor_21", "sensor_20", "sensor_10", "sensor_9", "sensor_15",
            "sensor_17", "sensor_3", "sensor_4", "sensor_11", "sensor_14", "sensor_16"
        ]

    for sensor in sensors:
        if sensor not in df_train.columns:
            print(f"{sensor} not found in data")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for cond in sorted(df_train["condition_id"].unique()):
            cond_data = df_train[df_train["condition_id"] == cond]
            cycle_mean = cond_data.groupby("cycle")[sensor].mean()
            axes[0].plot(cycle_mean.index, cycle_mean.values, label=f"Cond {cond}", alpha=0.7)
        axes[0].set_xlabel("Cycle")
        axes[0].set_ylabel(sensor)
        axes[0].set_title(f"{sensor} - Mean Trend by Cycle per Condition")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

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


def individual_engine_trends(sensors=None, n_sample=20):
    """Individual engine trends by condition (raw cycles and normalized life)."""
    if sensors is None:
        sensors = ["sensor_16", "sensor_11", "sensor_14", "sensor_4", "sensor_3"]

    np.random.seed(42)

    for sensor in sensors:
        if sensor not in df_train.columns:
            print(f"{sensor} not found in data")
            continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, cond in enumerate(sorted(df_train["condition_id"].unique())):
            ax = axes[idx]
            cond_data = df_train[df_train["condition_id"] == cond]

            engines = cond_data["unit_id"].unique()
            n_sample = min(n_sample, len(engines))
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

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for idx, cond in enumerate(sorted(df_train["condition_id"].unique())):
            ax = axes[idx]
            cond_data = df_train[df_train["condition_id"] == cond]

            engines = cond_data["unit_id"].unique()
            n_sample = min(n_sample, len(engines))
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


# ============================================================
# Main - Call functions you want to run
# ============================================================

if __name__ == "__main__":
    # Uncomment the functions you want to run:
    
    # basic_summary()
    # engine_lifecycle_analysis()
    # condition_distribution()
    # setting_space_analysis()
    # condition_dependent_sensors()
    # sensor_trends_by_condition()
    # individual_engine_trends()
    # lifetime_vs_condition()
    #sensor_rul_correlation()
    sensor_std_by_condition()
    pass