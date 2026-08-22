import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None
)
columns = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)]
]

df.columns = columns
sensors_to_plot = [
    "sensor_1",
    "sensor_5",
    "sensor_4",
    "sensor_9",
    "sensor_15",
    "sensor_8"
]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))

for sensor, ax in zip(sensors_to_plot, axes.flat):

    engine_data = df[df["unit_id"] == 1]

    ax.plot(
        engine_data["cycle"],
        engine_data[sensor],
        marker=".",
        linewidth=1
    )

    ax.set_title(f"{sensor} - Engine 1")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Value")

plt.tight_layout()
plt.show()
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]



# Calculate cycle-to-cycle changes
for sensor in sensor_cols:
    df[f"{sensor}_diff"] = (
        df.groupby("unit_id")[sensor].diff()
    )
noise_std = {}

for sensor in sensor_cols:
    diff = df[f"{sensor}_diff"].dropna()
    noise_std[sensor] = diff.std()

noise_std = pd.Series(noise_std).sort_values()

print(noise_std)
plt.figure(figsize=(12, 6))

noise_std.sort_values().plot(kind="bar")

plt.xlabel("Sensor")
plt.ylabel("Std of Cycle-to-Cycle Change")
plt.title("Sensor Short-Term Variability")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
sensor = "sensor_15"
engine_id = 88

engine_data = df[
    df["unit_id"] == engine_id
].copy()

engine_data["rolling_mean"] = (
    engine_data[sensor]
    .rolling(window=10, center=True)
    .mean()
)

plt.figure(figsize=(12, 6))

plt.plot(
    engine_data["cycle"],
    engine_data[sensor],
    label="Raw signal"
)

plt.plot(
    engine_data["cycle"],
    engine_data["rolling_mean"],
    label="Rolling mean"
)

plt.xlabel("Cycle")
plt.ylabel(sensor)
plt.title(f"Noise Analysis - {sensor}, Engine {engine_id}")
plt.legend()

plt.show()




settings = ["setting_1", "setting_2", "setting_3"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for setting, ax in zip(settings, axes):
    sns.histplot(df[setting], bins=30, kde=True, ax=ax)
    ax.set_title(f"{setting} Distribution")
    ax.set_xlabel(setting)

plt.tight_layout()
plt.show()
corr = df[settings + sensor_cols].corr()

setting_sensor_corr = corr.loc[settings, sensor_cols]

plt.figure(figsize=(16, 4))

sns.heatmap(
    setting_sensor_corr,
    annot=True,
    cmap="coolwarm",
    center=0
)

plt.title("Operating Settings vs Sensors Correlation")
plt.xlabel("Sensors")
plt.ylabel("Operating Settings")

plt.tight_layout()
plt.show()
# Maximum cycle of each engine
max_cycle = df.groupby("unit_id")["cycle"].transform("max")
# Relative life of each observation
df["life_progress"] = df["cycle"] / max_cycle
df["life_stage"] = pd.cut(
    df["life_progress"],
    bins=[0, 0.2, 0.8, 1.0],
    labels=["Early-life", "Mid-life", "Late-life"],
    include_lowest=True
)
print(df["life_stage"].value_counts())
stage_means = df.groupby("life_stage", observed=True)[sensor_cols].mean()

print(stage_means.T)


stage_data = df.groupby(
    "life_stage",
    observed=True
)[sensor_cols].mean()

scaler = StandardScaler()

stage_scaled = pd.DataFrame(
    scaler.fit_transform(stage_data),
    index=stage_data.index,
    columns=stage_data.columns
)

plt.figure(figsize=(16, 8))

for sensor in sensor_cols:
    plt.plot(
        stage_scaled.index,
        stage_scaled[sensor],
        marker="o",
        label=sensor
    )

plt.xlabel("Life Stage")
plt.ylabel("Standardized Mean Sensor Value")
plt.title("Sensor Behavior Across Engine Life Stages")

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    ncol=2
)

plt.tight_layout()
plt.show()
sensor = "sensor_11"

plt.figure(figsize=(10, 6))

for stage in ["Early-life", "Mid-life", "Late-life"]:
    
    data = df[df["life_stage"] == stage][sensor]
    
    plt.hist(
        data,
        bins=30,
        alpha=0.5,
        label=stage
    )

plt.xlabel(sensor)
plt.ylabel("Frequency")
plt.title(f"{sensor} Distribution Across Life Stages")
plt.legend()

plt.show()
early_mean = (
    df[df["life_stage"] == "Early-life"]
    [sensor_cols]
    .mean()
)

late_mean = (
    df[df["life_stage"] == "Late-life"]
    [sensor_cols]
    .mean()
)

relative_change = (
    (late_mean - early_mean)
    / early_mean.abs()
)

print(
    relative_change
    .sort_values(key=abs, ascending=False)
)
sensor = "sensor_21"

engines = [1, 20,50, 60,70, 88,90]

plt.figure(figsize=(10, 6))

for engine_id in engines:

    engine_data = df[df["unit_id"] == engine_id].copy()

    max_cycle = engine_data["cycle"].max()
    engine_data["RUL"] = max_cycle - engine_data["cycle"]

    plt.scatter(
        engine_data["RUL"],
        engine_data[sensor],
        alpha=0.3,
        label=f"Engine {engine_id}"
    )

plt.xlabel("RUL")
plt.ylabel(sensor)
plt.title(f"{sensor} vs RUL")
plt.legend()
plt.show()


engine_stats = (
    df.groupby("unit_id")[sensor_cols]
      .agg(["mean", "std"])
)

print(engine_stats)
early_data = df[df["cycle"] <= 10]

engine_initial = (
    early_data
    .groupby("unit_id")[sensor_cols]
    .mean()
)
sensor = "sensor_20"

plt.figure(figsize=(14, 6))

plt.bar(
    engine_initial.index,
    engine_initial[sensor]
)

plt.xlabel("Engine ID")
plt.ylabel(f"Mean {sensor} - First 10 Cycles")
plt.title(f"Initial Condition Comparison Across Engines")
plt.show()