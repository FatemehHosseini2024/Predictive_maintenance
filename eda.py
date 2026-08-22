import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
numeric_cols = df.select_dtypes(include="number").columns

corr = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)

plt.title("Correlation Matrix")
plt.show()

cycle_corr = df[numeric_cols].corrwith(df["cycle"])

print(cycle_corr.sort_values())

engine_lifetimes = df.groupby("unit_id")["cycle"].max()

# Plot distribution
plt.figure(figsize=(10, 6))

sns.histplot(
    engine_lifetimes,
    bins=15,
    kde=True
)

plt.xlabel("Engine Lifetime (Cycles)")
plt.ylabel("Number of Engines")
plt.title("Distribution of Engine Lifetimes - FD001")

plt.show()
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
engine_sensor_means = df.groupby("unit_id")[sensor_cols].mean()

print(F"Engine sensor means : {engine_sensor_means}")



sensor = "sensor_14"

engine_sensor_means[sensor].plot(
    kind="bar",
    figsize=(12, 5)
)

plt.xlabel("Engine (unit_id)")
plt.ylabel("Mean Sensor Value")
plt.title(f"Engine-level Variability - {sensor}")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
max_cycles = df.groupby("unit_id")["cycle"].max()

df["RUL"] = df.apply(
    lambda row: max_cycles[row["unit_id"]] - row["cycle"],
    axis=1
)

sensor_rul_corr = df[sensor_cols + ["RUL"]].corr()["RUL"].drop("RUL")

print(sensor_rul_corr.sort_values())

sensor = "sensor_9"

plt.figure(figsize=(8, 5))

plt.scatter(
    df[sensor],
    df["RUL"],
    alpha=0.3
)

plt.xlabel(sensor)
plt.ylabel("RUL")
plt.title(f"{sensor} vs RUL")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

sensor_rul_corr.plot(kind="barh")

plt.xlabel("Pearson Correlation with RUL")
plt.ylabel("Sensor")
plt.title("Sensor-RUL Correlation")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

sensor_corr = df[sensor_cols].corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    sensor_corr,
    cmap="coolwarm",
    center=0,
    annot=False
)

plt.title("Sensor-to-Sensor Correlation")
plt.tight_layout()
plt.show()

high_corr_pairs = (
    sensor_corr
    .where(np.triu(np.ones(sensor_corr.shape), k=1).astype(bool))
    .stack()
    .sort_values(ascending=False)
)

print(high_corr_pairs[high_corr_pairs > 0.9])
print(
    high_corr_pairs[high_corr_pairs.abs() > 0.9]
)
plt.figure(figsize=(10, 6))

sns.histplot(
    df["RUL"],
    bins=30,
    kde=True
)

plt.xlabel("RUL (Cycles)")
plt.ylabel("Number of Observations")
plt.title("RUL Distribution - FD001 Train")

plt.show()


Q1 = df[sensor_cols].quantile(0.25)
Q3 = df[sensor_cols].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Number of outliers for each sensor
outlier_counts = (
    ((df[sensor_cols] < lower_bound) |
     (df[sensor_cols] > upper_bound))
    .sum()
)

print(outlier_counts.sort_values(ascending=False))
outlier_percentage = (
    outlier_counts / len(df) * 100
).sort_values(ascending=False)

print(outlier_percentage)
# Sensor with the most outliers
#sensor = outlier_counts.idxmax()
sensor = "sensor_8"

print("Sensor with most outliers:", sensor)

# Select one engine
engine_id = 1

engine_data = df[df["unit_id"] == engine_id]

plt.figure(figsize=(10, 6))

plt.plot(
    engine_data["cycle"],
    engine_data[sensor],
    marker=".",
    linewidth=1
)

plt.xlabel("Cycle")
plt.ylabel(sensor)
plt.title(f"{sensor} Trend - Engine {engine_id}")

plt.show()
