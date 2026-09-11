import pandas as pd
import numpy as np


def _rolling_slope(series, window):
    def slope_of_window(vals):
        n = len(vals)
        if n < 2:
            return np.nan
        x = np.arange(n)
        x_mean = x.mean()
        y_mean = vals.mean()
        numerator = ((x - x_mean) * (vals - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        return numerator / denominator if denominator != 0 else 0.0

    return series.rolling(window=window, min_periods=1).apply(slope_of_window, raw=True)


def compute_rolling_features(df, windows=[5, 20], sensor_prefix="sensor"):
    sensor_cols = [c for c in df.columns if c.startswith(sensor_prefix)]

    for window in windows:
        for col in sensor_cols:
            df[f"{col}_roll_mean_{window}"] = (
                df.groupby("unit_id")[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            df[f"{col}_roll_std_{window}"] = (
                df.groupby("unit_id")[col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
            df[f"{col}_roll_std_{window}"] = df[f"{col}_roll_std_{window}"].fillna(0)

    print(f"Rolling features added with window sizes = {windows}")
    print(f"Number of new columns: {len(windows) * 2 * len(sensor_cols)}")
    print(df.filter(like="_roll_").columns.tolist())

    return df


def compute_trend_features(df, sensors, windows=[5, 20]):
    for window in windows:
        for col in sensors:
            if col in df.columns:
                df[f"{col}_slope_{window}"] = (
                    df.groupby("unit_id")[col]
                    .transform(lambda s: _rolling_slope(s, window))
                )
                df[f"{col}_slope_{window}"] = df[f"{col}_slope_{window}"].fillna(0)

    print(f"Slope features added for {len(sensors)} sensors with window sizes = {windows}")
    print(f"Number of new columns: {len(windows) * len(sensors)}")
    print(df.filter(like="_slope_").columns.tolist())

    return df
