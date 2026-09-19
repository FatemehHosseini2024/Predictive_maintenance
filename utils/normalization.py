import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def normalize_global(train_df, target_df, sensor_cols):
    scaler = StandardScaler()
    scaler.fit(train_df[sensor_cols])
    train_df[sensor_cols] = scaler.transform(train_df[sensor_cols])
    target_df[sensor_cols] = scaler.transform(target_df[sensor_cols])

    print("Standardization done on sensor columns (global)")
    print(sensor_cols)
    print("\nMean after standardization (should be ~0):")
    print(train_df[sensor_cols].mean())
    print("\nStd after standardization (should be ~1):")
    print(train_df[sensor_cols].std())

    return train_df, target_df, scaler


def normalize_by_condition(train_df, target_df, sensor_cols, condition_col, n_conditions=None, epsilon=1e-8):
    train_conds = sorted(train_df[condition_col].unique())
    if n_conditions is not None:
        train_conds = train_conds[:n_conditions]

    scalers = {}
    for cond in train_conds:
        mask = train_df[condition_col] == cond
        mean = train_df.loc[mask, sensor_cols].mean()
        std = train_df.loc[mask, sensor_cols].std()
        std = std.where(std > epsilon, epsilon)
        scalers[cond] = {"mean": mean, "std": std}
        print(f"Fitted scaler for condition '{cond}' ({mask.sum()} train rows)")

    global_mean = train_df[sensor_cols].mean()
    global_std = train_df[sensor_cols].std()
    global_std = global_std.where(global_std > epsilon, epsilon)

    def _apply(df):
        df = df.copy()
        df[sensor_cols] = df[sensor_cols].astype(float)
        for cond, scaler in scalers.items():
            mask = df[condition_col] == cond
            df.loc[mask, sensor_cols] = (df.loc[mask, sensor_cols] - scaler["mean"]) / scaler["std"]
        remaining_mask = ~df[condition_col].isin(scalers.keys())
        if remaining_mask.any():
            print(f"Warning: {remaining_mask.sum()} rows in target with unseen conditions -> using global scaler")
            df.loc[remaining_mask, sensor_cols] = (df.loc[remaining_mask, sensor_cols] - global_mean) / global_std
        return df

    train_df = _apply(train_df)
    target_df = _apply(target_df)

    print("\nCondition-aware standardization complete.")
    print("\nMean after standardization per condition (train):")
    print(train_df.groupby(condition_col)[sensor_cols].mean().abs().max().max())
    print("\nStd after standardization per condition (train):")
    print(train_df.groupby(condition_col)[sensor_cols].std().mean().mean())

    return train_df, target_df, scalers
