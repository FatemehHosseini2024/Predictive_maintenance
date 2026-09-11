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


def normalize_by_condition(train_df, target_df, sensor_cols, condition_col, n_conditions=None):
    """
    Normalizes sensor_cols per unique condition value in condition_col.
    - Fits scaler on train_df only (no leakage).
    - Applies same scaler to target_df grouped by matching conditions.
    - If target_df contains conditions not in train_df, falls back to a global scaler.
    """
    train_conds = sorted(train_df[condition_col].unique())
    if n_conditions is not None:
        train_conds = train_conds[:n_conditions]

    scalers = {}
    for cond in train_conds:
        mask = train_df[condition_col] == cond
        scaler = StandardScaler()
        scaler.fit(train_df.loc[mask, sensor_cols])
        scalers[cond] = scaler
        print(f"Fitted scaler for condition '{cond}' ({mask.sum()} train rows)")

    global_scaler = StandardScaler()
    global_scaler.fit(train_df[sensor_cols])

    def _apply(df):
        df = df.copy()
        for cond, scaler in scalers.items():
            mask = df[condition_col] == cond
            df.loc[mask, sensor_cols] = scaler.transform(df.loc[mask, sensor_cols])
        remaining_mask = ~df[condition_col].isin(scalers.keys())
        if remaining_mask.any():
            print(f"Warning: {remaining_mask.sum()} rows in target with unseen conditions -> using global scaler")
            df.loc[remaining_mask, sensor_cols] = global_scaler.transform(df.loc[remaining_mask, sensor_cols])
        return df

    train_df = _apply(train_df)
    target_df = _apply(target_df)

    print("\nCondition-aware standardization complete.")
    print("\nMean after standardization per condition (train):")
    print(train_df.groupby(condition_col)[sensor_cols].mean().abs().max().max())
    print("\nStd after standardization per condition (train):")
    print(train_df.groupby(condition_col)[sensor_cols].std().mean().mean())

    return train_df, target_df, scalers
