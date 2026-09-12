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