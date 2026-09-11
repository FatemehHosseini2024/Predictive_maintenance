import pandas as pd
import numpy as np


COLUMNS = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


def load_data(train_path, test_path, rul_path):
    df_train = pd.read_csv(train_path, sep=r"\s+", header=None)
    df_test = pd.read_csv(test_path, sep=r"\s+", header=None)
    rul = pd.read_csv(rul_path, header=None, names=["RUL"])

    df_train.columns = COLUMNS
    df_test.columns = COLUMNS

    return df_train, df_test, rul


def validate_data(df, name, expected_engines=100, expected_columns=None):
    if expected_columns is None:
        expected_columns = COLUMNS

    print(f"\n{'='*60}\nValidating {name}\n{'='*60}")
    issues = []

    if df.shape[1] != len(expected_columns):
        issues.append(f"Unexpected column count: {df.shape[1]} (expected {len(expected_columns)})")

    n_engines = df["unit_id"].nunique()
    print(f"Rows: {df.shape[0]}, Engines (unique unit_id): {n_engines}")
    if n_engines != expected_engines:
        issues.append(f"Expected {expected_engines} engines, found {n_engines}")

    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        issues.append(f"Found {n_missing} missing values")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("No missing values.")

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        issues.append(f"Non-numeric columns found: {non_numeric}")
    else:
        print("All columns numeric.")

    if (df["unit_id"] < 1).any():
        issues.append("Found unit_id values < 1")
    if (df["cycle"] < 1).any():
        issues.append("Found cycle values < 1")

    bad_engines = []
    for unit, g in df.groupby("unit_id"):
        cycles = g["cycle"].sort_values().to_numpy()
        expected = np.arange(1, len(cycles) + 1)
        if not np.array_equal(cycles, expected):
            bad_engines.append(unit)
    if bad_engines:
        issues.append(f"Engines with non-contiguous cycle sequence: {bad_engines}")
    else:
        print("All engines have contiguous cycle sequences (1..max_cycle).")

    dup_count = df.duplicated(subset=["unit_id", "cycle"]).sum()
    if dup_count > 0:
        issues.append(f"Found {dup_count} duplicate (unit_id, cycle) rows")
    else:
        print("No duplicate (unit_id, cycle) rows.")

    return issues


def identify_constant_sensors(df, variance_threshold=0.005, exact_zero_threshold=1e-10, sensor_prefix="sensor", setting_cols=None):
    if setting_cols is None:
        setting_cols = ["setting_1", "setting_2", "setting_3"]

    sensor_cols = [c for c in df.columns if c.startswith(sensor_prefix)]
    columns_to_check = sensor_cols + setting_cols
    constant_cols = []

    for col in columns_to_check:
        if col not in df.columns:
            continue
        variance = df[col].var()

        if col in sensor_cols:
            is_constant = abs(variance) < exact_zero_threshold
        elif col in setting_cols:
            is_constant = abs(variance) < variance_threshold
        else:
            is_constant = False

        if is_constant:
            print(f"Column '{col}' has variance {variance:.6f}, marked as constant")
            constant_cols.append(col)
            
            

    return constant_cols


def drop_columns(df, cols):
    cols_to_drop = [c for c in cols if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"Dropped columns: {cols_to_drop}")
    else:
        print("No columns to drop (empty or all already absent).")
    return df


def compute_rul(df, is_test=False, rul_last_cycles=None):
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")

    if is_test:
        if rul_last_cycles is None:
            raise ValueError("rul_last_cycles must be provided for test set RUL computation")
        df = df.copy()
        if not isinstance(rul_last_cycles, pd.Series):
            rul_last_cycles = pd.Series(rul_last_cycles)
        df["RUL_last_cycle"] = df["unit_id"].map(rul_last_cycles)
        df["RUL"] = df["RUL_last_cycle"] + (max_cycles - df["cycle"])
        df = drop_columns(df, ["RUL_last_cycle"])
    else:
        df = df.copy()
        df["RUL"] = max_cycles - df["cycle"]

    return df


def clip_rul(series, clip_value=125):
    return series.clip(upper=clip_value)
