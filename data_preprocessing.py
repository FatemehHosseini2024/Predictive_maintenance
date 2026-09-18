import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from utils import (
    load_data,
    validate_data,
    identify_constant_sensors,
    drop_columns,
    compute_rul,
)

VARIANCE_THRESHOLD = 0.005
SENSOR_EXACT_ZERO_THRESHOLD = 1e-10


def load_fd_data(dataset_name="FD001"):
    paths = {
        "FD001": ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt", 100),
        "FD002": ("train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt", 260),
    }

    if dataset_name not in paths:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(paths.keys())}")

    train_p, test_p, rul_p, expected_eng = paths[dataset_name]

    df_train, df_test, rul = load_data(train_p, test_p, rul_p)
    validate_data(df_train, train_p, expected_engines=expected_eng)
    validate_data(df_test, test_p, expected_engines=None)

    const_cols = identify_constant_sensors(
        df_train,
        variance_threshold=VARIANCE_THRESHOLD,
        exact_zero_threshold=SENSOR_EXACT_ZERO_THRESHOLD,
    )
    print(f"\nConstant columns to drop ({dataset_name}): {const_cols}")
    df_train = drop_columns(df_train, const_cols)
    df_test = drop_columns(df_test, const_cols)

    df_train = compute_rul(df_train, is_test=False)
    df_test = compute_rul(df_test, is_test=True, rul_last_cycles=rul["RUL"])

    print(f"\nFinal shapes ({dataset_name}):")
    print(f"  train: {df_train.shape}")
    print(f"  test: {df_test.shape}")

    return df_train, df_test, rul
