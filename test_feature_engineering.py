"""Test script for feature_engineering.add_feature_engineering"""

import sys
sys.path.insert(0, ".")

from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering

def test_feature_engineering():
    print("=" * 60)
    print("Test 1: FD001 without conditions")
    print("=" * 60)
    try:
        df_train, df_test, rul = load_fd_data("FD001")
        df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(
            df_train, df_test, dataset_name="FD001"
        )
        assert df_train_fe is not None
        assert df_test_fe is not None
        assert scaler is not None
        assert len(sensor_cols) > 0
        assert "condition_id" not in df_train_fe.columns
        assert "condition_id" not in df_test_fe.columns
        print(f"✓ FD001 without conditions")
        print(f"  train: {df_train_fe.shape}")
        print(f"  test: {df_test_fe.shape}")
        print(f"  sensor cols: {len(sensor_cols)}")
        print(f"  NaN in train: {df_train_fe.isnull().sum().sum()}")
        print(f"  NaN in test: {df_test_fe.isnull().sum().sum()}")
    except Exception as e:
        print(f"✗ FD001 without conditions failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test 2: FD002 with 6 conditions")
    print("=" * 60)
    try:
        df_train, df_test, rul = load_fd_data("FD002")
        df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(
            df_train, df_test, dataset_name="FD002", n_conditions=6
        )
        assert df_train_fe is not None
        assert df_test_fe is not None
        assert "condition_id" in df_train_fe.columns
        assert "condition_id" in df_test_fe.columns
        assert df_train_fe["condition_id"].nunique() <= 6
        assert df_test_fe["condition_id"].nunique() <= 6
        print(f"✓ FD002 with 6 conditions")
        print(f"  train: {df_train_fe.shape}")
        print(f"  test: {df_test_fe.shape}")
        print(f"  conditions: {df_train_fe['condition_id'].nunique()}")
        print(f"  sensor cols: {len(sensor_cols)}")
        print(f"  NaN in train: {df_train_fe.isnull().sum().sum()}")
        print(f"  NaN in test: {df_test_fe.isnull().sum().sum()}")
    except Exception as e:
        print(f"✗ FD002 with conditions failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test 3: Verify rolling and slope features exist")
    print("=" * 60)
    try:
        df_train, df_test, rul = load_fd_data("FD001")
        df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(
            df_train, df_test, dataset_name="FD001"
        )
        roll_cols = [c for c in df_train_fe.columns if "_roll_" in c]
        slope_cols = [c for c in df_train_fe.columns if "_slope_" in c]
        assert len(roll_cols) > 0, "No rolling features found"
        assert len(slope_cols) > 0, "No slope features found"
        assert df_train_fe.shape[1] == df_test_fe.shape[1], \
            f"Column mismatch: train={df_train_fe.shape[1]}, test={df_test_fe.shape[1]}"
        print(f"✓ Rolling and slope features exist")
        print(f"  rolling features: {len(roll_cols)}")
        print(f"  slope features: {len(slope_cols)}")
        print(f"  total features: {df_train_fe.shape[1]}")
    except Exception as e:
        print(f"✗ Feature verification failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test 4: Invalid dataset_name")
    print("=" * 60)
    try:
        load_fd_data("FD999")
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    test_feature_engineering()
