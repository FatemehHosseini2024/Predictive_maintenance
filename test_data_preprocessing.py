"""Test script for data_preprocessing.load_fd_data on FD001 and FD002"""

import sys
sys.path.insert(0, ".")

from data_preprocessing import load_fd_data

def test_load_fd_data():
    print("=" * 60)
    print("Testing load_fd_data on FD001")
    print("=" * 60)
    try:
        df_train_1, df_test_1, rul_1, scaler_1, sensor_cols_1 = load_fd_data("FD001")
        assert df_train_1 is not None
        assert df_test_1 is not None
        assert rul_1 is not None
        assert scaler_1 is not None
        assert len(sensor_cols_1) > 0
        print(f"✓ FD001 loaded successfully")
        print(f"  train shape: {df_train_1.shape}")
        print(f"  test shape: {df_test_1.shape}")
        print(f"  sensor cols: {len(sensor_cols_1)}")
    except Exception as e:
        print(f"✗ FD001 test failed: {e}")

    print("\n" + "=" * 60)
    print("Testing load_fd_data on FD002")
    print("=" * 60)
    try:
        df_train_2, df_test_2, rul_2, scaler_2, sensor_cols_2 = load_fd_data("FD002")
        assert df_train_2 is not None
        assert df_test_2 is not None
        assert rul_2 is not None
        assert scaler_2 is not None
        assert len(sensor_cols_2) > 0
        print(f"✓ FD002 loaded successfully")
        print(f"  train shape: {df_train_2.shape}")
        print(f"  test shape: {df_test_2.shape}")
        print(f"  sensor cols: {len(sensor_cols_2)}")
    except Exception as e:
        print(f"✗ FD002 test failed: {e}")

    print("\n" + "=" * 60)
    print("Testing invalid dataset_name")
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
    test_load_fd_data()
