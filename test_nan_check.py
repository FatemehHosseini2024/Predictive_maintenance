"""Test script to check NaN counts in df_train and df_test from data_preprocessing"""

import sys
sys.path.insert(0, ".")

from data_preprocessing import df_train, df_test

print("=" * 60)
print("NaN counts in df_train")
print("=" * 60)
nan_train = df_train.isnull().sum()
nan_train_cols = nan_train[nan_train > 0]
if nan_train_cols.empty:
    print("No NaN values in df_train")
else:
    print(nan_train_cols)
    print(f"\nTotal NaN in df_train: {nan_train_cols.sum()}")

print(f"\ndf_train shape: {df_train.shape}")
print(f"df_train columns: {len(df_train.columns)}")

print("\n" + "=" * 60)
print("NaN counts in df_test")
print("=" * 60)
nan_test = df_test.isnull().sum()
nan_test_cols = nan_test[nan_test > 0]
if nan_test_cols.empty:
    print("No NaN values in df_test")
else:
    print(nan_test_cols)
    print(f"\nTotal NaN in df_test: {nan_test_cols.sum()}")

print(f"\ndf_test shape: {df_test.shape}")
print(f"df_test columns: {len(df_test.columns)}")

print("\n" + "=" * 60)
print("Columns with NaN in both train and test")
print("=" * 60)
common_nan = nan_train_cols.index.intersection(nan_test_cols.index)
if common_nan.empty:
    print("No common NaN columns")
else:
    print(common_nan)

print("\n" + "=" * 60)
print("Column dtypes")
print("=" * 60)
print(df_train.dtypes)
