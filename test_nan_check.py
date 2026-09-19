from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering

df_train, df_test, rul = load_fd_data("FD002")
df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(df_train, df_test, dataset_name="FD002")

print("=" * 60)
print("NaN counts in df_train_fe")
print("=" * 60)
nan_train = df_train_fe.isnull().sum()
nan_train_cols = nan_train[nan_train > 0]
if nan_train_cols.empty:
    print("No NaN values in df_train_fe")
else:
    print(nan_train_cols)
    print(f"\nTotal NaN in df_train_fe: {nan_train_cols.sum()}")

print(f"\ndf_train_fe shape: {df_train_fe.shape}")

print("\n" + "=" * 60)
print("NaN counts in df_test_fe")
print("=" * 60)
nan_test = df_test_fe.isnull().sum()
nan_test_cols = nan_test[nan_test > 0]
if nan_test_cols.empty:
    print("No NaN values in df_test_fe")
else:
    print(nan_test_cols)
    print(f"\nTotal NaN in df_test_fe: {nan_test_cols.sum()}")

print(f"\ndf_test_fe shape: {df_test_fe.shape}")

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
print(df_train_fe.dtypes)
