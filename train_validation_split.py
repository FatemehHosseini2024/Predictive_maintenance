from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering
from utils import stratified_engine_split

df_train, df_test, rul = load_fd_data("FD001")
df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(df_train, df_test, dataset_name="FD001")

df_train_final, df_val = stratified_engine_split(
    df_train_fe,
    test_size=0.2,
    n_bins=4,
    random_state=42,
)
