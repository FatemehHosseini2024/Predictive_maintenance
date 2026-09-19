from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering
from utils import stratified_engine_split, clip_rul
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

RUL_CLIP = 125


def get_baseline_results(dataset_name="FD001", n_conditions=None):
    df_train, df_test, rul = load_fd_data(dataset_name)
    df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(
        df_train, df_test, dataset_name=dataset_name, n_conditions=n_conditions
    )

    df_train_final, df_val = stratified_engine_split(
        df_train_fe,
        test_size=0.2,
        n_bins=4,
        random_state=42,
    )

    exclude_cols = ["unit_id", "RUL"]
    feature_cols = [c for c in df_train_final.columns if c not in exclude_cols]

    X_train = df_train_final[feature_cols]
    y_train = df_train_final["RUL"]
    y_train_clipped = clip_rul(y_train, RUL_CLIP)

    X_val = df_val[feature_cols]
    y_val = df_val["RUL"]
    y_val_clipped = clip_rul(y_val, RUL_CLIP)

    if dataset_name == "FD002":
        baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_label = "Random Forest"
    else:
        baseline_model = LinearRegression()
        model_label = "Linear Regression"

    baseline_model.fit(X_train, y_train_clipped)

    y_pred = baseline_model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val_clipped, y_pred))
    mae = mean_absolute_error(y_val_clipped, y_pred)

    print(f"Number of features used: {len(feature_cols)}")
    print(f"Baseline ({model_label}) - Validation RMSE: {rmse:.3f}")
    print(f"Baseline ({model_label}) - Validation MAE: {mae:.3f}")

    return {
        "df_train_final": df_train_final,
        "df_val": df_val,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "y_train_clipped": y_train_clipped,
        "X_val": X_val,
        "y_val_clipped": y_val_clipped,
        "baseline_model": baseline_model,
        "rmse": rmse,
        "mae": mae,
    }
    
get_baseline_results(dataset_name="FD002", n_conditions=6)
