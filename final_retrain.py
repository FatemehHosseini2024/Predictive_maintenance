from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering
from baseline import feature_cols, RUL_CLIP
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd

df_train, df_test, rul = load_fd_data("FD001")
df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(df_train, df_test, dataset_name="FD001")

X_train_full = df_train_fe[feature_cols]
y_train_full = df_train_fe["RUL"].clip(upper=RUL_CLIP)

final_model = RandomForestRegressor(
    max_depth=24,
    max_features="sqrt",
    min_samples_leaf=3,
    min_samples_split=6,
    n_estimators=406,
    random_state=42,
    n_jobs=-1,
)

final_model.fit(X_train_full, y_train_full)

print("مدل نهایی روی کل train (train_final + validation) fit شد.")
print(f"تعداد نمونه‌های train نهایی: {X_train_full.shape[0]}")

X_test = df_test_fe[feature_cols]
y_test = df_test_fe["RUL"]
y_test_clipped = y_test.clip(upper=RUL_CLIP)

y_test_pred = final_model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test_clipped, y_test_pred))
test_mae = mean_absolute_error(y_test_clipped, y_test_pred)

print(f"Test RMSE: {test_rmse:.3f}")
print(f"Test MAE: {test_mae:.3f}")

comparison = pd.DataFrame({
    "Stage": ["Baseline (Linear Regression) - Validation", "Random Forest (tuned) - Validation", "Random Forest (final) - Test"],
    "RMSE": [16.534, 14.429, test_rmse],
    "MAE": [13.248, None, test_mae],
})
print("\n", comparison.to_string(index=False))
