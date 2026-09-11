from data_preprocessing import df_train, df_test
from train_validation_split import df_train_final, df_val
from utils import clip_rul
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

RUL_CLIP = 125
exclude_cols = ["unit_id", "RUL"]
feature_cols = [c for c in df_train_final.columns if c not in exclude_cols]

X_train = df_train_final[feature_cols]
y_train = df_train_final["RUL"]
y_train_clipped = clip_rul(y_train, RUL_CLIP)

X_val = df_val[feature_cols]
y_val = df_val["RUL"]
y_val_clipped = clip_rul(y_val, RUL_CLIP)

baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train_clipped)

y_pred = baseline_model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val_clipped, y_pred))
mae = mean_absolute_error(y_val_clipped, y_pred)

print(f"Number of features used: {len(feature_cols)}")
print(f"Baseline (Linear Regression) - Validation RMSE: {rmse:.3f}")
print(f"Baseline (Linear Regression) - Validation MAE: {mae:.3f}")
