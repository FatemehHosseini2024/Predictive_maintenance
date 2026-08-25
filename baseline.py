# ============================================================
# Baseline Model: Linear Regression
# ============================================================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from train_validation_split import df_train_final,df_val

# ستون‌هایی که نباید به‌عنوان feature وارد مدل بشن
exclude_cols = ["unit_id", "RUL"]
# اگر ستون کمکی life_pct در مرحله‌ی بررسی setting_1/2 اضافه شده و هنوز مونده، اونم کنار می‌ذاریم
if "life_pct" in df_train_final.columns:
    exclude_cols.append("life_pct")

feature_cols = [c for c in df_train_final.columns if c not in exclude_cols]

X_train = df_train_final[feature_cols]
y_train = df_train_final["RUL"]

X_val = df_val[feature_cols]
y_val = df_val["RUL"]
# ============================================================
# RUL Clipping
# ============================================================
RUL_CLIP = 125

y_train_clipped = y_train.clip(upper=RUL_CLIP)
y_val_clipped = y_val.clip(upper=RUL_CLIP)



# فیت مدل روی train
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train_clipped)

# پیش‌بینی روی validation
y_pred = baseline_model.predict(X_val)

# ارزیابی
rmse = np.sqrt(mean_squared_error(y_val_clipped, y_pred))
mae = mean_absolute_error(y_val_clipped, y_pred)

print(f"تعداد feature های استفاده‌شده: {len(feature_cols)}")
print(f"Baseline (Linear Regression) - Validation RMSE: {rmse:.3f}")
print(f"Baseline (Linear Regression) - Validation MAE: {mae:.3f}")
#print(y_train_clipped.describe())
#print(y_val_clipped.describe())