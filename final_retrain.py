
from feature_engineering import df_train,df_test
from baseline import feature_cols,RUL_CLIP
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
# ============================================================
# Retrain نهایی روی کل train (train_final + val) با بهترین hyperparameter ها
# ============================================================

# آماده‌سازی X, y روی کل df_train (همون ستون‌های feature قبلی)
X_train_full = df_train[feature_cols]
y_train_full = df_train["RUL"].clip(upper=RUL_CLIP)

# مدل نهایی با بهترین پارامترهایی که از RandomizedSearchCV به‌دست اومد
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
# ============================================================
# Evaluation: ارزیابی نهایی روی df_test
# ============================================================
X_test = df_test[feature_cols]
y_test = df_test["RUL"]
y_test_clipped = y_test.clip(upper=RUL_CLIP)

y_test_pred = final_model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test_clipped, y_test_pred))
test_mae = mean_absolute_error(y_test_clipped, y_test_pred)

print(f"Test RMSE: {test_rmse:.3f}")
print(f"Test MAE: {test_mae:.3f}")

# ============================================================
# جدول مقایسه‌ی سه‌مرحله‌ای: Baseline -> Validation -> Test
# ============================================================
comparison = pd.DataFrame({
    "Stage": ["Baseline (Linear Regression) - Validation", "Random Forest (tuned) - Validation", "Random Forest (final) - Test"],
    "RMSE": [16.534, 14.429, test_rmse],
    "MAE": [13.248, None, test_mae],  # MAE مدل tuned روی validation رو اگه داری جایگزین None کن
})
print("\n", comparison.to_string(index=False))