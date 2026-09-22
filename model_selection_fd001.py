# ============================================================
# مقایسه‌ی چند مدل: Random Forest, Gradient Boosting, Ridge/Lasso
# ============================================================
from baseline import get_baseline_results
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import time
import pandas as pd

results = get_baseline_results("FD001")
baseline_model = results["baseline_model"]
X_train = results["X_train"]
y_train_clipped = results["y_train_clipped"]
X_val = results["X_val"]
y_val_clipped = results["y_val_clipped"]

models = {
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

model_results = []

for name, model in models.items():
    start = time.time()

    model.fit(X_train, y_train_clipped)

    y_pred = model.predict(X_val)
    elapsed = time.time() - start

    rmse = np.sqrt(mean_squared_error(y_val_clipped, y_pred))
    mae = mean_absolute_error(y_val_clipped, y_pred)

    model_results.append({
        "Model": name,
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "Train time (s)": round(elapsed, 2),
    })

model_results_df = pd.DataFrame(model_results).sort_values("RMSE").reset_index(drop=True)
print(model_results_df.to_string(index=False))

y_train_pred_rf = models["Random Forest"].predict(X_train)
rmse_train_rf = np.sqrt(mean_squared_error(y_train_clipped, y_train_pred_rf))
print(f"Random Forest - Train RMSE: {rmse_train_rf:.3f}")
rf_limited = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
rf_limited.fit(X_train, y_train_clipped)

y_train_pred = rf_limited.predict(X_train)
y_val_pred = rf_limited.predict(X_val)

print(f"Random Forest (limited) - Train RMSE: {np.sqrt(mean_squared_error(y_train_clipped, y_train_pred)):.3f}")
print(f"Random Forest (limited) - Validation RMSE: {np.sqrt(mean_squared_error(y_val_clipped, y_val_pred)):.3f}")
val_errors = pd.DataFrame({
    "y_true": y_val_clipped.values,
    "y_pred": rf_limited.predict(X_val),
})
val_errors["abs_error"] = (val_errors["y_true"] - val_errors["y_pred"]).abs()

val_errors["RUL_bin"] = pd.cut(val_errors["y_true"], bins=[0, 50, 100, 150, 200, 400])
print(val_errors.groupby("RUL_bin", observed=True)["abs_error"].agg(["mean", "count"]))
