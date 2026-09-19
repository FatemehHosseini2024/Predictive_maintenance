from baseline import get_baseline_results
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import randint, uniform
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

results = get_baseline_results(dataset_name="FD001", n_conditions=6)
df_train_final = results["df_train_final"]
df_val = results["df_val"]
X_train = results["X_train"]
y_train_clipped = results["y_train_clipped"]
X_val = results["X_val"]
y_val_clipped = results["y_val_clipped"]

groups_train = df_train_final["unit_id"]

gkf = GroupKFold(n_splits=5)

# ============================================================
# Random Forest
# ============================================================
def rf(n):
    rf_param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(5, 30),
        "min_samples_leaf": randint(1, 20),
        "min_samples_split": randint(2, 20),
        "max_features": ["sqrt", "log2", None],
    }

    rf_search = RandomizedSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_distributions=rf_param_dist,
        n_iter=n,
        scoring="neg_root_mean_squared_error",
        cv=gkf.split(X_train, y_train_clipped, groups=groups_train),
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    rf_search.fit(X_train, y_train_clipped)

    print("=== Random Forest ===")
    print("Best params:", rf_search.best_params_)
    print("Best CV RMSE:", -rf_search.best_score_)

    rf_best = rf_search.best_estimator_
    y_val_pred_rf = rf_best.predict(X_val)
    rf_val_rmse = np.sqrt(mean_squared_error(y_val_clipped, y_val_pred_rf))
    print(f"Validation RMSE (best RF): {rf_val_rmse:.3f}")


# ============================================================
# Gradient Boosting
# ============================================================
def gb():
    gb_param_dist = {
        "n_estimators": randint(100, 400),
        "max_depth": randint(2, 8),
        "learning_rate": uniform(0.01, 0.29),
        "min_samples_leaf": randint(1, 20),
        "subsample": uniform(0.6, 0.4),
    }

    gb_search = RandomizedSearchCV(
        estimator=GradientBoostingRegressor(random_state=42),
        param_distributions=gb_param_dist,
        n_iter=5,
        scoring="neg_root_mean_squared_error",
        cv=gkf.split(X_train, y_train_clipped, groups=groups_train),
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    gb_search.fit(X_train, y_train_clipped)

    print("\n=== Gradient Boosting ===")
    print("Best params:", gb_search.best_params_)
    print("Best CV RMSE:", -gb_search.best_score_)

    gb_best = gb_search.best_estimator_
    y_val_pred_gb = gb_best.predict(X_val)
    gb_val_rmse = np.sqrt(mean_squared_error(y_val_clipped, y_val_pred_gb))
    print(f"Validation RMSE (best GB): {gb_val_rmse:.3f}")


# ============================================================
# مقایسه‌ی نهایی
# ============================================================
#print("\n=== Final Comparison ===")
#print(f"Random Forest (tuned)     - Validation RMSE: {rf_val_rmse:.3f}")
#print(f"Gradient Boosting (tuned) - Validation RMSE: {gb_val_rmse:.3f}")
rf(20)
