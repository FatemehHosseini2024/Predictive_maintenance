from final_retrain import y_test_clipped, y_test_pred, final_model
from data_preprocessing import load_fd_data
from feature_engineering import add_feature_engineering
import matplotlib.pyplot as plt
import pandas as pd
from baseline import feature_cols

df_train, df_test, rul = load_fd_data("FD001")
df_train_fe, df_test_fe, scaler, sensor_cols = add_feature_engineering(df_train, df_test, dataset_name="FD001")

# ============================================================
# Error Analysis - بخش 1: توزیع کلی خطا (Residual Distribution)
# ============================================================
residuals = y_test_clipped - y_test_pred

print("آمار خلاصه‌ی residual ها (y_true - y_pred):")
print(residuals.describe())

print(f"\nمیانگین residual: {residuals.mean():.3f}")
print(f"(اگه مثبت باشه یعنی مدل به‌طور میانگین underestimate می‌کنه، اگه منفی باشه overestimate می‌کنه)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
axes[0].axvline(0, color="red", linestyle="--", label="zero error")
axes[0].axvline(residuals.mean(), color="green", linestyle="--", label=f"mean = {residuals.mean():.2f}")
axes[0].set_title("Histogram of Residuals")
axes[0].set_xlabel("Residual (y_true - y_pred)")
axes[0].legend()

axes[1].boxplot(residuals, vert=True)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Boxplot of Residuals")
axes[1].set_ylabel("Residual")

plt.tight_layout()
plt.show()

# ============================================================
# Error Analysis - بخش 2: خطا بر اساس بازه‌ی RUL
# ============================================================
error_by_rul = pd.DataFrame({
    "y_true": y_test_clipped.values,
    "y_pred": y_test_pred,
})
error_by_rul["residual"] = error_by_rul["y_true"] - error_by_rul["y_pred"]
error_by_rul["abs_error"] = error_by_rul["residual"].abs()

bins = [0, 25, 50, 75, 100, 125]
error_by_rul["RUL_bin"] = pd.cut(error_by_rul["y_true"], bins=bins, include_lowest=True)

summary = error_by_rul.groupby("RUL_bin", observed=True).agg(
    mean_residual=("residual", "mean"),
    mean_abs_error=("abs_error", "mean"),
    count=("residual", "count"),
)
print(summary)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

summary["mean_residual"].plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("Mean Residual by RUL Bin\n(مثبت = underestimate, منفی = overestimate)")
axes[0].set_ylabel("Mean Residual")

summary["mean_abs_error"].plot(kind="bar", ax=axes[1], color="darkorange")
axes[1].set_title("Mean Absolute Error by RUL Bin")
axes[1].set_ylabel("Mean Absolute Error")

plt.tight_layout()
plt.show()

# ============================================================
# Error Analysis - بخش 3: خطا بر اساس هر engine (per-unit error)
# ============================================================
error_by_unit = pd.DataFrame({
    "unit_id": df_test_fe["unit_id"].values,
    "y_true": y_test_clipped.values,
    "y_pred": y_test_pred,
})
error_by_unit["residual"] = error_by_unit["y_true"] - error_by_unit["y_pred"]
error_by_unit["abs_error"] = error_by_unit["residual"].abs()

per_engine_summary = error_by_unit.groupby("unit_id").agg(
    mean_residual=("residual", "mean"),
    mean_abs_error=("abs_error", "mean"),
    max_abs_error=("abs_error", "max"),
    n_records=("residual", "count"),
).reset_index()

worst_engines = per_engine_summary.sort_values("mean_abs_error", ascending=False)
print("10 تا engine با بیشترین میانگین خطا:")
print(worst_engines.head(10).to_string(index=False))

print("\n10 تا engine با کمترین میانگین خطا:")
print(per_engine_summary.sort_values("mean_abs_error").head(10).to_string(index=False))

print("\nآمار توزیع mean_abs_error بین engine ها:")
print(per_engine_summary["mean_abs_error"].describe())

fig, ax = plt.subplots(figsize=(12, 5))
sorted_summary = per_engine_summary.sort_values("mean_abs_error").reset_index(drop=True)
ax.bar(range(len(sorted_summary)), sorted_summary["mean_abs_error"], color="steelblue")
ax.set_xlabel("Engine (sorted by error)")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Per-Engine Mean Absolute Error (sorted)")
ax.axhline(per_engine_summary["mean_abs_error"].mean(), color="red", linestyle="--", label="میانگین کلی")
ax.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Error Analysis - بخش 5: Feature Importance
# ============================================================
importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": final_model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("Top 20 feature بر اساس اهمیت:")
print(importances.head(20).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 8))
top20 = importances.head(20).sort_values("importance")
ax.barh(top20["feature"], top20["importance"], color="steelblue")
ax.set_xlabel("Feature Importance")
ax.set_title("Top 20 Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

def classify_feature(name):
    if "_roll_mean_" in name:
        return "rolling_mean"
    elif "_roll_std_" in name:
        return "rolling_std"
    elif "_slope_" in name:
        return "slope"
    elif name in ["cycle"]:
        return "cycle"
    else:
        return "raw_sensor"

importances["feature_type"] = importances["feature"].apply(classify_feature)
type_summary = importances.groupby("feature_type")["importance"].sum().sort_values(ascending=False)
print("\nمجموع اهمیت بر اساس نوع feature:")
print(type_summary)

# ============================================================
# SHAP Feature Importance per RUL Bin (FD001) - sampled for speed
# ============================================================
import shap

print("\n" + "=" * 60)
print("SHAP Feature Importance on df_test (FD001)")
print("=" * 60)

SAMPLE_SIZE = 1000
df_test_sample = df_test_fe.sample(SAMPLE_SIZE, random_state=42)
X_sample = df_test_sample[feature_cols]

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=df_test_sample.index)

print(f"SHAP values computed on {SAMPLE_SIZE} samples")

bins = [0, 25, 50, 75, 100, 125]
df_test_sample["RUL_bin"] = pd.cut(df_test_sample["RUL"], bins=bins, include_lowest=True)

overall_mean_abs = shap_df.abs().mean().sort_values(ascending=False)
top_feats = overall_mean_abs.head(10).index.tolist()

comparison = {}
for bin_label in df_test_sample["RUL_bin"].cat.categories:
    mask = df_test_sample["RUL_bin"] == bin_label
    mean_shap = shap_df[mask].mean(axis=0)
    comparison[str(bin_label)] = mean_shap

comparison_df = pd.DataFrame(comparison)
comparison_df.loc["overall"] = shap_df.mean(axis=0)

print("\nMean SHAP per feature per RUL bin (top 10 features):")
for feat in top_feats:
    vals = []
    for col in comparison_df.columns:
        v = comparison_df.loc[feat, col]
        sign = "+" if v >= 0 else "-"
        vals.append(f"{sign}{v:.4f}")
    print(f"  {feat}: {vals}")

print("\nSHAP analysis complete.")
