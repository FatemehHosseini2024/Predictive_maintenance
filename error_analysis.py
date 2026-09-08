from final_retrain import y_test_clipped,y_test_pred
import matplotlib.pyplot as plt
import pandas as pd
from feature_engineering import df_train,df_test
from baseline import feature_cols
from final_retrain import final_model
# ============================================================
# Error Analysis - بخش 1: توزیع کلی خطا (Residual Distribution)
# ============================================================
residuals = y_test_clipped - y_test_pred

print("آمار خلاصه‌ی residual ها (y_true - y_pred):")
print(residuals.describe())

print(f"\nمیانگین residual: {residuals.mean():.3f}")
print(f"(اگه مثبت باشه یعنی مدل به‌طور میانگین underestimate می‌کنه، اگه منفی باشه overestimate می‌کنه)")

# رسم histogram و boxplot
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

# بازه‌بندی بر اساس RUL واقعی
bins = [0, 25, 50, 75, 100, 125]
error_by_rul["RUL_bin"] = pd.cut(error_by_rul["y_true"], bins=bins, include_lowest=True)

summary = error_by_rul.groupby("RUL_bin", observed=True).agg(
    mean_residual=("residual", "mean"),
    mean_abs_error=("abs_error", "mean"),
    count=("residual", "count"),
)
print(summary)

# نمودار برای دیدن بصری روند
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
    "unit_id": df_test["unit_id"].values,
    "y_true": y_test_clipped.values,
    "y_pred": y_test_pred,
})
error_by_unit["residual"] = error_by_unit["y_true"] - error_by_unit["y_pred"]
error_by_unit["abs_error"] = error_by_unit["residual"].abs()

# میانگین خطا برای هر engine
per_engine_summary = error_by_unit.groupby("unit_id").agg(
    mean_residual=("residual", "mean"),
    mean_abs_error=("abs_error", "mean"),
    max_abs_error=("abs_error", "max"),
    n_records=("residual", "count"),
).reset_index()

# مرتب‌سازی بر اساس بدترین عملکرد (بیشترین mean_abs_error)
worst_engines = per_engine_summary.sort_values("mean_abs_error", ascending=False)
print("10 تا engine با بیشترین میانگین خطا:")
print(worst_engines.head(10).to_string(index=False))

print("\n10 تا engine با کمترین میانگین خطا:")
print(per_engine_summary.sort_values("mean_abs_error").head(10).to_string(index=False))

# آمار کلی توزیع خطا بین engine ها
print("\nآمار توزیع mean_abs_error بین engine ها:")
print(per_engine_summary["mean_abs_error"].describe())

# نمودار: میانگین خطای هر engine، مرتب‌شده
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

# نمودار Top 20
fig, ax = plt.subplots(figsize=(10, 8))
top20 = importances.head(20).sort_values("importance")
ax.barh(top20["feature"], top20["importance"], color="steelblue")
ax.set_xlabel("Feature Importance")
ax.set_title("Top 20 Feature Importances - Random Forest")
plt.tight_layout()
plt.show()

# خلاصه بر اساس نوع feature (خام / rolling / slope)
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