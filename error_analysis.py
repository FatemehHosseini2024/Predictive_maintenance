from final_retrain import y_test_clipped,y_test_pred
import matplotlib.pyplot as plt

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