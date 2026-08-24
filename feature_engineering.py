from data_preprocessing import df_test , df_train
import pandas as pd
import numpy as np

# ============================================================
# Rolling features (mean & std) برای سنسورها - چند window مختلف
# ============================================================
windows = [5, 20]  # کوتاه‌مدت و بلندمدت

sensor_cols = [c for c in df_train.columns if c.startswith("sensor_")]

for window in windows:
    for col in sensor_cols:
        # --- Rolling mean ---
        df_train[f"{col}_roll_mean_{window}"] = (
            df_train.groupby("unit_id")[col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df_test[f"{col}_roll_mean_{window}"] = (
            df_test.groupby("unit_id")[col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # --- Rolling std ---
        df_train[f"{col}_roll_std_{window}"] = (
            df_train.groupby("unit_id")[col]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
        df_test[f"{col}_roll_std_{window}"] = (
            df_test.groupby("unit_id")[col]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

        # رکورد اول هر engine در roll_std همیشه NaN می‌شه، با 0 پر می‌کنیم
        df_train[f"{col}_roll_std_{window}"] = df_train[f"{col}_roll_std_{window}"].fillna(0)
        df_test[f"{col}_roll_std_{window}"] = df_test[f"{col}_roll_std_{window}"].fillna(0)

print(f"Rolling features اضافه شدن با window sizes = {windows}")
print(f"تعداد ستون‌های جدید: {len(windows) * 2 * len(sensor_cols)}")
print(df_train.filter(like="_roll_").columns.tolist())

# ============================================================
# Trend / Slope features برای سنسورهای منتخب (روند degradation واضح)
# ============================================================
trend_sensors = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11",
    "sensor_12", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]

def rolling_slope(series, window):
    """
    شیب رگرسیون خطی (y = slope*x + intercept) رو برای هر پنجره
    محاسبه می‌کنه. x همون شماره‌ی نسبی نقطه‌ی داخل پنجره‌ست (0, 1, 2, ...).
    برای پنجره‌هایی با کمتر از 2 نقطه، NaN برمی‌گرده.
    """
    def slope_of_window(vals):
        n = len(vals)
        if n < 2:
            return np.nan
        x = np.arange(n)
        # فرمول بسته‌ی شیب رگرسیون خطی ساده
        x_mean = x.mean()
        y_mean = vals.mean()
        numerator = ((x - x_mean) * (vals - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()
        return numerator / denominator if denominator != 0 else 0.0

    return series.rolling(window=window, min_periods=1).apply(slope_of_window, raw=True)


for window in windows:  # windows = [5, 20]
    for col in trend_sensors:
        df_train[f"{col}_slope_{window}"] = (
            df_train.groupby("unit_id")[col]
            .transform(lambda s: rolling_slope(s, window))
        )
        df_test[f"{col}_slope_{window}"] = (
            df_test.groupby("unit_id")[col]
            .transform(lambda s: rolling_slope(s, window))
        )

        # اولین رکورد هر engine (فقط 1 نقطه، شیب تعریف‌نشده) با 0 پر می‌شه یعنی "بدون تغییر"
        df_train[f"{col}_slope_{window}"] = df_train[f"{col}_slope_{window}"].fillna(0)
        df_test[f"{col}_slope_{window}"] = df_test[f"{col}_slope_{window}"].fillna(0)

print(f"Slope features اضافه شدن برای {len(trend_sensors)} سنسور با window sizes = {windows}")
print(f"تعداد ستون‌های جدید: {len(windows) * len(trend_sensors)}")
print(df_train.filter(like="_slope_").columns.tolist())
print(df_train.filter(like="_slope_").describe())
