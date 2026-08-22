import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================================================
# خواندن دیتاست
# ============================================================
df_train = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None
)
df_test = pd.read_csv(
    "test_FD001.txt",
    sep=r"\s+",
    header=None
)

columns = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)]
]

df_train.columns = columns
df_test.columns = columns

rul = pd.read_csv(
    "RUL_FD001.txt",
    header=None,
    names=["RUL"]
)

print(rul.head())
print("Shape:", rul.shape)
print("\nData types:")
print(rul.dtypes)


# ============================================================
# توابع کمکی برای چک dtype و whitespace/parsing
# ============================================================
def check_column_dtype_consistency(df):
    """
    برای هر ستون چک می‌کنه که آیا نوع داده‌ی سطرهاش یکدست هست یا نه.
    اگه dtype ستون 'object' باشه، یعنی pandas نتونسته یه نوع واحد
    (مثل int64 یا float64) براش تشخیص بده و این خودش نشونه‌ی
    مشکل توی داده‌هاست (مثلاً یه سلول رشته وسط ستون عددی).
    """
    problems = {}
    for col in df.columns:
        if df[col].dtype == object:
            # نوع پایتونی هر مقدار غیر-null رو جمع می‌کنیم
            types_found = df[col].dropna().apply(type).unique()
            problems[col] = {
                "dtype": str(df[col].dtype),
                "python_types": [t.__name__ for t in types_found],
            }
    return problems


def check_whitespace_issues(df):
    """
    توی ستون‌های رشته‌ای (object)، دنبال whitespace اضافه
    (فاصله‌ی ابتدا/انتهای مقدار) می‌گرده که می‌تونه نشونه‌ی
    مشکل parsing باشه (مثلاً هنگام delimiter نامناسب).
    """
    issues = {}
    for col in df.select_dtypes(include="object").columns:
        stripped = df[col].astype(str).str.strip()
        mismatch = (stripped != df[col].astype(str)).sum()
        if mismatch > 0:
            issues[col] = mismatch
    return issues





# ============================================================
# توابع اعتبارسنجی (Validation)
# ============================================================
def validate_train_test(df, name, expected_engines=100, raw_path=None):
    print(f"\n{'='*60}\nValidating {name}\n{'='*60}")
    issues = []

    # 1. تعداد ستون‌ها
    if df.shape[1] != len(columns):
        issues.append(f"Unexpected column count: {df.shape[1]} (expected {len(columns)})")

    # 2. تعداد engine ها
    n_engines = df["unit_id"].nunique()
    print(f"Rows: {df.shape[0]}, Engines (unique unit_id): {n_engines}")
    if n_engines != expected_engines:
        issues.append(f"Expected {expected_engines} engines, found {n_engines}")

    # 3. مقادیر گم‌شده
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        issues.append(f"Found {n_missing} missing values")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("No missing values.")

    # 4. نوع داده - همه‌ی ستون‌ها باید عددی باشن
    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        issues.append(f"Non-numeric columns found: {non_numeric}")
    else:
        print("All columns numeric.")

    # 5. محدوده‌ی منطقی مقادیر
    if (df["unit_id"] < 1).any():
        issues.append("Found unit_id values < 1")
    if (df["cycle"] < 1).any():
        issues.append("Found cycle values < 1")

    # 6. پیوستگی cycle برای هر engine (باید از 1 شروع بشه و بدون gap باشه)
    bad_engines = []
    for unit, g in df.groupby("unit_id"):
        cycles = g["cycle"].sort_values().to_numpy()
        expected = np.arange(1, len(cycles) + 1)
        if not np.array_equal(cycles, expected):
            bad_engines.append(unit)
    if bad_engines:
        issues.append(f"Engines with non-contiguous cycle sequence: {bad_engines}")
    else:
        print("All engines have contiguous cycle sequences (1..max_cycle).")

    # 7. ردیف‌های تکراری (unit_id, cycle)
    dup_count = df.duplicated(subset=["unit_id", "cycle"]).sum()
    if dup_count > 0:
        issues.append(f"Found {dup_count} duplicate (unit_id, cycle) rows")
    else:
        print("No duplicate (unit_id, cycle) rows.")

    # 8. یکدست بودن dtype هر ستون
    dtype_problems = check_column_dtype_consistency(df)
    if dtype_problems:
        for col, info in dtype_problems.items():
            issues.append(f"Column '{col}' has inconsistent/object dtype: {info}")
    else:
        print("All columns have a single consistent dtype.")

    # 9. whitespace issues داخل مقادیر (بعد از parse)
    ws_issues = check_whitespace_issues(df)
    if ws_issues:
        for col, count in ws_issues.items():
            issues.append(f"Column '{col}' has {count} values with leading/trailing whitespace")
    else:
        print("No whitespace issues found in parsed values.")

   
    return issues


def validate_rul(df, name, expected_count=100, raw_path=None):
    print(f"\n{'='*60}\nValidating {name}\n{'='*60}")
    issues = []

    # 1. تعداد ردیف‌ها
    if df.shape[0] != expected_count:
        issues.append(f"Expected {expected_count} rows, found {df.shape[0]}")

    # 2. مقادیر گم‌شده
    n_missing = df["RUL"].isnull().sum()
    if n_missing > 0:
        issues.append(f"Found {n_missing} missing values")
    else:
        print("No missing values.")

    # 3. نوع داده - باید عدد صحیح باشه
    is_integer = df["RUL"].dropna().apply(lambda x: float(x).is_integer())
    if not is_integer.all():
        issues.append("Found non-integer RUL values")
    else:
        print("All RUL values are integers.")

    # 4. محدوده‌ی منطقی - نباید منفی باشه
    negative = df[df["RUL"] < 0]
    if not negative.empty:
        issues.append(f"Found {len(negative)} negative RUL values")
    else:
        print("No negative RUL values.")

    # 5. یکدست بودن dtype ستون RUL
    dtype_problems = check_column_dtype_consistency(df)
    if dtype_problems:
        for col, info in dtype_problems.items():
            issues.append(f"Column '{col}' has inconsistent/object dtype: {info}")
    else:
        print("RUL column has a single consistent dtype.")

    # 6. whitespace issues
    ws_issues = check_whitespace_issues(df)
    if ws_issues:
        for col, count in ws_issues.items():
            issues.append(f"Column '{col}' has {count} values with leading/trailing whitespace")
    else:
        print("No whitespace issues found in parsed values.")

   
    return issues


# ============================================================
# اجرای اعتبارسنجی روی هر سه فایل
# ============================================================
all_issues = {
    "train": validate_train_test(df_train, "train_FD001.txt", raw_path="train_FD001.txt"),
    "test": validate_train_test(df_test, "test_FD001.txt", raw_path="test_FD001.txt"),
    "rul": validate_rul(rul, "RUL_FD001.txt", raw_path="RUL_FD001.txt"),
}

print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
for k, v in all_issues.items():
    status = "OK" if not v else f"{len(v)} issue(s)"
    print(f"{k}: {status}")


# ============================================================
# حذف سنسورهای ثابت (constant sensors)
# ============================================================
# این سنسورها توی EDA تشخیص داده شدن که مقدارشون در کل دیتاست
# ثابته (واریانس صفر یا نزدیک صفر) و در نتیجه هیچ سیگنالی برای
# مدل ندارن.
constant_features = ["setting_3","sensor_1", "sensor_5", "sensor_10", "sensor_16", "sensor_18", "sensor_19"]

df_train = df_train.drop(columns=constant_features)
df_test = df_test.drop(columns=constant_features)

print(f"\nDropped constant sensor columns: {constant_features}")
print(f"df_train shape after drop: {df_train.shape}")
print(f"df_test shape after drop: {df_test.shape}")
max_cycles = df_train.groupby("unit_id")["cycle"].max()

df_train["RUL"] = df_train.apply(
    lambda row: max_cycles[row["unit_id"]] - row["cycle"],
    axis=1
)
# ============================================================
# اضافه کردن RUL به df_test برای همه‌ی cycle ها (نه فقط آخرین)
# ============================================================
# محاسبه‌ی آخرین cycle ثبت‌شده برای هر engine در df_test
max_cycle_test = df_test.groupby("unit_id")["cycle"].max().reset_index()
max_cycle_test.columns = ["unit_id", "max_cycle"]

# اضافه کردن RUL_last_cycle (مقدار فایل RUL) به همون جدول، بر اساس ترتیب engine ها
max_cycle_test["RUL_last_cycle"] = rul["RUL"].values

# merge با df_test روی unit_id
df_test = df_test.merge(max_cycle_test, on="unit_id", how="left")

# محاسبه‌ی RUL برای تک‌تک ردیف‌ها با توجه به فاصله از آخرین cycle
df_test["RUL"] = df_test["RUL_last_cycle"] + (df_test["max_cycle"] - df_test["cycle"])

# حذف ستون‌های کمکی که دیگه لازم نیستن
df_test = df_test.drop(columns=["max_cycle", "RUL_last_cycle"])