import pandas as pd

# خواندن دیتاست
df = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None
)
df_test =  pd.read_csv(
    "test_FD001.txt",
    sep=r"\s+",
    header=None
)
columns = [
    "unit_number",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)]
]

df.columns = columns

df_test.columns=columns
# نمایش نوع داده هر ستون
print(df.dtypes)
print(df.shape)
print(df_test.dtypes)
print(df_test.shape)
print(df.info())
print(df_test.info())
cycle_counts = df.groupby("unit_number")["cycle"].count()

print(f"cycle counts for train :{cycle_counts}")
print(df.isnull().sum())
duplicates = df[df.duplicated()]
print(f"duplicates :{duplicates}")
constant_features = df.columns[df.nunique() == 1]

print("Constant Features:")
print(constant_features)

