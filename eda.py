import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv(
    "train_FD001.txt",
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

df.columns = columns
numeric_cols = df.select_dtypes(include="number").columns

corr = df[numeric_cols].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap="coolwarm", center=0)

plt.title("Correlation Matrix")
plt.show()

cycle_corr = df[numeric_cols].corrwith(df["cycle"])

print(cycle_corr.sort_values())

