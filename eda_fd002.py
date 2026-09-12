from utils import (
    load_data, validate_data, find_constant_features,
    dataset_summary, missing_and_duplicates, cycle_counts_statistics,
)
from utils.preprocessing import compute_rul
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

DATASET_NAME = "FD002"
TRAIN_PATH = "train_FD002.txt"
TEST_PATH = "test_FD002.txt"
RUL_PATH = "RUL_FD002.txt"

df_train, df_test, rul = load_data(TRAIN_PATH, TEST_PATH, RUL_PATH)
validate_data(df_train, TRAIN_PATH, expected_engines=None)
validate_data(df_test, TEST_PATH, expected_engines=None)

df_train = compute_rul(df_train, is_test=False)
df_test = compute_rul(df_test, is_test=True, rul_last_cycles=rul["RUL"])

dataset_summary(df_train, f"{DATASET_NAME} Train")
missing_and_duplicates(df_train)
cycle_counts_statistics(df_train)

constant_cols = find_constant_features(df_train, exclude_cols=["RUL"])
print(f"\nConstant columns to consider dropping: {constant_cols}")

# KMeans clustering on settings to create condition_id
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
df_train["condition_id"] = kmeans.fit_predict(df_train[["setting_1", "setting_2", "setting_3"]])

# Condition ID distribution
if "condition_id" in df_train.columns:
    print(f"\nCondition ID distribution:")
    cond_dist = df_train["condition_id"].value_counts().sort_index()
    print(cond_dist)
    print(f"\nEngines per condition:")
    engines_per_cond = df_train.groupby("condition_id")["unit_id"].nunique()
    print(engines_per_cond)
    
    # Plot distribution
    plt.figure(figsize=(8, 5))
    cond_dist.plot(kind="bar")
    plt.xlabel("Condition ID")
    plt.ylabel("Number of Rows")
    plt.title(f"{DATASET_NAME} - Rows per Condition ID")
    plt.show()
    
    plt.figure(figsize=(8, 5))
    engines_per_cond.plot(kind="bar")
    plt.xlabel("Condition ID")
    plt.ylabel("Number of Engines")
    plt.title(f"{DATASET_NAME} - Engines per Condition ID")
    plt.show()

# Check for 6 distinct clusters in setting space
settings = ["setting_1", "setting_2", "setting_3"]
print(f"\nUnique setting combinations:")
setting_combos = df_train[settings].drop_duplicates()
print(setting_combos)
print(f"Number of unique combinations: {len(setting_combos)}")

# Pair plot of settings (all rows)
sns.pairplot(df_train[settings], plot_kws={'alpha': 0.5, 's': 10})
plt.suptitle(f"{DATASET_NAME} - Setting Pair Plot", y=1.02)
plt.show()

# 3D scatter plot (all rows)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_train["setting_1"], df_train["setting_2"], df_train["setting_3"], alpha=0.5, s=10)
ax.set_xlabel("setting_1")
ax.set_ylabel("setting_2")
ax.set_zlabel("setting_3")
ax.set_title(f"{DATASET_NAME} - 3D Setting Space")
plt.show()