# ============================================================
# Train / Validation split (stratified by engine lifetime, at unit_id level)
# ============================================================
from sklearn.model_selection import train_test_split
from feature_engineering import df_train
import pandas as pd
import numpy as np

# 1. طول عمر (max_cycle) هر engine در df_train
engine_life = df_train.groupby("unit_id")["cycle"].max().reset_index()
engine_life.columns = ["unit_id", "max_cycle"]

# 2. تقسیم طول عمرها به چند دسته (bin) برای stratification
#    تعداد bin ها رو می‌شه تنظیم کرد؛ اینجا 4 دسته (کوتاه تا خیلی بلند) در نظر گرفتیم
engine_life["life_bin"] = pd.qcut(engine_life["max_cycle"], q=4, labels=False)

# 3. stratified split در سطح unit_id (نه در سطح ردیف!)
train_units, val_units = train_test_split(
    engine_life["unit_id"],
    test_size=0.2,
    stratify=engine_life["life_bin"],
    random_state=42,
)

# 4. اعمال این تقسیم روی df_train اصلی
df_val = df_train[df_train["unit_id"].isin(val_units)].reset_index(drop=True)
df_train_final = df_train[df_train["unit_id"].isin(train_units)].reset_index(drop=True)

print(f"تعداد engine در train: {train_units.nunique()}")
print(f"تعداد engine در validation: {val_units.nunique()}")
print(f"تعداد ردیف در train: {df_train_final.shape[0]}")
print(f"تعداد ردیف در validation: {df_val.shape[0]}")

# 5. چک توزیع طول عمر بین train و validation (باید مشابه باشن)
print("\nMiddle max_cycle - train engines:")
print(engine_life[engine_life["unit_id"].isin(train_units)]["max_cycle"].describe())
print("\nMiddle max_cycle - validation engines:")
print(engine_life[engine_life["unit_id"].isin(val_units)]["max_cycle"].describe())
