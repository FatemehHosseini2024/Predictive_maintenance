import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def stratified_engine_split(df, test_size=0.2, n_bins=4, random_state=42):
    engine_life = df.groupby("unit_id")["cycle"].max().reset_index()
    engine_life.columns = ["unit_id", "max_cycle"]

    engine_life["life_bin"] = pd.qcut(engine_life["max_cycle"], q=n_bins, labels=False, duplicates="drop")

    train_units, val_units = train_test_split(
        engine_life["unit_id"],
        test_size=test_size,
        stratify=engine_life["life_bin"],
        random_state=random_state,
    )

    df_train_final = df[df["unit_id"].isin(train_units)].reset_index(drop=True)
    df_val = df[df["unit_id"].isin(val_units)].reset_index(drop=True)

    print(f"Number of engines in train: {train_units.nunique()}")
    print(f"Number of engines in validation: {val_units.nunique()}")
    print(f"Rows in train: {df_train_final.shape[0]}")
    print(f"Rows in validation: {df_val.shape[0]}")

    train_life = engine_life[engine_life["unit_id"].isin(train_units)]["max_cycle"]
    val_life = engine_life[engine_life["unit_id"].isin(val_units)]["max_cycle"]

    print("\nMax cycle - train engines:")
    print(train_life.describe())
    print("\nMax cycle - validation engines:")
    print(val_life.describe())

    return df_train_final, df_val
